"""16 step 뒤 실제 이미지를 정책 입력과 같은 33장 시퀀스 안에서 VAE로 인코딩해, 예측 latent와 비교한다.

정책(get_action)이 만드는 시퀀스와 같은 순서로 이미지를 넣는다.
  0 빈 1장 | 1 관절 자리(빈) ×4 | 2 현재 손목 ×4 | 3 현재 3인칭 ×4 | 4 행동 자리(빈) ×4
  5 미래 관절 자리(빈) ×4 | 6 실제 미래 손목 ×4 | 7 실제 미래 3인칭 ×4 | 8 가치 자리(빈) ×4
Wan2.1 VAE는 시간 방향으로 causal이라, 6, 7번 latent는 앞 프레임(현재 이미지 등)의 영향을 받는다.
정책도 같은 앞 프레임을 조건으로 6, 7번을 생성했으므로 같은 조건에서 비교하게 된다.

이미지 처리도 정책과 같다: JPEG(q95) → 224 리사이즈 → 90% 가운데 자르기 → 224 → uint8/127.5−1 → model.encode.
HDF5의 i번째 관측은 step t = i + 10이다 (처음 10 step 대기는 저장 안 됨).

출력: 에피소드마다 <에피소드>_actual.npz
  chunk_t (K',)            비교한 chunk의 추론 step (16 step 뒤가 에피소드 안에 있는 chunk만)
  z_actual (K', 16, 2, 28, 28) float16   실제 미래 손목, 3인칭 latent
  d_img (K',)               (예측 − 실제)² 평균, 손목과 3인칭 평균
  d_img_wrist, d_img_primary (K',)

사용 (서버): python wac/encode_actual.py <데이터 폴더> [--gpu 2] [--shard 0 --num_shards 2]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("data_dir")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num_shards", type=int, default=1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--overwrite", action="store_true")
a = p.parse_args()

# 공식 평가와 같은 설정으로 정책을 불러온다 (VAE만 쓰지만 공식 로더가 모델 전체를 만든다)
from cosmos_policy.experiments.robot.libero import run_libero_eval as rle  # noqa: E402
from cosmos_policy.experiments.robot import cosmos_utils as cu  # noqa: E402
import draccus  # noqa: E402

CK = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
cfg = draccus.parse(config_class=rle.PolicyEvalConfig, args=[
    "--config", "cosmos_predict2_2b_480p_libero__inference_only", "--ckpt_path", CK,
    "--config_file", "cosmos_policy/config/config.py",
    "--use_wrist_image", "True", "--use_proprio", "True", "--normalize_proprio", "True",
    "--unnormalize_actions", "True",
    "--dataset_stats_path", f"{CK}/libero_dataset_statistics.json",
    "--t5_text_embeddings_path", f"{CK}/libero_t5_embeddings.pkl",
    "--trained_with_image_aug", "True", "--chunk_size", "16", "--num_open_loop_steps", "16",
    "--task_suite_name", "libero_spatial", "--use_jpeg_compression", "True", "--flip_images", "True",
])
model, _ = cu.get_model(cfg)
model.eval()

WAIT = 10       # 처음 대기 step (HDF5에 저장 안 됨)
CHUNK = 16
DUP = cu.COSMOS_TEMPORAL_COMPRESSION_FACTOR  # 4


def decode_jpeg(b):
    from PIL import Image
    import io
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))


def prep(img):
    """정책과 같은 전처리. prepare_images_for_model은 (T,H,W,C)를 받는다."""
    return cu.prepare_images_for_model([img], cfg)[0]


def sequence(wrist_t, prim_t, wrist_f, prim_f):
    blank = np.zeros_like(prim_t)
    rep = lambda x: np.repeat(x[None], DUP, axis=0)
    frames = [blank[None], rep(blank), rep(wrist_t), rep(prim_t), rep(blank),
              rep(blank), rep(wrist_f), rep(prim_f), rep(blank)]
    return np.concatenate(frames, axis=0)  # (33, H, W, C)


@torch.no_grad()
def encode(seqs):
    x = torch.from_numpy(np.stack(seqs)).permute(0, 4, 1, 2, 3).cuda()           # (B, C, T, H, W) uint8
    x = x.to(**model.tensor_kwargs) / 127.5 - 1.0
    return model.encode(x).float().cpu().numpy()                                   # (B, 16, 9, 28, 28)


eps = sorted(glob.glob(os.path.join(a.data_dir, "*", "*", "D*", "init*_*.npz")))
eps = [e for e in eps if not e.endswith("_actual.npz")][a.shard::a.num_shards]
print(f"에피소드 {len(eps)}개 (shard {a.shard}/{a.num_shards})", flush=True)

for n, ep in enumerate(eps):
    out = ep[:-4] + "_actual.npz"
    if os.path.exists(out) and not a.overwrite:
        continue
    z = np.load(ep)
    with h5py.File(ep[:-4] + ".hdf5") as h:
        prim = h["primary_images_jpeg"][:]
        wrist = h["wrist_images_jpeg"][:]
    T = len(prim)
    ts, seqs, idx = [], [], []
    for k, t in enumerate(z["chunk_t"]):
        i_now, i_fut = int(t) - WAIT, int(t) + CHUNK - WAIT
        if i_now < 0 or i_fut >= T:
            continue
        seqs.append(sequence(prep(decode_jpeg(wrist[i_now])), prep(decode_jpeg(prim[i_now])),
                             prep(decode_jpeg(wrist[i_fut])), prep(decode_jpeg(prim[i_fut]))))
        ts.append(int(t))
        idx.append(k)
    if not seqs:
        np.savez_compressed(out, chunk_t=np.zeros(0, int))
        continue
    lat = np.concatenate([encode(seqs[s:s + a.batch]) for s in range(0, len(seqs), a.batch)])  # (K', 16, 9, 28, 28)
    actual = lat[:, :, 6:8]                                      # 실제 미래 손목, 3인칭
    pred = z["latent"][idx][:, :, 1:3].astype(np.float32)        # 저장한 프레임 5~8 중 6, 7
    d_w = ((pred[:, :, 0] - actual[:, :, 0]) ** 2).mean(axis=(1, 2, 3))
    d_p = ((pred[:, :, 1] - actual[:, :, 1]) ** 2).mean(axis=(1, 2, 3))
    np.savez_compressed(out, chunk_t=np.array(ts), z_actual=actual.astype(np.float16),
                        d_img=(d_w + d_p) / 2, d_img_wrist=d_w, d_img_primary=d_p)
    if n % 20 == 0:
        print(f"[{n + 1}/{len(eps)}] {os.path.relpath(ep, a.data_dir)} chunk {len(ts)}개, "
              f"d_img 평균 {((d_w + d_p) / 2).mean():.4f}", flush=True)
print("끝", flush=True)
