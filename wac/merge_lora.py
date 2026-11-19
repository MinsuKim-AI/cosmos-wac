"""파인튜닝 DCP 체크포인트(LoRA)를 공개 체크포인트와 같은 형식의 .pt로 합친다.

W' = W + (alpha / r) · B · A  (rank 8, alpha 8 → 배율 1)
LoRA가 없는 키는 DCP 값을 쓰고, DCP에 없는 키는 공개 체크포인트 값을 그대로 둔다.

사용: python wac/merge_lora.py <DCP model 폴더> <공개 .pt> <출력 .pt> [--alpha 8 --rank 8]
"""

import argparse

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

p = argparse.ArgumentParser()
p.add_argument("dcp_dir")
p.add_argument("base_pt")
p.add_argument("out_pt")
p.add_argument("--alpha", type=float, default=8)
p.add_argument("--rank", type=float, default=8)
a = p.parse_args()

base = torch.load(a.base_pt, map_location="cpu", weights_only=False)
md = dcp.FileSystemReader(a.dcp_dir).read_metadata().state_dict_metadata
sd = {k: torch.empty(m.size, dtype=m.properties.dtype) for k, m in md.items() if isinstance(m, TensorStorageMetadata)}
dcp.load(sd, storage_reader=dcp.FileSystemReader(a.dcp_dir), no_dist=True)

scale = a.alpha / a.rank
out, merged, copied = dict(base), 0, 0
for k, v in sd.items():
    if ".lora_" in k:
        continue
    if k in base and base[k].shape == v.shape:
        out[k] = v.to(base[k].dtype)
        copied += 1
for k in sd:
    if ".lora_A.default.weight" in k:
        stem = k.replace(".lora_A.default.weight", "")
        A_, B_ = sd[k].float(), sd[stem + ".lora_B.default.weight"].float()
        W = out[stem + ".weight"]
        out[stem + ".weight"] = (W.float() + scale * (B_ @ A_)).to(W.dtype)
        merged += 1
print(f"DCP 키 {len(sd)}, 공개 키 {len(base)}, 덮어쓴 키 {copied}, LoRA 합친 층 {merged}")
torch.save(out, a.out_pt)
print("저장:", a.out_pt)
