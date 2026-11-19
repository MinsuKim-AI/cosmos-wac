"""LIBERO 평가에 실패 유도 교란과 시뮬레이터 화면 표시를 붙인 실행기 (PROPOSAL 5.1).

공식 run_libero_eval.py는 고치지 않는다. 그 모듈의 get_libero_env를 바꿔치기해서
환경을 PerturbedEnv로 감싼 뒤, 공식 eval_libero()를 그대로 호출한다.

교란 (에피소드마다 초기 상태를 세팅한 직후에 적용):
  --obj_xy_noise R    : 움직일 수 있는 물체 전부의 x, y를 [-R, R] m 균등분포만큼 옮긴다.
  --cam_shift D       : 카메라를 무작위 방향으로 D m 평행이동한다 (방향은 그대로).
  --wrist_roll_deg A  : 손목 카메라를 광축 기준으로 ±A도 회전한다.
  --cam_yaw_deg A     : agentview 카메라를 테이블 중심 수직축 기준으로 ±A도 회전 (부호 무작위).
  --perturb_seed S    : 교란 난수 seed. 정책 seed(--seed)와 따로 둔다.

화면:
  --show True         : 실행 중 3인칭 + 손목 화면을 창에 띄운다 (WSLg 필요).
  --video_dir DIR     : 에피소드마다 mp4 저장. 파일 이름에 과제, 순번, 성공 여부가 들어간다.

나머지 인자는 공식 평가 스크립트로 그대로 넘어간다.
"""

import glob
import json
import os
import re
import sys
import time

import numpy as np

WAC_ARGS = {
    "--obj_xy_noise": float,
    "--cam_shift": float,
    "--wrist_roll_deg": float,
    "--cam_yaw_deg": float,
    "--perturb_seed": int,
    "--show": lambda s: s.lower() in ("1", "true", "yes"),
    "--video_dir": str,
    "--perturb_log": str,
    "--init_offset": int,
    "--episode_dir": str,    # 묶음 폴더. 이 아래에 D01_<지시문>/ 폴더들이 생긴다
    "--live_dir": str,       # 원격 감시용: 몇 step마다 최신 화면을 <live_dir>/<live_name>.jpg로 덮어쓴다
    "--live_name": str,
}

# 지금 돌고 있는 에피소드의 환경. get_action 가로채기에서 chunk 값을 여기에 쌓는다.
CURRENT_ENV = None
# 공식 HDF5를 지시문 폴더로 옮기는 객체 (--data_collection True일 때만)
MOVER = None

# LIBERO latent 프레임 순서 (cosmos_utils.get_action 주석과 같음)
# 0 빈 프레임, 1 현재 관절, 2 현재 손목, 3 현재 3인칭, 4 행동, 5 미래 관절, 6 미래 손목, 7 미래 3인칭, 8 가치
SAVE_FRAMES = [5, 6, 7, 8]
PROPRIO_DIM = 9


def untile(frame, dim):
    """latent 프레임(C, H, W)에 반복해서 채운 벡터를 평균 내어 되돌린다 (공식 extract_action_chunk와 같은 방식)."""
    flat = frame.reshape(-1)
    n = flat.shape[0] // dim
    return flat[: n * dim].reshape(n, dim).mean(0)


def pop_wac_args(argv):
    """sys.argv에서 이 파일 전용 인자를 빼낸다. 공식 스크립트(draccus)는 모르는 인자를 거부하기 때문."""
    opts = {"obj_xy_noise": 0.0, "cam_shift": 0.0, "wrist_roll_deg": 0.0,
            "cam_yaw_deg": 0.0, "perturb_seed": 0,
            "show": False, "video_dir": "", "perturb_log": "", "init_offset": 0, "episode_dir": "",
            "live_dir": "", "live_name": "worker"}
    rest, i = [argv[0]], 1
    while i < len(argv):
        key, val = argv[i], None
        if "=" in key:
            key, val = key.split("=", 1)
        if key in WAC_ARGS:
            if val is None:
                i += 1
                val = argv[i]
            opts[key[2:]] = WAC_ARGS[key](val)
        else:
            rest.append(argv[i])
        i += 1
    return opts, rest


def yaw_quat(angle_rad):
    """z축 회전 쿼터니언 (w, x, y, z)."""
    return np.array([np.cos(angle_rad / 2), 0.0, 0.0, np.sin(angle_rad / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class Viewer:
    """실시간 창과 mp4 녹화를 맡는다. 프로세스에 하나만 만든다 (Tk 루트가 여러 개면 이미지 참조가 깨진다)."""

    _instance = None

    @classmethod
    def shared(cls, show, video_dir):
        if cls._instance is None:
            cls._instance = cls(show, video_dir)
        return cls._instance

    def __init__(self, show, video_dir):
        self.show = show
        self.video_dir = video_dir
        self.frames = []
        if show:
            # venv의 OpenCV가 headless라 창을 못 띄운다. 표준 라이브러리 tkinter를 쓴다 (WSLg로 Windows에 표시).
            try:
                import tkinter as tk
                from PIL import Image, ImageDraw, ImageTk
                self.tk, self.Image, self.ImageDraw, self.ImageTk = tk, Image, ImageDraw, ImageTk
                self.root = tk.Tk()
                self.root.title("LIBERO (왼쪽: 3인칭, 오른쪽: 손목)")
                self.label = tk.Label(self.root)
                self.label.pack()
            except Exception as e:  # 창을 못 띄워도 평가는 계속한다
                print(f"[viewer] 실시간 창을 띄울 수 없음: {e}")
                self.show = False
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

    @staticmethod
    def compose(obs, caption=""):
        # LIBERO 원본 이미지는 위아래가 뒤집혀 있어서 사람이 볼 때는 바로 세운다
        a = np.flipud(obs["agentview_image"])
        w = np.flipud(obs["robot0_eye_in_hand_image"])
        return np.ascontiguousarray(np.concatenate([a, w], axis=1))

    def add(self, obs, caption):
        if not (self.show or self.video_dir):
            return
        frame = self.compose(obs)
        if self.video_dir:
            self.frames.append(frame)
        if self.show:
            try:
                img = self.Image.fromarray(frame)
                img = img.resize((img.width * 2, img.height * 2), self.Image.NEAREST)
                self.ImageDraw.Draw(img).text((8, 8), caption, fill=(255, 255, 255))
                self._photo = self.ImageTk.PhotoImage(img, master=self.root)  # 참조를 붙잡아 둬야 화면에서 안 사라진다
                self.label.configure(image=self._photo)
                self.root.update()
            except Exception as e:  # 사용자가 창을 닫으면 표시만 끄고 평가는 계속
                print(f"[viewer] 창 표시 중단: {e}")
                self.show = False

    def flush(self, name):
        if self.video_dir and self.frames:
            import imageio
            path = os.path.join(self.video_dir, name + ".mp4")
            imageio.mimwrite(path, self.frames, fps=20)
            print(f"[viewer] 저장: {path}")
        self.frames = []


CHUNK = 16
STATS_PATH = os.environ.get(
    "WAC_STATS",
    "/mnt/c/Users/rlaal/Desktop/PML/cosmos-wac/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
_STATS = None


def unnormalize_proprio(p):
    """정규화 공간(-1~1)의 관절을 원래 단위로 되돌린다. 공식 rescale_proprio의 역변환."""
    global _STATS
    if _STATS is None:
        with open(STATS_PATH) as f:
            s = json.load(f)
        _STATS = (np.array(s["proprio_min"]), np.array(s["proprio_max"]))
    lo, hi = _STATS
    return (np.asarray(p) + 1) / 2 * (hi - lo) + lo


def _label(img, text):
    """이미지 위에 띠를 붙이고 글자를 쓴다."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    out = Image.new("RGB", (im.width, im.height + 22), (30, 30, 30))
    out.paste(im, (0, 22))
    ImageDraw.Draw(out).text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(out)


def task_folder_name(task_id, instruction, max_len=60):
    """지시문 폴더의 숫자 부분을 두 자리로 맞춘다."""
    slug = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")[:max_len].rstrip("_")
    return f"D{task_id + 1:02d}_{slug}"


PRED_COLUMNS = ["episode", "init_state_idx", "success", "chunk", "t", "value",
                "pred_eef_x", "pred_eef_y", "pred_eef_z", "actual_eef_x", "actual_eef_y", "actual_eef_z",
                "eef_err_cm", "pred_gripper", "actual_gripper"]


def write_episode_files(task_dir, name, env):
    """에피소드 하나를 지시문 폴더 안에 같은 이름의 파일들로 저장한다.
      <name>.mp4          시뮬레이션 영상 [3인칭 | 손목]
      <name>_frames/      chunk마다 '현재 | 예측한 16 step 뒤 | 실제 16 step 뒤' 그림
      <name>.npz          WAC 계산용 값 (생성 latent, 가치, 예측 관절, 실제 관절)
      predictions.csv     추론마다 한 줄 (지시문 폴더 공용, 뒤에 덧붙임)
      episodes.jsonl      에피소드마다 한 줄 (지시문 폴더 공용, 뒤에 덧붙임)
    """
    import csv
    import imageio
    from PIL import Image
    os.makedirs(os.path.join(task_dir, name + "_frames"), exist_ok=True)
    rec, frames = env.record, env.frames
    imageio.mimwrite(os.path.join(task_dir, name + ".mp4"), frames, fps=20)

    rows = []
    for k, c in enumerate(env.chunks):
        t, tf = c["t"], c["t"] + CHUNK
        has_actual = tf < len(frames)
        pred = unnormalize_proprio(c["future_proprio_norm"])           # 원래 단위
        act = env.proprio[tf] if has_actual else None
        err_cm = float(np.linalg.norm(pred[2:5] - act[2:5]) * 100) if has_actual else None
        rows.append({
            "episode": name, "init_state_idx": rec["init_state_idx"], "success": env.success,
            "chunk": k, "t": t, "value": round(c["value"], 4),
            "pred_eef_x": round(pred[2], 4), "pred_eef_y": round(pred[3], 4), "pred_eef_z": round(pred[4], 4),
            "actual_eef_x": "" if act is None else round(float(act[2]), 4),
            "actual_eef_y": "" if act is None else round(float(act[3]), 4),
            "actual_eef_z": "" if act is None else round(float(act[4]), 4),
            "eef_err_cm": "" if err_cm is None else round(err_cm, 2),
            "pred_gripper": round(float(pred[0]), 4),
            "actual_gripper": "" if act is None else round(float(act[0]), 4),
        })

        h, w = frames[0].shape[:2]
        panels = [_label(frames[t], f"now  t={t}")]
        if c["pred_primary"] is not None:
            pp = np.asarray(Image.fromarray(c["pred_primary"]).resize((w // 2, h)))
            pw = np.asarray(Image.fromarray(c["pred_wrist"]).resize((w // 2, h))) if c["pred_wrist"] is not None \
                else np.zeros_like(pp)
            panels.append(_label(np.concatenate([pp, pw], 1), f"predicted  t={tf}"))
        if has_actual:
            panels.append(_label(frames[tf], f"actual  t={tf}"))
        Image.fromarray(np.concatenate(panels, 0)).save(os.path.join(task_dir, name + "_frames", f"q{k:02d}.png"))

    if env.chunks:
        np.savez_compressed(
            os.path.join(task_dir, name + ".npz"),
            chunk_t=np.array([c["t"] for c in env.chunks]),
            value=np.array([c["value"] for c in env.chunks], dtype=np.float32),
            future_proprio_norm=np.stack([c["future_proprio_norm"] for c in env.chunks]).astype(np.float32),
            proprio_norm_at_query=np.stack([c["proprio_norm"] for c in env.chunks]),
            latent=np.stack([c["latent"] for c in env.chunks]),          # (K, 16, 4, 28, 28) 프레임 5~8
            proprio=np.stack(env.proprio).astype(np.float32),            # (T+1, 9) step마다 실제 관절
            meta=json.dumps(rec, ensure_ascii=False),
        )

    pred_path = os.path.join(task_dir, "predictions.csv")
    try:  # 여러 프로세스가 같은 폴더에 쓸 때 머리줄은 처음 만든 쪽만 쓴다
        with open(pred_path, "x", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PRED_COLUMNS).writeheader()
    except FileExistsError:
        pass
    with open(pred_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=PRED_COLUMNS).writerows(rows)

    errs = [r["eef_err_cm"] for r in rows if r["eef_err_cm"] != ""]
    ep = {**rec, "episode": name, "success": env.success, "steps": env.t, "num_chunks": len(env.chunks),
          "seconds": round(time.time() - env.t_start, 1),
          "min_value": min((r["value"] for r in rows), default=None),
          "max_eef_err_cm": max(errs) if errs else None}
    with open(os.path.join(task_dir, "episodes.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def describe_perturbation(rec):
    out = []
    if rec["obj_xy_noise"]:
        out.append(f"물체 위치 ±{rec['obj_xy_noise'] * 100:.0f}cm")
    if rec["cam_yaw_deg"]:
        out.append(f"카메라 회전 {rec['cam_yaw_deg']:+.0f}도")
    if any(rec["cam_shift"]):
        out.append(f"3인칭 카메라 이동 {np.round(np.array(rec['cam_shift']) * 100, 1).tolist()}cm")
    if rec.get("wrist_roll_deg"):
        out.append(f"손목 카메라 회전 {rec['wrist_roll_deg']:+g}도")
    return ", ".join(out) or "없음"


def write_task_result(task_dir):
    """지시문 폴더의 result.md를 episodes.jsonl로부터 다시 쓴다. 에피소드가 끝날 때마다 갱신된다."""
    path = os.path.join(task_dir, "episodes.jsonl")
    eps = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not eps:
        return
    eps.sort(key=lambda e: e["init_state_idx"])
    n, k = len(eps), sum(e["success"] for e in eps)
    first = eps[0]
    condition = []
    if first["obj_xy_noise"]:
        condition.append(f"물체 위치 ±{first['obj_xy_noise'] * 100:g}cm")
    if first["cam_yaw_deg"]:
        condition.append(f"카메라 회전 ±{abs(first['cam_yaw_deg']):g}도")
    if any(first["cam_shift"]):
        condition.append(f"3인칭 카메라 이동 {np.linalg.norm(first['cam_shift']) * 100:.1f}cm")
    if first.get("wrist_roll_deg"):
        condition.append(f"손목 카메라 회전 ±{abs(first['wrist_roll_deg']):g}도")
    lines = [
        f"# D{first['task_id'] + 1}: {first['task']}",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 과제 | {eps[0]['suite']} {eps[0]['task_id']}번 |",
        f"| 교란 조건 | {', '.join(condition) or '없음'} |",
        f"| 교란 seed | {eps[0]['perturb_seed']} |",
        f"| 에피소드 | {n} |",
        f"| 성공 | {k} ({k / n:.0%}) |",
        "",
        "## 에피소드",
        "",
        "| 초기 상태 | 결과 | step | 추론 횟수 | 최소 가치 | 최대 위치 오차 | 실제 교란 | 시간 | 영상 | 그림 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in eps:
        mv = f"{e['min_value']:.2f}" if e["min_value"] is not None else "-"
        er = f"{e['max_eef_err_cm']:.1f} cm" if e["max_eef_err_cm"] is not None else "-"
        lines.append(f"| {e['init_state_idx']} | {'성공' if e['success'] else '**실패**'} | {e['steps']} | "
                     f"{e['num_chunks']} | {mv} | {er} | {describe_perturbation(e)} | {e['seconds']:.0f}초 | "
                     f"[mp4]({e['episode']}.mp4) | [보기]({e['episode']}_frames/) |")
    lines += [
        "",
        "## 열 설명",
        "",
        "- step: 실행한 step 수. 처음 10 step은 물체가 자리 잡게 기다리는 구간이다. 실패하면 최대 step까지 간다.",
        "- 추론 횟수: 정책을 부른 횟수. 한 번 부를 때마다 16 step을 실행한다.",
        "- 최소 가치: 에피소드 중 가장 낮았던 가치 예측 (0~1, 1이 성공에 가깝다는 뜻).",
        "- 최대 위치 오차: 예측한 16 step 뒤 엔드이펙터 위치와 실제 위치의 거리 중 가장 큰 값.",
        "- 그림: 추론마다 위에서부터 현재 화면, 예측한 16 step 뒤, 실제 16 step 뒤. 왼쪽 3인칭, 오른쪽 손목.",
        "- 추론마다의 값은 `predictions.csv`, WAC 계산용 latent는 `<에피소드>.npz`, 파인튜닝용 데이터는 `<에피소드>.hdf5`.",
        "",
    ]
    with open(os.path.join(task_dir, "result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class HDF5Mover:
    """공식 코드가 쓴 에피소드 HDF5를 지시문 폴더로 옮겨 이름을 맞춘다.
    공식 코드는 run_episode가 끝난 뒤에 HDF5를 쓰므로, 다음 에피소드 시작 때와 평가가 끝났을 때 옮긴다.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.pending = []      # (지시문 폴더, 에피소드 이름) 순서대로

    def add(self, task_dir, name):
        self.pending.append((task_dir, name))

    def flush(self):
        if not self.src_dir or not self.pending:
            return
        files = sorted(glob.glob(os.path.join(self.src_dir, "*.hdf5")), key=os.path.getmtime)
        while files and self.pending:
            task_dir, name = self.pending.pop(0)
            os.replace(files.pop(0), os.path.join(task_dir, name + ".hdf5"))


class PerturbedEnv:
    """OffScreenRenderEnv를 감싸 set_init_state 직후 교란을 넣고, step마다 화면을 갱신한다."""

    def __init__(self, env, task_description, opts, task_tag):
        self._env = env
        self.task_description = task_description
        self.opts = opts
        self.task_tag = task_tag
        self.viewer = Viewer.shared(opts["show"], opts["video_dir"])
        self.episode = -1
        self.t = 0
        self.success = False
        self.record = None
        self.chunks = []       # chunk마다 get_action 결과에서 뽑은 값
        self.proprio = []      # step마다 실제 관절 (원래 단위)
        self.frames = []       # step마다 [3인칭 | 손목] 화면 (바로 세운 RGB). 에피소드 정리용
        self.t_start = time.time()

    def __getattr__(self, name):
        return getattr(self._env, name)

    # ---- 교란 ----
    def _perturb(self):
        o = self.opts
        # 난수는 (교란 seed, 과제 번호, 초기 상태 번호)로 정한다. 작업을 초기 상태별로 나눠 돌려도 같은 교란이 나온다.
        rng = np.random.default_rng([o["perturb_seed"], self.task_tag[1], o["init_offset"] + self.episode])
        inner = self._env.env
        sim = inner.sim
        rec = {"objects": {}, "cam_shift": [0.0, 0.0, 0.0], "cam_yaw_deg": 0.0, "wrist_roll_deg": 0.0}

        if o["obj_xy_noise"] > 0:
            for name, obj in inner.objects_dict.items():
                if not obj.joints:
                    continue
                j = obj.joints[0]
                qpos = sim.data.get_joint_qpos(j).copy()
                d = rng.uniform(-o["obj_xy_noise"], o["obj_xy_noise"], size=2)
                qpos[:2] += d
                sim.data.set_joint_qpos(j, qpos)
                sim.data.set_joint_qvel(j, np.zeros(6))
                rec["objects"][name] = d.round(4).tolist()

        cam_id = sim.model.camera_name2id("agentview")
        if o["cam_yaw_deg"] > 0:
            ang = np.deg2rad(o["cam_yaw_deg"]) * rng.choice([-1.0, 1.0])
            # 테이블 중심(대략 원점 위)을 지나는 수직축으로 카메라를 공전시킨다
            c, s = np.cos(ang), np.sin(ang)
            p = sim.model.cam_pos[cam_id].copy()
            sim.model.cam_pos[cam_id][:2] = [c * p[0] - s * p[1], s * p[0] + c * p[1]]
            sim.model.cam_quat[cam_id] = quat_mul(yaw_quat(ang), sim.model.cam_quat[cam_id])
            rec["cam_yaw_deg"] = float(np.rad2deg(ang))
        if o["cam_shift"] > 0:
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * o["cam_shift"]
            sim.model.cam_pos[cam_id] += v
            rec["cam_shift"] = v.round(4).tolist()
        if o["wrist_roll_deg"] > 0:
            wrist_id = sim.model.camera_name2id("robot0_eye_in_hand")
            roll = np.deg2rad(o["wrist_roll_deg"]) * rng.choice([-1.0, 1.0])
            sim.model.cam_quat[wrist_id] = quat_mul(sim.model.cam_quat[wrist_id], yaw_quat(roll))
            rec["wrist_roll_deg"] = float(np.rad2deg(roll))

        sim.forward()
        return rec

    def set_init_state(self, state):
        global CURRENT_ENV
        self._finish_episode()
        self.episode += 1
        self.t = 0
        self.success = False
        self.chunks, self.proprio, self.frames = [], [], []
        self.t_start = time.time()
        CURRENT_ENV = self
        obs = self._env.set_init_state(state)
        rec = self._perturb()
        obs = self._env.env._get_observations(force_update=True)
        self.record = {"suite": self.task_tag[0], "task_id": self.task_tag[1], "episode": self.episode,
                       "init_state_idx": self.opts["init_offset"] + self.episode,
                       "task": self.task_description, **{k: self.opts[k] for k in
                       ("obj_xy_noise", "cam_shift", "cam_yaw_deg", "wrist_roll_deg", "perturb_seed")}, **rec}
        self.proprio.append(self._proprio(obs))
        self.frames.append(Viewer.compose(obs))
        self.viewer.add(obs, self._caption())
        return obs

    @staticmethod
    def _proprio(obs):
        # 공식 prepare_observation과 같은 순서: 그리퍼 2 + 엔드이펙터 위치 3 + 쿼터니언 4
        return np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))

    def log_chunk(self, rd):
        """get_action 반환값에서 WAC 계산에 필요한 값을 뽑아 둔다. t는 이 chunk를 추론한 시점의 step 번호."""
        lat = rd["generated_latent"][0].float().cpu().numpy()        # (16, 9, 28, 28)
        fut = rd.get("future_image_predictions") or {}
        self.chunks.append({
            # 공식 코드가 디코딩한 예측 미래 이미지 (224x224, 바로 세운 방향). 결과 정리용
            "pred_primary": fut.get("future_image"),
            "pred_wrist": fut.get("future_wrist_image"),
            "t": self.t,
            "value": float(rd["value_prediction"]),
            "future_proprio_norm": untile(lat[:, 5], PROPRIO_DIM),     # 정규화 공간 (-1~1)
            "proprio_norm": np.asarray(rd["proprio"], dtype=np.float32),
            "latent": lat[:, SAVE_FRAMES].astype(np.float16),         # (16, 4, 28, 28)
        })

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.t += 1
        self.proprio.append(self._proprio(obs))
        self.frames.append(Viewer.compose(obs))
        if done:
            self.success = True
        self.viewer.add(obs, self._caption())
        if self.opts["live_dir"] and (self.t % 4 == 0 or done):
            self._write_live()
        return obs, reward, done, info

    def _caption(self):
        return f"task {self.task_tag[1]} ep {self.episode} t={self.t}" + (" SUCCESS" if self.success else "")

    def _write_live(self):
        """원격 감시용 최신 화면. 임시 파일에 쓰고 바꿔치기해서 읽는 쪽이 반쯤 쓴 파일을 보지 않게 한다."""
        try:
            from PIL import Image
            d = self.opts["live_dir"]
            os.makedirs(d, exist_ok=True)
            img = _label(self.frames[-1], f"{self.opts['live_name']}  D{self.task_tag[1] + 1}  "
                                          f"init{self.opts['init_offset'] + self.episode}  {self._caption()}")
            tmp = os.path.join(d, f".{self.opts['live_name']}.jpg")
            Image.fromarray(img).save(tmp, quality=70)
            os.replace(tmp, os.path.join(d, f"{self.opts['live_name']}.jpg"))
        except Exception:
            pass

    def _finish_episode(self):
        if self.record is None:
            return
        self.record.update(success=self.success, steps=self.t)
        if self.opts["perturb_log"]:
            os.makedirs(os.path.dirname(self.opts["perturb_log"]) or ".", exist_ok=True)
            with open(self.opts["perturb_log"], "a") as f:
                f.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        name = f"init{self.record['init_state_idx']:02d}_{'succ' if self.success else 'fail'}"
        self.viewer.flush(f"{self.task_tag[0]}_task{self.task_tag[1]:02d}_{name}")
        if self.opts["episode_dir"]:
            folder = task_folder_name(self.task_tag[1], self.task_description)
            split_dir = self.opts["episode_dir"]
            task_dir = os.path.join(split_dir, folder)
            legacy = os.path.join(split_dir, f"D{self.task_tag[1] + 1}_{folder.split('_', 1)[1]}")
            if not os.path.isdir(task_dir) and os.path.isdir(legacy):
                task_dir = legacy
            os.makedirs(task_dir, exist_ok=True)
            try:
                write_episode_files(task_dir, name, self)
                write_task_result(task_dir)
                print(f"[dataset] {os.path.basename(task_dir)}/{name} 저장")
            except Exception as e:  # 정리 실패가 수집을 멈추게 하지 않는다
                import traceback
                print(f"[dataset] 에피소드 저장 실패: {e}\n{traceback.format_exc()}")
            if MOVER is not None:
                MOVER.add(task_dir, name)
        self.record = None

    def close(self):
        self._finish_episode()
        return self._env.close()


def main():
    opts, sys.argv = pop_wac_args(sys.argv)
    if not opts["perturb_log"]:
        opts["perturb_log"] = os.path.expanduser(f"~/wac/logs/perturb_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    print(f"[wac] 교란 설정: {opts}")

    from cosmos_policy.experiments.robot.libero import run_libero_eval as rle

    suite = next((a.split("=", 1)[1] if "=" in a else sys.argv[i + 1])
                 for i, a in enumerate(sys.argv) if a.startswith("--task_suite_name"))
    orig_get_env = rle.get_libero_env

    def get_env(task, model_family, resolution=256):
        env, desc = orig_get_env(task, model_family, resolution)
        task_id = next(i for i, n in enumerate(rle.benchmark.get_benchmark_dict()[suite]().get_task_names())
                       if n == task.name)
        return PerturbedEnv(env, desc, opts, (suite, task_id)), desc

    # 에피소드가 끝나는 시점을 공식 run_episode의 반환으로 잡는다.
    # 성공 여부도 공식 판정(반환값 첫 번째)을 그대로 쓴다.
    orig_run_episode = rle.run_episode

    def run_episode(cfg, env, *args, **kwargs):
        if MOVER is not None:
            MOVER.flush()   # 직전 에피소드의 HDF5는 run_episode가 끝난 뒤에 쓰이므로 여기서 옮긴다
        result = orig_run_episode(cfg, env, *args, **kwargs)
        if isinstance(env, PerturbedEnv):
            env.success = bool(result[0])
            env._finish_episode()
        return result

    # chunk마다 정책 출력(생성 latent, 가치)을 가로채서 저장한다.
    orig_get_action = rle.get_action

    def get_action(*args, **kwargs):
        rd = orig_get_action(*args, **kwargs)
        if CURRENT_ENV is not None and opts["episode_dir"]:
            CURRENT_ENV.log_chunk(rd)
        return rd

    # 묶음(학습용, 보정용, 평가용)마다 다른 초기 상태를 쓰도록 앞에서부터 init_offset개를 건너뛴다.
    orig_load_initial_states = rle.load_initial_states

    def load_initial_states(cfg, task_suite, task_id, log_file=None):
        states, all_states = orig_load_initial_states(cfg, task_suite, task_id, log_file)
        off = opts["init_offset"]
        assert off + cfg.num_trials_per_task <= len(states), \
            f"초기 상태가 부족함: offset {off} + trials {cfg.num_trials_per_task} > {len(states)}"
        return states[off:], all_states

    # --data_collection True면 공식 코드가 <local_log_dir>/rollout_data/에 쓴 HDF5를 지시문 폴더로 옮긴다
    global MOVER
    # 같은 인자가 여러 번 오면 공식 코드(draccus)처럼 마지막 값을 쓴다
    arg = lambda key: ([a.split("=", 1)[1] if "=" in a else sys.argv[i + 1]
                        for i, a in enumerate(sys.argv) if a.startswith(key)] or [None])[-1]
    if opts["episode_dir"] and (arg("--data_collection") or "").lower() == "true" and arg("--local_log_dir"):
        MOVER = HDF5Mover(os.path.join(os.path.expanduser(arg("--local_log_dir")), "rollout_data"))

    rle.get_libero_env = get_env
    rle.run_episode = run_episode
    rle.get_action = get_action
    rle.load_initial_states = load_initial_states
    rle.eval_libero()
    if MOVER is not None:
        MOVER.flush()


if __name__ == "__main__":
    main()
