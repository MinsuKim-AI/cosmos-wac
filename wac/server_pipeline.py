"""서버에서 혼자 끝까지 도는 수집 순서. 노트북을 꺼도 계속된다.

1. 교란 크기 찾기 (물체, 카메라 동시에)
   - 크기마다 20번 (지시문 10개 × 초기 상태 0, 1), 교란 seed 9000
   - 40~60%인 크기가 있으면 50%에 가장 가까운 것으로 확정
   - 없으면 50%를 사이에 두는 두 크기 사이를 더 잘게 나눠 다시 잰다 (최대 3번)
   - 처음 잰 카메라 각도에서 크기를 못 찾으면 105~180도를 15도 간격으로 잰다.
   - 180도까지 잰 뒤에도 후보가 없으면 성공률이 가장 낮은 각도 주변을 3도 간격으로 잰다.
2. 정한 크기로 두 조건을 동시에 수집 (calib 10, eval 10, train 5 × 지시문 10개 × 조건 2개 = 500)
3. 묶음별 result.md, 전체 index.csv, README.md 작성

사용: nohup setsid python server_pipeline.py --gpus 2,3 --per_gpu 3 > ~/wac_cosmos/pipeline.log 2>&1 &
상태: ~/wac_cosmos/status.json
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~/wac_cosmos")
CALIB = os.path.join(HOME, "calib")
ROOT = os.path.join(HOME, "repo", "data", "libero_spatial")
PY = sys.executable

p = argparse.ArgumentParser()
p.add_argument("--gpus", default="2,3")
p.add_argument("--per_gpu", type=int, default=3)
a = p.parse_args()

KINDS = {
    "obj": {"flag": "--obj_xy_noise", "prefix": "obj_", "fmt": "{:.3f}", "grid": [0.08, 0.09, 0.10, 0.11, 0.12],
            "step": 0.005, "max": 0.30, "grow": 0.04},
    "cam": {"flag": "--cam_yaw_deg", "prefix": "camyaw_", "fmt": "{:.0f}", "grid": [45, 60, 75, 90],
            "step": 1, "max": 180, "grow": 15},
}
SPLITS = [("calib", 10, 10, 2), ("eval", 20, 10, 3), ("train", 0, 5, 1)]
status = {"stage": "start", "sizes": {}, "measured": {}}


def save_status(**kw):
    status.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(HOME, "status.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1, ensure_ascii=False)
    os.replace(temporary, path)
    print(f"[{time.strftime('%H:%M:%S')}] {kw}", flush=True)


def name(kind, s):
    c = KINDS[kind]
    return c["prefix"] + c["fmt"].format(s)


def run_jobs(jobs):
    path = os.path.join(HOME, f"jobs_{int(time.time())}.json")
    json.dump(jobs, open(path, "w"), indent=1)
    subprocess.run([PY, os.path.join(HERE, "parallel_runner.py"), "--gpus", a.gpus,
                    "--per_gpu", str(a.per_gpu), "--jobs", path], check=False)


def calib_jobs(kind, sizes, root=CALIB, initial_states=(0, 1), seed=9000):
    jobs = []
    for s in sizes:
        out = os.path.join(root, name(kind, s))
        for init in initial_states:
            if os.path.exists(os.path.join(out, "_logs", f"{name(kind, s)}_i{init}", "DONE")):
                continue  # 끝난 작업은 건너뛴다 (같은 에피소드가 두 번 기록되지 않게)
            jobs.append({"name": f"{name(kind, s)}_i{init}", "trials": 1, "init_offset": init, "out": out,
                         "args": perturb_args(kind, s), "perturb_seed": seed})
    return jobs


def rate(kind, s, root=CALIB):
    out = os.path.join(root, name(kind, s))
    rows = [json.loads(l) for f in glob.glob(os.path.join(out, "D*", "episodes.jsonl")) for l in open(f) if l.strip()]
    return (sum(r["success"] for r in rows), len(rows)) if rows else None


def measured(kind, root=CALIB):
    m = {}
    for d in glob.glob(os.path.join(root, KINDS[kind]["prefix"] + "*")):
        s = float(os.path.basename(d)[len(KINDS[kind]["prefix"]):])
        r = rate(kind, s, root=root)
        if r and r[1] >= 20:
            m[s] = r
    return dict(sorted(m.items()))


def next_sizes(kind, root=CALIB):
    """확정 크기 또는 다음에 잴 크기 목록."""
    c, m = KINDS[kind], measured(kind, root=root)
    ok = [(abs(k / n - 0.5), s) for s, (k, n) in m.items() if 0.4 <= k / n <= 0.6]
    if ok:
        return "done", min(ok)[1]
    sizes = list(m)
    for lo, hi in zip(sizes, sizes[1:]):
        if m[lo][0] / m[lo][1] >= 0.5 > m[hi][0] / m[hi][1]:
            # 두 크기 사이를 3등분해 가운데 두 점을 동시에 잰다
            pts = sorted({
                round(round((lo + (hi - lo) * f) / c["step"]) * c["step"], 3)
                for f in (1 / 3, 2 / 3)
            } - set(sizes))
            if pts:
                return "run", pts
    next_size = round(sizes[-1] + c["grow"], 3) if sizes else None
    if sizes and all(k / n >= 0.5 for k, n in m.values()) and next_size <= c["max"]:
        if kind == "cam":
            return "run", list(range(int(sizes[-1]) + c["grow"], c["max"] + 1, c["grow"]))
        return "run", [next_size]
    if sizes and all(k / n < 0.5 for k, n in m.values()) and sizes[0] / 2 >= c["step"]:
        return "run", [round(sizes[0] / 2 / c["step"]) * c["step"]]
    if kind == "cam" and sizes and sizes[-1] == c["max"]:
        best = int(min(m, key=lambda s: abs(m[s][0] / m[s][1] - 0.5)))
        nearby = [
            s for s in range(best - 9, best + 10, 3)
            if 0 < s <= c["max"] and s not in m
        ]
        if nearby:
            return "run", nearby
        remaining = [s for s in (15, 30, best - 2, best - 1, best + 1, best + 2)
                     if 0 < s <= c["max"] and s not in m]
        if remaining:
            return "run", sorted(set(remaining))
    return "fail", None


def mark_calib_done():
    for d in glob.glob(os.path.join(CALIB, "*")):
        r = sum(1 for f in glob.glob(os.path.join(d, "D*", "episodes.jsonl")) for l in open(f) if l.strip())
        if r >= 20:
            os.makedirs(os.path.join(d, "_logs"), exist_ok=True)
            open(os.path.join(d, "_logs", "ALL_DONE"), "w").close()


def perturb_args(kind, size):
    spec = KINDS[kind]
    return [*spec.get("fixed_args", []), spec["flag"], spec["fmt"].format(size)]


def search_camera_perturbation(kind, spec):
    KINDS[kind] = spec
    root = os.path.join(HOME, "calib_" + kind)
    sizes = spec["grid"]
    for _ in range(4):
        jobs = calib_jobs(kind, sizes, root=root)
        if jobs:
            save_status(stage=f"size_search_{kind}", camera_candidates=sizes)
            run_jobs(jobs)
        result, candidate = next_sizes(kind, root=root)
        status["measured"][kind] = {
            spec["fmt"].format(size): counts for size, counts in measured(kind, root=root).items()
        }
        if result == "done":
            return candidate
        if result == "fail":
            break
        sizes = candidate
    return None


def confirm_reproduction(kinds):
    groups = [("a", 30, 19001), ("b", 35, 19002)]
    roots = {group: os.path.join(HOME, "reproduction", group) for group, _, _ in groups}
    pending = {kind: status["sizes"][kind] for kind in kinds}
    tested = {kind: {} for kind in kinds}
    status["reproduction"] = tested
    for _ in range(4):
        if not pending:
            return True
        save_status(stage="size_search_reproduction", checking=pending)
        jobs = [
            job
            for kind, value in pending.items()
            for group, start, seed in groups
            for job in calib_jobs(kind, [value], root=roots[group], initial_states=range(start, start + 5), seed=seed)
        ]
        if jobs:
            run_jobs(jobs)
        next_pending = {}
        for kind, value in pending.items():
            spec = KINDS[kind]
            counts = {group: rate(kind, value, root=roots[group]) for group, _, _ in groups}
            tested[kind][spec["fmt"].format(value)] = counts
            if any(not result or result[1] != 50 for result in counts.values()):
                save_status(stage="stopped_incomplete_reproduction")
                return False
            if all(20 <= successes <= 30 for successes, _ in counts.values()):
                status["sizes"][kind] = value
                continue

            previous = {float(v): sum(k for k, _ in rs.values()) for v, rs in tested[kind].items()}
            step = spec["step"]
            options = []
            for lo, hi in zip(sorted(previous), sorted(previous)[1:]):
                if previous[lo] >= 50 >= previous[hi]:
                    options.append(round(round((lo + hi) / (2 * step)) * step, 3))
            direction = 1 if previous[value] >= 50 else -1
            delta = {"obj": 0.005, "cam": 3, "camera_shift": 0.05, "wrist_roll": 5}[kind]
            options.extend(round(value + direction * delta * n, 3) for n in (1, 2, -1, -2))
            valid = [candidate for candidate in options if 0 < candidate <= spec["max"] and candidate not in previous]
            if not valid:
                save_status(stage="stopped_size_not_found")
                return False
            next_pending[kind] = valid[0]
        pending = next_pending
        save_status()
    return not pending


# ---------- 1. 교란 크기 찾기 ----------
save_status(stage="size_search")
todo = {k: KINDS[k]["grid"] for k in KINDS}
for rnd in range(4):
    jobs = [j for k, sizes in todo.items() for j in calib_jobs(k, sizes)]
    if jobs:
        save_status(stage=f"size_search_round{rnd}", running={k: v for k, v in todo.items()})
        run_jobs(jobs)
        mark_calib_done()
    todo = {}
    for k in KINDS:
        if k in status["sizes"]:
            continue
        st, val = next_sizes(k)
        status["measured"][k] = {KINDS[k]["fmt"].format(s): v for s, v in measured(k).items()}
        if st == "done":
            status["sizes"][k] = val
        elif st == "run":
            todo[k] = val
        else:
            status["sizes"][k] = None
    save_status(stage=f"size_search_round{rnd}_done")
    if not todo:
        break

camera_kind = "cam"
if status["sizes"].get("cam") is None and status["sizes"].get("obj") is not None:
    # 후보 선택에 쓴 상태와 본수집 상태를 피해서 가장 가까운 각도를 별도로 확인한다.
    camera_results = measured("cam")
    if camera_results:
        angle = min(camera_results, key=lambda s: abs(camera_results[s][0] / camera_results[s][1] - 0.5))
        confirmation_dir = os.path.join(HOME, "calib_confirmation")
        save_status(stage="size_search_camera_confirmation", camera_confirmation_angle=angle)
        jobs = calib_jobs("cam", [angle], root=confirmation_dir, initial_states=range(5, 10))
        if jobs:
            run_jobs(jobs)
        confirmation = rate("cam", angle, root=confirmation_dir)
        save_status(camera_confirmation=confirmation)
        if confirmation and confirmation[1] == 50 and 0.4 <= confirmation[0] / 50 <= 0.6:
            status["sizes"]["cam"] = angle
        elif confirmation and confirmation[1] == 50:
            camera_kind = "camera_shift"
            status["sizes"][camera_kind] = search_camera_perturbation(camera_kind, {
                "flag": "--cam_shift", "fixed_args": ["--cam_yaw_deg", f"{angle:.0f}"],
                "prefix": f"camyaw_{angle:.0f}_shift_", "fmt": "{:.3f}",
                "grid": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                "step": 0.025, "max": 0.90, "grow": 0.15,
            })
            if status["sizes"][camera_kind] is None:
                camera_kind = "wrist_roll"
                status["sizes"][camera_kind] = search_camera_perturbation(camera_kind, {
                    "flag": "--wrist_roll_deg", "prefix": "wristroll_", "fmt": "{:.0f}",
                    "grid": [10, 20, 30, 45, 60, 90],
                    "step": 1, "max": 180, "grow": 15,
                })

active_kinds = ("obj", camera_kind)
if any(status["sizes"].get(k) is None for k in active_kinds):
    save_status(stage="stopped_size_not_found")
    sys.exit(1)
if not confirm_reproduction(active_kinds):
    if not status["stage"].startswith("stopped"):
        save_status(stage="stopped_size_not_found")
    sys.exit(1)

# ---------- 2. 수집 ----------
save_status(stage="collect")
jobs = []
for k in active_kinds:
    s = status["sizes"][k]
    for split, start, n, seed in SPLITS:
        out = os.path.join(ROOT, name(k, s), split)
        for init in range(start, start + n):
            if os.path.exists(os.path.join(out, "_logs", f"{split}_i{init}", "DONE")):
                continue
            jobs.append({"name": f"{split}_i{init}", "trials": 1, "init_offset": init, "out": out,
                         "args": perturb_args(k, s), "perturb_seed": seed, "data_collection": True})
# calib, eval을 먼저 돌리도록 순서를 섞는다 (조건 둘을 번갈아)
order = {"calib": 0, "eval": 1, "train": 2}
jobs.sort(key=lambda j: (order[j["name"].split("_")[0]], j["init_offset"]))
run_jobs(jobs)

# ---------- 3. 요약 ----------
expected = set()
for k in active_kinds:
    size = status["sizes"][k]
    condition = name(k, size)
    for split, start, n, seed in SPLITS:
        split_dir = os.path.join(ROOT, condition, split)
        subprocess.run([PY, os.path.join(HERE, "summarize_split.py"), split_dir], check=True)
        expected.update(
            (condition, split, task, init, seed)
            for task in range(10) for init in range(start, start + n)
        )
subprocess.run([PY, os.path.join(HERE, "summarize_split.py"), "--index", ROOT], check=True)
with open(os.path.join(ROOT, "index.csv"), newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
observed = {
    (r["condition"], r["split"], int(r["task_id"]), int(r["init_state_idx"]), int(r["perturb_seed"]))
    for r in rows
}
missing_files = sum(
    not r[ext] or not os.path.isfile(os.path.join(ROOT, r[ext]))
    for r in rows for ext in ("mp4", "npz", "hdf5")
)
if observed != expected or len(rows) != len(expected) or missing_files:
    save_status(stage="stopped_incomplete_collection", rows=len(rows), missing_files=missing_files)
    sys.exit(1)
save_status(stage="finished")
