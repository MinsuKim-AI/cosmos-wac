"""데이터셋 요약 만들기.

  python summarize_split.py <묶음 폴더>     예: data/libero_spatial/obj_0.12/eval
      → <묶음 폴더>/result.md  (지시문별 성공률 표)
  python summarize_split.py --index <과제 묶음 폴더>   예: data/libero_spatial
      → index.csv  (전체 에피소드 한 줄씩), README.md  (조건별, 묶음별 성공률)
"""

import csv
import glob
import json
import os
import sys


def load(pattern):
    eps = []
    for p in sorted(glob.glob(pattern)):
        d = os.path.dirname(p)
        for line in open(p, encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                e["_task_dir"] = d
                eps.append(e)
    return eps


def relocate_hdf5(split_dir):
    """_logs/official/rollout_data/에 남은 공식 HDF5를 지시문 폴더로 옮긴다.
    파일 이름의 task=<번호>, ep=<전체 순번>, success=<결과>로 같은 과제 안의 실행 순서대로 에피소드와 짝을 맞춘다."""
    import re
    src = os.path.join(split_dir, "_logs", "official", "rollout_data")
    files = glob.glob(os.path.join(src, "*.hdf5"))
    if not files:
        return
    by_task = {}
    for f in files:
        m = re.search(r"task=(\d+)--ep=(\d+)--success=(True|False)", os.path.basename(f))
        if m:
            by_task.setdefault(int(m.group(1)), []).append((int(m.group(2)), m.group(3) == "True", f))
    eps = load(os.path.join(split_dir, "D*", "episodes.jsonl"))
    moved = 0
    for tid, hs in by_task.items():
        hs.sort()
        # episodes.jsonl은 실행 순서대로 쌓인다. 이미 HDF5가 옮겨진 에피소드는 짝 맞추기에서 뺀다
        es = [e for e in eps if e["task_id"] == tid
              and not os.path.exists(os.path.join(e["_task_dir"], e["episode"] + ".hdf5"))]
        for (_, succ, f), e in zip(hs, es):
            dst = os.path.join(e["_task_dir"], e["episode"] + ".hdf5")
            if succ != e["success"]:
                print(f"[hdf5] 성공 여부가 안 맞아 건너뜀: {os.path.basename(f)} / {e['episode']}")
                continue
            if not os.path.exists(dst):
                os.replace(f, dst)
                moved += 1
    print(f"[hdf5] {moved}개 옮김")


def split_summary(split_dir):
    relocate_hdf5(split_dir)
    eps = load(os.path.join(split_dir, "D*", "episodes.jsonl"))
    if not eps:
        sys.exit(f"에피소드 없음: {split_dir}")
    n, k = len(eps), sum(e["success"] for e in eps)
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task_id"], []).append(e)
    lines = [
        f"# {os.path.relpath(split_dir, os.path.dirname(os.path.dirname(os.path.dirname(split_dir))))}",
        "",
        f"에피소드 {n}개, 성공 {k}개 ({k / n:.0%}).",
        "",
        "| D | 지시문 | 성공 | 에피소드 (O 성공, x 실패, 초기 상태 순) |",
        "|---|---|---|---|",
    ]
    for tid, es in sorted(by_task.items()):
        es.sort(key=lambda e: e["init_state_idx"])
        folder = os.path.basename(es[0]["_task_dir"])
        marks = "".join("O" if e["success"] else "x" for e in es)
        lines.append(f"| [D{tid + 1}]({folder}/result.md) | {es[0]['task']} | "
                     f"{sum(e['success'] for e in es)}/{len(es)} | `{marks}` |")
    lines.append("")
    with open(os.path.join(split_dir, "result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{split_dir}: {k}/{n} 성공")


INDEX_COLUMNS = ["condition", "split", "D", "task_id", "task", "init_state_idx", "perturb_seed",
                 "obj_xy_noise", "cam_yaw_deg", "cam_shift", "wrist_roll_deg", "success", "steps", "num_chunks", "seconds",
                 "min_value", "max_eef_err_cm", "mp4", "npz", "hdf5"]


def build_index(suite_dir):
    eps = load(os.path.join(suite_dir, "*", "*", "D*", "episodes.jsonl"))
    rows = []
    for e in eps:
        td = e["_task_dir"]
        split_dir = os.path.dirname(td)
        rel = lambda ext: os.path.relpath(os.path.join(td, e["episode"] + ext), suite_dir).replace("\\", "/")
        rows.append({
            "condition": os.path.basename(os.path.dirname(split_dir)), "split": os.path.basename(split_dir),
            "D": e["task_id"] + 1, "task_id": e["task_id"], "task": e["task"],
            "init_state_idx": e["init_state_idx"], "perturb_seed": e["perturb_seed"],
            "obj_xy_noise": e["obj_xy_noise"], "cam_yaw_deg": e["cam_yaw_deg"],
            "cam_shift": json.dumps(e.get("cam_shift", [0.0, 0.0, 0.0])),
            "wrist_roll_deg": e.get("wrist_roll_deg", 0.0),
            "success": int(e["success"]), "steps": e["steps"], "num_chunks": e["num_chunks"],
            "seconds": e["seconds"], "min_value": e["min_value"], "max_eef_err_cm": e["max_eef_err_cm"],
            "mp4": rel(".mp4"), "npz": rel(".npz"),
            "hdf5": rel(".hdf5") if os.path.exists(os.path.join(td, e["episode"] + ".hdf5")) else "",
        })
    rows.sort(key=lambda r: (r["condition"], r["split"], r["task_id"], r["init_state_idx"]))
    with open(os.path.join(suite_dir, "index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["split"]), []).append(r["success"])
    lines = [f"# {os.path.basename(suite_dir)} 데이터셋", "",
             f"전체 에피소드 {len(rows)}개. 한 줄씩 정리한 목록은 `index.csv`.", "",
             "| 교란 조건 | 묶음 | 에피소드 | 성공률 | 요약 |", "|---|---|---|---|---|"]
    for (c, s), v in sorted(groups.items()):
        lines.append(f"| {c} | {s} | {len(v)} | {sum(v) / len(v):.0%} | [result.md]({c}/{s}/result.md) |")
    lines.append("")
    with open(os.path.join(suite_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"index.csv: {len(rows)}개 에피소드")


if __name__ == "__main__":
    if sys.argv[1] == "--index":
        build_index(sys.argv[2])
    else:
        split_summary(sys.argv[1])
