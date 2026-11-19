"""교란 sweep 결과 요약: 단계별 성공률과 과제별 결과. 사용: python summarize_calib.py ~/wac/logs/calib"""

import glob
import json
import os
import sys

d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/wac/logs/calib")
for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        continue
    n, s = len(rows), sum(r["success"] for r in rows)
    fail_steps = [r["steps"] for r in rows if not r["success"]]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append("O" if r["success"] else "x")
    tasks = " ".join(f"{t}:{''.join(v)}" for t, v in sorted(by_task.items()))
    print(f"{os.path.basename(path)[:-6]:<12} {s:>3}/{n:<3} = {s / n:5.0%}   과제별 {tasks}")
