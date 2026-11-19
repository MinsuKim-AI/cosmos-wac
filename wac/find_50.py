"""교란 크기를 직접 돌려 보며 성공률 50%가 나오는 크기를 찾는다.

규칙
  1. 이미 잰 크기 중 성공률이 40~60%인 것이 있으면 그 크기로 확정한다.
  2. 없으면 50%를 사이에 두는 두 크기(위는 50% 이상, 아래는 50% 미만)를 찾아 가운데 크기를 직접 잰다.
  3. 1로 돌아간다. 최대 4번까지 잰다.
각 크기는 calib_one.sh로 같은 조건(seed 9000, 초기 상태 0~1, 지시문당 2번 = 20번)에서 잰다.

사용: python find_50.py obj      (물체 이동, m 단위, 0.01 간격)
      python find_50.py cam      (카메라 회전, 도 단위, 1도 간격)
결과: ~/wac/logs/calib/find_50_<obj|cam>.json
"""

import glob
import json
import os
import subprocess
import sys

KIND = sys.argv[1]
CALIB = os.path.expanduser("~/wac/logs/calib")
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX, ARG, STEP, FMT, GROW, MAX_SIZE = {
    # 잰 크기가 모두 50% 이상이면 가장 큰 크기에 GROW를 더해 더 큰 크기를 잰다 (MAX_SIZE까지)
    "obj": ("obj_", "--obj_xy_noise", 0.01, "{:.2f}", 0.04, 0.30),
    "cam": ("cam_yaw", "--cam_yaw_deg", 1, "{:.0f}", 15, 90),
}[KIND]
LOW, HIGH, MAX_RUNS = 0.40, 0.60, 6


def measured():
    out = {}
    for p in glob.glob(os.path.join(CALIB, PREFIX + "*.jsonl")):
        name = os.path.basename(p)[:-6]
        if not os.path.exists(os.path.join(CALIB, name, "DONE")):
            continue  # 끝까지 못 돈 크기는 쓰지 않는다
        rows = [json.loads(l) for l in open(p) if l.strip()]
        out[float(name[len(PREFIX):])] = (sum(r["success"] for r in rows), len(rows))
    return dict(sorted(out.items()))


def decide(m):
    ok = [(abs(k / n - 0.5), s) for s, (k, n) in m.items() if LOW <= k / n <= HIGH]
    if ok:
        return "done", min(ok)[1]
    sizes = list(m)
    for a, b in zip(sizes, sizes[1:]):
        ra, rb = m[a][0] / m[a][1], m[b][0] / m[b][1]
        if ra >= 0.5 > rb:  # 크기가 클수록 성공률이 낮다
            mid = round((a + b) / 2 / STEP) * STEP
            if mid in (a, b):
                return "done", a if abs(ra - 0.5) <= abs(rb - 0.5) else b  # 더 쪼갤 수 없음
            return "run", mid
    if sizes and all(m[s][0] / m[s][1] >= 0.5 for s in sizes):
        bigger = round((sizes[-1] + GROW) / STEP) * STEP
        if bigger <= MAX_SIZE:
            return "run", bigger  # 가장 큰 크기에서도 50% 이상이면 더 키워서 잰다
    return "fail", None  # 50%를 사이에 두는 두 크기가 없음


runs = 0
while True:
    m = measured()
    state, size = decide(m)
    print(f"[find_50 {KIND}] 측정값 " + ", ".join(f"{FMT.format(s)}: {k}/{n}" for s, (k, n) in m.items()), flush=True)
    if state != "run" or runs >= MAX_RUNS:
        break
    name = PREFIX + FMT.format(size)
    print(f"[find_50 {KIND}] {name} 측정", flush=True)
    subprocess.run(["bash", os.path.join(HERE, "calib_one.sh"), name, ARG, FMT.format(size)])
    runs += 1

result = {"kind": KIND, "state": state, "size": size, "measured": {FMT.format(s): v for s, v in m.items()}}
with open(os.path.join(CALIB, f"find_50_{KIND}.json"), "w") as f:
    json.dump(result, f, indent=1)
print(f"[find_50 {KIND}] 결과: {result}", flush=True)
