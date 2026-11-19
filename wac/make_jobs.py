"""서버 병렬 실행용 작업 목록(jobs.json) 만들기.

  python make_jobs.py calib cam 45 60 75 90      카메라 각도 시험 (각 20번 = 초기 상태 0, 1 × 지시문 10개)
  python make_jobs.py collect cam 60             카메라 조건 본 수집 (calib/eval/train)
출력은 표준 출력. 각 작업은 지시문 10개 × 1번 (초기 상태 하나)이라 슬롯 수만큼 동시에 돈다.
"""

import json
import os
import sys

ROOT = os.path.expanduser("~/wac_cosmos/repo/data/libero_spatial")
CALIB = os.path.expanduser("~/wac_cosmos/calib")
mode, kind, *sizes = sys.argv[1:]
flag, prefix = {"cam": ("--cam_yaw_deg", "camyaw_"), "obj": ("--obj_xy_noise", "obj_")}[kind]

jobs = []
if mode == "calib":
    for s in sizes:
        for init in (0, 1):
            jobs.append({"name": f"{prefix}{s}_i{init}", "trials": 1, "init_offset": init,
                         "out": os.path.join(CALIB, f"{prefix}{s}"), "args": [flag, s], "perturb_seed": 9000})
elif mode == "collect":
    (s,) = sizes
    for split, start, n, seed in [("calib", 10, 10, 2), ("eval", 20, 10, 3), ("train", 0, 5, 1)]:
        for init in range(start, start + n):
            jobs.append({"name": f"{split}_i{init}", "trials": 1, "init_offset": init,
                         "out": os.path.join(ROOT, f"{prefix}{s}", split), "args": [flag, s],
                         "perturb_seed": seed, "data_collection": True})
print(json.dumps(jobs, indent=1, ensure_ascii=False))
