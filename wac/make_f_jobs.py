"""신호 F(샘플 간 흩어짐) 수집 작업 목록 (parallel_runner.py 입력).

본수집과 같은 교란 조건, 같은 묶음(calib 초기 상태 10~19 seed 2, eval 20~29 seed 3)을 다시 돌리면서
chunk마다 seed만 바꾼 표본을 3개 더 만들어 저장한다(--n_samples 4).
실행하는 행동은 원래 seed의 표본이라 에피소드는 본수집과 똑같이 진행된다.

사용: python wac/make_f_jobs.py > ~/wac_cosmos/jobs_f.json
출력 폴더: ~/wac_cosmos/repo/data/libero_spatial_F/<조건>/<묶음>/
"""

import json
import os
import sys

ROOT = os.path.expanduser(os.environ.get("WAC_F_ROOT", "~/wac_cosmos/repo/data/libero_spatial_F"))  # 파인튜닝 모델은 다른 폴더
CONDITIONS = {"obj_0.090": ["--obj_xy_noise", "0.09"], "wristroll_50": ["--wrist_roll_deg", "50"]}
SPLITS = [("calib", 10, 10, 2), ("eval", 20, 10, 3)]
only = sys.argv[1:]  # 예: calib_i10 하나만 (시험용)

jobs = []
for split, start, n, seed in SPLITS:
    for cond, args in CONDITIONS.items():
        out = os.path.join(ROOT, cond, split)
        for init in range(start, start + n):
            name = f"{split}_i{init}"
            if only and name not in only:
                continue
            if os.path.exists(os.path.join(out, "_logs", name, "DONE")):
                continue
            jobs.append({"name": name, "trials": 1, "init_offset": init, "out": out,
                         "args": [*args, "--n_samples", "4", "--hidden", "6,13,20,27"], "perturb_seed": seed,
                         "data_collection": os.environ.get("WAC_HDF5") == "1"})  # 파인튜닝 모델: 신호 C용 HDF5도 저장
print(json.dumps(jobs, indent=1))
