"""PROPOSAL 5.7 denoising step 수 실험의 작업 목록 (parallel_runner.py 입력).

본수집과 같은 교란 조건, 같은 묶음(calib 초기 상태 10~19 seed 2, eval 20~29 seed 3)에서
--num_denoising_steps_action만 바꿔 다시 돌린다. 5 step은 본수집 데이터를 그대로 쓴다.
train 묶음은 탐지 평가에 쓰지 않으므로 돌리지 않는다.

사용: python wac/make_step_jobs.py > ~/wac_cosmos/jobs_steps.json
출력 폴더: ~/wac_cosmos/repo/data/libero_spatial_steps/steps<k>/<조건>/<묶음>/
"""

import json
import os

ROOT = os.path.expanduser("~/wac_cosmos/repo/data/libero_spatial_steps")
CONDITIONS = {"obj_0.090": ["--obj_xy_noise", "0.09"], "wristroll_50": ["--wrist_roll_deg", "50"]}
SPLITS = [("calib", 10, 10, 2), ("eval", 20, 10, 3)]
STEPS = [1, 2, 10]

jobs = []
for split, start, n, seed in SPLITS:          # calib을 먼저 모두 돌려 보정이 먼저 가능하게 한다
    for k in STEPS:
        for cond, args in CONDITIONS.items():
            out = os.path.join(ROOT, f"steps{k}", cond, split)
            for init in range(start, start + n):
                if os.path.exists(os.path.join(out, "_logs", f"{split}_i{init}", "DONE")):
                    continue
                jobs.append({"name": f"{split}_i{init}", "trials": 1, "init_offset": init, "out": out,
                             "args": [*args, "--num_denoising_steps_action", str(k)],
                             "perturb_seed": seed})
print(json.dumps(jobs, indent=1))
