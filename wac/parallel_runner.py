"""서버에서 여러 시뮬레이터를 동시에 돌린다.

작업(job) 하나 = run_perturbed_eval.sh 한 번 실행 (지시문 10개 × trials번, 초기 상태 init_offset부터).
슬롯 하나 = GPU 하나 위의 프로세스 하나. 슬롯이 비면 다음 작업을 넣는다.

사용:
  python parallel_runner.py --gpus 2,3 --per_gpu 3 --jobs jobs.json
jobs.json: [{"name": "...", "trials": 2, "init_offset": 0, "out": "<묶음 폴더>",
             "args": ["--cam_yaw_deg", "30"], "perturb_seed": 9000, "data_collection": false}, ...]
"""

import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.expanduser("~/wac_cosmos/live")

p = argparse.ArgumentParser()
p.add_argument("--gpus", default="2,3")
p.add_argument("--per_gpu", type=int, default=3)
p.add_argument("--jobs", required=True)
a = p.parse_args()

jobs = json.load(open(a.jobs))
slots = [(g, i) for g in a.gpus.split(",") for i in range(a.per_gpu)]
running = {}   # slot -> (Popen, job)
queue = list(jobs)
t0 = time.time()


def start(slot, job):
    gpu, i = slot
    out = job["out"]
    logs = os.path.join(out, "_logs", job["name"])
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu, PYTHONUNBUFFERED="1")
    cmd = ["bash", os.path.join(HERE, "run_perturbed_eval.sh"), "libero_spatial", str(job["trials"]),
           *job["args"],
           "--init_offset", str(job["init_offset"]), "--perturb_seed", str(job["perturb_seed"]),
           "--perturb_log", os.path.join(logs, "perturb.jsonl"),
           "--episode_dir", out, "--show", "False",
           "--live_dir", LIVE, "--live_name", f"g{gpu}s{i}_{job['name']}",
           "--local_log_dir", os.path.join(logs, "official") + "/"]
    if job.get("data_collection"):
        cmd += ["--data_collection", "True"]
    f = open(os.path.join(logs, "run.log"), "w")
    running[slot] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), job)
    print(f"[{time.strftime('%H:%M:%S')}] 시작 {job['name']} (GPU {gpu} 슬롯 {i})", flush=True)


while queue or running:
    for slot in slots:
        if slot in running:
            proc, job = running[slot]
            if proc.poll() is None:
                continue
            code = proc.returncode
            print(f"[{time.strftime('%H:%M:%S')}] 끝 {job['name']} exit={code}", flush=True)
            logs = os.path.join(job["out"], "_logs", job["name"])
            if code == 0:
                open(os.path.join(logs, "DONE"), "w").close()
            else:
                # 에피소드를 하나도 못 남기고 실패한 작업(예: 모델 로드 중 메모리 부족)은 다시 넣는다.
                # 에피소드를 남긴 뒤 실패한 작업은 다시 돌리면 같은 에피소드가 두 번 기록되므로 넣지 않는다.
                pl = os.path.join(logs, "perturb.jsonl")
                done_eps = sum(1 for _ in open(pl)) if os.path.exists(pl) else 0
                if done_eps == 0 and job.get("retries", 0) < 2:
                    job["retries"] = job.get("retries", 0) + 1
                    queue.append(job)
                    print(f"[{time.strftime('%H:%M:%S')}] 다시 넣음 {job['name']} ({job['retries']}번째)", flush=True)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 실패로 남김 {job['name']} (에피소드 {done_eps}개 기록됨)", flush=True)
            del running[slot]
        if queue:
            start(slot, queue.pop(0))
            time.sleep(20)  # 모델을 동시에 올리면 메모리가 순간적으로 몰려서 조금씩 띄운다
    time.sleep(5)
print(f"[{time.strftime('%H:%M:%S')}] 전체 끝 ({(time.time() - t0) / 60:.0f}분)", flush=True)
