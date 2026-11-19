#!/usr/bin/env bash
# 파인튜닝(4090_lora_v2)이 끝나면 GPU 0, 1에서 남은 신호 F, hidden feature 수집을 시작한다.
# 끝은 학습 로그의 "Done with training" 문구로 판단한다.
cd ~/wac_cosmos/repo
f=data/finetune/4090_lora_v2/train.log
until tail -n 300 "$f" 2>/dev/null | grep -q "Done with training"; do sleep 60; done
sleep 60   # 학습 프로세스가 GPU 메모리를 놓을 때까지
python3 wac/make_f_jobs.py > ~/wac_cosmos/jobs_f.json
~/wac_cosmos/cosmos-policy/.venv/bin/python wac/parallel_runner.py --gpus 0,1 --per_gpu 3 \
  --jobs ~/wac_cosmos/jobs_f.json >> ~/wac_cosmos/f_run.log 2>&1
