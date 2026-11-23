#!/usr/bin/env bash
# 확인 평가용 수집: 공개 모델로 train 묶음(조건당 50 에피소드)에서 신호 F와 hidden을 모은다 (GPU 0, 1).
# 작업 2개(두 조건의 초기 상태 0)로 먼저 시험해 결과 파일을 확인하고, 통과하면 나머지를 돌린다.
cd ~/wac_cosmos/repo
W=~/wac_cosmos; L=$W/confirm_run.log
log() { echo "[$(date +%m-%d\ %H:%M)] $*" >> $L; }
unset WAC_CKPT
export WAC_F_SPLITS=train
PY=~/wac_cosmos/cosmos-policy/.venv/bin/python
R=data/libero_spatial_F
python3 wac/make_f_jobs.py train_i0 > $W/jobs_confirm_test.json
$PY wac/parallel_runner.py --gpus 0,1 --per_gpu 1 --jobs $W/jobs_confirm_test.json >> $L 2>&1
n_done=$(find $R/*/train -name DONE | wc -l)
n_ok=$($PY -c "
import glob, numpy as np
fs=[f for f in glob.glob('$R/*/train/D*/init*_*.npz')]
print(sum(1 for f in fs if {'hidden','extra_latent'} <= set(np.load(f).files)))")
log "시험: DONE $n_done, hidden+extra_latent npz $n_ok"
if [ "$n_done" -lt 2 ] || [ "$n_ok" -lt 20 ]; then echo FAIL > $W/CONFIRM_TEST_FAIL; exit 1; fi
for try in 1 2 3; do
  python3 wac/make_f_jobs.py > $W/jobs_confirm.json
  n=$(python3 -c "import json;print(len(json.load(open('$W/jobs_confirm.json'))))")
  log "수집 $try회차: 남은 작업 $n"
  [ "$n" -eq 0 ] && break
  $PY wac/parallel_runner.py --gpus 0,1 --per_gpu 3 --jobs $W/jobs_confirm.json >> $L 2>&1
done
n_done=$(find $R/*/train -name DONE | wc -l); log "수집 끝: DONE $n_done/10"
[ "$n_done" -ge 10 ] && touch $W/CONFIRM_COLLECT_DONE || echo FAIL > $W/CONFIRM_COLLECT_FAIL
