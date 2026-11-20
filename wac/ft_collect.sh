#!/usr/bin/env bash
# 기본 모델의 신호 F, hidden 수집(40개 작업)이 끝나면 파인튜닝 모델로 같은 에피소드를 수집한다 (GPU 0, 1).
# 먼저 작업 하나로 시험해 결과 파일(DONE, npz)을 확인하고, 통과하면 나머지를 돌린다.
cd ~/wac_cosmos/repo
until [ "$(find data/libero_spatial_F -name DONE | wc -l)" -ge 40 ]; do sleep 120; done
sleep 60
export WAC_CKPT=~/wac_cosmos/ft_ckpt/v2_iter6576.pt
export WAC_F_ROOT=~/wac_cosmos/repo/data/libero_spatial_F_ft
PY=~/wac_cosmos/cosmos-policy/.venv/bin/python
python3 wac/make_f_jobs.py > ~/wac_cosmos/jobs_ft.json
python3 -c "import json;j=json.load(open('$HOME/wac_cosmos/jobs_ft.json'));json.dump(j[:1],open('$HOME/wac_cosmos/jobs_ft_test.json','w'),indent=1)"
$PY wac/parallel_runner.py --gpus 0 --per_gpu 1 --jobs ~/wac_cosmos/jobs_ft_test.json >> ~/wac_cosmos/ft_run.log 2>&1
n_done=$(find $WAC_F_ROOT -name DONE | wc -l); n_npz=$(find $WAC_F_ROOT -name "*.npz" | wc -l)
echo "[$(date +%H:%M)] 시험: DONE $n_done, npz $n_npz" >> ~/wac_cosmos/ft_run.log
if [ "$n_done" -lt 1 ] || [ "$n_npz" -lt 1 ]; then echo FAIL > ~/wac_cosmos/FT_TEST_FAIL; exit 1; fi
$PY wac/parallel_runner.py --gpus 0,1 --per_gpu 3 --jobs ~/wac_cosmos/jobs_ft.json >> ~/wac_cosmos/ft_run.log 2>&1
touch ~/wac_cosmos/FT_COLLECT_DONE
