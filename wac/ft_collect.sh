#!/usr/bin/env bash
# 파인튜닝 모델로 기본 모델과 같은 에피소드를 수집하고(신호 F, hidden, HDF5), 신호 C를 계산한 뒤 평가한다 (GPU 0, 1).
# 먼저 작업 하나로 수집·C 계산을 시험하고, 통과하면 나머지를 돌린다. 수집은 실패한 작업만 최대 3번 다시 돌린다.
cd ~/wac_cosmos/repo
source wac/env.sh
W=~/wac_cosmos; L=$W/ft_run.log
log() { echo "[$(date +%m-%d\ %H:%M)] $*" >> $L; }
export WAC_CKPT=$W/ft_ckpt/v2_iter6576.pt
export WAC_F_ROOT=$W/repo/data/libero_spatial_F_ft
export WAC_HDF5=1
PY=$VENV/bin/python
enc() {  # 신호 C: 실제 미래 이미지 latent (GPU 0, 1 나눠서)
  (cd $COSMOS_POLICY_DIR && for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $PY $WAC_DIR/wac/encode_actual.py $1 --shard $g --num_shards 2 > $W/ft_encode_$g.log 2>&1 & done; wait)
}

python3 wac/make_f_jobs.py calib_i10 > $W/jobs_ft_test.json
$PY wac/parallel_runner.py --gpus 0,1 --per_gpu 1 --jobs $W/jobs_ft_test.json >> $L 2>&1
enc $WAC_F_ROOT
n_done=$(find $WAC_F_ROOT -name DONE | wc -l); n_npz=$(find $WAC_F_ROOT -name "*.npz" ! -name "*_actual.npz" | wc -l)
n_h5=$(find $WAC_F_ROOT -name "*.hdf5" | wc -l); n_act=$(find $WAC_F_ROOT -name "*_actual.npz" | wc -l)
log "시험: DONE $n_done, npz $n_npz, hdf5 $n_h5, actual $n_act"
if [ "$n_done" -lt 1 ] || [ "$n_npz" -lt 1 ] || [ "$n_h5" -lt 1 ] || [ "$n_act" -lt 1 ]; then echo FAIL > $W/FT_TEST_FAIL; exit 1; fi

for try in 1 2 3; do
  python3 wac/make_f_jobs.py > $W/jobs_ft.json
  n=$(python3 -c "import json;print(len(json.load(open('$W/jobs_ft.json'))))")
  log "수집 $try회차: 남은 작업 $n"
  [ "$n" -eq 0 ] && break
  $PY wac/parallel_runner.py --gpus 0,1 --per_gpu 3 --jobs $W/jobs_ft.json >> $L 2>&1
done
n_done=$(find $WAC_F_ROOT -name DONE | wc -l); log "수집 끝: DONE $n_done/40"
[ "$n_done" -ge 40 ] || { echo FAIL > $W/FT_COLLECT_FAIL; exit 1; }
touch $W/FT_COLLECT_DONE

enc $WAC_F_ROOT
log "C 계산 끝: actual $(find $WAC_F_ROOT -name '*_actual.npz' | wc -l)"
OMP_NUM_THREADS=8 $PY wac/evaluate.py $WAC_F_ROOT $WAC_STATS results/detection_ft_full.json --f_dir $WAC_F_ROOT > results/detection_ft_full.log 2>&1 \
  && touch $W/FT_EVAL_DONE || { echo FAIL > $W/FT_EVAL_FAIL; log "평가 실패"; }
log "평가 끝"
# 비교표와 정성 분석 그림 (노트북 없이 미리 만들어 둔다)
[ -f $W/FT_EVAL_DONE ] && {
  $PY wac/compare_models.py results/detection_base_full_v2.json results/detection_ft_full.json results/compare_base_ft.md > results/compare_base_ft.log 2>&1
  MPLBACKEND=Agg $PY wac/make_figures.py results/detection_ft_full.json $WAC_F_ROOT results/figures_ft --det D > results/figures_ft.log 2>&1
  log "비교표·그림 끝 (results/compare_base_ft.md, results/figures_ft)"; touch $W/FT_REPORT_DONE; }
