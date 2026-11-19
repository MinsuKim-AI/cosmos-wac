#!/usr/bin/env bash
# PROPOSAL 5.2 데이터 수집. 교란 조건 2개 × 묶음 3개(보정용, 평가용, 학습용).
# 사용: nohup setsid bash collect.sh <obj_xy_noise> <cam_yaw_deg> > ~/wac/logs/collect.log 2>&1 &
#
# 저장 위치: 저장소의 data/libero_spatial/  (구조는 docs/Experiment.md 5장)
#   index.csv, README.md                         전체 에피소드 목록, 조건별 성공률
#   <조건>/<묶음>/result.md                       지시문별 성공률
#   <조건>/<묶음>/D<n>_<지시문>/result.md          지시문 하나의 에피소드 표
#   <조건>/<묶음>/D<n>_<지시문>/init<번호>_<succ|fail>.{mp4,npz,hdf5}, ..._frames/
#   <조건>/<묶음>/_logs/                          실행 로그, 교란 기록
# 실시간 감시 창은 기본으로 뜬다. 끄려면 SHOW=False bash collect.sh ...
#
# 묶음은 초기 상태 번호와 교란 seed로 나눈다. 크기 정하기(seed 9000)와도 겹치지 않는다.
#   보정용  init 10~19, seed 2, 지시문당 10회
#   평가용  init 20~29, seed 3, 지시문당 10회
#   학습용  init  0~4,  seed 1, 지시문당 5회
# 이미 끝난 (조건, 묶음)은 건너뛴다. 중간에 멈춰도 다시 실행하면 이어서 한다.
set -u
OBJ=${1:?obj_xy_noise 필요}
YAW=${2:?cam_yaw_deg 필요}
SUITE=libero_spatial
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$DIR")/data/$SUITE      # 저장소의 data/ 폴더 (git에는 안 올림)
PY="$HOME/cosmos-venv/bin/python"
mkdir -p "$ROOT"

run() {  # run <조건> <묶음> <init_offset> <seed> <trials> <교란 인자...>
  local cond=$1 split=$2 off=$3 seed=$4 trials=$5; shift 5
  local out=$ROOT/$cond/$split
  if [ -f "$out/_logs/DONE" ]; then echo "[skip] $cond/$split"; return; fi
  rm -rf "$out"; mkdir -p "$out/_logs"
  echo "[start] $cond/$split $(date '+%m-%d %H:%M:%S')"
  bash "$DIR/run_perturbed_eval.sh" "$SUITE" "$trials" "$@" \
    --init_offset "$off" --perturb_seed "$seed" \
    --perturb_log "$out/_logs/perturb.jsonl" \
    --episode_dir "$out" --show "${SHOW:-True}" \
    --data_collection True --local_log_dir "$out/_logs/official/" > "$out/_logs/run.log" 2>&1
  local code=$?
  [ $code -eq 0 ] && touch "$out/_logs/DONE"
  "$PY" -I "$DIR/summarize_split.py" "$out" || true
  "$PY" -I "$DIR/summarize_split.py" --index "$ROOT" || true
  echo "[done] $cond/$split exit=$code $(date '+%m-%d %H:%M:%S') 성공 $(grep -c '"success": true' "$out/_logs/perturb.jsonl" 2>/dev/null)/$(wc -l < "$out/_logs/perturb.jsonl" 2>/dev/null)"
}

for split in calib eval train; do
  case $split in
    calib) off=10; seed=2; n=10 ;;
    eval)  off=20; seed=3; n=10 ;;
    train) off=0;  seed=1; n=5 ;;
  esac
  # 크기 자리에 - 를 주면 그 조건은 건너뛴다
  [ "$OBJ" != "-" ] && run "obj_$OBJ"   $split $off $seed $n --obj_xy_noise "$OBJ"
  [ "$YAW" != "-" ] && run "camyaw_$YAW" $split $off $seed $n --cam_yaw_deg "$YAW"
done
echo "[all done] $(date '+%m-%d %H:%M:%S')"
