#!/usr/bin/env bash
# 교란 크기 정하기 (PROPOSAL 5.2의 2번). 단계마다 따로 평가를 돌리고 결과를 남긴다.
# 사용: nohup setsid bash calibrate_sweep.sh [지시문당 횟수] > ~/wac/logs/calib/sweep.log 2>&1 &
# 크기 정하기 전용 seed(9000)는 본 수집에 쓰지 않는다.
# 감시 창은 기본으로 뜬다. 끄려면 SHOW=False. 끝난 단계(DONE 파일 있음)는 건너뛴다.
SUITE=libero_spatial
TRIALS=${1:-2}
SEED=9000
OUT=$HOME/wac/logs/calib
mkdir -p "$OUT"
DIR=$(cd "$(dirname "$0")" && pwd)

run() {  # run <이름> <교란 인자...>
  local name=$1; shift
  if [ -f "$OUT/$name/DONE" ]; then echo "[skip] $name"; return; fi
  rm -rf "$OUT/$name" "$OUT/$name.jsonl"; mkdir -p "$OUT/$name"
  echo "[start] $name $(date +%H:%M:%S)"
  bash "$DIR/run_perturbed_eval.sh" "$SUITE" "$TRIALS" "$@" \
    --perturb_seed $SEED --perturb_log "$OUT/$name.jsonl" \
    --episode_dir "$OUT/$name" --show "${SHOW:-True}" \
    --local_log_dir "$OUT/$name/official/" > "$OUT/$name.log" 2>&1
  local code=$?
  [ $code -eq 0 ] && touch "$OUT/$name/DONE"
  echo "[done] $name $(date +%H:%M:%S) exit=$code"
}

# 시험할 조건: 인자 없으면 카메라만 (물체는 ±8, ±12cm로 끝냄)
WHICH=${2:-cam}
if [ "$WHICH" = "obj" ] || [ "$WHICH" = "all" ]; then
  run obj_0.08 --obj_xy_noise 0.08
  run obj_0.12 --obj_xy_noise 0.12
fi
if [ "$WHICH" = "cam" ] || [ "$WHICH" = "all" ]; then
  run cam_yaw15 --cam_yaw_deg 15
  run cam_yaw30 --cam_yaw_deg 30
  run cam_yaw45 --cam_yaw_deg 45
fi
echo "[all done] $(date +%H:%M:%S)"
