#!/usr/bin/env bash
# 교란 크기 하나만 시험한다 (calibrate_sweep.sh와 같은 조건: seed 9000, 초기 상태 0~1, 지시문당 2번).
# 사용: bash calib_one.sh <이름> <교란 인자...>     예: bash calib_one.sh obj_0.10 --obj_xy_noise 0.10
# 다른 시험이 돌고 있으면 끝날 때까지 기다렸다가 시작한다.
NAME=${1:?이름 필요}; shift
OUT=$HOME/wac/logs/calib
DIR=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "[c]alibrate_sweep.sh" >/dev/null || pgrep -f "[w]ac/perturbed_eval.py" >/dev/null; do sleep 30; done
if [ -f "$OUT/$NAME/DONE" ]; then echo "[skip] $NAME"; exit 0; fi
rm -rf "$OUT/$NAME" "$OUT/$NAME.jsonl"; mkdir -p "$OUT/$NAME"
echo "[start] $NAME $(date +%H:%M:%S)"
bash "$DIR/run_perturbed_eval.sh" libero_spatial 2 "$@" \
  --perturb_seed 9000 --perturb_log "$OUT/$NAME.jsonl" \
  --episode_dir "$OUT/$NAME" --show "${SHOW:-True}" \
  --local_log_dir "$OUT/$NAME/official/" > "$OUT/$NAME.log" 2>&1
code=$?
[ $code -eq 0 ] && touch "$OUT/$NAME/DONE"
echo "[done] $NAME $(date +%H:%M:%S) exit=$code"
