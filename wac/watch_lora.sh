#!/usr/bin/env bash
# 서버에서 파인튜닝을 감시하다가 프로세스가 죽으면 마지막 체크포인트부터 다시 띄운다.
# 같은 실행 이름으로 다시 시작하면 공식 학습 코드가 latest_checkpoint.txt를 읽어
# 모델, optimizer, 스케줄러, 갱신 수를 이어 받는다.
# 사용: nohup setsid bash wac/watch_lora.sh 4090_lora_v1 > ~/wac_cosmos/watch_lora.log 2>&1 < /dev/null &
RUN=${1:-4090_lora_v1}
MAX_RESTART=${MAX_RESTART:-5}
DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$DIR")
LOG=$REPO/data/finetune/$RUN/train.log
CKDIR=$(ls -d "$REPO"/data/finetune/$RUN/raw/*/*/*/checkpoints 2>/dev/null | head -1)
restarts=0

running() { pgrep -f "wac/run_lora.sh $RUN" >/dev/null; }
finished() { tail -n 200 "$LOG" 2>/dev/null | grep -q "Done with training"; }
last_ckpt() { cat "${CKDIR:-/nonexistent}/latest_checkpoint.txt" 2>/dev/null || echo "없음"; }
gpus_free() {  # GPU 0, 1번 모두 사용 메모리가 2GB 미만
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '$1 >= 2048 {busy=1} END {exit busy}'
}

echo "[$(date '+%m-%d %H:%M')] 감시 시작: $RUN"
while true; do
  sleep 120
  running && continue
  if finished; then
    echo "[$(date '+%m-%d %H:%M')] 학습 끝 (마지막 체크포인트 $(last_ckpt)). 감시 종료"
    exit 0
  fi
  if (( restarts >= MAX_RESTART )); then
    echo "[$(date '+%m-%d %H:%M')] ${MAX_RESTART}번 다시 띄워도 끝나지 않아 멈춤. 마지막 체크포인트 $(last_ckpt)"
    exit 1
  fi
  echo "[$(date '+%m-%d %H:%M')] 학습 프로세스가 없음. 마지막 체크포인트 $(last_ckpt). 로그 끝:"
  tail -n 5 "$LOG" | cut -c1-200
  until gpus_free; do
    echo "[$(date '+%m-%d %H:%M')] GPU 0, 1번을 다른 프로세스가 쓰는 중. 기다림"
    sleep 300
  done
  restarts=$((restarts + 1))
  echo "[$(date '+%m-%d %H:%M')] 다시 시작 (${restarts}번째)"
  ( cd "$REPO" && nohup setsid bash wac/run_lora.sh "$RUN" epoch >> "$HOME/wac_cosmos/lora_v1.out" 2>&1 < /dev/null & )
  sleep 300   # 모델을 불러오는 동안은 다시 확인하지 않는다
  CKDIR=$(ls -d "$REPO"/data/finetune/$RUN/raw/*/*/*/checkpoints 2>/dev/null | head -1)
done
