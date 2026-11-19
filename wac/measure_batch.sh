#!/usr/bin/env bash
# GPU 0, 1번에서 배치별로 한 스텝씩 학습하며 최대 GPU 메모리를 잰다.
# 사용: nohup setsid bash wac/measure_batch.sh 1 2 4 > ~/wac_cosmos/measure_batch.log 2>&1 < /dev/null &
# 결과: data/finetune/measure_batch.tsv (배치, 종료 코드, GPU0 최대 MiB, GPU1 최대 MiB, 걸린 초)
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=$(dirname "$DIR")/data/finetune/measure_batch.tsv
mkdir -p "$(dirname "$OUT")"
echo -e "batch\texit\tgpu0_peak_mib\tgpu1_peak_mib\tseconds" > "$OUT"
for B in "$@"; do
  RUN=4090_lora_mem_b$B
  rm -rf "$(dirname "$DIR")/data/finetune/$RUN"
  mkdir -p "$(dirname "$DIR")/data/finetune/$RUN"
  POLL=$(mktemp)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | paste -sd' ' >> "$POLL"; sleep 1; done ) &
  PID=$!
  T0=$(date +%s)
  # 6 step (누적 없음). 첫 step은 모델 로드와 준비가 섞여서 속도 계산에서 뺀다.
  LOCAL_BATCH=$B bash "$DIR/run_lora.sh" "$RUN" 6 1 1000 > /dev/null 2>&1
  CODE=$?
  T1=$(date +%s)
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  P0=$(awk '{if($1>m)m=$1} END{print m+0}' "$POLL")
  P1=$(awk '{if($2>m)m=$2} END{print m+0}' "$POLL")
  rm -f "$POLL"
  echo -e "$B\t$CODE\t$P0\t$P1\t$((T1-T0))" | tee -a "$OUT"
  sleep 10   # 다음 배치 전에 GPU 메모리가 풀리게 기다린다
done
echo "끝"
