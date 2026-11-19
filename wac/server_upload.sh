#!/usr/bin/env bash
# 서버 수집(server_pipeline.py)이 끝나면 데이터셋을 Hugging Face에 올린다. 세션이나 노트북과 상관없이 서버에서 돈다.
# 사용: nohup setsid bash server_upload.sh > ~/wac_cosmos/upload.log 2>&1 < /dev/null &
set -e

REPO=MINSUAI/cosmos-wac-libero-spatial
HF=$HOME/.local/bin/hf
DATA=$HOME/wac_cosmos/repo/data/libero_spatial
W=$(cd "$(dirname "$0")" && pwd)

until grep -q '"stage": "finished"' "$HOME/wac_cosmos/status.json" 2>/dev/null; do
  if grep -q '"stage": "stopped' "$HOME/wac_cosmos/status.json" 2>/dev/null; then echo "[upload] 수집이 멈춰서 올리지 않음"; exit 1; fi
  sleep 60
done
echo "[upload] 수집 끝 확인 $(date '+%m-%d %H:%M')"

if ! $HF auth whoami >/dev/null 2>&1; then echo "[upload] 서버 Hugging Face 로그인이 없어 올리지 않음"; exit 1; fi

# 데이터셋 카드: 카드 본문 + 수집 결과 요약(README.md)
if ! grep -q '^## 수집 결과$' "$DATA/README.md"; then
  { cat "$W/dataset_card.md"; echo; echo "## 수집 결과"; echo; tail -n +3 "$DATA/README.md"; } > "$DATA/README.card.md"
  mv "$DATA/README.card.md" "$DATA/README.md"
fi

# 실행 로그 폴더(_logs)는 올리지 않는다
$HF repo create "$REPO" --repo-type dataset --public --exist-ok
$HF upload-large-folder "$REPO" "$DATA" --repo-type dataset --exclude "*/_logs/*" --exclude "_logs/*" \
  --num-workers 8 --no-bars
echo "[upload] 끝 $(date '+%m-%d %H:%M') https://huggingface.co/datasets/$REPO"
