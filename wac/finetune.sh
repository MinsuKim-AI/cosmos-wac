#!/usr/bin/env bash
# DRY=1이면 데이터 로더와 설정만 확인한다.
set -euo pipefail

RUN=${1:?run 이름 필요}
EPOCHS=${2:-3}
NGPU=${NGPU:-2}
LOCAL_BATCH=${LOCAL_BATCH:-1}
EFF_BATCH=${EFF_BATCH:-32}
LR=${LR:-2e-5}
DEMO_PROB=${DEMO_PROB:-0.25}

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$DIR")
OUT=$REPO/data/finetune/$RUN
# 기존 환경 파일에서 쓰는 선택 환경 변수를 허용한다.
set +u
source "$DIR/env.sh"
set -u

DATA=$REPO/data/libero_spatial
TRAIN_DIR=$REPO/data/finetune/train_rollouts
DEMOS=$REPO/data/finetune/base/success_only
SNAPSHOT=$(dirname "${WAC_STATS:?LIBERO checkpoint snapshot not found}")
PUBLISHED=$SNAPSHOT/Cosmos-Policy-LIBERO-Predict2-2B.pt
CHECKPOINT=${CHECKPOINT:-$REPO/data/finetune/base/libero_policy_dcp}
EMBEDDINGS=$SNAPSHOT/libero_t5_embeddings.pkl
test -s "$PUBLISHED"
test -s "$EMBEDDINGS"
test -s "$DEMOS/dataset_statistics.json"
test -s "$DEMOS/dataset_statistics_post_norm.json"

MICRO_BATCH=$(( LOCAL_BATCH * NGPU ))
if (( MICRO_BATCH < 1 || EFF_BATCH < MICRO_BATCH || EFF_BATCH % MICRO_BATCH != 0 || EPOCHS < 1 )); then
  echo '유효 배치는 GPU 수 × GPU당 배치의 양의 배수여야 하고 epoch는 1 이상이어야 한다.' >&2
  exit 1
fi
mkdir -p "$OUT/raw"
CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$DIR/prepare_checkpoint.py" "$PUBLISHED" "$CHECKPOINT"
CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$DIR/prepare_finetune.py" \
  "$DATA" "$TRAIN_DIR" "$DEMOS" "$EMBEDDINGS" "$DEMO_PROB" | tee "$OUT/dataset.log"
tail -1 "$OUT/dataset.log" > "$OUT/dataset.json"
N_SAMPLES=$("$VENV/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["samples_per_epoch"])' "$OUT/dataset.json")
ACCUM=$(( EFF_BATCH / MICRO_BATCH ))
SPE=$(( (N_SAMPLES + NGPU - 1) / NGPU / LOCAL_BATCH / ACCUM ))
if (( SPE < 1 )); then
  echo '샘플 수가 유효 배치보다 적다.' >&2
  exit 1
fi
MAX_ITER=$(( SPE * EPOCHS ))
echo "1 epoch = $SPE step, $EPOCHS epoch = $MAX_ITER step"

export BASE_DATASETS_DIR=$REPO/data/finetune/base
export IMAGINAIRE_OUTPUT_ROOT=$OUT/raw
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$COSMOS_POLICY_DIR"
CONFIG=cosmos_policy/config/config.py
OVERRIDES=(
  experiment=cosmos_predict2_2b_480p_libero
  "checkpoint.load_path=$CHECKPOINT"
  checkpoint.load_ema_to_reg=False
  "checkpoint.save_iter=$SPE"
  "trainer.max_iter=$MAX_ITER"
  "trainer.grad_accum_iter=$ACCUM"
  "model.config.fsdp_shard_size=$NGPU"
  "dataloader_train.batch_size=$LOCAL_BATCH"
  dataloader_train.num_workers=2
  "dataloader_train.dataset.data_dir=$DEMOS"
  "dataloader_train.dataset.t5_text_embeddings_path=$EMBEDDINGS"
  "dataloader_train.dataset.rollout_data_dir=$TRAIN_DIR"
  "dataloader_train.dataset.demonstration_sampling_prob=$DEMO_PROB"
  "optimizer.lr=$LR"
  "job.name=$RUN"
)
if [[ ${DRY:-0} == 1 ]]; then
  "$VENV/bin/python" -m cosmos_policy.scripts.train \
    --config="$CONFIG" --dryrun -- "${OVERRIDES[@]}" > "$OUT/config.log" 2>&1
  tail -1 "$OUT/config.log"
  exit 0
fi

"$VENV/bin/torchrun" --nproc_per_node="$NGPU" --standalone -m cosmos_policy.scripts.train \
  --config="$CONFIG" -- "${OVERRIDES[@]}" 2>&1 | tee "$OUT/train.log"
"$VENV/bin/python" "$DIR/organize_epochs.py" "$OUT" "$SPE"
touch "$OUT/TRAINING_DONE"
