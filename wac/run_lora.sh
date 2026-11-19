#!/usr/bin/env bash
# 서버 OASYS에서 LoRA 파인튜닝 (DDP). GPU는 GPUS 환경변수로 고른다 (기본 0,1).
# 사용: bash wac/run_lora.sh <run 이름> <max_iter> <grad_accum> <save_iter>
#   몇 스텝 확인: bash wac/run_lora.sh 4090_lora_check 8 1 1000
#   본 학습:     bash wac/run_lora.sh 4090_lora_v1 epoch    (1 epoch, GPU당 배치 4, 유효 배치 32)
# 서버 사본에 wac/patches의 두 패치를 적용해 둔다 (로그 집계 멈춤, LoRA 초기화).
set -eo pipefail
RUN=${1:?run 이름 필요}; MAX_ITER=${2:?max_iter 필요}; ACCUM=${3:-16}; SAVE_ITER=${4:-}
DIR=$(cd "$(dirname "$0")" && pwd)
WAC_REPO=$(dirname "$DIR")
source "$DIR/env.sh"
EMB=$(dirname "$WAC_STATS")/libero_t5_embeddings.pkl
BASE=$WAC_REPO/data/finetune/base
GPUS=${GPUS:-0,1}                         # 쓸 GPU 번호
NGPU=$(echo "$GPUS" | awk -F, '{print NF}')
LOCAL_BATCH=${LOCAL_BATCH:-4}               # GPU당 배치. 4090에서 4가 최적 (8은 메모리 부족)
if [ "$MAX_ITER" = "epoch" ]; then
  SAMPLES=$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["samples_per_epoch"])' \
    "$WAC_REPO/data/finetune/wac_v1/dataset.json")
  ACCUM=$(( 32 / (NGPU * LOCAL_BATCH) ))    # 유효 배치 32 = GPU당 배치 × 2 GPU × 누적
  MAX_ITER=$(( SAMPLES / 32 ))
  SAVE_ITER=$(( MAX_ITER / 6 ))             # 1 epoch를 6번 나눠 저장 (체크포인트 하나 3.8GB)
fi
SAVE_ITER=${SAVE_ITER:-$MAX_ITER}
echo "run=$RUN batch=$LOCAL_BATCH max_iter=$MAX_ITER accum=$ACCUM save_iter=$SAVE_ITER (GPU $GPUS)"

export BASE_DATASETS_DIR=$BASE
export IMAGINAIRE_OUTPUT_ROOT=$WAC_REPO/data/finetune/$RUN/raw
export CUDA_VISIBLE_DEVICES=$GPUS WANDB_MODE=disabled
# RTX 4090은 GPU 사이 P2P를 지원하지 않아 NCCL이 P2P 경로에서 멈출 수 있다. 공유 메모리 경유로 통신하게 한다.
export NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
cd "$COSMOS_POLICY_DIR"
"$VENV/bin/torchrun" --standalone --nproc_per_node=$NGPU -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero \
  "checkpoint.load_path=$BASE/libero_policy_dcp" \
  checkpoint.load_ema_to_reg=False "checkpoint.save_iter=$SAVE_ITER" \
  "trainer.max_iter=$MAX_ITER" "trainer.grad_accum_iter=$ACCUM" trainer.logging_iter=1 \
  trainer.distributed_parallelism=ddp model.config.fsdp_shard_size=1 \
  model.config.use_lora=True model.config.lora_rank=8 model.config.lora_alpha=8 \
  "dataloader_train.batch_size=$LOCAL_BATCH" dataloader_train.num_workers=2 \
  "dataloader_train.dataset.data_dir=$BASE/success_only" \
  "dataloader_train.dataset.t5_text_embeddings_path=$EMB" \
  "dataloader_train.dataset.rollout_data_dir=$WAC_REPO/data/finetune/train_rollouts" \
  dataloader_train.dataset.demonstration_sampling_prob=0.25 \
  optimizer.lr=2e-5 job.project=cosmos_policy_minsu_solo "job.name=$RUN" \
  2>&1 | tee -a "$WAC_REPO/data/finetune/$RUN/train.log"   # 다시 시작해도 로그를 이어 붙인다
