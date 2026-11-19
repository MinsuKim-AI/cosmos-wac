#!/usr/bin/env bash
# 공식 LIBERO 평가 (교란 없음). 사용: bash run_official_eval.sh <suite> <trials_per_task> [추가 인자...]
set -e
source "$(dirname "$0")/env.sh"
SUITE=${1:-libero_spatial}; TRIALS=${2:-1}; shift 2 || true
cd "$COSMOS_POLICY_DIR"
CK=nvidia/Cosmos-Policy-LIBERO-Predict2-2B
exec "$VENV/bin/python" -m cosmos_policy.experiments.robot.libero.run_libero_eval \
  --config cosmos_predict2_2b_480p_libero__inference_only --ckpt_path $CK \
  --config_file cosmos_policy/config/config.py \
  --use_wrist_image True --use_proprio True --normalize_proprio True --unnormalize_actions True \
  --dataset_stats_path $CK/libero_dataset_statistics.json \
  --t5_text_embeddings_path $CK/libero_t5_embeddings.pkl \
  --trained_with_image_aug True --chunk_size 16 --num_open_loop_steps 16 \
  --task_suite_name "$SUITE" --local_log_dir "$HOME/wac/logs/" \
  --randomize_seed False --data_collection False --available_gpus 0 \
  --seed 195 --use_variance_scale False --deterministic True \
  --ar_future_prediction False --ar_value_prediction False \
  --use_jpeg_compression True --flip_images True \
  --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
  --num_trials_per_task "$TRIALS" "$@"
