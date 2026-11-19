# 서버 OASYS용 환경 변수. 서버에서는 이 파일이 env.sh 자리에 복사된다.
export VENV=$HOME/wac_cosmos/cosmos-policy/.venv
export COSMOS_POLICY_DIR=$HOME/wac_cosmos/cosmos-policy
export WAC_DIR=$HOME/wac_cosmos/repo
export HF_HOME=$HOME/wac_cosmos/hf                     # models--* 와 hub/ 를 둘 다 기존 캐시로 링크
export WAC_STATS=$(ls $HF_HOME/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/*/libero_dataset_statistics.json | head -1)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_HOME=$VENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export PYTHONPATH=$COSMOS_POLICY_DIR:$WAC_DIR
