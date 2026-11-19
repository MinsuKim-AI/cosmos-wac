# WSL 안에서 source 해서 쓴다: source /mnt/c/Users/rlaal/Desktop/PML/cosmos-wac/wac/env.sh
export VENV=$HOME/cosmos-venv
export COSMOS_POLICY_DIR=$HOME/wac/cosmos-policy        # 공식 코드 사본 (external/cosmos-policy를 rsync)
export WAC_DIR=/mnt/c/Users/rlaal/Desktop/PML/cosmos-wac
export HF_HOME=$HOME/hf                                  # 체크포인트 캐시
export MUJOCO_GL=egl                                     # 오프스크린 렌더링
export PYOPENGL_PLATFORM=egl
# transformer_engine이 ldconfig로 libnvrtc를 못 찾아서 venv 안 pip 패키지 위치를 알려 준다
export CUDA_HOME=$VENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export PYTHONPATH=$COSMOS_POLICY_DIR:$WAC_DIR
