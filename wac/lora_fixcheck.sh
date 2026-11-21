#!/usr/bin/env bash
# LoRA 초기화 수정 확인: 신호 F 수집이 끝나 GPU 2, 3이 비면 20번 갱신만 학습하고,
# 저장된 체크포인트의 LoRA A, B가 0이 아닌지 잰다. 결과는 ~/wac_cosmos/lora_fixcheck.txt
set -u
cd ~/wac_cosmos/repo
until [ "$(find data/libero_spatial_F -name DONE | wc -l)" -ge 40 ] || ! pgrep -f "jobs_f.json" >/dev/null; do sleep 60; done
rm -rf data/finetune/4090_lora_fixcheck
GPUS=2,3 bash wac/run_lora.sh 4090_lora_fixcheck 20 1 20 > ~/wac_cosmos/lora_fixcheck.log 2>&1
C=$(ls -d data/finetune/4090_lora_fixcheck/raw/*/*/*/checkpoints/iter_000000020/model 2>/dev/null)
{
  date "+%H:%M"
  grep -E "LoRA 초기화|LoRA injection" ~/wac_cosmos/lora_fixcheck.log | sed 's/.*\] //'
  [ -n "$C" ] && CUDA_VISIBLE_DEVICES= ~/wac_cosmos/cosmos-policy/.venv/bin/python - "$C" <<'EOF'
import sys, torch, torch.distributed.checkpoint as dcp
md = dcp.FileSystemReader(sys.argv[1]).read_metadata().state_dict_metadata
ks = [k for k in md if ("blocks.0.self_attn.q_proj.lora" in k or "blocks.20.mlp.layer1.lora" in k)]
sd = {k: torch.empty(md[k].size, dtype=md[k].properties.dtype) for k in ks}
dcp.load(sd, storage_reader=dcp.FileSystemReader(sys.argv[1]), no_dist=True)
for k, v in sd.items():
    print(k, "norm", round(v.float().norm().item(), 6))
EOF
  [ -z "$C" ] && { echo "체크포인트 없음"; tail -20 ~/wac_cosmos/lora_fixcheck.log; }
} > ~/wac_cosmos/lora_fixcheck.txt 2>&1
