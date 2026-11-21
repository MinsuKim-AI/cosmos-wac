#!/usr/bin/env bash
# 신호 F, hidden feature 수집(작업 40개)이 끝나면 LoRA 초기화를 고친 코드로 1 epoch 파인튜닝을 GPU 4장에서 시작한다.
# 프로세스 이름 대신 DONE 파일 개수로 끝을 판단한다.
cd ~/wac_cosmos/repo
until [ "$(find data/libero_spatial_F -name DONE | wc -l)" -ge 40 ]; do sleep 60; done
sleep 30
GPUS=0,1,2,3 bash wac/run_lora.sh 4090_lora_v2 epoch > ~/wac_cosmos/ft_v2.log 2>&1
