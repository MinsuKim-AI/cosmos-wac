---
license: other
license_name: nvidia-open-model-license-derived
task_categories:
- robotics
tags:
- libero
- cosmos-policy
- failure-detection
- diffusion-policy
pretty_name: cosmos-wac LIBERO-Spatial perturbed rollouts
size_categories:
- n<1K
---

# cosmos-wac LIBERO-Spatial perturbed rollouts

Cosmos Policy(`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`)를 LIBERO-Spatial에서 교란을 넣어 돌린 에피소드 모음이다. 정책이 추론마다 생성한 미래 예측(16 step 뒤 이미지 latent, 관절, 가치)과 실제 관측을 함께 저장해서, 예측과 실제의 차이로 실패를 미리 알아채는 연구(cosmos-wac)에 쓴다.

코드와 실험 기록은 GitHub 저장소 [MinsuKim-AI/cosmos-wac](https://github.com/MinsuKim-AI/cosmos-wac)에 있다. 데이터를 모은 방법과 교란 크기를 정한 과정은 저장소의 [Experiment.md](https://github.com/MinsuKim-AI/cosmos-wac/blob/main/docs/Experiment.md)에 적었다.

## 교란 조건

한 에피소드에는 교란을 하나만 넣는다. 크기는 성공률이 50% 근처가 되도록 직접 재서 정했다.

| 폴더 | 교란 |
|---|---|
| `obj_<m>` | 움직일 수 있는 물체마다 초기 위치의 x, y를 ±m 범위 균등분포로 옮긴다 |
| `camyaw_<도>` | 3인칭 카메라를 테이블 중심 수직축 둘레로 그 각도만큼 돌린다 (방향 무작위) |
| `camyaw_<도>_shift_<m>` | 3인칭 카메라를 회전한 뒤 무작위 방향으로 m미터 옮긴다 |
| `wristroll_<도>` | 손목 카메라를 광축 둘레로 그 각도만큼 돌린다 (방향 무작위) |

## 묶음

| 묶음 | LIBERO 초기 상태 | 교란 seed | 지시문당 에피소드 |
|---|---|---|---|
| calib | 10~19 | 2 | 10 |
| eval | 20~29 | 3 | 10 |
| train | 0~4 | 1 | 5 |

## 파일

```
<조건>/<묶음>/D<두 자리 번호>_<지시문>/
  result.md            에피소드 표
  predictions.csv      추론마다 가치, 예측 엔드이펙터 위치, 실제 위치, 오차
  init<번호>_<succ|fail>.mp4     시뮬레이션 영상 (왼쪽 3인칭, 오른쪽 손목)
  init<번호>_<succ|fail>.hdf5    이미지, 관절, 행동 (Cosmos Policy 공식 data_collection 형식)
  init<번호>_<succ|fail>.npz     생성 latent(미래 관절, 미래 손목, 미래 3인칭, 가치 프레임), 가치, 예측 관절, 실제 관절
  init<번호>_<succ|fail>_frames/ 추론마다 '현재 | 예측한 16 step 뒤 | 실제 16 step 뒤' 그림
index.csv              전체 에피소드 목록
```

정책 설정은 공식 LIBERO 평가와 같다(denoising 5 step, chunk 16, seed 195, deterministic). 수집은 RTX 4090에서 했다.

정책 체크포인트는 NVIDIA 라이선스를 따른다. 이 데이터셋은 연구용이다.
