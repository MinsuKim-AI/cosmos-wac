# Diffusion 기반 로봇 정책의 자기 예측 불일치를 이용한 실패 조기 탐지

## Proposed Idea

Cosmos Policy [1] is a robot policy built on a video diffusion model. Each time it is queried, it generates not only an action chunk but also a prediction of what the robot will observe after executing that chunk (a future camera image and joint state) and a value estimate for that future state. We propose to use the policy's own prediction as a reference for runtime failure detection. After a chunk is executed, we compare the predicted future observation with the observation that actually occurs; we call this discrepancy the World-Action Consistency (WAC) score. If the policy's understanding of the scene is correct, the two should agree. A large discrepancy indicates that the episode is deviating from what the policy expected, which we hypothesize precedes task failure. The approach requires no additional training and no failure data: scores are normalized using successful rollouts only. We evaluate on LIBERO [2] whether the WAC score separates failed episodes from successful ones, and, critically, whether it does so better or earlier than the value estimate the policy already provides.

## 1. 연구 질문

- **Q1.** WAC 점수는 실패할 에피소드와 성공할 에피소드를 구분하는가?
- **Q2.** 모델이 원래 출력하는 가치 예측보다 더 잘, 또는 더 일찍 구분하는가?

가치 예측만으로 실패를 알 수 있다면 WAC 점수는 필요 없다. 따라서 Q2가 핵심 비교다.

## 2. 관련 연구

| 연구 | 신호 | 실패 데이터 |
|---|---|---|
| Sentinel [3] | 시점 간 행동 묶음의 일관성, VLM 진행 판단 | 불필요 |
| FAIL-Detect [4] | 정책 입출력의 불확실성 점수 | 불필요 |
| SAFE [5] | 정책 내부 feature로 학습한 분류기 | 필요 |

이 과제는 행동이나 내부 feature 대신 정책이 스스로 내놓은 미래 관측 예측의 오차를 신호로 쓴다. 성공 데이터만으로 점수를 보정하고 임계값을 정하는 절차는 FAIL-Detect [4]를 따른다.

## 3. 방법

### 3.1 모델

공개 체크포인트 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`를 그대로 쓴다(LIBERO 평균 성공률 98.5% [1]). 모델은 한 번 추론할 때 다음 16 step의 행동(chunk), chunk 실행 후의 미래 이미지(VAE latent)와 관절 상태, 그 미래 상태의 가치를 함께 출력한다. 로봇은 chunk를 다 실행한 뒤 다시 추론한다.

### 3.2 기호

| 기호 | 뜻 |
|---|---|
| $k$ | 에피소드 안에서 chunk 순번 |
| $\hat{z}_k$, $z_k$ | chunk $k$ 종료 시점의 이미지 latent (예측 / 실제 이미지를 같은 VAE로 인코딩한 값) |
| $\hat{p}_k$, $p_k$ | chunk $k$ 종료 시점의 관절 상태 (예측 / 실제) |
| $\hat{V}_k$ | 예측 가치 |

### 3.3 신호

$$
d^{\text{img}}_k = \text{mean}\left( (\hat{z}_k - z_k)^2 \right), \qquad
d^{\text{prop}}_k = \lVert \hat{p}_k - p_k \rVert_2, \qquad
d^{\text{val}}_k = 1 - \hat{V}_k
$$

- $d^{\text{img}}$: 예상한 장면과 실제 장면의 차이 (latent 평균 제곱 오차, 카메라 평균)
- $d^{\text{prop}}$: 예상한 팔 자세와 실제 팔 자세의 거리
- $d^{\text{val}}$: 비교 기준. 모델이 성공에서 멀다고 볼수록 크다.

$d^{\text{val}}$은 chunk 실행 전에, 나머지 둘은 실행 후에 알 수 있다.

### 3.4 정규화와 판정

신호의 평소 크기는 과제와 chunk 순번에 따라 다르므로, 보정용 성공 에피소드에서 과제별·순번별 평균 $\mu_k$, 표준편차 $\sigma_k$를 구해 세 신호에 똑같이 적용한다.

$$
\tilde{d}_k = \frac{d_k - \mu_k}{\sigma_k}, \qquad R_k = \max_{j \le k} \tilde{d}_j
$$

$\tilde{d}_k$는 성공 에피소드 평균에서 표준편차 몇 배만큼 벗어났는지, $R_k$는 chunk $k$까지의 최댓값이며 에피소드 점수로 쓴다. 임계값은 보정 세트 성공 에피소드 최종 점수의 95% 분위수로 정한다.

### 3.5 한계

$d^{\text{img}}$, $d^{\text{prop}}$는 chunk 실행 후에 나오므로 탐지가 최소 16 step 늦고, 첫 chunk 안에서 결정된 실패는 잡을 수 없다.

## 4. 실험

### 4.1 설정

기본 설정은 성공률이 98.5%라 실패 사례가 부족하다. 물체 초기 위치를 넓게 흩뜨리는 조건과 카메라 위치를 옮기는 조건에서 성공률이 50~70%가 되도록 변화 크기를 맞춘다. 이런 변화로 LIBERO 성능이 떨어진다는 점은 LIBERO-Plus [6]에서 보고되었다.

조건 변화만으로 성공 에피소드의 신호도 커질 수 있으므로 보정, 임계값, 평가는 모두 조건별로 따로 한다. 에피소드는 100~200개를 수집하고, chunk마다 3.2절의 값을, 에피소드마다 과제·조건·seed·성공 여부를 저장한다.

### 4.2 비교 조건

| 조건 | 신호 | 시점 |
|---|---|---|
| A | $d^{\text{val}}$ | 실행 전 |
| B | $d^{\text{prop}}$ | 실행 후 |
| C | $d^{\text{img}}$ | 실행 후 |
| D | A+B+C logistic regression (과제 단위 train/test 분리) | 실행 후 |

### 4.3 지표

| 지표 | 정의 |
|---|---|
| AUROC, AUPRC | 최종 점수로 성공·실패를 구분하는 정도 |
| 시점별 AUROC | chunk $k$까지의 점수 $R_k$로 계산한 AUROC |
| 오경보율 / 탐지율 | 임계값을 넘은 성공 / 실패 에피소드 비율 |

시점은 에피소드 길이 비율이 아니라 chunk 순번으로 잰다. 실패 에피소드는 항상 최대 길이까지 가므로, 비율로 재면 길이가 정답을 누설한다.

모든 지표에 bootstrap 95% 신뢰구간을 붙이고, Q2는 A 대 B, A 대 C의 AUROC 차이를 paired bootstrap으로 검정한다.

## 5. 주의할 점

- 유도한 조건에서의 결론이 기본 설정의 자연 실패에도 성립한다는 보장은 없다. 자연 실패는 사례 분석으로 확인한다.
- 조건 안에서도 변화 크기가 에피소드마다 다르면 신호가 실패가 아니라 변화 크기를 잡을 수 있다. 변화 크기와 점수의 상관을 따로 확인한다.

## 참고문헌

[1] M. J. Kim et al. "Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning." arXiv:2601.16163, 2026.

[2] B. Liu et al. "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." arXiv:2306.03310, 2023.

[3] C. Agia et al. "Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress." CoRL, 2024.

[4] C. Xu et al. "Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies." RSS, 2025.

[5] Q. Gu et al. "SAFE: Multitask Failure Detection for Vision-Language-Action Models." NeurIPS, 2025.

[6] S. Fei et al. "LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models." arXiv:2510.13626, 2025.
