# Cosmos Policy 정리: 원리부터 LIBERO 실험까지

이 문서는 Cosmos Policy를 처음 보는 사람이 이것만 읽고 LIBERO 평가를 돌리고, WAC 실험에 필요한 값을 뽑을 수 있도록 정리한 것이다.

근거는 세 가지다.

- 논문: arXiv 2601.16163
- 체크포인트: `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` (README, config.json)
- 공식 코드: `external/cosmos-policy` (NVlabs/cosmos-policy, 2026-11-13에 clone)

코드에서 직접 확인한 내용에는 파일 위치를 달았다. 확인하지 못한 내용은 **(미확인)**으로 표시했다.

> 수식은 렌더러 없이도 읽히도록 일반 텍스트로 적었다. `x_σ`는 "σ만큼 노이즈가 섞인 x", `‖v‖`는 벡터 길이, `E[·]`는 평균이다.

---

## 목차

1. 한눈에 보기
2. Diffusion: 노이즈를 걷어내며 만든다
3. Latent diffusion과 DiT
4. Cosmos Policy는 비디오 모델을 어떻게 정책으로 바꿨나
5. 학습은 어떻게 했나
6. 추론은 어떻게 돌아가나
7. 실행 환경과 평가 명령
8. WAC 실험에 필요한 값 꺼내기
9. 주의할 점 모음

---

## 1. 한눈에 보기

Cosmos Policy는 NVIDIA의 **비디오 생성 모델**(Cosmos-Predict2-2B-Video2World)을 로봇 정책으로 파인튜닝한 모델이다. 구조는 한 줄도 바꾸지 않았다. 바꾼 건 **입력을 만드는 방식**뿐이다.

비디오 모델은 원래 "앞부분 프레임을 보고 뒷부분 프레임을 그리는" 일을 한다. Cosmos Policy는 로봇의 관절값, 행동(action), 가치(value) 같은 숫자를 **영상 프레임처럼 생긴 텐서에 채워 넣는다**. 그러면 모델은 평소처럼 "다음 프레임"을 그릴 뿐인데, 그 결과가 곧 행동과 미래 예측이 된다.

```
입력 (조건)                          출력 (생성)
─────────────                       ──────────────────────────────
작업 지시문 ─ T5 임베딩 ─┐           action 16 step × 7차원
현재 관절 9차원 ──┐      │            미래 관절 9차원 (16 step 뒤)
손목 카메라 ──────┼─▶ DiT 2B ─▶       미래 손목 이미지 (16 step 뒤)
3인칭 카메라 ─────┘   (5번 반복)       미래 3인칭 이미지 (16 step 뒤)
                                     value: 이대로 가면 성공할 확률 비슷한 값
```

이 다섯 가지 출력이 **한 번의 생성에서 동시에** 나온다. WAC 과제가 노리는 지점이 바로 여기다. 행동과 함께 "이렇게 하면 이렇게 될 것"이라는 예측이 공짜로 나오니, 실제 결과와 비교만 하면 된다.

LIBERO 성공률은 Spatial 98.1%, Object 100%, Goal 98.2%, Long 97.6%로 평균 98.5%다.

---

## 2. Diffusion: 노이즈를 걷어내며 만든다

### 2.1 왜 로봇 정책에 diffusion을 쓰나

행동을 회귀(MSE)로 바로 출력하면 문제가 하나 있다. 같은 상황에서 정답이 여러 개일 때 모델이 그 **평균**을 낸다. 장애물을 왼쪽으로 피한 시연과 오른쪽으로 피한 시연이 섞여 있으면 "정면으로 돌진"을 배우는 식이다.

Diffusion은 정답 하나를 외우는 대신 **정답들의 분포**를 배우고, 거기서 하나를 뽑는다. 그래서 왼쪽이든 오른쪽이든 둘 중 하나를 확실하게 고른다.

### 2.2 노이즈 섞기

깨끗한 데이터 `x₀`에 크기 σ인 가우시안 노이즈를 더한다.

```
x_σ = x₀ + σ · ε        (ε ~ 표준정규분포)
```

σ가 작으면 원본이 거의 보이고, 크면 노이즈뿐이다. Cosmos Policy LIBERO 추론에서는 σ를 80에서 4까지 쓴다. 데이터 값이 대략 -1~1 범위라서, σ = 80이면 사실상 순수 노이즈다.

### 2.3 무엇을 학습하나

네트워크에게 "노이즈 섞인 입력을 보고 원본을 맞혀 봐"라고 시킨다. Cosmos는 **EDM**(Karras et al., 2022) 방식을 쓰는데, 네트워크 `F`를 그대로 쓰지 않고 σ에 맞춰 스케일을 조정하는 껍데기를 씌운다.

```
D(x_σ, σ) = c_skip(σ) · x_σ  +  c_out(σ) · F( c_in(σ) · x_σ , σ, 조건 )
```

각 계수의 역할은 이렇다.

| 계수 | 역할 |
|---|---|
| `c_in` | 입력 크기를 σ와 상관없이 1 근처로 맞춘다. |
| `c_skip` | σ가 작을 때 입력을 거의 그대로 통과시킨다. 이미 거의 깨끗하니까. |
| `c_out` | 네트워크 출력을 얼마나 믿을지 정한다. σ가 클수록 네트워크 몫이 커진다. |

이렇게 하면 네트워크는 노이즈 크기와 상관없이 늘 "비슷한 크기의 보정값"만 내면 되니 학습이 안정적이다. (`cosmos_policy/models/policy_video2world_model.py`의 `denoise()`에 `x0_pred = c_skip · x_t + c_out · net_output`가 그대로 있다.)

손실은 맞힌 원본과 진짜 원본의 차이다.

```
L = E[ λ(σ) · ‖ D(x_σ, σ) − x₀ ‖² ]
```

`λ(σ)`는 σ마다 손실 크기를 맞추는 가중치다. 학습할 때 σ는 매번 어떤 분포에서 무작위로 뽑는데, 이 분포가 "어느 노이즈 구간을 많이 연습할지"를 정한다. Cosmos Policy는 이 분포를 바꿨다(5.3절).

### 2.4 생성은 어떻게 하나

순수 노이즈에서 출발해 σ를 조금씩 줄이면서 반복한다.

```
x ← 80 · ε                              # 순수 노이즈에서 시작
σ를 80 → … → 4 로 줄여 가며:
    x̂₀ ← D(x, σ)                        # 지금 상태에서 원본 추정
    x  ← 원본 추정 쪽으로 한 걸음 이동     # 노이즈를 조금 걷어냄
마지막에 x̂₀를 그대로 출력 (sample_clean)
```

σ 간격은 EDM 표준 방식(ρ = 7)으로 큰 σ 쪽은 듬성듬성, 작은 σ 쪽은 촘촘하게 잡는다. Solver는 2차 다단계 방식(`2ab`)이 기본값이다. (`cosmos_policy/modules/cosmos_sampler.py`)

σ_min이 4로 꽤 큰데도 결과가 깨끗한 이유는 마지막 단계에서 노이즈를 다시 더하지 않고 원본 추정값 `x̂₀`를 바로 쓰기 때문이다. 코드에서는 `sample_clean = True`로 되어 있고, 이것도 한 step으로 치기 때문에 실제 solver 반복은 `num_steps − 1`번이다.

보통 이미지 생성은 수십 step을 쓰지만 여기서는 **5 step**만 쓴다. 로봇을 실시간으로 움직여야 하기 때문이다. Step 수는 재학습 없이 바꿀 수 있다(`--num_denoising_steps_action`). 줄이면 빨라지는 대신 추정이 거칠어진다.

---

## 3. Latent diffusion과 DiT

### 3.1 VAE: 영상을 작게 압축한다

224×224 RGB 영상에서 diffusion을 바로 돌리면 너무 비싸다. 그래서 **VAE**로 압축한 공간(latent)에서 돌린다. Cosmos-Predict2는 **Wan2.1 VAE**를 쓴다.

| 항목 | 값 |
|---|---|
| 공간 압축 | 8배: 224×224 → 28×28 |
| 채널 | 16 |
| 시간 압축 | 4배: 영상 4장 → latent 1장 |
| 첫 프레임 | 1장 → latent 1장 (따로 처리) |

그래서 영상 `1 + 4k`장을 넣으면 latent는 `1 + k`장이 된다. 코드 주석의 `(B, C'=16, T', H'=28, W'=28)`이 이 모양이다.

Wan2.1 VAE는 시간 방향으로 **causal**하다. 각 latent 프레임은 자기 앞에 있는 프레임들의 영향도 받는다. 8장에서 실제 이미지를 인코딩할 때 이 점이 중요해진다.

### 3.2 DiT: Transformer로 노이즈를 걷어낸다

압축한 latent를 작은 패치로 잘라 토큰으로 만들고, Transformer 블록을 여러 번 통과시킨다.

```
latent (16 × T' × 28 × 28)
  → 패치로 잘라 토큰 시퀀스로
  → 블록 × 28
      ├ Self-attention   : 모든 프레임·모든 위치의 토큰끼리 서로 참조 (위치 정보는 3D RoPE)
      ├ Cross-attention  : 작업 지시문의 T5 임베딩을 참조
      ├ MLP
      └ adaLN            : σ에 따라 각 층의 scale·shift를 조절
  → 다시 latent 모양으로
```

2B 모델 설정은 블록 28개, hidden 2048, head 16이다. 베이스 모델 계열 설정 파일(`_src/predict2/.../net.py`)에서 확인했다. Cosmos Policy가 정확히 같은 값을 쓰는지는 체크포인트를 로드해서 다시 확인할 것. **(미확인)**

블록마다 하는 일은 이렇다.

- **Self-attention이 핵심이다.** 모든 토큰이 서로를 보기 때문에, action 프레임의 토큰이 현재 이미지와 미래 이미지 토큰을 직접 참조한다. 그래서 행동, 미래 예측, value가 한 번의 생성 안에서 서로 맞물린다.
- **Cross-attention**은 "그릇을 접시 위에 올려라" 같은 지시문을 반영한다. LIBERO는 과제별 T5 임베딩을 미리 계산해 `libero_t5_embeddings.pkl`에 저장해 두었다.
- **adaLN**은 `adaLN(h) = γ(σ) · LayerNorm(h) + β(σ)` 꼴이다. 같은 네트워크가 "지금 노이즈가 얼마나 남았나"에 따라 다르게 동작하게 해 준다.

### 3.3 조건 프레임은 노이즈 없이 넣는다

Video2World는 "처음 몇 프레임을 주고 나머지를 생성"하는 모델이다. 조건으로 줄 프레임은 노이즈를 섞지 않은 **깨끗한 latent 그대로** 넣고, 나머지만 노이즈 상태에서 출발한다. 이를 `frame_replace` 방식이라고 부른다.

매 denoise 단계마다 조건 프레임 위치를 원래 값으로 덮어쓰기 때문에, 조건은 끝까지 그대로 유지된다. (`denoise()`의 `condition_video_mask` 부분, 설정 `sigma_conditional=0.0`, `denoise_replace_gt_frames=True`)

이 성질 덕분에 **어느 프레임을 조건으로 주느냐만 바꿔서** 한 모델로 여러 일을 시킬 수 있다.

---

## 4. Cosmos Policy는 비디오 모델을 어떻게 정책으로 바꿨나

### 4.1 latent 시퀀스 구성 (LIBERO, 코드 확인)

LIBERO에서는 latent 프레임 9장(`state_t=9`)을 쓴다. 각 자리의 의미는 다음과 같다. (`cosmos_utils.py`의 `get_action()`, `libero_dataset.py`의 `__getitem__`)

| idx | 내용 | 원본 영상에 넣는 것 | 추론 시 역할 |
|---|---|---|---|
| 0 | 빈 프레임 | 검은 이미지 1장 | 조건 |
| 1 | 현재 관절 p_t | 검은 이미지 4장, 인코딩 후 관절값으로 덮어씀 | 조건 |
| 2 | 현재 손목 이미지 | 손목 이미지 4장 복제 | 조건 |
| 3 | 현재 3인칭 이미지 | 3인칭 이미지 4장 복제 | 조건 |
| 4 | action chunk | 검은 이미지 4장, 인코딩 후 action으로 덮어씀 | **생성** |
| 5 | 미래 관절 p′ (t+16) | 검은 이미지 4장, 덮어씀 | **생성** |
| 6 | 미래 손목 이미지 (t+16) | 이미지 4장 | **생성** |
| 7 | 미래 3인칭 이미지 (t+16) | 이미지 4장 | **생성** |
| 8 | value V(s′) | 검은 이미지 4장, 덮어씀 | **생성** |

원본 영상은 1 + 8×4 = 33장이고(`chunk_duration=33`), VAE를 거치면 latent 9장이 된다. 조건 프레임은 0~3번 4장이다(`min/max_num_conditional_frames=4`).

0번 빈 프레임이 따로 있는 이유는 VAE가 첫 프레임만 1장 단위로 압축하기 때문이다. 0번에 검은 이미지 1장을 두면, 그 뒤부터 4장씩 정확히 latent 1장에 대응한다.

이미지를 4장씩 복제하는 이유도 같다. 정지 이미지 한 장을 latent 프레임 하나에 깔끔하게 담으려면 VAE의 시간 압축 단위인 4장을 채워야 한다.

### 4.2 숫자를 프레임에 채우는 법 (latent injection)

관절, action, value는 이미지가 아니다. 그래서 일단 검은 이미지로 자리만 잡아 VAE 인코딩을 하고, **인코딩이 끝난 뒤** 그 latent 프레임(16×28×28 = 12,544칸)을 숫자로 덮어쓴다. (`policy_text2world_model.py`의 `replace_latent_with_action_chunk`, `replace_latent_with_proprio`)

1. **정규화.** 데이터셋 통계(`libero_dataset_statistics.json`)의 min/max로 -1~1 범위로 맞춘다.
   ```
   x_norm = 2 · (x − min) / (max − min) − 1
   ```
2. **반복해서 채우기.** 벡터를 펼친 뒤 12,544칸이 찰 때까지 반복해서 붙인다. action은 16×7 = 112개 값이므로 112번 반복되고, 관절은 9개 값이 약 1,393번 반복된다.
3. **꺼낼 때는 평균.** 생성이 끝나면 반복된 사본들의 평균을 내서 원래 벡터로 되돌린다. (`extract_action_chunk_from_latent_sequence`)

같은 값을 수백 번 넣어 두면 생성 결과에 오차가 조금씩 섞여도 평균을 내면 안정적으로 복원된다.

value는 스칼라 하나라서 프레임 전체를 같은 값으로 채우고, 꺼낼 때 전체 평균을 낸다. 출력은 -1~1 범위이고 `(v + 1) / 2`로 0~1로 바꾼 뒤 잘라낸다. (`extract_value_from_latent_sequence`, `get_action()`)

덮어쓴 프레임을 그대로 VAE로 디코딩하면 이미지가 심하게 깨진다. 그래서 미래 이미지를 그릴 때는 0, 1, 4, 5번 프레임을 원래의 검은 이미지 latent로 되돌린 뒤 디코딩한다. (`undo_latent_injection`)

---

## 5. 학습은 어떻게 했나

### 5.1 한 모델, 세 가지 역할

학습 데이터는 두 종류다. 사람 시연(demo, 모두 성공)과 정책을 직접 돌려 모은 rollout(성공과 실패가 섞임)이다. 샘플 종류마다 조건 프레임과 생성 프레임을 다르게 잡는다.

| 역할 | 주는 것 (조건) | 맞히는 것 (생성) | 데이터 | 배치 비율 |
|---|---|---|---|---|
| Policy | 현재 상태 s | a, s′, V | 시연 (성공만) | 50% |
| World model | s, a | s′, V | rollout (성공·실패) | 25% |
| Value | s, a, s′ | V | rollout (성공·실패) | 25% |

World model과 value를 실패 rollout으로도 학습시키는 이유는 이렇다. 따라 하면 안 되는 행동이라도 "그렇게 하면 어떻게 되는지"는 실패 사례를 봐야 배울 수 있다.

LIBERO 기본 설정(`mask_loss_for_action_future_state_prediction=False`)에서 policy 샘플은 생성하는 프레임 전부(a, s′, V)에 손실을 받는다. 그래서 **정책 모드에서 나오는 s′도 학습된 예측**이다. WAC가 이 s′를 쓸 수 있는 근거다.

### 5.2 value의 정답은 무엇인가

보상은 에피소드가 성공으로 끝날 때 마지막 step에서 1을 주고, 나머지는 0이다. 할인율 γ = 0.99로 몬테카를로 return을 계산한다.

```
G_t = γ^(T−1−t)      (성공 에피소드, T = 에피소드 길이)
G_t = 0              (실패 에피소드)
정답 value = 2·G − 1   (-1~1로 변환)
```

(`dataset_common.py`의 `compute_monte_carlo_returns`)

value 정답으로는 **미래 시점 t+16의 return**을 쓴다. 즉 V는 "16 step 뒤 상태가 성공에 얼마나 가까운가"이고, 성공 직전일수록 1에 가깝다.

### 5.3 노이즈 분포를 바꿨다

학습 때 σ를 뽑는 분포를 이렇게 섞었다. (`modules/hybrid_edm_sde.py`, LIBERO 설정)

```
70% 확률: log σ ~ 정규분포(평균 1.386, 표준편차 1.2)    ← 베이스 모델과 같음
30% 확률: σ ~ 균등분포(1, 85)
```

log-normal만 쓰면 중간 σ에 학습이 몰려서 큰 σ 구간을 덜 연습한다. 균등분포를 섞으면 큰 σ에서도 원본을 잘 맞히게 되고, 그래서 **σ = 80에서 시작해 5 step만에** 정확한 행동을 뽑을 수 있다.

학습 때 σ 범위는 0.01~200이고, 추론 때는 4~80으로 줄인다(`__inference_only` 설정).

### 5.4 학습 규모

| 항목 | 값 |
|---|---|
| 학습 step | 40,000 |
| 배치 | 1,920 |
| 하드웨어 | H100 64장, 약 48시간 |
| 학습 범위 | 전체 가중치 |
| 학습률 | 1e-4 |
| 텍스트 dropout | 0 (지시문을 무시하지 않도록) |

이 과제는 추가 학습을 하지 않으니 참고만 하면 된다.

---

## 6. 추론은 어떻게 돌아가나

### 6.1 에피소드 루프

`run_libero_eval.py`의 `run_episode()`를 정리하면 다음과 같다.

```
처음 10 step: 아무것도 안 함 (물체가 자리 잡게)
t = 10부터, 최대 step까지:
    관측 → 이미지 2장(224, 상하 반전, JPEG 압축), 관절 9차원
    action 큐가 비었으면:
        get_action() 호출 → action 16개, 미래 이미지, value
        action 16개를 큐에 넣음
    큐에서 action 하나를 꺼내 env.step()
    done이면 성공으로 기록하고 종료
```

과제별 최대 step(대기 10 step 제외)은 다음과 같다.

| 과제 | 최대 step |
|---|---|
| Spatial | 220 |
| Object | 280 |
| Goal | 300 |
| Long (libero_10) | 520 |

16 step마다 모델을 한 번씩 부르고, 그 사이에는 다시 계획하지 않는다(open-loop, `--num_open_loop_steps 16`).

### 6.2 get_action() 안에서 일어나는 일

1. 관절을 정규화한다.
2. 33장짜리 원본 시퀀스를 만든다: 빈 이미지, 관절 자리, 현재 이미지, action 자리, 미래 자리, value 자리.
3. VAE로 인코딩하고, 관절값을 1번 프레임에 주입한다.
4. 0~3번은 조건으로 고정하고 4~8번은 σ = 80 노이즈로 채운 뒤 5 step denoise한다.
5. 4번 프레임에서 action을 꺼내고 정규화를 푼다.
6. 6, 7번 프레임을 디코딩해 미래 이미지를 만들고, 8번에서 value를 꺼낸다.

반환값(`return_dict`)에는 WAC에 필요한 것이 다 들어 있다.

| 키 | 내용 |
|---|---|
| `actions` | action 16개 |
| `generated_latent` | 생성된 latent 전체 (B, 16, 9, 28, 28) |
| `orig_clean_latent_frames` | 주입 전 깨끗한 latent |
| `latent_indices` | 각 프레임 위치 |
| `future_image_predictions` | 디코딩한 미래 이미지 |
| `value_prediction` | 0~1 value |
| `proprio` | 정규화한 현재 관절 |

**미래 관절 p̂′는 반환값에 따로 없다.** `generated_latent[:, :, 5]`에서 직접 꺼내야 한다(8.2절).

---

## 7. 실행 환경과 평가 명령

### 7.1 현재 PC 상태와 해야 할 일

| 항목 | 상태 |
|---|---|
| GPU | RTX 5070 Laptop, VRAM 8GB, 드라이버 577.13 |
| 필요 VRAM | 약 6.8GB (bf16) → 들어가긴 하지만 여유가 거의 없음 |
| OS | Windows 11. 공식 코드는 **Linux 전용** (Docker + NVIDIA Container Toolkit) |
| WSL | **설치 안 됨** |
| 체크포인트 | `checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/`에 받아 둠 |
| 공식 코드 | `external/cosmos-policy/`에 받아 둠 |

그래서 바로 돌리려면 순서가 이렇다.

1. **WSL2 설치.** 관리자 PowerShell에서 `wsl --install -d Ubuntu-22.04`를 실행하고 재부팅한다.
2. **Docker 설치.** WSL 안에 Docker Engine을 설치하거나 Docker Desktop의 WSL 백엔드를 쓰고, NVIDIA Container Toolkit도 설치한다.
3. **GPU 확인.** WSL 안에서 `nvidia-smi`가 GPU를 보여야 한다. RTX 50 시리즈는 CUDA 12.8 이상이 필요한데, 공식 설치 명령이 `cu128`을 쓰니 맞는다.
4. **이미지 빌드와 실행.** 저장소 루트에서 아래를 실행한다.
   ```bash
   cd external/cosmos-policy
   docker build -t cosmos-policy docker
   docker run -u root -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) \
     -v $HOME/.cache:/home/cosmos/.cache -v $(pwd):/workspace \
     --gpus all --ipc=host -it --rm -w /workspace --entrypoint bash cosmos-policy
   ```
5. **LIBERO 의존성 설치.** 컨테이너 안에서 실행한다.
   ```bash
   uv sync --extra cu128 --group libero --python 3.10
   ```

8GB에서 메모리가 부족하면 `--use_parallel_inference`는 끄고(기본값이 꺼짐), 다른 GPU 프로그램을 모두 닫는다. 그래도 안 되면 미래 이미지 디코딩(VAE decode)이 추가 메모리를 쓰는 지점이니 그 부분을 먼저 의심한다. **(이 PC에서 실제로 돌려 보지는 않았다.)**

### 7.2 공식 평가 명령 (성공률 재현)

`LIBERO.md`의 명령에서 GPU 목록만 1장으로 바꿨다. 이 과제는 `--task_suite_name libero_spatial`만 쓴다.

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --use_proprio True \
    --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 --num_open_loop_steps 16 \
    --task_suite_name libero_spatial \
    --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
    --randomize_seed False --data_collection False \
    --available_gpus "0" \
    --seed 195 --use_variance_scale False --deterministic True \
    --ar_future_prediction False --ar_value_prediction False \
    --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 --num_denoising_steps_value 1
```

기본값은 10과제 × 50 에피소드 = 500회다. 처음에는 `--num_trials_per_task 2` 정도로 짧게 돌려서 동작부터 확인한다.

`--ckpt_path`에 HF 경로를 주면 자동으로 받는다. 이미 받아 둔 체크포인트를 쓰려면 `HF_HOME`을 맞추거나 로컬 경로를 준다. **(로컬 경로 지원 여부는 `resolve_path()` 기준으로 확인 필요)**

### 7.3 실험에서 바꾸게 될 플래그

| 플래그 | 의미 | WAC 실험에서 |
|---|---|---|
| `--ar_future_prediction`, `--ar_value_prediction` | False면 a, s′, V를 한 번에 생성. True면 action을 먼저 만들고 s′, V를 따로 다시 생성 | **False 유지.** 정책이 실제로 쓴 생성과 같은 s′를 봐야 함 |
| `--num_denoising_steps_action` | denoise step 수 | step 수 실험 (5 → 1~2) |
| `--num_denoising_steps_future_state`, `--num_denoising_steps_value` | ar 모드에서만 쓰임 | 무시 |
| `--deterministic`, `--seed` | 재현성 | 실패 유도 실험에서는 seed를 바꿔 여러 번 |
| `--num_trials_per_task` | 과제당 에피소드 수 | rollout 100~200개에 맞춰 조절 |
| `--data_collection` | True면 이미지, 관절, action, 성공 여부, 예측 미래 이미지를 저장 | 출발점으로 쓸 만하지만 latent, p̂′, V̂는 저장 안 함 → 8장처럼 추가 필요 |
| `--use_variance_scale` | 초기 노이즈 크기를 흔들어 다양성 증가 | 샘플링 분산 확장 실험 때만 |

---

## 8. WAC 실험에 필요한 값 꺼내기

신호마다 필요한 값과 위치를 정리하면 다음과 같다.

| 신호 | 필요한 값 | 위치 |
|---|---|---|
| A: 1 − V̂ | value | `return_dict["value_prediction"]` (0~1) |
| B: d_prop | 예측 미래 관절 p̂′, 실제 관절 p_{t+16} | latent 5번 프레임, 16 step 뒤 관측 |
| C: d_img | 예측 미래 latent ẑ′, 실제 이미지의 latent | latent 6·7번 프레임, 실제 관측 재인코딩 |
| E: probe | DiT 중간층 feature | 블록에 forward hook (모듈 이름 미확인) |

### 8.1 로깅을 끼워 넣을 자리

`run_libero_eval.py`의 `run_episode()`에서 `get_action()`을 부른 직후가 가장 간단하다. 여기서 chunk 하나마다 다음을 저장한다.

- `return_dict["generated_latent"]`의 5·6·7·8번 프레임 (CPU로 옮겨서)
- `return_dict["value_prediction"]`
- 현재 t, 그리고 16 step 실행 후의 관측(이미지 2장, 관절)
- 에피소드가 끝나면 성공 여부

16 step 실행 후의 관측은 바로 다음 `get_action()` 호출 때의 관측과 같다. 그래서 chunk k의 예측을 chunk k+1의 입력과 짝지으면 된다. 에피소드가 chunk 중간에 끝나면 마지막 chunk는 짝이 없으니 버리거나 종료 시점 관측을 쓴다.

### 8.2 미래 관절 p̂′ 꺼내기

action을 꺼내는 함수와 같은 방식으로 5번 프레임을 평균 낸다. 아래는 **스케치이고 아직 실행해 보지 않았다.**

```python
def extract_proprio(latent, idx=5, dim=9):
    f = latent[:, :, idx].reshape(latent.shape[0], -1)   # (B, 12544)
    n = f.shape[1] // dim
    return f[:, : n * dim].reshape(-1, n, dim).mean(1)   # (B, 9), 정규화 공간
```

비교는 **정규화 공간에서** 한다. 실제 관절 p_{t+16}도 `rescale_proprio(p, dataset_stats)`로 똑같이 정규화한 뒤 거리를 잰다. 단위가 다른 위치, 쿼터니언, 그리퍼가 섞여 있어서 정규화 공간에서 재는 편이 공정하다.

주입 함수(`replace_latent_with_proprio`)는 9개 값을 펼친 채로 처음부터 반복해 붙이고 남는 부분을 잘라낸다. action 쪽과 같은 방식이라 위 코드로 되돌릴 수 있다. 끝에 잘린 불완전한 사본(12,544 = 9 × 1,393 + 7)은 버린다.

### 8.3 실제 이미지를 latent로 만들기 (주의)

VAE가 시간 방향으로 causal이라서, 같은 이미지라도 **앞에 무엇이 있었는지에 따라 latent가 달라진다.** 실제 미래 이미지를 단독으로 인코딩하면 ẑ′와 같은 조건에서 비교한 게 아니게 된다.

공정하게 비교하려면 `get_action()`이 만드는 33장 시퀀스를 똑같이 만들되, 6·7번 자리에 **실제 관측 o_{t+16}을 4장씩** 넣고 인코딩해서 6·7번 latent를 꺼낸다. 그러면 ẑ′와 같은 위치, 같은 앞 문맥에서 비교하게 된다.

이미지 전처리도 정책 입력과 똑같이 맞춘다. 상하 반전, JPEG 압축, 224 리사이즈를 그대로 적용한다.

```
d_img = 카메라 평균 of  (1/D) · ‖ ẑ′_c − E(실제 시퀀스)_c ‖²      (D = 16·28·28)
```

생성 결과에 정규화 계수(`sigma_data` 등)가 곱해져 있는지, `model.encode()` 출력과 스케일이 같은지도 먼저 확인한다. 성공 rollout 하나에서 미래 예측이 맞을 법한 chunk를 골라, d_img가 작게 나오는지 보는 것으로 검증할 수 있다.

### 8.4 비용

parallel 모드에서는 s′와 V가 action과 같은 생성에서 나오니 추가 생성 비용이 없다. 늘어나는 것은 실제 관측을 다시 인코딩하는 VAE encode 1회뿐이다.

공식 코드는 매 호출마다 미래 이미지를 디코딩(VAE decode)한다. 정성 분석용 그림이 필요 없을 때 이 디코딩을 끄면 시간과 메모리를 아낄 수 있다.

---

## 9. 주의할 점 모음

- **WAC는 최소 한 chunk 늦다.** 예측을 검증하려면 16 step을 실행해야 하니, 첫 chunk에서 생긴 실패는 원리상 미리 못 잡는다.
- **parallel 모드를 유지한다.** ar 모드를 켜면 s′를 다시 생성하기 때문에, 정책이 실제로 쓴 생성과 다른 예측을 보게 된다.
- **value는 0~1로 잘린 값이다.** 성공 직전 상태에서 1에 붙어 있을 수 있으니 분포를 먼저 확인한다.
- **seed와 deterministic.** `--deterministic True`는 매 step마다 seed를 0으로 다시 맞춘다. 실패 유도 실험에서 다양성이 필요하면 이 설정의 영향을 생각해야 한다.
- **실행 환경.** 논문 결과는 H100, PyTorch 2.7.0 기준이다. 노트북 GPU에서는 성공률이 조금 다를 수 있으니, 이 PC에서 재현한 성공률을 기준선으로 따로 기록한다.
- **라이선스.** 체크포인트는 NVIDIA 비상업 라이선스다. 연구 용도로만 쓴다.

**아직 확인할 것**

- [ ] 이 PC(WSL2, 8GB)에서 평가가 실제로 돌아가는지
- [ ] DiT 블록 수와 모듈 경로 (조건 E용 hook 위치)
- [ ] 8.2 스케치를 실제로 돌려서 p̂′가 현실적인 값인지
- [ ] 생성 latent와 `encode()` 출력의 스케일 일치 → 8.3
- [ ] `--ckpt_path`에 로컬 경로를 줄 수 있는지
