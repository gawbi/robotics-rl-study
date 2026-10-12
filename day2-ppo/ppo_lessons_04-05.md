# PPO 공부 노트 4~5강: Return·Advantage·GAE와 손실함수

> 목표: `compute_returns_and_advantage()`와 `value_loss`, `loss.backward()`, `optimizer.step()`의 역할을 이해한다.

## 공부 순서

1. 4강에서 **Critic의 예측, Advantage, Return**의 차이를 확실히 구분한다.
2. GAE 수식과 역순 반복문을 연결한다.
3. 5강에서 손실이 만들어지고 파라미터 업데이트로 이어지는 순서를 익힌다.
4. 마지막 문제를 풀고 답을 확인한다.

---

# 4강. Return, TD Error, Advantage, GAE

## 1. 가장 먼저 구분해야 할 값들

| 이름 | 기호 | 쉬운 뜻 |
|---|---|---|
| 즉시 보상 | `r_t` | 지금 한 행동 직후 환경에서 받은 보상 |
| 상태 가치 | `V(s_t)` | 현재 상태에서 앞으로 얻을 할인 누적 보상의 추정치 |
| Return | `G_t` 또는 코드의 `returns` | 보상과 미래 가치에 근거해 계산한 목표값 |
| TD Error | `δ_t` | 한 단계 보상과 다음 상태 가치로 계산한 예측 오차 |
| Advantage | `A_t` | 행동의 결과가 기대보다 얼마나 좋거나 나빴는지 나타내는 값 |

**Reward는 환경에서 오고, Value는 Critic이 예측하고, Return/Advantage는 경험을 토대로 계산한다.**

## 2. Return: 미래 보상은 현재보다 덜 중요하게 볼 수 있다

일반적인 할인 누적 보상은 다음과 같이 표현한다.

`G_t = r_t + γ r_(t+1) + γ² r_(t+2) + ...`

- `γ`는 `gamma`라고 읽는다.
- `γ=0.99`이면 미래 보상의 상당 부분을 고려한다.
- `γ`가 작아질수록 먼 미래의 보상은 현재 Return에 덜 반영된다.

### 간단한 계산

현재 보상이 1, 다음 보상이 2이고 `gamma=0.5`라면, 두 시점까지만 고려한 Return은 다음과 같다.

`G_t = 1 + 0.5 × 2 = 2`

이 예시는 계산 방식만 설명한다. 실제 PPO 코드의 Return은 에피소드 종료와 마지막 상태 가치 등도 처리한다.

## 3. TD Error: 예측과 한 단계 목표의 차이

TD Error(Temporal-Difference Error)는 다음과 같이 계산한다.

`δ_t = r_t + γ × V(s_(t+1)) × (1 - done_t) - V(s_t)`

코드로는 다음과 같은 구조다.

```python
delta = (
    reward
    + gamma * next_value * next_non_terminal
    - value
)
```

각 요소의 뜻:

- `reward`: 방금 실제로 받은 보상
- `next_value`: 다음 상태에 대한 Critic 예측
- `next_non_terminal`: 다음 상태의 가치 예측을 이어서 사용할지 나타내는 마스크(일반적으로 계속 진행 가능하면 1, 진짜 종료면 0)
- `value`: 현재 상태에 대한 Critic 예측

### 숫자로 계산해 보기

`reward=1`, `gamma=0.9`, `next_value=4`, `value=3`, 종료되지 않았다고 하자.

`δ = 1 + 0.9 × 4 - 3 = 1.6`

양의 TD Error는 한 단계 목표가 현재 예측값보다 컸다는 뜻이다. 단, TD Error 그 자체가 모든 미래 보상을 완벽히 나타내는 것은 아니다.

## 4. Advantage: 기대보다 좋았나, 나빴나?

이론적으로 Advantage는 다음 관계로 이해할 수 있다.

`A(s, a) = Q(s, a) - V(s)`

- `Q(s, a)`: 상태 `s`에서 행동 `a`를 선택했을 때의 가치
- `V(s)`: 그 상태 자체의 가치

직관적으로는 **해당 행동의 결과가 그 상태에서 기대했던 수준에 비해 어땠는지** 나타낸다.

| Advantage | 직관적 해석 |
|---|---|
| 양수 | 기대보다 좋은 결과였다는 신호 |
| 음수 | 기대보다 나쁜 결과였다는 신호 |
| 0 부근 | 기대와 비슷한 결과였다는 신호 |

Advantage는 환경이 직접 주는 숫자가 아니다. PPO 코드에서는 Return이나 TD Error 등의 정보를 바탕으로 추정한다.

## 5. GAE: 여러 TD Error를 누적해 Advantage를 추정

GAE는 Generalized Advantage Estimation의 약자다. 단일 TD Error만 사용하면 편향과 분산 사이의 선택이 제한적일 수 있으므로, 여러 시점의 TD Error를 감쇠하면서 누적해 Advantage를 추정한다.

개념적인 식은 다음과 같다.

`A_t = δ_t + γ λ δ_(t+1) + (γ λ)² δ_(t+2) + ...`

여기서 `λ`는 `gae_lambda`다. 실제 구현은 에피소드 종료 경계를 고려하면서 역순 재귀식으로 계산한다.

```python
last_gae = 0.0

for step in reversed(range(n_steps)):
    # 해당 transition의 next_value와 종료 마스크를 가져온다.
    delta = (
        rewards[step]
        + gamma * next_values[step] * next_non_terminal[step]
        - values[step]
    )

    last_gae = (
        delta
        + gamma * gae_lambda
        * next_non_terminal[step]
        * last_gae
    )
    advantages[step] = last_gae

returns = advantages + values
```

> **주의:** 위 코드는 GAE의 구조를 설명하기 위한 예시다. 실제 구현에서는 `next_values`와 `next_non_terminal`을 각 transition에 맞게 구성해야 한다. 마지막 Rollout 스텝에는 `last_value`가 필요할 수 있고, 종료 마스크 정의도 `dones`를 저장하는 방식에 맞춰야 한다.

### 왜 `reversed(range(n_steps))`인가?

GAE는 현재 Advantage를 계산할 때 다음 시점의 누적값 `last_gae`를 필요로 한다. 그러므로 마지막 스텝에서 시작해 첫 스텝 방향으로 역순 계산한다.

```python
for step in reversed(range(n_steps)):
    ...
```

예를 들어 `n_steps=5`이면 순서는 `4, 3, 2, 1, 0`이다.

## 6. Return을 `advantages + values`로 만드는 이유

```python
returns = advantages + values
```

이 식은 GAE 기반 Advantage와 기존 상태 가치 예측을 합쳐 Critic이 학습할 목표값을 구성한다.

예시:

- 기존 `value = 5`
- 계산한 `advantage = 2`
- 따라서 `return = 7`

**이 예시에서 Return 7은 환경이 실제로 즉시 준 보상이 아니다.** Critic이 자기 예측을 개선할 때 맞추도록 사용하는 목표값이다.

## 7. `terminated`와 `truncated` 주의사항

Gymnasium은 두 종료 신호를 구분한다.

- `terminated=True`: 과제 자체의 종료 조건에 도달했다. 이 경우 일반적으로 다음 상태 가치로 부트스트랩하지 않는다.
- `truncated=True`: 시간 제한 등 외부 조건으로 에피소드가 끊겼다. 이때는 종료 직전 상태의 가치 예측을 이용해야 하는 경우가 있다.

네가 공부한 구현은 `truncated` 때 `gamma * next_value`를 보상에 더하는 처리가 있었다. 이 방식은 그 다음 GAE 계산에서 같은 다음 상태 가치를 또 반영하는지, 종료 마스크가 어떻게 만들어지는지 함께 확인해야 한다. **같은 부트스트랩 값을 보상에 더하고 TD 식에서도 중복 반영하면 목표값이 잘못될 수 있다.**

따라서 종료 처리 코드를 검토할 때는 다음을 확인한다.

1. `terminated`와 `truncated`를 구분하고 있는가?
2. Rollout의 각 transition에 맞는 `next_value`가 사용되는가?
3. GAE 누적이 에피소드 경계를 넘어가지 않도록 마스크가 올바른가?
4. 다음 상태 가치를 보상에 미리 더했다면, 이후 계산에서 같은 값을 중복 반영하지 않는가?

이 부분은 코드가 저장하는 `dones`의 의미를 정확히 알아야 검증할 수 있다. 단순히 모든 종료 신호를 `done=True` 하나로 뭉쳐 처리하는 구현은 의도에 맞는지 추가 점검이 필요하다.

---

# 5강. Value Loss, 전체 Loss, 역전파와 업데이트

## 1. Critic은 무엇을 맞추려고 하는가?

Critic은 `V(s)`를 예측한다. 학습할 때는 예측값 `values`와 계산해 둔 목표값 `returns`를 비교한다.

```python
value_loss = F.mse_loss(returns, values)
```

MSE(Mean Squared Error, 평균제곱오차)는 두 값의 차이를 제곱하고 평균한 손실이다.

`MSE = mean((target - prediction)²)`

여기서:

- `returns`: 목표값(target)
- `values`: 현재 Critic의 예측값(prediction)

예를 들어 `returns=[7, 4]`, `values=[5, 5]`라면 오차는 `[2, -1]`, 제곱 오차는 `[4, 1]`, 평균은 `2.5`다.

**`values`라는 변수명을 보고 Rollout 때 저장한 값을 뜻한다고 생각하면 안 된다.** 업데이트 미니배치에서 다음과 같이 다시 계산한 `values`는 현재 모델의 Critic 예측값이다.

```python
values = model.get_value(obs)
```

경험 수집 때 저장한 `buffer.values`와 업데이트 중 재계산한 `values`는 시점과 용도가 다르다.

## 2. Actor와 Critic의 손실

PPO 업데이트에서는 보통 다음 요소를 조합한다.

```python
loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
```

| 요소 | 무엇을 학습하는가? | 네 코드에서의 의미 |
|---|---|---|
| `policy_loss` | Actor의 정책 | PPO Clipped Surrogate Objective를 최소화 형태로 변환한 손실 |
| `value_loss` | Critic의 가치 예측 | Return과 현재 Value 예측의 MSE |
| `entropy_loss` | Actor의 행동 분포 엔트로피 | 탐험 정도에 영향을 줄 수 있는 항 |
| `vf_coef` | Critic 손실의 비중 | 예: `0.5` |
| `ent_coef` | 엔트로피 항의 비중 | `0.0`이면 전체 손실에 해당 항 영향이 없다 |

사용한 구현에서 `entropy_loss = -entropy.mean()`이라면, `ent_coef`가 양수일 때 더 높은 엔트로피를 유도하는 효과를 낼 수 있다. 현재 설정처럼 `ent_coef=0.0`이면 이 항은 total loss에 영향을 주지 않는다.

## 3. Gradient와 역전파는 무엇인가?

Gradient(기울기)는 각 파라미터를 조금 바꾸었을 때 손실이 어떤 방향으로 얼마나 변하는지 알려주는 정보다.

`loss.backward()`는 PyTorch의 자동 미분을 통해 손실의 gradient를 계산한다. 이 함수가 그 자체로 파라미터를 업데이트하는 것은 아니다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

세 단계는 각각 다르다.

1. `zero_grad()`: 이전 업데이트에서 남아 있던 gradient를 지운다.
2. `backward()`: 현재 Loss에 대한 gradient를 계산한다.
3. `step()`: 옵티마이저가 gradient를 사용해 파라미터를 업데이트한다.

## 4. 왜 optimizer는 Loss를 최소화하는데 Actor 목적함수는 최대화인가?

Actor는 좋은 행동의 확률을 적절히 높이는 방향으로 목적함수를 최대화하고 싶다. 하지만 보편적인 학습 루프는 손실(Loss)을 최소화하도록 작성한다.

따라서 Actor의 목적함수 앞에 마이너스를 붙여 최소화 문제로 바꾼다.

`policy_loss = - objective.mean()`

예를 들어 목적함수 값이 `3`이면 손실은 `-3`, 목적함수 값이 `4`이면 손실은 `-4`다. 목적함수가 커질수록 음수 손실은 작아지므로, Loss를 최소화하면 원래 목적함수를 최대화하는 것과 같은 방향이 된다.

## 5. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_grad_norm
)
```

이 코드는 역전파로 계산한 gradient의 전체 노름이 설정값보다 커지지 않도록 제한한다. 너무 큰 gradient가 업데이트를 불안정하게 만드는 것을 완화하는 데 도움이 된다.

이것은 PPO의 `torch.clamp(ratio, ...)`와 다르다.

- **Ratio Clipping:** PPO 목적함수에서 Ratio가 주는 이득을 제한한다.
- **Gradient Clipping:** 옵티마이저가 쓸 gradient 크기를 제한한다.

## 6. 한 미니배치 업데이트 순서

```python
# 1. 현재 정책과 현재 Critic 예측을 구한다.
dist = model.get_dist(obs)
log_prob = dist.log_prob(actions)
values = model.get_value(obs)

# 2. Actor의 손실 계산 (6강에서 세부 설명)
ratio = torch.exp(log_prob - old_log_probs)

surr1 = advantages * ratio
surr2 = advantages * torch.clamp(
    ratio, 1 - clip_range, 1 + clip_range
)
policy_loss = -torch.minimum(surr1, surr2).mean()

# 3. Critic의 손실 계산
value_loss = F.mse_loss(returns, values)

# 4. 전체 손실 구성
loss = policy_loss + vf_coef * value_loss

# 5. gradient 계산 및 파라미터 업데이트
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
```

위는 핵심 흐름을 보여주는 학습용 발췌 코드다. 원본에 `entropy_loss` 항이 있을 경우 `ent_coef * entropy_loss`도 전체 손실에 더한다.

하나의 통합 Optimizer가 Actor와 Critic 파라미터를 모두 관리할 수 있다. 각 Loss의 gradient는 연결된 계산 경로를 통해 해당 네트워크에 전달된다. Actor와 Critic이 별도 네트워크라면 `policy_loss`는 주로 Actor를, `value_loss`는 Critic을 학습시킨다.

---

# 4~5강 셀프 체크

1. `reward`, `value`, `returns`의 차이를 설명하라.
2. `reward=1`, `gamma=0.9`, `next_value=4`, `value=3`, 종료되지 않았다면 TD Error는 얼마인가?
3. Advantage가 -2라면 행동은 기대보다 어떤 결과였다는 신호인가?
4. GAE 계산을 역순으로 하는 이유는 무엇인가?
5. `F.mse_loss(returns, values)`에서 `returns`는 목표값인가, 예측값인가?
6. `loss.backward()`와 `optimizer.step()`의 차이는 무엇인가?
7. `ent_coef=0.0`이면 무엇이 달라지는가?
8. Ratio Clipping과 Gradient Clipping은 무엇이 다른가?

<details>
<summary>정답과 해설 보기</summary>

1. Reward는 환경이 준 즉시 보상, Value는 Critic의 상태 가치 예측, Return은 Critic을 학습할 목표값이다.
2. `1 + 0.9 × 4 - 3 = 1.6`.
3. 해당 행동의 결과가 기대보다 나빴다는 신호다.
4. 현재 Advantage 계산에 이후 스텝의 누적 GAE가 필요하기 때문이다.
5. `returns`가 목표값, `values`가 현재 Critic의 예측값이다.
6. `backward()`는 gradient를 계산하고, `step()`은 그 gradient를 이용해 파라미터를 바꾼다.
7. 엔트로피 항이 전체 Loss에 영향을 주지 않는다. 탐험 자체가 반드시 사라진다는 뜻은 아니다.
8. Ratio Clipping은 PPO 목적함수에서 Ratio의 이득을 제한하고, Gradient Clipping은 역전파 gradient의 크기를 제한한다.

</details>

