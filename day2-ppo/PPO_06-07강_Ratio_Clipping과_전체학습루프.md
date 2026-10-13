# PPO 공부 노트 6~7강: Ratio·Clipping부터 전체 학습 루프까지

> 목표: `ratio`, `torch.clamp`, Clipped Surrogate Objective를 이해하고, PPO의 `train()` 흐름을 처음부터 끝까지 설명할 수 있다.

## 이 노트의 큰 질문

1. PPO는 왜 기존 정책의 확률을 저장해 두는가?
2. `ratio`와 `clip_range`는 어떻게 사용되는가?
3. 왜 Actor 목적함수 앞에 마이너스 부호를 붙이는가?
4. 경험 수집에서 `optimizer.step()`까지 어떻게 연결되는가?

---

# 6강. Ratio와 Clipped Surrogate Objective

## 1. 기존 정책과 현재 정책 비교

PPO는 현재 정책으로 경험을 모은 다음, 같은 경험을 이용해 정책을 업데이트한다. 업데이트 중 정책은 바뀔 수 있으므로, 당시 정책이 행동에 부여했던 확률을 저장해 현재 정책과 비교한다.

```python
# 경험 수집 때 저장한 값
old_log_prob = ...

# 업데이트 시 현재 모델로 다시 계산한 값
log_prob = dist.log_prob(actions)
```

- `old_log_prob`: 경험을 수집했을 당시 정책이 해당 행동에 부여한 로그 확률
- `log_prob`: 업데이트 시점의 현재 정책이 같은 상태·행동에 부여한 로그 확률

이 둘은 서로 다른 시점의 정책을 나타낸다. `actions`는 버퍼에 저장한 당시 행동을 사용해야 비교가 성립한다.

## 2. Ratio 수식

정책 확률 비율은 다음과 같다.

`r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)`

로그 확률을 사용하면 다음과 같이 계산할 수 있다.

```python
ratio = torch.exp(log_prob - old_log_probs)
```

왜 가능한가?

`log(p_new) - log(p_old) = log(p_new / p_old)` 이므로, 지수 함수를 적용하면 `p_new / p_old`가 된다.

| Ratio | 해석 |
|---:|---|
| `1.0` | 해당 행동의 확률이 기존 정책과 같다 |
| `1.5` | 해당 행동의 확률이 기존의 1.5배다 |
| `0.5` | 해당 행동의 확률이 기존의 절반이다 |

예: 기존 확률이 `0.4`, 현재 확률이 `0.6`이면 `ratio = 0.6 / 0.4 = 1.5`다.

로그 확률을 이용하는 실질적인 구현상의 이유는 분포 API가 로그 확률을 제공하고, 로그 공간에서 차이를 계산해 비율을 구하는 것이 확률을 여러 번 곱하는 방식보다 수치적으로 안정적인 데 도움이 되기 때문이다. 로그를 취한다고 무조건 더 정확한 정보가 새로 생기는 것은 아니다.

## 3. Ratio만 보면 좋은 행동인지 알 수 없다

정책은 행동 확률을 바꾸지만, 그 행동이 좋은지 나쁜지는 Advantage의 부호와 함께 판단해야 한다.

- `advantage > 0`: 기대보다 좋았다는 추정. 그 행동의 확률을 높이는 방향이 유리할 수 있다.
- `advantage < 0`: 기대보다 나빴다는 추정. 그 행동의 확률을 낮추는 방향이 유리할 수 있다.

따라서 `ratio > 1`이 무조건 좋은 업데이트라는 뜻은 아니다. **Ratio는 확률 변화량, Advantage는 그 행동 결과의 상대적 평가**를 나타낸다.

## 4. Clipping 범위

네 설정은 다음과 같다.

```python
clip_range = 0.2
```

이 값은 다음 범위를 만든다.

`[1 - clip_range, 1 + clip_range] = [0.8, 1.2]`

코드:

```python
clipped_ratio = torch.clamp(
    ratio,
    1 - clip_range,
    1 + clip_range,
)
```

`torch.clamp(x, min, max)`는 `x`가 min보다 작으면 min, max보다 크면 max, 사이에 있으면 x를 반환한다.

| 원래 Ratio | Clipped Ratio (`clip_range=0.2`) |
|---:|---:|
| 0.5 | 0.8 |
| 0.9 | 0.9 |
| 1.0 | 1.0 |
| 1.1 | 1.1 |
| 1.6 | 1.2 |

**핵심:** 원래 Ratio 텐서를 무조건 `[0.8, 1.2]` 안에 강제로 고정하는 뜻이 아니다. PPO 목적함수 계산을 위해 제한한 Ratio를 따로 만든다.

## 5. Clipped Surrogate Objective

PPO는 Ratio와 Advantage를 곱한 값을 사용하고, 원래 Ratio를 사용한 값과 Clipped Ratio를 사용한 값 중 작은 값을 취한다.

```python
surr1 = ratio * advantages
surr2 = torch.clamp(
    ratio, 1 - clip_range, 1 + clip_range
) * advantages

policy_loss = -torch.minimum(surr1, surr2).mean()
```

수식 형태로는 다음과 같이 이해할 수 있다.

`L_clip = E[min(ratio × A, clip(ratio) × A)]`

일반적으로 PPO의 정책 목적함수는 크게 만들고 싶다. 하지만 코드에서는 옵티마이저로 Loss를 최소화하므로 앞에 마이너스 부호를 붙인다.

### 예시 A: Advantage가 양수

- `advantage = +2`
- `ratio = 1.5`
- `clipped_ratio = 1.2`

```text
surr1 = 1.5 × 2 = 3.0
surr2 = 1.2 × 2 = 2.4
min(3.0, 2.4) = 2.4
```

정책이 해당 행동의 확률을 크게 높여도 목적함수에서 그 변화로 얻는 이득은 제한된다.

### 예시 B: Advantage가 음수

- `advantage = -2`
- `ratio = 0.5`
- `clipped_ratio = 0.8`

```text
surr1 = 0.5 × (-2) = -1.0
surr2 = 0.8 × (-2) = -1.6
min(-1.0, -1.6) = -1.6
```

음수인 Advantage에서는 부호 때문에 어떤 값이 더 작은지 달라진다. 그래서 `min()`을 단순히 “항상 Ratio가 더 작은 값을 선택한다”고 외우면 안 된다. **곱셈 결과 두 개의 실제 값을 비교해야 한다.**

## 6. 왜 `policy_loss` 앞에 마이너스를 붙이는가?

```python
policy_loss = -torch.minimum(surr1, surr2).mean()
```

- PPO 목적함수: 최대화하고 싶다.
- 일반적인 옵티마이저: Loss를 최소화한다.
- 해결 방법: 목적함수에 `-1`을 곱해 최소화 문제로 바꾼다.

목적함수가 `3`이면 Loss는 `-3`, 목적함수가 `4`이면 Loss는 `-4`다. Loss를 더 작게 만드는 방향은 원래 목적함수를 더 크게 만드는 방향과 일치한다.

## 7. Ratio Clipping과 Gradient Clipping 구별

| 기능 | 코드 예시 | 제한하는 대상 |
|---|---|---|
| PPO Ratio Clipping | `torch.clamp(ratio, ...)` | 정책 목적함수에서 사용되는 확률 비율의 영향 |
| Gradient Clipping | `clip_grad_norm_(model.parameters(), ...)` | 역전파 후 파라미터 업데이트에 사용되는 gradient 크기 |

두 기능은 적용 시점과 목적이 다르다.

---

# 7강. 전체 학습 코드 연결하기

## 1. `train()`을 큰 블록으로 나눠 읽기

PPO의 `train()` 함수는 다음과 같은 큰 단계로 읽는다.

```text
A. Rollout 수집
   └─ Actor가 행동 선택, 환경이 보상 반환, 버퍼에 저장

B. Advantage / Return 계산
   └─ GAE와 마지막 상태 가치 이용

C. Epoch와 미니배치 반복
   ├─ D. 현재 정책 / 가치 예측 계산
   ├─ E. Policy Loss와 Value Loss 계산
   └─ F. backward → gradient 제한 → optimizer.step

G. 통계 출력 및 모델 저장
```

이 구조를 먼저 머릿속에 두고 실제 코드의 반복문을 읽는다.

## 2. A단계 — 경험 수집

```python
for step in range(n_steps):
    with torch.no_grad():
        dist = model.get_dist(obs)
        action = dist.sample()
        old_log_prob = dist.log_prob(action)
        value = model.get_value(obs)

    next_obs, reward, terminated, truncated, info = env.step(
        action.item()
    )

    # 버퍼에 obs, action, reward, 종료 정보, value, old_log_prob 저장
    buffer.add(obs, action, reward, done, value, old_log_prob)

    obs = next_obs
```

이 부분에서 핵심은 **현재 정책의 확률을 `old_log_prob`로 보존하는 것**이다. 업데이트 미니배치에서는 정책이 달라진 뒤 같은 행동을 다시 평가하므로 원래 값을 잃어버리면 Ratio를 계산할 수 없다.

위 코드는 흐름을 설명하는 축약본이다. 실제 코드에서는 `terminated`/`truncated` 처리, 텐서 자료형, 환경 reset 등이 구현되어야 한다.

## 3. B단계 — 마지막 가치·GAE 계산

Rollout의 마지막 상태는 아직 환경의 진짜 최종 상태가 아닐 수 있다. 이 경우 마지막 상태의 Critic 예측인 `last_value`를 이용해 Return과 Advantage를 계산해야 할 수 있다.

```python
with torch.no_grad():
    last_value = model.get_value(obs)

buffer.compute_returns_and_advantage(last_value)
```

이후 버퍼에는 업데이트에 쓸 `advantages`와 `returns`가 준비된다.

- `advantages`: Actor 목적함수에 사용
- `returns`: Critic Value Loss의 목표값으로 사용

## 4. C단계 — Epoch와 미니배치 반복

```python
for epoch in range(n_epochs):
    for batch in buffer.get(batch_size):
        # 현재 미니배치 데이터를 텐서로 준비
        ...
```

앞에서 본 설정 `n_steps=2048`, `batch_size=64`, `n_epochs=10`이라면 한 Rollout당 320번의 미니배치 업데이트가 수행된다(데이터가 정확히 나누어떨어지는 설정일 때).

같은 경험을 여러 epoch 재사용하지만, 매 업데이트에서 Actor와 Critic의 파라미터가 바뀔 수 있다. 따라서 `old_log_probs`와 현재 `log_prob`가 달라지고 Ratio도 업데이트마다 달라질 수 있다.

## 5. D단계 — 현재 정책과 Critic 예측 계산

```python
dist = model.get_dist(obs)
log_prob = dist.log_prob(actions)
entropy = dist.entropy().mean()
values = model.get_value(obs)
```

여기서 `obs`, `actions`는 현재 미니배치의 저장된 데이터다.

- `dist`: 현재 Actor가 만든 행동 분포
- `log_prob`: 현재 Actor가 저장된 행동에 부여하는 로그 확률
- `entropy`: 현재 행동 분포의 엔트로피
- `values`: 현재 Critic이 계산한 상태 가치 예측

주의: 이 시점의 `values`는 새로 계산한 예측이다. Rollout 수집 때 저장했던 `buffer.values`와 혼동하지 않는다.

## 6. E단계 — Loss 계산

아래는 핵심 업데이트 계산의 통합 예시다.

```python
# Ratio: 현재 정책과 기존 정책의 행동 로그 확률을 비교
ratio = torch.exp(log_prob - old_log_probs)

# 원래 정책 목적함수와 clipped 정책 목적함수의 각 항
surr1 = ratio * advantages
surr2 = torch.clamp(
    ratio,
    1 - clip_range,
    1 + clip_range,
) * advantages

# 최대화 목표를 최소화 Loss로 변환
policy_loss = -torch.minimum(surr1, surr2).mean()

# Critic의 현재 예측과 목표 Return 비교
value_loss = F.mse_loss(returns, values)

# 구현에 entropy_loss = -entropy 로 정의했다면 이 항을 더한다.
entropy_loss = -entropy
loss = (
    policy_loss
    + ent_coef * entropy_loss
    + vf_coef * value_loss
)
```

이 코드가 각 네트워크에 주는 신호를 요약하면 다음과 같다.

- `policy_loss`: Actor가 행동 분포를 어떻게 바꿀지 학습한다.
- `value_loss`: Critic이 Return을 더 잘 예측하도록 학습한다.
- `entropy_loss`: `ent_coef`가 0보다 클 때 행동 분포의 무작위성/탐험에 영향을 줄 수 있다.

`ent_coef=0.0`이라면 entropy 항의 값은 계산하더라도 전체 `loss`에 영향이 없다.

## 7. F단계 — 파라미터 업데이트

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_grad_norm,
)
optimizer.step()
```

실행 순서를 바꾸어 읽지 않는다.

1. 남은 gradient 제거
2. 현재 Loss 기준으로 gradient 계산
3. gradient 크기 제한
4. 파라미터 업데이트

`optimizer.step()`이 실행된 뒤 모델의 파라미터는 바뀐다. 다음 미니배치에서는 바뀐 파라미터를 사용해 새로운 분포와 가치를 계산한다.

## 8. G단계 — 출력과 저장

학습 중 평균 에피소드 점수, Loss 또는 기타 통계를 출력하면 학습이 진행되는 양상을 모니터링할 수 있다. 모델 파라미터를 저장하는 코드가 `torch.save(model.state_dict(), model_dir)` 형태라면, 학습된 `state_dict`를 파일로 저장하는 것이다.

`state_dict`는 학습된 파라미터 값들의 묶음이지, 환경이나 학습 코드 전체를 자동으로 저장하는 것은 아니다. 모델을 다시 불러올 때는 동일한 네트워크 구조를 만든 뒤 가중치를 불러와야 한다.

---

# 전체 코드 읽기 체크리스트

`train()` 함수를 다시 열고 각 항목에 체크한다.

- [ ] Actor가 상태를 받아 행동 분포를 만든다.
- [ ] 행동을 샘플링하고 `old_log_prob`를 저장한다.
- [ ] Critic의 `value`를 저장한다.
- [ ] 환경에서 `reward`와 종료 신호를 받는다.
- [ ] 경험을 버퍼에 저장한다.
- [ ] Rollout이 끝나면 마지막 상태 가치 및 GAE를 계산한다.
- [ ] `returns`와 `advantages`를 미니배치에서 가져온다.
- [ ] 현재 정책의 `log_prob`와 저장된 `old_log_probs`로 Ratio를 계산한다.
- [ ] Ratio Clipping을 목적함수 계산에 적용한다.
- [ ] Actor의 `policy_loss`와 Critic의 `value_loss`를 계산한다.
- [ ] `zero_grad() → backward() → clip_grad_norm_() → step()` 순서로 업데이트한다.
- [ ] 종료 처리에서 `terminated`와 `truncated`를 올바르게 구분한다.

## 구현 점검: 가장 조심해야 할 부분

네가 공부한 코드에서는 `truncated`일 때 `gamma * next_value`를 보상에 더하는 처리가 있었다. 이 방식을 사용할 경우 GAE 쪽에서 같은 가치가 중복 계산되는지, 종료 마스크가 올바른지 반드시 확인한다. Rollout 수집, 마지막 가치 계산, GAE 마스크를 서로 독립된 코드 조각으로 읽으면 오류를 놓치기 쉽다.

또한 `done` 하나에 `terminated or truncated`를 넣었는지, GAE가 `done`을 어떤 인덱스로 참조하는지에 따라 결과가 달라질 수 있다. 구현 전체를 확인하지 않고 종료 처리를 무작정 한 줄만 바꾸는 것은 권장하지 않는다.

---

# 6~7강 셀프 체크

1. 기존 `old_log_prob`를 저장하는 이유는 무엇인가?
2. 기존 행동 확률이 0.4이고 현재 정책의 확률이 0.6이라면 Ratio는 얼마인가?
3. `clip_range=0.2`, `ratio=1.6`일 때 Clipped Ratio는 얼마인가?
4. Advantage가 양수일 때 Ratio가 지나치게 커지면 Clipped Surrogate Objective는 어떤 역할을 하는가?
5. `policy_loss` 앞에 마이너스 부호를 붙이는 이유는 무엇인가?
6. `value_loss`에서 `returns`와 `values`는 각각 무엇인가?
7. `backward()` 후 `optimizer.step()`을 실행하는 이유는 무엇인가?
8. Ratio Clipping과 Gradient Clipping의 차이는 무엇인가?
9. 수집된 경험에 대해 같은 Rollout을 여러 epoch 학습하면 왜 현재 `log_prob`와 `old_log_prob`가 달라질 수 있는가?
10. `truncated` 처리에서 무엇을 점검해야 하는가?

<details>
<summary>정답과 해설 보기</summary>

1. 현재 정책의 로그 확률과 비교해 기존 대비 행동 확률 비율(Ratio)을 계산하기 위해서다.
2. `0.6 / 0.4 = 1.5`.
3. `1 + 0.2 = 1.2`로 제한된다.
4. 그 행동의 확률이 크게 바뀌어 얻는 정책 목적함수상의 이득을 제한한다. 실제 Ratio 자체를 강제로 고정하는 것은 아니다.
5. 원래 정책 목적함수는 최대화하지만 옵티마이저는 Loss를 최소화하므로, 목적함수에 음수를 곱해 최소화 문제로 바꾼다.
6. `returns`는 Critic의 목표값이고, `values`는 현재 Critic 예측값이다.
7. `backward()`는 gradient를 계산하고 `optimizer.step()`은 그 gradient로 파라미터를 업데이트한다.
8. Ratio Clipping은 PPO 목적함수에서 Ratio의 영향을 제한하고, Gradient Clipping은 gradient 크기를 제한한다.
9. 각 업데이트가 Actor의 파라미터를 바꾸므로 이후 같은 상태·행동의 현재 로그 확률이 달라질 수 있다.
10. `terminated`와 `truncated`의 구분, 다음 상태 가치의 부트스트랩, GAE 마스크, 보상에 미리 더한 가치가 중복 반영되는지를 확인한다.

</details>

## 마지막 연습: 코드 없이 설명하기

아래 네 문장을 자료를 보지 않고 설명해 본다.

1. “PPO는 경험을 모은 다음, 그 경험으로 여러 번 정책을 업데이트한다.”
2. “Ratio는 정책이 바뀐 정도를 행동 확률 기준으로 비교한다.”
3. “Advantage는 행동 결과를 기대와 비교하는 신호이고, Return은 Critic의 학습 목표로 사용된다.”
4. “`backward()`는 gradient를 계산하고 `optimizer.step()`은 파라미터를 바꾼다.”

각 문장을 네 말로 설명할 수 있다면 1~7강의 핵심 연결을 이해하고 있는 것이다.
