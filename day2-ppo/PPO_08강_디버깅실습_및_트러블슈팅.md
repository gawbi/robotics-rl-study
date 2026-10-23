# PPO 8강 — 디버깅 실습 및 트러블슈팅 기록

> 목적: PPO 코드가 실제로 어떤 데이터를 처리하는지 확인하고, 학습 중 비정상 수치나 정책 변화가 있는지 점검한다.  
> 환경: macOS / VS Code 터미널 / Python / Gymnasium / PyTorch / CartPole-v1

---

## 1. 8강에서 배운 것

이번 강의는 PPO 알고리즘을 새로 구현하기보다는, 이미 작성한 `my_ppo.py`를 실행하면서 내부 값들을 확인하는 실습이다.

- 미니배치 데이터의 `shape`를 출력해 데이터 차원을 확인했다.
- `th.isfinite(loss)`로 전체 손실값이 `NaN`이나 무한대인지 검사했다.
- `ratio`와 `clip_fraction`을 출력해 기존 정책과 현재 정책의 차이를 관찰했다.
- 첫 번째 업데이트의 첫 번째 Epoch와 두 번째 Epoch에서 Ratio를 비교했다.
- CartPole 학습 결과 및 저장된 모델 실행을 확인했다.

## 2. 실습 환경 준비와 실행 명령어

### 2.1 프로젝트 파일 위치

실습 파일의 실제 경로:

```text
day2-ppo/my_ppo.py
```

터미널에서 프로젝트 루트인 `robotics-rl-study` 폴더에 있을 때 실행하는 명령어:

```bash
python day2-ppo/my_ppo.py train
```

학습된 모델을 화면으로 확인하는 명령어:

```bash
python day2-ppo/my_ppo.py watch --episodes 5
```

> 실제 파일명이나 폴더 위치가 다르면 명령어도 실제 경로에 맞게 바꾼다.

### 2.2 필요한 라이브러리

`gymnasium`이 설치되어 있지 않다는 오류가 나면 현재 실행 중인 Python 환경에 설치한다.

```bash
python -m pip install gymnasium numpy torch
```

`watch` 실행 중 `pygame is not installed` 오류가 나면 CartPole 화면 렌더링 의존성을 설치한다.

```bash
python -m pip install --upgrade "gymnasium[classic-control]"
```

설치 여부 확인:

```bash
python -c "import gymnasium, numpy, torch; print('gymnasium:', gymnasium.__version__); print('numpy:', numpy.__version__); print('torch:', torch.__version__)"
```

## 3. 실습 1 — 미니배치 데이터의 크기 확인

### 3.1 코드 위치

`train()` 함수 안에서 다음 반복문을 찾는다.

```python
for epoch in range(n_epochs):
    for rollout_data in rollout_buffer.get(batch_size):
        actions = rollout_data.actions
```

이 안쪽 반복문은 `rollout_buffer.get(batch_size)`에서 미니배치 하나를 가져온다.

### 3.2 출력문을 넣은 코드

현재 코드는 `batch_idx`를 이용해 첫 번째 업데이트의 첫 번째 Epoch에서 첫 번째 미니배치만 출력하도록 구성했다.

```python
for epoch in range(n_epochs):
    for batch_idx, rollout_data in enumerate(
        rollout_buffer.get(batch_size)
    ):
        if update == 1 and epoch == 0 and batch_idx == 0:
            print("observations:", rollout_data.observations.shape)
            print("actions:", rollout_data.actions.shape)
            print("old_log_prob:", rollout_data.old_log_prob.shape)
            print("advantages:", rollout_data.advantages.shape)
            print("returns:", rollout_data.returns.shape)

        actions = rollout_data.actions
        # 아래 기존 학습 코드 계속...
```

### 3.3 예상 출력과 의미

```text
observations: torch.Size([64, 4])
actions: torch.Size([64])
old_log_prob: torch.Size([64])
advantages: torch.Size([64])
returns: torch.Size([64])
```

| 변수 | 출력 크기 | 의미 |
|---|---|---|
| `observations` | `[64, 4]` | 상태 데이터 64개, 상태마다 숫자 4개 |
| `actions` | `[64]` | 경험별 행동 64개 |
| `old_log_prob` | `[64]` | 경험별 기존 정책의 로그 확률 64개 |
| `advantages` | `[64]` | 경험별 Advantage 64개 |
| `returns` | `[64]` | 경험별 Critic 학습 목표값 64개 |

이유:

- `batch_size = 64`이므로 미니배치에 경험 64개가 들어간다.
- CartPole의 관측 상태는 4개 숫자로 표현되므로 `observations`의 모양은 `[64, 4]`다.
- 나머지 값들은 경험 하나당 값 하나씩이므로 크기가 `[64]`다.

### 3.4 왜 `enumerate()`를 썼는가?

기존 코드:

```python
for rollout_data in rollout_buffer.get(batch_size):
```

실습 코드:

```python
for batch_idx, rollout_data in enumerate(
    rollout_buffer.get(batch_size)
):
```

`enumerate()`는 데이터와 함께 순서 번호도 제공한다. `batch_idx`는 0부터 시작하므로 `batch_idx == 0` 조건을 이용해 첫 번째 미니배치만 골라낼 수 있다.

조건의 의미:

```python
if update == 1 and epoch == 0 and batch_idx == 0:
```

- `update == 1`: 첫 번째 Rollout 업데이트
- `epoch == 0`: 그 업데이트의 첫 번째 Epoch
- `batch_idx == 0`: 그 Epoch의 첫 번째 미니배치

이 조건을 붙이지 않으면 `print()`가 미니배치마다 실행된다. 현재 설정에서 한 Epoch에는 `2048 / 64 = 32`개 미니배치가 있으므로 출력이 크게 늘어난다.

## 4. 실습 2 — Loss가 비정상적인 수치인지 검사

### 4.1 추가한 코드

`loss`를 계산한 바로 다음 줄, `optimizer.zero_grad()` 전에 넣는다.

```python
loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

if not th.isfinite(loss).item():
    raise ValueError(f"비정상적인 Loss 발생: {loss.item()}")

optimizer.zero_grad()
loss.backward()
```

### 4.2 어떻게 검사하는가?

- `th.isfinite(loss)`: 값이 유한한 숫자인지 검사한다.
- `.item()`: 텐서에 들어 있는 단일 결과를 Python 값으로 가져온다.
- `not`: 참/거짓을 반대로 바꾼다.
- `raise ValueError(...)`: 비정상적인 값이면 오류를 발생시켜 실행을 중단하고 원인을 조사할 수 있게 한다.

예시 확인:

```bash
python -c "import torch as th; print(th.isfinite(th.tensor(2.5)).item()); print(th.isfinite(th.tensor(float('nan'))).item()); print(th.isfinite(th.tensor(float('inf'))).item())"
```

예상 출력:

```text
True
False
False
```

실습 실행 중 이 검사에서 오류가 발생하지 않고 학습이 완료됐다면, 검사한 시점의 **전체 `loss`는 `NaN` 또는 무한대가 아니었다**는 뜻이다. 모든 종류의 버그나 학습 문제까지 없다는 보증은 아니다.

## 5. 실습 3 — PPO Ratio와 Clipping 상태 확인

### 5.1 코드 위치

`train()`의 미니배치 학습 루프에서 다음 줄을 찾는다.

```python
ratio = th.exp(log_prob - rollout_data.old_log_prob)
```

바로 아래에 다음 진단 코드를 추가했다.

```python
if update == 1 and epoch == 1 and batch_idx == 0:
    clip_fraction = (
        (ratio < 1 - clip_range) |
        (ratio > 1 + clip_range)
    ).float().mean()

    print("ratio min:", ratio.min().item())
    print("ratio mean:", ratio.mean().item())
    print("ratio max:", ratio.max().item())
    print("clip_fraction:", clip_fraction.item())
```

설정값:

```python
clip_range = 0.2
batch_size = 64
```

따라서 Clipping 구간은 `[0.8, 1.2]`다.

### 5.2 지표가 의미하는 것

- `ratio min`: 미니배치에서 가장 작은 Ratio
- `ratio mean`: 미니배치의 Ratio 평균
- `ratio max`: 미니배치에서 가장 큰 Ratio
- `clip_fraction`: Ratio가 0.8보다 작거나 1.2보다 큰 데이터의 비율

Ratio는 현재 정책과 경험 수집 당시 정책이 **같은 상태에서 같은 행동을 선택할 확률의 비율**이다.

- Ratio가 1에 가까움: 그 상태-행동 조합에서 현재와 기존 정책의 확률이 비슷함.
- Ratio가 0.8보다 작음: 현재 정책에서 해당 행동의 확률이 기존보다 크게 낮아짐.
- Ratio가 1.2보다 큼: 현재 정책에서 해당 행동의 확률이 기존보다 크게 높아짐.

중요: `torch.clamp()`는 Ratio 원본을 강제로 바꾸는 것이 아니라, PPO 목적함수를 계산할 때 제한된 Ratio를 사용하도록 한다.

### 5.3 실제 첫 번째 관찰 결과

`epoch == 1`에서 관찰한 결과:

```text
ratio min: 0.670654833316803
ratio mean: 0.9988953471183777
ratio max: 1.3701021671295166
clip_fraction: 0.125
```

해석:

- 최소 Ratio는 약 `0.671`로 0.8보다 낮다.
- 평균 Ratio는 약 `0.999`로 1에 가깝다. 그렇다고 모든 Ratio가 1인 것은 아니다.
- 최대 Ratio는 약 `1.370`으로 1.2보다 높다.
- `clip_fraction = 0.125`는 해당 미니배치의 12.5%가 Clipping 구간 바깥이라는 뜻이다.
- 배치가 64개이므로 `64 × 0.125 = 8`; 해당 미니배치에서 8개가 구간 바깥이었다.

이 결과는 오류라고 단정할 수 없다. 특정 미니배치의 Ratio 분포를 보여주는 진단 정보다.

### 5.4 업데이트 전과 후 비교

조건을 다음처럼 바꿔 관찰했다.

```python
if update == 1 and epoch == 0 and batch_idx == 0:
```

결과:

```text
ratio min: 0.9999999403953552
ratio mean: 1.0
ratio max: 1.000000238418579
clip_fraction: 0.0
```

첫 번째 Epoch의 첫 번째 미니배치는 경험을 수집했던 정책의 파라미터가 아직 업데이트되지 않은 시점에서 평가된다. 따라서 현재 정책과 기존 정책이 사실상 같아서 Ratio가 1에 매우 가깝게 나타난다. 소수점 끝자리의 미세한 차이는 부동소수점 연산에서 생길 수 있다.

`epoch == 1`은 두 번째 Epoch다. 첫 번째 Epoch에서 여러 미니배치로 파라미터를 업데이트한 뒤이므로, 기존 정책과 현재 정책의 확률이 달라져 Ratio가 1에서 벗어날 수 있다.

**직접 확인한 핵심:** 정책 파라미터가 바뀌면 같은 저장 데이터에 대한 현재 정책의 행동 확률도 바뀌고, 이 변화가 Ratio에 나타난다.

## 6. CartPole 학습 및 모델 실행 결과

학습 명령어:

```bash
python day2-ppo/my_ppo.py train
```

실행 로그에서 확인한 결과:

- `update`가 `1/48`부터 `48/48`까지 진행됐다.
- `ep_rew_mean`이 초기 약 25점대에서 상승해, 후반에 500.0을 기록했다.
- 모델이 `./my_ppo_model.pt`에 저장됐다.

모델 실행 명령어:

```bash
python day2-ppo/my_ppo.py watch --episodes 5
```

5개 에피소드 모두 500점을 기록했다. CartPole-v1에서는 에피소드 최대 점수가 500이므로, 저장한 모델이 테스트 실행에서도 제한 시간까지 막대를 유지한 결과다.

참고: `watch()`에서 정책은 `argmax()`로 행동을 선택한다. 이는 학습 중 `dist.sample()`로 확률적으로 행동을 선택하는 것과 다르다.

## 7. 이번 실습에서 기억할 구분

| 지표/도구 | 확인하는 대상 | 주의사항 |
|---|---|---|
| `shape` | 텐서 차원 | 올바른 크기라고 학습이 잘된다는 뜻은 아님 |
| `th.isfinite(loss)` | 전체 Loss가 유한한지 | 모든 버그를 잡는 것은 아님 |
| `ratio` | 기존 정책 대비 현재 정책의 행동 확률 비율 | Ratio가 1에서 벗어나는 것이 자동으로 오류는 아님 |
| `clip_fraction` | Ratio가 Clipping 구간 밖인 비율 | 높거나 낮은 값 하나만으로 학습 품질을 단정하지 않음 |
| `ep_rew_mean` | 최근 에피소드의 실제 보상 평균 | 실제 환경 성능을 판단하는 핵심 지표 중 하나 |
| `value_loss` | Critic 예측값과 Return 목표의 MSE | 출력은 마지막 미니배치 값일 수 있으며 전체 평균이 아닐 수 있음 |
| `policy_loss` | PPO 정책 목적함수에서 만들어진 손실 | 0에 가깝다고 무조건 정책이 완벽하다는 뜻은 아님 |

---

## 8. 트러블슈팅 기록

아래는 이번 실습에서 발생한 문제와 해결 방법이다. 다음 강의에서 새 문제가 발생하면 이어서 추가한다.

### 문제 1 — `my_ppo.py` 파일을 찾을 수 없음

**오류 예시**

```text
python: can't open file 'my_ppo.py': [Errno 2] No such file or directory
```

**원인**: 터미널의 현재 폴더에 `my_ppo.py`가 없었다. 실제 파일은 `day2-ppo` 하위 폴더에 있었다.

**확인 방법**

```bash
find . -type f -name "*.py"
```

**해결 명령어**

```bash
python day2-ppo/my_ppo.py train
```

**교훈**: 파이썬 파일 경로는 터미널의 현재 폴더를 기준으로 해석한다. 파일명을 추측하기보다 실제 경로를 먼저 찾는다.

### 문제 2 — `ModuleNotFoundError: No module named 'gymnasium'`

**원인**: 실행에 사용한 Python 환경에 Gymnasium이 설치되어 있지 않았다.

**해결 명령어**

```bash
python -m pip install gymnasium numpy torch
```

**교훈**: `python -m pip`를 사용하면 해당 `python` 실행 환경에 라이브러리를 설치하기 쉽다.

### 문제 3 — `pygame is not installed`

**발생 위치**: `watch` 모드에서 CartPole 화면을 띄울 때 발생했다. 학습용 `train` 모드는 화면 렌더링이 필요하지 않았지만, `watch` 모드는 `render_mode="human"`을 사용한다.

**해결 명령어**

```bash
python -m pip install --upgrade "gymnasium[classic-control]"
```

**교훈**: 환경 학습과 GUI 렌더링에는 필요한 의존성이 다를 수 있다. 화면 실행이 필요한 경우 렌더링 추가 패키지를 확인한다.

### 문제 4 — `shape`가 너무 많이 출력됨

**원인**: `print()`가 미니배치 반복문 안에 있어서 매 미니배치마다 실행됐다. `n_steps=2048`, `batch_size=64`이면 Epoch마다 32개 미니배치가 생성된다.

**해결 방법**: `enumerate()`로 미니배치 번호 `batch_idx`를 받고, 첫 번째 미니배치에서만 출력한다.

```python
for batch_idx, rollout_data in enumerate(
    rollout_buffer.get(batch_size)
):
    if update == 1 and epoch == 0 and batch_idx == 0:
        print(rollout_data.observations.shape)
```

**교훈**: 반복문 안의 코드는 반복 횟수만큼 실행된다. 디버깅 출력은 조건을 붙여 제한할 수 있다.

### 문제 5 — Ratio 통계가 Epoch에 따라 달라짐

**관찰**: `epoch == 0`에서 Ratio는 거의 1, `epoch == 1`에서 최소 약 0.671, 최대 약 1.370으로 변했다.

**원인**: 첫 번째 Epoch에서 여러 미니배치를 학습하며 정책 파라미터가 업데이트됐기 때문이다.

**교훈**: PPO는 경험을 수집할 당시의 로그 확률 `old_log_prob`를 저장하고, 업데이트된 정책의 현재 로그 확률과 비교해 Ratio를 계산한다. 정책이 바뀌면 Ratio도 변할 수 있다.

---

## 9. 다음에 문제를 만나면 확인할 순서

1. 오류 메시지에서 가장 아래쪽의 핵심 오류 유형과 메시지를 읽는다.
2. 파일 경로 문제인지, 패키지 설치 문제인지, 코드 내부 오류인지 분류한다.
3. 오류가 발생한 파일과 줄 번호를 확인한다.
4. 최근에 추가하거나 수정한 코드부터 확인한다.
5. 한 번에 여러 곳을 수정하지 말고, 하나씩 수정한 뒤 다시 실행한다.
6. 해결한 방법과 원인을 이 문서의 트러블슈팅 기록에 추가한다.

---


## 10. 8강 완료 체크리스트

- [x] 미니배치 데이터의 `shape`를 출력했다.
- [x] `th.isfinite(loss)`를 이용한 Loss 검사를 추가했다.
- [x] Ratio 최솟값·평균·최댓값과 `clip_fraction`을 관찰했다.
- [x] Epoch 0과 Epoch 1의 Ratio 차이를 비교했다.
- [x] CartPole 학습이 끝나고 모델 파일이 저장되는 것을 확인했다.
- [x] 저장한 모델을 `watch` 모드로 실행해 5개 에피소드에서 500점을 확인했다.
- [x] 이번 실습의 트러블슈팅과 해결 방법을 기록했다.

**다음 강의 예고 — 9강:** 학습 결과를 분석하고 `learning_rate`, `n_epochs`, `clip_range` 같은 하이퍼파라미터를 하나씩 바꿔 결과를 비교한다.
