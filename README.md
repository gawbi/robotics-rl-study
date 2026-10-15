# Robotics RL Study

휴머노이드 Whole-Body Control 강화학습을 목표로, **PPO를 직접 구현**하고 **Isaac Lab에서 Unitree G1 보행 정책을 학습해 보상 실험**까지 한 기록입니다.

![Isaac Lab에서 학습한 Unitree G1 평지 보행](day4-g1-reward-baseline/images/exp0_baseline.gif)

## 핵심 결과물

| 결과물 | 내용 | 바로가기 |
|---|---|---|
| **G1 보행 학습** | Isaac Lab `Isaac-Velocity-Flat-G1-v0`, RSL-RL PPO, 4096 환경 × 1500 iteration (27분). 99.6% 환경이 20초 동안 넘어지지 않고 속도 명령을 따라 걸음 | [day4](day4-g1-reward-baseline/notes.md) |
| **보상 실험** | `feet_air_time` 가중치만 0으로 바꿔 비교. 넘어짐·명령 추종은 그대로, 발 미끄러짐 벌점이 −0.0123 → −0.0162로 커짐 (크기 +32%, 발이 더 미끄러짐) | [day5](day5-g1-reward-experiment/notes.md) |
| 실험 종합·다음 계획 | 배운 점, 다음 보상·하이퍼파라미터·sim-to-sim 실험 설계 | [day6](day6-g1-analysis-portfolio/notes.md) |
| PPO 직접 구현 | PyTorch로 PPO 구현, CartPole-v1에서 500점 (최대 점수) | [day2-ppo/my_ppo.py](day2-ppo/my_ppo.py) |
| PPO 노트 | 1~3강 기초 (Actor-Critic, 경험 수집), 4~7강 이론 (GAE, Ratio Clipping), 8·9강 디버깅과 하이퍼파라미터 실험 | [day1](day1-rl-basics/), [day2](day2-ppo/) |
| Isaac Lab Cartpole 분석 | 학습 곡선, RSL-RL PPO와 직접 구현한 PPO의 코드 단위 비교 | [day3](day3-isaaclab-cartpole/notes.md) |
| 트러블슈팅 | Isaac Lab 설치 후 `No module named 'isaaclab'` — 의존성 빌드 실패 추적 | [troubleshooting](troubleshooting/isaaclab_install_flatdict.md) |

## G1 보상 실험 요약

- **실험 0 (기본)**: 넘어짐 0.41%, 이동 명령 추종 보상 0.94 (마지막 100 iteration 평균).
- **실험 1**: `feet_air_time`(한 발로 서 있는 시간에 주는 상점) 가중치를 0.75 → 0으로 **하나만** 바꿈 (설정 파일 `diff`로 확인, 같은 seed).
  넘어짐(0.39%)과 명령 추종(0.93)은 그대로였지만, 발 미끄러짐 벌점(`feet_slide`)이 −0.0123 → −0.0162로 **크기가 32% 커졌다.** 즉 실험 1에서 발이 더 미끄러졌다.
- **해석**: 이 상점은 "걸을 수 있느냐"보다 "어떻게 걷느냐"에 영향을 주는 것으로 보인다. 단, 각 조건을 시드 1개로 한 번씩만 학습해서 이 차이가 잡음일 수 있다. 시드 3개 반복으로 확인할 예정이다.

![왼쪽 실험 0(기본), 오른쪽 실험 1(feet_air_time=0)](day5-g1-reward-experiment/images/exp0_vs_exp1.gif)

## 진행 현황

| 단계 | 내용 | 상태 |
|---|---|---|
| Day 1 | PPO 1~3강: 강화학습 큰 그림, Actor·Critic 네트워크, 경험 수집(RolloutBuffer) | ✅ 완료 |
| Day 2 | PPO 4~9강: Return·TD·GAE, Ratio Clipping, 직접 구현(`my_ppo.py`), CartPole-v1 디버깅·하이퍼파라미터 실험 | ✅ 완료 |
| Day 3 | Isaac Lab Cartpole 학습, RSL-RL PPO 설정·코드 분석 | ✅ 완료 |
| Day 4 | G1 평지 보행 기본 학습 (실험 0), 보상 구조 정리 | ✅ 완료 |
| Day 5 | 보상 실험 1: `feet_air_time` 끄기 | ✅ 완료 |
| Day 6 | 실험 종합, 배운 점, 다음 계획 | ✅ 완료 |
| 다음 | seed 반복, `action_rate_l2`·`flat_orientation_l2` 끄기, `clip_param` 비교, 관측 노이즈 끄기, MuJoCo sim-to-sim, SO-101 sim-to-real | ⬜ 예정 |

> 상태 표시: ✅ 완료 / 🔄 진행 중 / ⬜ 예정

## 저장소 구조

```
robotics-rl-study/
├── day1-rl-basics/              # PPO 노트 1~3강 (강화학습 기초, Actor-Critic, 경험 수집)
├── day2-ppo/                    # PPO 노트 4~9강, 직접 구현(my_ppo.py)
├── day3-isaaclab-cartpole/      # Isaac Lab Cartpole 학습 곡선, RSL-RL vs my_ppo.py 코드 비교
├── day4-g1-reward-baseline/     # G1 기본 학습(실험 0), 보상 구조 노트, 보행 GIF
├── day5-g1-reward-experiment/   # 보상 실험 1, 지표 추출 스크립트, 비교 그래프·GIF
├── day6-g1-analysis-portfolio/  # 실험 종합, 배운 점, 다음 실험 계획
└── troubleshooting/             # 설치·환경 문제 해결 기록
```

## 실습 환경

- OS: Ubuntu 24.04 (Day 2 `my_ppo.py` 실습은 macOS)
- GPU: NVIDIA RTX 5070 Ti
- Simulator: Isaac Sim 5.1, Isaac Lab 2.3.2
- RL: RSL-RL 3.1.2 (Isaac Lab), 직접 구현한 PPO (PyTorch, Gymnasium)

## 학습 자료

- K-MOOC 한양대학교 「강화학습」 강의
- Isaac Lab 공식 문서와 소스 코드
- RSL-RL 소스 코드
- Stable-Baselines3 소스 코드 (`my_ppo.py`의 RolloutBuffer 구조 참고)

## 주요 정리 및 트러블슈팅

| 주제 | 요약 | 문서 |
|---|---|---|
| Isaac Lab 설치 실패 | 설치는 정상 종료처럼 보였지만, 의존성 `flatdict` 빌드가 실패해 본체 `isaaclab`만 빠져 있었음. 로그를 거슬러 올라가 근본 원인을 찾음 | [troubleshooting](troubleshooting/isaaclab_install_flatdict.md) |
| PPO 디버깅 | 미니배치 shape, Loss NaN 검사, epoch별 ratio·clip_fraction 관찰 | [8강 노트](day2-ppo/PPO_08강_디버깅실습_및_트러블슈팅.md) |
| 끈 보상은 로그도 0 | Isaac Lab은 가중치 0인 보상 항목을 계산하지 않음 → 끈 행동은 별도 지표로 재야 함 | [day6](day6-g1-analysis-portfolio/notes.md) |
| 총 보상 비교의 함정 | 실험 1의 평균 보상 감소는 대부분 빠진 상점 자체 → 개별 지표로 비교 | [day5](day5-g1-reward-experiment/notes.md) |
| 학습률이 설정값과 다름 | RSL-RL은 KL을 보고 학습률을 자동 조절(`schedule: adaptive`) | [day3](day3-isaaclab-cartpole/notes.md) |

## 배운 점

- **Day 1~2**: PPO를 직접 구현하고 epoch마다 ratio와 clip_fraction을 찍어보면서, clipping이 "정책이 한 번에 너무 많이 바뀌는 것"을 어떻게 막는지 확인했다. 한 번의 실행 결과로 하이퍼파라미터의 우열을 결론 내리지 않는다.
- **Day 3**: RSL-RL과 직접 구현한 PPO는 GAE와 clipped surrogate가 같은 식이다. 차이는 advantage 정규화 시점, value loss clipping, KL 기반 학습률 조절이다. GPU 병렬 환경 4096개로 약 980만 step을 19초에 학습했다.
- **Day 4**: G1은 "넘어지지 않기 → 명령 따라가기 → 부드럽게 움직이기" 순서로 학습했다.
- **Day 5**: 보상 항목 하나를 끄면 걸음의 질이 바뀐다 (발 미끄러짐 벌점 −0.0123 → −0.0162, 실험 1에서 발이 더 미끄러짐). 보상 함수가 다른 실험은 총 보상이 아니라 개별 지표로 비교해야 하고, 끈 항목은 로그도 0이 되므로 무엇으로 잴지 먼저 정해야 한다.
- **설치**: 설치 명령이 끝났다고 성공한 게 아니다. 직접 원인과 근본 원인은 다를 수 있다.
