# Robotics RL Study

휴머노이드 로봇 강화학습(Whole-Body Control)을 목표로, 강화학습 기초부터 PPO, Isaac Lab 실습까지 직접 공부하고 실험한 기록입니다.

## 목표

- 강화학습 기초 개념과 PPO 알고리즘을 원리부터 이해하기
- Isaac Lab에서 PPO 설정과 보상(reward)이 학습 결과에 미치는 영향을 직접 실험하기
- 휴머노이드 보행 정책 학습 및 sim-to-real까지 확장하기

## 진행 현황

| 단계 | 내용 | 상태 |
|---|---|---|
| Day 1 | RL 기초: MDP, 가치 함수, MC/TD, On/Off-policy | 🔄 진행 중 |
| Day 2 | REINFORCE → A2C → PPO, SB3로 LunarLander 학습 | ⬜ 예정 |
| Day 3 | Isaac Lab Cartpole로 PPO 하이퍼파라미터 실험 | ⬜ 예정 |
| Day 4~6 | 휴머노이드(G1) 보행 학습 및 보상 실험 → [별도 저장소](링크) | ⬜ 예정 |
| 이후 | Sim-to-sim(MuJoCo), SO-101 로봇팔 sim-to-real | ⬜ 예정 |

> 상태 표시: ✅ 완료 / 🔄 진행 중 / ⬜ 예정

## 저장소 구조

```
robotics-rl-study/
├── day1-rl-basics/          # RL 기초 개념 정리, 셀프 체크
├── day2-ppo/                # 정책 경사 ~ PPO 정리, LunarLander 실습 코드
└── day3-isaaclab-cartpole/  # RSL-RL PPO 설정 분석, 하이퍼파라미터 실험
```

## 실습 환경

- OS: Ubuntu 24.04
- GPU: NVIDIA RTX 5070 Ti
- Simulator: Isaac Sim 5.1, Isaac Lab
- RL Library: Stable-Baselines3, RSL-RL

## 학습 자료

- K-MOOC 한양대학교 「강화학습」 강의 
- Isaac Lab 공식 문서
- Stable-Baselines3 소스 코드

## 주요 정리 및 트러블슈팅

공부하면서 막혔던 부분과 해결 과정을 기록합니다.

## 배운 점

각 단계를 마칠 때마다 핵심적으로 배운 점을 한두 줄씩 추가합니다.
