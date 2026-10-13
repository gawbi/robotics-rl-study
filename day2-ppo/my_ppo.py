import argparse
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

### input parameter ###################
env_name = "CartPole-v1"
total_timesteps = 100_000
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.1
learning_rate = 3e-4
vf_coef = 0.5
ent_coef = 0.0
max_grad_norm = 0.5
model_dir = "./my_ppo_model.pt"
########################################


# ---------------------------------------------------------------
# [1단계] Actor와 Critic 네트워크 정의 및 파라미터(theta, w) 초기화
# ---------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_dist(self, obs):
        return Categorical(logits=self.actor(obs))


# ---------------------------------------------------------------
# 데이터 저장소 (SB3의 rollout_buffer를 단순하게 흉내 냄)
# ---------------------------------------------------------------
RolloutData = namedtuple(
    "RolloutData", ["observations", "actions", "old_log_prob", "advantages", "returns"]
)


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim):
        self.observations = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

    def compute_returns_and_advantage(self, last_value):
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            next_value = last_value if t == n_steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get(self, batch_size):
        # [3단계] 배치를 섞은 뒤 미니배치(M개)씩 잘라서 꺼낸다
        indices = np.random.permutation(n_steps)
        for start in range(0, n_steps, batch_size):
            idx = indices[start:start + batch_size]
            yield RolloutData(
                observations=th.as_tensor(self.observations[idx]),
                actions=th.as_tensor(self.actions[idx]),
                old_log_prob=th.as_tensor(self.log_probs[idx]),
                advantages=th.as_tensor(self.advantages[idx]),
                returns=th.as_tensor(self.returns[idx]),
            )


def train():
    # 실험 재현성을 높이기 위한 난수 시드 설정
    np.random.seed(0)
    th.manual_seed(0)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = ActorCritic(obs_dim, n_actions)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    rollout_buffer = RolloutBuffer(n_steps, obs_dim)

    obs, _ = env.reset(seed=0)
    episode_reward, episode_rewards = 0.0, []
    num_updates = total_timesteps // n_steps

    for update in range(1, num_updates + 1):
        # -------------------------------------------------------
        # [2단계] 업데이트 전 정책으로 N timestep 동안 데이터 수집
        # -------------------------------------------------------
        for t in range(n_steps):
            with th.no_grad():
                obs_tensor = th.as_tensor(obs, dtype=th.float32)
                dist = policy.get_dist(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = policy.get_value(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            if truncated and not terminated:
                with th.no_grad():
                    reward += gamma * policy.get_value(th.as_tensor(next_obs, dtype=th.float32)).item()

            rollout_buffer.observations[t] = obs
            rollout_buffer.actions[t] = action.item()
            rollout_buffer.rewards[t] = reward
            rollout_buffer.dones[t] = float(terminated or truncated)
            rollout_buffer.values[t] = value.item()
            rollout_buffer.log_probs[t] = log_prob.item()

            obs = next_obs
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

        with th.no_grad():
            last_value = policy.get_value(th.as_tensor(obs, dtype=th.float32)).item()
        rollout_buffer.compute_returns_and_advantage(last_value)

        # -------------------------------------------------------
        # 같은 배치로 n_epochs번 반복 학습
        # -------------------------------------------------------
        for epoch in range(n_epochs):
            # [3단계] 배치에서 미니배치 크기(M)만큼 데이터 추출
            for batch_idx, rollout_data in enumerate(rollout_buffer.get(batch_size)):
                        # [실습 1] 미니배치 데이터 크기 확인
                if update == 1 and epoch == 0 and batch_idx == 0:
                    print("observations:", rollout_data.observations.shape)
                    print("actions:", rollout_data.actions.shape)
                    print("old_log_prob:", rollout_data.old_log_prob.shape)
                    print("advantages:", rollout_data.advantages.shape)
                    print("returns:", rollout_data.returns.shape)

                actions = rollout_data.actions

                dist = policy.get_dist(rollout_data.observations)
                log_prob = dist.log_prob(actions)
                values_pred = policy.get_value(rollout_data.observations)
                entropy = dist.entropy()

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # [4단계] Critic 손실: L(w) = 1/2M * sum (y_i - V_w(s_i))^2
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # [6단계] 비율 r(theta) = pi_theta / pi_old = exp(log pi_theta - log pi_old)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                
                if update == 1 and epoch == 0 and batch_idx == 0:
                    clip_fraction = (
                            (ratio < 1 - clip_range) |
                            (ratio > 1 + clip_range)
                    ).float().mean()

                    print("ratio min:", ratio.min().item())
                    print("ratio mean:", ratio.mean().item())
                    print("ratio max:", ratio.max().item())
                    print("clip_fraction:", clip_fraction.item())
                
                # [7단계] Clipped Surrogate Objective (앞에 - 를 붙여 최소화 문제로 바꿈)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
                if not th.isfinite(loss).item():
                    raise ValueError(f"비정상적인 Loss 발생: {loss.item()}")
                # [5단계 + 8단계] backward()로 Actor와 Critic을 한 번에 업데이트
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        recent = episode_rewards[-20:]
        print(
            f"update {update:3d}/{num_updates} | timesteps {update * n_steps:7d} | "
            f"ep_rew_mean {np.mean(recent) if recent else 0:6.1f} | "
            f"value_loss {value_loss.item():8.3f} | policy_loss {policy_loss.item():7.4f}"
        )

    th.save(policy.state_dict(), model_dir)
    print(f"saved: {model_dir}")
    env.close()


def watch(episodes):
    env = gym.make(env_name, render_mode="human")
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    policy.load_state_dict(th.load(model_dir))
    for ep in range(episodes):
        obs, _ = env.reset()
        game_score, done = 0.0, False
        while not done:
            with th.no_grad():
                action = policy.actor(th.as_tensor(obs, dtype=th.float32)).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            game_score += reward
            done = terminated or truncated
        print(f"episode {ep + 1}: game score = {game_score:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "watch"])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        watch(args.episodes)
