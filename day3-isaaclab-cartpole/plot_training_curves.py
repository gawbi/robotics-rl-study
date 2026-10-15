"""Isaac Lab Cartpole TensorBoard 로그 -> 학습 곡선 PNG (small multiples)."""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir, out_path = sys.argv[1], sys.argv[2]

# reference palette (light)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange

font_path = subprocess.run(
    ["fc-match", "-f", "%{file}", "Noto Sans CJK KR:style=Regular"], capture_output=True, text=True
).stdout
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=font_path).get_name(),
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": TEXT2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": TEXT2,
    "ytick.labelcolor": TEXT2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
ea.Reload()


def series(tag):
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


LW = 1.4
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), dpi=150)
fig.suptitle("Isaac Lab Cartpole · RSL-RL PPO 학습 곡선 (150 iteration, 4096 env)",
             x=0.01, ha="left", color=TEXT, fontsize=13, fontweight="bold")


def style(ax, title, sub=None):
    ax.set_title(title, loc="left", color=TEXT, fontsize=10.5, fontweight="bold", pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=TEXT2, fontsize=8.5, va="bottom")
    ax.set_xlim(0, 149)
    ax.set_xlabel("iteration")


# 1. 평균 에피소드 길이
ax = axes[0, 0]
x, y = series("Train/mean_episode_length")
ax.plot(x, y, color=S1, lw=LW)
ax.axhline(300, color=MUTED, lw=0.9, ls="--")
ax.text(148, 300, "최대 300 step (5초)", color=TEXT2, fontsize=8, ha="right", va="bottom")
first = next(s for s, v in zip(x, y) if v >= 299.9)
ax.plot([first], [300], "o", ms=5, color=S1, mec=SURFACE, mew=1.5, zorder=5)
ax.annotate(f"{first}번째에 처음 도달", (first, 300), xytext=(first + 8, 215),
            color=TEXT, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(0, 330)
style(ax, "평균 에피소드 길이", "끝까지 버티면 300")

# 2. 평균 보상
ax = axes[0, 1]
x, y = series("Train/mean_reward")
ax.plot(x, y, color=S1, lw=LW)
ax.axhline(0, color=BASE, lw=0.9)
ax.axhline(5, color=MUTED, lw=0.9, ls="--")
ax.text(148, 5, "alive만 받으면 약 5", color=TEXT2, fontsize=8, ha="right", va="bottom")
i_min = min(range(len(y)), key=lambda i: y[i])
ax.plot([x[i_min]], [y[i_min]], "o", ms=5, color=S1, mec=SURFACE, mew=1.5, zorder=5)
ax.annotate(f"최저 {y[i_min]:.2f} ({x[i_min]}번째)", (x[i_min], y[i_min]), xytext=(x[i_min] + 12, y[i_min] - 0.3),
            color=TEXT, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(-4.5, 6)
style(ax, "평균 보상 (에피소드 합계)", "초반에 음수로 떨어졌다가 회복")

# 3. 종료 원인 비율
ax = axes[0, 2]
x1, y1 = series("Episode_Termination/time_out")
x2, y2 = series("Episode_Termination/cart_out_of_bounds")
ax.plot(x1, y1, color=S1, lw=LW, label="time_out (5초 버팀)")
ax.plot(x2, y2, color=S2, lw=LW, label="cart_out_of_bounds (범위 이탈)")
ax.text(148, y1[-1] - 0.04, "time_out", color=TEXT, fontsize=8.5, ha="right", va="top")
peak = max(range(len(y2)), key=lambda i: y2[i])
ax.text(14, 0.62, "범위 이탈", color=TEXT, fontsize=8.5, va="center")
ax.set_ylim(0, 1.08)
ax.legend(loc="center right", frameon=False, fontsize=8, labelcolor=TEXT2)
style(ax, "에피소드 종료 원인 비율", "끝날 때 무엇 때문에 끝났나")

# 4. 행동 노이즈 표준편차
ax = axes[1, 0]
x, y = series("Policy/mean_noise_std")
ax.plot(x, y, color=S1, lw=LW)
ax.text(x[0] + 2, y[0], f"{y[0]:.2f}", color=TEXT, fontsize=8.5, va="bottom")
ax.text(x[-1], y[-1] + 0.03, f"{y[-1]:.3f}", color=TEXT, fontsize=8.5, ha="right", va="bottom")
ax.set_ylim(0, 1.1)
style(ax, "행동 노이즈 표준편차", "탐색이 줄고 정책이 확신을 가짐")

# 5. 학습률 (adaptive)
ax = axes[1, 1]
x, y = series("Loss/learning_rate")
ax.plot(x, y, color=S1, lw=LW, drawstyle="steps-post")
ax.set_yscale("log")
ax.axhline(1e-3, color=MUTED, lw=0.9, ls="--")
ax.text(3, 1.15e-3, "설정값 0.001", color=TEXT2, fontsize=8, ha="left", va="bottom")
ax.set_ylim(5e-6, 2e-2)
style(ax, "학습률 (log)", "schedule: adaptive — KL 보고 자동 조절")

# 6. Value loss
ax = axes[1, 2]
x, y = series("Loss/value_function")
ax.plot(x, y, color=S1, lw=LW)
ax.set_yscale("log")
style(ax, "Value loss (log)", "Critic 예측 오차")

fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=2.0)
fig.savefig(out_path, facecolor=SURFACE)
print("saved", out_path)
