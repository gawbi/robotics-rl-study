"""G1 실험 0(기본)과 실험 1(feet_air_time=0)의 TensorBoard 로그 비교.

- 각 스칼라의 마지막 값과 마지막 100 iteration 평균을 비교표(metrics_table.md)로 저장
- mean reward / episode length / feet_slide 학습 곡선을 겹쳐 그린 PNG 저장

실행 (isaaclab conda 환경의 python):
    ~/miniforge3/envs/isaaclab/bin/python day5-g1-reward-experiment/extract_metrics.py
"""
import argparse
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = os.path.expanduser("~/IsaacLab/logs/rsl_rl/g1_flat")
HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--exp0", default=os.path.join(LOG_ROOT, "2026-10-11_18-36-30"))
parser.add_argument("--exp1", default=os.path.join(LOG_ROOT, "2026-10-11_20-25-28_no_feet_air_time"))
parser.add_argument("--out_dir", default=HERE)
parser.add_argument("--window", type=int, default=100, help="마지막 N iteration 평균")
args = parser.parse_args()

# 비교표에 넣을 지표 (순서 = 표 순서)
TAGS = [
    ("Train/mean_reward", "mean reward"),
    ("Train/mean_episode_length", "mean episode length (step)"),
    ("Episode_Termination/base_contact", "종료: base_contact (넘어짐)"),
    ("Episode_Termination/time_out", "종료: time_out (20초 버팀)"),
    ("Metrics/base_velocity/error_vel_xy", "속도 명령 오차 xy"),
    ("Metrics/base_velocity/error_vel_yaw", "속도 명령 오차 yaw"),
    ("Episode_Reward/track_lin_vel_xy_exp", "track_lin_vel_xy_exp"),
    ("Episode_Reward/track_ang_vel_z_exp", "track_ang_vel_z_exp"),
    ("Episode_Reward/feet_air_time", "feet_air_time"),
    ("Episode_Reward/feet_slide", "feet_slide"),
    ("Episode_Reward/action_rate_l2", "action_rate_l2"),
    ("Episode_Reward/dof_torques_l2", "dof_torques_l2"),
    ("Episode_Reward/dof_acc_l2", "dof_acc_l2"),
    ("Episode_Reward/lin_vel_z_l2", "lin_vel_z_l2"),
    ("Episode_Reward/ang_vel_xy_l2", "ang_vel_xy_l2"),
    ("Episode_Reward/flat_orientation_l2", "flat_orientation_l2"),
    ("Episode_Reward/dof_pos_limits", "dof_pos_limits"),
    ("Episode_Reward/termination_penalty", "termination_penalty"),
    ("Episode_Reward/joint_deviation_hip", "joint_deviation_hip"),
    ("Episode_Reward/joint_deviation_arms", "joint_deviation_arms"),
    ("Episode_Reward/joint_deviation_fingers", "joint_deviation_fingers"),
    ("Episode_Reward/joint_deviation_torso", "joint_deviation_torso"),
    ("Policy/mean_noise_std", "행동 노이즈 std"),
]


def load(log_dir):
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea, tag):
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


def fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    return f"{v:.2e}"


ea0, ea1 = load(args.exp0), load(args.exp1)
N = args.window

rows = []
for tag, name in TAGS:
    _, y0 = series(ea0, tag)
    _, y1 = series(ea1, tag)
    last0, last1 = y0[-1], y1[-1]
    avg0, avg1 = sum(y0[-N:]) / N, sum(y1[-N:]) / N
    # 크기(절댓값) 기준 변화: 벌점은 음수라서 부호 기준이면 헷갈림. +면 실험 1에서 값/벌점이 커짐
    change = f"{(abs(last1) - abs(last0)) / abs(last0) * 100:+.0f}%" if last0 != 0 else "—"
    sd0 = (sum((v - avg0) ** 2 for v in y0[-N:]) / N) ** 0.5
    sd1 = (sum((v - avg1) ** 2 for v in y1[-N:]) / N) ** 0.5
    change_avg = f"{(abs(avg1) - abs(avg0)) / abs(avg0) * 100:+.0f}%" if avg0 != 0 else "—"
    rows.append((name, last0, last1, change, avg0, avg1, sd0, sd1, change_avg))

steps0, _ = series(ea0, "Train/mean_reward")
steps1, _ = series(ea1, "Train/mean_reward")


def minutes(ea):
    ev = ea.Scalars("Train/mean_reward")
    return (ev[-1].wall_time - ev[0].wall_time) / 60


lines = [
    f"| 지표 | 실험 0 (기본) | 실험 1 (feet_air_time=0) | 크기 변화 | 실험 0 마지막 {N}회 평균 ± 표준편차 | 실험 1 마지막 {N}회 평균 ± 표준편차 | 크기 변화 ({N}회 평균) |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, l0, l1, ch, a0, a1, d0, d1, cha in rows:
    lines.append(f"| {name} | {fmt(l0)} | {fmt(l1)} | {ch} | {fmt(a0)} ± {fmt(d0)} | {fmt(a1)} ± {fmt(d1)} | {cha} |")
lines += [
    "",
    f"- 마지막 iteration: 실험 0 = {steps0[-1]}, 실험 1 = {steps1[-1]}",
    f"- 학습 시간 (첫 로그 ~ 마지막 로그): 실험 0 = {minutes(ea0):.1f}분, 실험 1 = {minutes(ea1):.1f}분",
    "- 크기 변화 = (|실험 1| − |실험 0|) / |실험 0|. 벌점 항목에서 +는 벌점이 커졌다는 뜻",
    "- `Episode_Reward/*`: 가중치를 곱한 보상의 에피소드 합 평균 ÷ 최대 에피소드 길이(20초)",
    "- `종료: *`: 직전 에피소드가 그 이유로 끝난 환경의 비율",
]
table = "\n".join(lines)
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "metrics_table.md"), "w") as f:
    f.write(table + "\n")
print(table)

# ---------------- 학습 곡선 ----------------
SURFACE, TEXT, TEXT2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"  # 실험 0 = blue, 실험 1 = orange

font_path = subprocess.run(
    ["fc-match", "-f", "%{file}", "Noto Sans CJK KR:style=Regular"], capture_output=True, text=True
).stdout
if font_path:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASE,
    "axes.labelcolor": TEXT2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": TEXT2, "ytick.labelcolor": TEXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})

PANELS = [
    ("Train/mean_reward", "평균 보상", "실험 1은 상점 하나(feet_air_time)가 빠져 있어 직접 비교 주의"),
    ("Train/mean_episode_length", "평균 에피소드 길이 (step)", "최대 1000 step = 20초"),
    ("Episode_Reward/feet_slide", "feet_slide 벌점", "0에 가까울수록 발이 덜 미끄러짐"),
]
LABEL0, LABEL1 = "실험 0: 기본", "실험 1: feet_air_time=0"

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=150)
fig.suptitle("G1 평지 보행 · 실험 0 vs 실험 1 학습 곡선 (각 1500 iteration, 4096 env, seed 42)",
             x=0.01, ha="left", color=TEXT, fontsize=12.5, fontweight="bold")
for ax, (tag, title, sub) in zip(axes, PANELS):
    x0, y0 = series(ea0, tag)
    x1, y1 = series(ea1, tag)
    ax.plot(x0, y0, color=S1, lw=1.2, label=LABEL0)
    ax.plot(x1, y1, color=S2, lw=1.2, label=LABEL1)
    ax.set_title(title, loc="left", color=TEXT, fontsize=10.5, fontweight="bold", pad=16)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=TEXT2, fontsize=8.3, va="bottom")
    ax.set_xlim(0, max(x0[-1], x1[-1]))
    ax.set_xlabel("iteration")
    # 패널 오른쪽 아래 빈 곳에 마지막 값 (색 선분 = 실험 구분, 글자는 텍스트 색)
    ax.text(0.97, 0.25, "마지막 값", transform=ax.transAxes, color=TEXT2, fontsize=8, ha="right")
    for i, (y, c, lab) in enumerate(((y0, S1, "실험 0"), (y1, S2, "실험 1"))):
        yy = 0.17 - i * 0.08
        ax.plot([0.55, 0.61], [yy + 0.012] * 2, color=c, lw=2, transform=ax.transAxes, clip_on=False)
        ax.text(0.97, yy, f"{lab}  {fmt(y[-1])}", transform=ax.transAxes, color=TEXT, fontsize=8.5, ha="right")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT2,
           bbox_to_anchor=(0.995, 0.995))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
png = os.path.join(args.out_dir, "images", "learning_curves_exp0_vs_exp1.png")
os.makedirs(os.path.dirname(png), exist_ok=True)
fig.savefig(png, facecolor=SURFACE)
print("saved", png)
