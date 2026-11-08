"""论文用图（matplotlib，中文字体 PingFang SC）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C_MAIN = "#1f4e79"
C_ACC = "#c00000"
C_GREY = "#7f7f7f"


def fig_history(g: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.plot(g.index, g.values, color=C_MAIN, lw=1.2)
    ax.axhline(0, color=C_GREY, lw=0.7, ls="--")
    marks = {"2020-03": "COVID", "2021-03": "Suez", "2022-02": "俄乌", "2023-12": "红海"}
    for m, lab in marks.items():
        t = pd.Timestamp(m + "-01") + pd.offsets.MonthEnd(1)
        if t in g.index:
            ax.axvline(t, color=C_ACC, lw=0.8, ls=":", alpha=0.8)
            ax.annotate(lab, (t, g.max() * 0.92), fontsize=8, color=C_ACC,
                        ha="center", rotation=90, va="top")
    ax.set_title("官方 GSCPI（1998-01 ~ 2026-10）与关键冲击时点", fontsize=11)
    ax.set_ylabel("标准差单位")
    ax.grid(alpha=0.2)
    fig.savefig(out / "fig1_gscpi_history.png")
    plt.close(fig)


def fig_impulse(parts: dict, actual: dict, g: pd.Series, out: Path):
    """机制冲量 vs 实际累计变动：检验"冲量方向是否有信息"。"""
    xs, ys, labs = [], [], []
    for k, path in actual.items():
        if k not in parts:
            continue
        for h in range(len(path)):
            if np.isnan(path[h]):
                continue
            xs.append(parts[k]["cum_F"][h] + parts[k]["cum_P"][h])
            ys.append(path[h])
            labs.append(k)
    xs, ys = np.array(xs), np.array(ys)
    corr = float(np.corrcoef(xs, ys)[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(xs, ys, s=26, color=C_MAIN, alpha=0.8, edgecolor="white", linewidth=0.4)
    b = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 20)
    ax.plot(xx, np.polyval(b, xx), color=C_ACC, lw=1.3, ls="--",
            label=f"线性拟合（r = {corr:+.2f}）")
    ax.axhline(0, color=C_GREY, lw=0.7)
    ax.set_xlabel("机制模型预测的 3 个月累计冲量（单位增益）")
    ax.set_ylabel("实际累计 ΔGSCPI")
    ax.set_title("事件窗口：预测冲量与实际变动（12 事件 × 3 期）", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(out / "fig2_impulse_vs_actual.png")
    plt.close(fig)


def fig_trajectory(snaps: list[dict], out: Path):
    from .profiles import ECONOMY_LABEL
    econ = ["ea", "uk", "china", "us"]
    months = np.arange(1, len(snaps) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for e, c in zip(econ, [C_MAIN, C_ACC, "#2e75b6", "#548235"]):
        axes[0].plot(months, [s["mfr"][e]["panic"] for s in snaps], marker="o",
                     color=c, label=ECONOMY_LABEL[e])
        axes[1].plot(months, [s["mfr"][e]["delay"] for s in snaps], marker="o",
                     color=c, label=ECONOMY_LABEL[e])
    axes[0].set_title("制造企业紧迫度（信念量）", fontsize=10)
    axes[1].set_title("感知到货延迟（天）", fontsize=10)
    for a in axes:
        a.set_xlabel("冲击后月数")
        a.set_xticks(list(months))
        a.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("红海情景下制造企业的状态轨迹（2 轮黑板博弈/月）", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig3_agent_trajectory.png")
    plt.close(fig)


def fig_contrib(c: pd.Series, out: Path, top: int = 12):
    s = c.reindex(c.abs().sort_values(ascending=False).index).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = [C_ACC if v > 0 else C_MAIN for v in s.values]
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("对 ΔGSCPI 的贡献（标准差单位）")
    ax.set_title("首月 ΔGSCPI 的指标级归因（精确可加）", fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(out / "fig4_contribution.png")
    plt.close(fig)


def fig_calibration(cal: dict, g: pd.Series, out: Path):
    d = g.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    axes[0].hist(d.values, bins=40, color=C_MAIN, alpha=0.85)
    axes[0].axvline(0, color=C_GREY, lw=0.8, ls="--")
    axes[0].set_title(f"官方月度变动分布（σ = {d.std():.3f}）", fontsize=10)
    axes[0].set_xlabel("ΔGSCPI（标准差单位）")

    gains = np.linspace(0, 1.2, 50)
    axes[1].plot(gains, cal["sim_sd_unit_gain"] * gains, color=C_MAIN, lw=1.4,
                 label="模拟 σ = 单位增益σ × gain")
    axes[1].axhline(cal["hist_sd_delta"], color=C_GREY, ls="--", lw=1.0,
                    label=f"历史 σ = {cal['hist_sd_delta']:.3f}")
    gi = cal["implied_common_gain"]
    axes[1].axvline(gi, color=C_ACC, lw=1.0, ls=":")
    axes[1].plot([gi], [cal["sim_sd_unit_gain"] * gi], "o", color=C_ACC,
                 label=f"隐含增益 = {gi:.3f}")
    axes[1].set_xlabel("分通道共同增益")
    axes[1].set_ylabel("ΔGSCPI 标准差")
    axes[1].set_title("增益标定：模拟波动率对齐历史", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig5_calibration.png")
    plt.close(fig)


def make_all(g, out_obj, contrib, loo, cal, out: Path, parts=None, actual=None):
    fig_history(g, out)
    if parts is not None and actual is not None:
        fig_impulse(parts, actual, g, out)
    fig_trajectory(out_obj["snapshots"], out)
    fig_contrib(contrib, out)
    fig_calibration(cal, g, out)
