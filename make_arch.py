import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.5); ax.axis("off")

layers = [
    ("L1  信号层", "新闻 / 数据  →  结构化冲击\n{区域, 渠道, 严重度, 运力变化, 持续时间}", "#E8EEF7"),
    ("L2  智能体层", "13 个集聚智能体 × N 轮黑板博弈\n7 制造 + 3 承运人 + 3 港口  →  定性意图", "#DCE7F5"),
    ("L3  结算层", "意图 → 驱动量 → 27 个子指标水平量 → Δz\n查表弹性 + 部分调整 + 饱和（LLM 不参与算术）", "#D0E0F2"),
    ("L4  同化层", "真实观测覆写（BDI / Harpex / 空运价）\n来源掩码：real / sim", "#C5D8EE"),
    ("L5  聚合层", "需求剥离（单方程 OLS）→ PCA / EM → 标准化\n= 官方 GSCPI 管道", "#B9D0EA"),
    ("L6  归因层", "指标级精确分解 + 智能体级反事实分解\n+ 不确定性带", "#ADC8E6"),
]
top, h, gap = 9.95, 1.18, 0.28
for i, (title, body, color) in enumerate(layers):
    y = top - i * (h + gap) - h
    box = FancyBboxPatch((1.35, y), 7.3, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                         linewidth=1.0, edgecolor="#1F3B63", facecolor=color)
    ax.add_patch(box)
    ax.text(1.55, y + h/2 + 0.20, title, fontsize=10.5, fontweight="bold", color="#1F3B63", va="center")
    ax.text(1.55, y + h/2 - 0.22, body, fontsize=8.2, color="#222", va="center", linespacing=1.5)
    if i < len(layers) - 1:
        ax.add_patch(FancyArrowPatch((5.0, y - 0.03), (5.0, y - gap + 0.03),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.1, color="#1F3B63"))

# 左侧输入
ax.add_patch(FancyBboxPatch((0.05, 7.55), 0.95, 1.6, boxstyle="round,pad=0.03,rounding_size=0.08",
                            linewidth=1.0, edgecolor="#7F7F7F", facecolor="#F4F4F4"))
ax.text(0.52, 8.35, "新闻\n数据", fontsize=8.5, ha="center", va="center", color="#333")
ax.add_patch(FancyArrowPatch((1.02, 8.35), (1.33, 8.35), arrowstyle="-|>", mutation_scale=11,
                             linewidth=1.0, color="#7F7F7F"))

# 右侧输出
ax.add_patch(FancyBboxPatch((9.0, 2.05), 0.95, 2.6, boxstyle="round,pad=0.03,rounding_size=0.08",
                            linewidth=1.0, edgecolor="#B02A2A", facecolor="#FBEDEC"))
ax.text(9.47, 3.35, "GSCPI\n重构值\n\n归因\n报告", fontsize=8.3, ha="center", va="center", color="#B02A2A")
ax.add_patch(FancyArrowPatch((8.67, 3.40), (8.98, 3.40), arrowstyle="-|>", mutation_scale=11,
                             linewidth=1.0, color="#B02A2A"))

# 横切
ax.add_patch(FancyBboxPatch((1.35, 0.30), 7.3, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=0.9, edgecolor="#7F7F7F", facecolor="#F7F7F7", linestyle="--"))
ax.text(5.0, 0.575, "横切能力：校准（增益 / 弹性 / 阈值） · 版本（载荷与参数快照） · 评估", fontsize=8.0,
        ha="center", va="center", color="#444")

plt.tight_layout()
plt.savefig("results/fig0_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
