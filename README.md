# SupplyChainSim-GSCPI v2

基于多智能体博弈与数据同化的全球供应链压力指数（GSCPI）预测系统——**可执行参考实现**。

> **版本说明（2026-11-08）**：本仓库 main 分支已重构为 v2（`scsim` 包）。v1 代码保留在
> [`archive/v1-legacy`](https://github.com/chenweibo045/SupplyChainSim-GSCPI/tree/archive/v1-legacy) 分支。
> 重构原因：v1 的结算口径（把百分比变化当 z 分）与回测口径（drift3 含未来信息、
> 方向准确率错位一期）存在三个会改变结论的错误，详见 `docs/PROJECT_DOC.md` 与 v3 项目文档。

## 核心思想

让 LLM 负责"看懂新闻"和"模拟微观主体的定性决策"，让物理查表与官方计量管道负责
"精准算数"——从机制上规避 LLM 数值幻觉（agent bullwhip）。

**关键概念澄清**：PCA 不是"我们选的预测方法"，而是 GSCPI **官方定义**的一部分
（NY Fed Staff Report 1017：需求剥离 → 第一主成分 → 标准化）。任何预测系统必须复用
同一条聚合管道，预测值才与官方发布值可比。

## 快速开始

```bash
python -m venv .venv && .venv/bin/pip install numpy pandas matplotlib
.venv/bin/python -m scsim.demo --selftest    # 自测（官方管道 corr=0.992 合成验证 + 红海情景）
.venv/bin/python -m scsim.demo               # 全流程 + 出图（results/）
```

只需 `numpy / pandas / matplotlib`，不需要 sklearn、statsmodels、LLM API，Python 3.9+ 离线可复现。

## 目录

| 文件 | 作用 |
|---|---|
| `scsim/profiles.py` | 13 个智能体的 Profile（身份/禀赋暴露/行为参数三层，字段带单位与来源标注） |
| `scsim/actions.py` | 动作空间：12 个动作 × 强度档位 0–3（u(k)=k/3），含前置条件与效应表 |
| `scsim/dynamics.py` | 状态更新算子：perceive → belief → decide → transition 四段式 + 黑板传导矩阵 TRANSMISSION |
| `scsim/settle.py` | 状态 → 27 个 GSCPI 子指标 → σ 归一 → ΔGSCPI（分通道增益） |
| `scsim/aggregate.py` | 官方三步管道复现（单方程 OLS 需求剥离 → SVD PCA/EM 填补 → 标准化）+ 精确线性归因 |
| `scsim/backtest.py` | 诚实回测：修正 look-ahead 与方向指标；事件窗口留一校准；DM 检验 |
| `scsim/events.py` | 12 个历史冲击的结构化编码（与在线 LLM 输出同格式） |
| `scsim/demo.py` | 端到端演示：管道复现 / 红海情景 / 反事实归因 / 留一事件回测 / 增益标定 |
| `scsim/viz.py` | 论文用图（历史序列、冲量检验、状态轨迹、归因、标定） |
| `data/gscpi_official.csv` | 官方 GSCPI 发布值（1998-01 ~ 2026-10，346 期，来源见 data/SOURCES.md） |
| `results/` | 可复现实验输出（csv/json；运行 demo 重新生成，png 图亦同） |

## 主要实证结论（全部可用 `python -m scsim.demo` 复现）

1. **官方管道复现正确**：合成面板上与真实因子 corr = 0.992（仅证明管道写对，不代表接上真实 27 序列后同样准确）。
2. **红海情景（2023-12）**：模型 ΔGSCPI 首月 +0.305，但实际指数连续三个月下跌
   （−0.184 / −0.041 / −0.207）——单一运价通道冲击不足以推动 27 维主成分指数。
3. **GSCPI 近似随机游走**：修正口径后 naive RMSE 0.377 已是强基准；唯一略优的是
   均值回复基准 0.9·G（0.371）。旧版"drift3 最优"是 look-ahead 的产物（0.444 修正后显著更差，DM = −3.81）。
4. **机制冲量暂无增量信息**：12 事件 × 3 期留一检验，模型冲量与实际变动 r = −0.31，
   0/12 跑赢"不变"基准；剔除均值回复后增量解释力 ≈ 0。
5. **增益标定**：按波动率对齐历史的隐含增益为 0.83，当前演示增益 0.30 偏保守约 2.8 倍。
6. **归因可用**：载荷冻结时指标级归因精确可加；智能体级反事实归因（红海 3 个月：
   两家绕航承运人合计 76%，交互残差 −11.6%）。

> 结论 3–4 不否定系统价值，但否定"以点预测精度超越基线"的主张。
> 建议定位：**结构化重构 + 情景归因 + 实时监测**（见 v3 项目文档第 9 章）。

## 与 v1 的关键差异

1. **单位修正**：v1 把"百分比变化 / 100"当作 z 分增量 → 红海情景算出 ΔGSCPI≈+3.7
   （历史单月最大仅 +1.40）。v2 用 `Δz = Δx / σ_x` + 分通道增益标定。
2. **回测修正**：drift3 含未来信息（`g.diff().rolling(3)`）→ 修正后 RMSE 0.314→0.444；
   方向准确率错位一期 → 修正后 AR(1) 方向命中 0.532。
3. **状态更新重构**：运价由"每月乘 1.22"改为"压力池 + 部分调整"，避免指数爆炸；
   指标是有记忆的状态量，不是当期动作的函数。
4. **智能体规范化**：Profile 三层结构（身份/禀赋/行为）、12 动作 × 4 档位、
   TRANSMISSION 黑板传导矩阵，与 v3 项目文档第 5 章一一对应。

## 免责声明

学术原型。合成数据演示仅验证工程链路；一切结论以接入真实 27 个子指标序列后的回测为准。
底层分项多为商业数据（Haver/Bloomberg/S&P Global），使用须遵守各数据商许可。
