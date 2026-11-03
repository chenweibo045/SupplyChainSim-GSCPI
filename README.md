# SupplyChainSim-GSCPI

基于多智能体博弈与数据同化的全球供应链压力指数（GSCPI）预测系统。

**核心思想**：让 LLM 负责"看懂新闻"和"模拟微观主体的定性决策"，让物理方程与
官方计量管道负责"精准算数"——从机制上规避 LLM 数值幻觉（agent bullwhip），
同时保留对红海危机等外生结构突变的行为模拟能力与因果归因能力。

## 系统流程（5 步）

```
[新闻/数据] → Step1 信号解析(LLM→Shock四元组)
           → Step2 13个智能体多轮消息循环博弈(只输出定性意图)
           → Step3 物理/运筹结算(查表: 绕航里程→延迟天数→运价弹性)
           → Step4 高频数据同化 + 27维指标矩阵合成
           → Step5 复现纽约联储官方管道(需求剥离→PCA→标准化) + 归因报告
```

## 快速开始

```bash
pip install -r requirements.txt
python selftest.py                    # 全部模块离线自测（7/7 应通过）
python -m sim.orchestrator --demo red_sea   # 红海案例端到端演示
python data/download_data.py          # 下载官方GSCPI + 27序列获取指引
streamlit run app/streamlit_app.py    # 情景沙盘（可选）
```

## 目录结构

| 路径 | 内容 |
|---|---|
| `pipeline/` | 官方 GSCPI 管道复现：需求剥离（单方程 OLS + 2 期滞后，注意**不是 VAR**）→ PCA 第一主成分（Stock-Watson 式 EM 填补缺失）→ 标准化；`validate.py` 对标官方发布值（目标 corr ≥ 0.95） |
| `agents/` | `parser.py` 新闻→Shock（LLM JSON Schema + 规则回退）；`base.py` 基类与 Intent 定性约定；`manufacturers/shippers/ports.py` 三类共 13 个集聚智能体 |
| `physics/` | 纯查表结算：`routing.py` 绕航（3500nm→~11.8 天）与运价弹性；`pmi_mapping.py` 意图→PMI 三分项增量（边际递减叠加） |
| `sim/` | `orchestrator.py` 消息黑板多轮博弈 + 结算 + 27 维合成；`assimilation.py` 按频率分层覆写（运价类同化/PMI 类模拟） |
| `backtest/` | baseline（naive/drift/ARIMA）+ RMSE/MAE/方向准确率排行榜 |
| `app/` | Streamlit 情景沙盘：新闻输入→预测值+归因+决策轨迹 |
| `data/` | `download_data.py` 官方数据下载 + `raw/SOURCES.md` 27 子指标获取指引 |
| `docs/` | 项目文档 v2（含方法修正与文献综述） |

## 设计纪律（为什么这样拆）

1. **LLM 不出数字**：Agent 只输出 `Intent(action, strength)` 枚举，所有定量
   变化在 `physics/` 查表完成 → 杜绝算术幻觉。
2. **真博弈而非并行独白**：意图经黑板广播，下游 Agent 下一轮感知
   （航运绕航 → 制造商恐慌 → 港口排队），支持 N 轮迭代。
3. **官方口径聚合**：预测值与纽约联储发布值走同一条数学管道，回测可比；
   载荷全样本一次估计并固定。
4. **离线可复现**：所有模块 `--selftest` 用固定随机种子合成数据自验，
   不依赖付费数据源即可演示全链路。

## 里程碑

- [x] M0 官方管道复现（含离线自测；真实数据对标见 `pipeline/validate.py`）
- [x] M1 信号解析（LLM 通道 + 规则回退）
- [x] M2 智能体集群 + 物理结算（红海案例验收：绕航 +11.8 天、Harpex 封顶 +60%）
- [x] M3 消息循环博弈 + 27 维矩阵对接官方管道
- [ ] M4 真实数据接入 + 事件窗口回测（2021 疫情峰值 / 2023Q4 红海危机）+ 论文

## 参考文献

1. Benigno, G., di Giovanni, J., Groen, J. J. J., & Noble, A. I. (2022).
   *The GSCPI: A New Barometer of Global Supply Chain Pressures.*
   NY Fed Staff Report No. 1017. —— 官方构建方法（需求剥离→PCA→标准化）
2. Benigno, G., di Giovanni, J., Groen, J. J. J., & Noble, A. I. (2022-01-04).
   *A New Barometer of Global Supply Chain Pressures.* Liberty Street Economics.
3. Benigno, G., Clark, H. L., Cross-Bermingham, W., & Nourbash, E. (2023-01-06).
   *Global Supply Chain Pressure Index: The China Factor.* Liberty Street Economics.
   —— 分项贡献分解范式（本项目归因报告口径）
4. Stock, J. H., & Watson, M. W. (2002). *Forecasting Using Principal Components
   from a Large Number of Predictors.* JASA 97(460), 1167–1179. —— PCA 聚合理论出处
5. Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). *Measuring the Effects of
   Monetary Policy: A FAVAR Approach.* QJE 120(1), 387–422.
6. Bryan, M. F., & Cecchetti, S. G. (1994). *Measuring Core Inflation.* NBER Macro Annual.
   —— "先净化、后聚合"的 CPI 版思想
7. Ishrak, A. M., et al. (2025). *Predicting the GSCPI from China's Sectoral Stock
   Returns.* IEOM Bangladesh Proceedings. —— 最接近的 GSCPI 预测文献

## 免责声明

本项目为学术研究原型。合成数据演示仅用于验证工程链路；任何结论以接入真实
数据后的回测为准。27 个底层分项多为商业数据，使用时请遵守各数据商许可。
