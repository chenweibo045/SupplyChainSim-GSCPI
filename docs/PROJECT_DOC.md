# SupplyChainSim-GSCPI 项目文档（v2 · 2026-11-03）

预测供应链指标 GSCPI：基于多智能体博弈与数据同化的全球供应链压力预测系统

> 本文档替代 2026-10 版《SupplyChainSim-GSCPI.docx》。相对旧版的主要更新：
> ① 官方聚合方法已调研闭环（NY Fed Staff Report 1017），"27 个指标如何聚合"的
> 卡点已解决；② 修正"VAR 剥离"表述——官方为带滞后单方程 OLS 回归；
> ③ 补充文献综述与创新性论证；④ 记录工程实现状态（7/7 模块自测通过）。

## 一、GSCPI 的含义与官方赋值流程

**概念**：全球供应链压力指数（GSCPI）由纽约联邦储备银行于 2022 年推出
（Benigno, di Giovanni, Groen & Noble, Staff Report No. 1017），将复杂的全球
供应链瓶颈提炼为单一综合指标。指数以历史均值为 0、历史标准差为单位归一化：
> 0 表示压力高于正常水平（疫情期间峰值曾超 +4.0），< 0 表示顺畅/过剩。

**数据构成（27 = 21 + 6）**：
- 运输成本 6 项：BDI（干散货）、Harpex（集装箱租船）、4 个 BLS 进/出口空运价格指数；
- 7 大经济体（中、美、欧元区、日、韩、台、英）× 3 个制造业 PMI 核心分项：
  供应商交货时间、积压订单、采购库存。

**官方聚合管道（三步，已调研确认）**：

```
27 个月度序列
  → ① 需求剥离：每个 PMI 分项对本经济体「新订单」做 OLS 回归（当期+2期滞后）取残差；
       运输成本对「GDP 加权新订单 + GDP 加权投入品采购量」回归取残差
       ⚠ 是单方程回归，不是 VAR 系统（旧版文档表述已修正）
  → ② PCA：27 个残差序列标准化后提取第一主成分（Stock & Watson 2002 扩散指数法，
       同时用公共成分 EM 式填补历史缺失值；载荷全样本一次估计）
  → ③ 标准化：符号归一（压力上升 = 指数上升），除以历史标准差，按月发布
```

与 CPI 聚合的关系：CPI 是人工权重加权平均（Laspeyres 支出份额；截尾均值/中位数
先去噪），GSCPI 是数据驱动因子加权（PCA 载荷，无人工权重）。两者共享"先净化、
后聚合"的思想骨架，理论根基同为动态因子模型（Stock & Watson 2002；
Bernanke-Boivin-Eliasz 2005 FAVAR）。

## 二、现有预测工作调研

| 路线 | 代表 | 局限 |
|---|---|---|
| 官方 Nowcasting | 高频运价 + 早期 PMI 线性外推 | 无法感知文本/地缘突变 |
| 传统计量 | VAR/VECM/MIDAS | 黑盒拟合，无行为博弈 |
| 机器学习 | LSTM/TFT/XGBoost（Ishrak et al. 2025 用中国行业股指预测 GSCPI，OOS R²≈0.85） | 需大量历史样本，外推弱 |
| LLM 多智能体 | OptiGuide、EconAgent；Long et al.(2026) 发现 "agent bullwhip" | **尚无针对 GSCPI 的 MAS 预测** |

**创新性**：本项目首次将 ConsumerSim 式 MAS 范式迁移至 GSCPI 预测，填补空白。

## 三、系统方案（5 步流水线）

```
Step1 信号解析：新闻 → LLM 结构化 Shock 四元组（位置/事件/渠道/严重度）
Step2 多智能体博弈：7 制造 + 3 航运 + 3 港口，消息黑板 N 轮循环，只输出定性意图
Step3 物理结算：查表（绕航 3500nm → +11.8 天；GRI 弹性 → Harpex +25%），LLM 不出数字
Step4 数据同化 + 矩阵合成：运价类真实覆写 / PMI 类模拟，产出 27 维矩阵
Step5 官方管道聚合 + 归因：复现 NY Fed 三步管道；贡献分解口径同官方"中国因素"博文
```

**设计纪律**：LLM 只做定性决策（防算术幻觉 + 防 agent bullwhip）；定量全走物理查表；
聚合走官方口径（预测值与发布值可比）；载荷固定保证回测口径稳定。

## 四、工程实现状态

仓库：`chenweibo045/SupplyChainSim-GSCPI`。全部模块 `--selftest` 离线通过（7/7）：

| 模块 | 内容 | 验收 |
|---|---|---|
| `pipeline/official_gscpi.py` | 官方三步管道复现（numpy OLS + sklearn PCA + EM 填补） | 合成数据上与真实因子 corr = 0.994 |
| `pipeline/validate.py` | 与官方发布值对标 | 目标 corr ≥ 0.95（待真实数据） |
| `agents/parser.py` | 新闻 → Shock（LLM JSON Schema + 规则回退） | 红海案例解析正确 |
| `agents/{manufacturers,shippers,ports}.py` | 13 个集聚智能体，消息黑板博弈 | 2 轮循环传导：绕航→恐慌采购 |
| `physics/{routing,pmi_mapping}.py` | 查表结算 + 边际递减叠加（封顶 ±60%） | 绕航 +11.8 天 / Harpex 弹性 |
| `sim/{orchestrator,assimilation}.py` | 编排 + 同化；`--demo red_sea` 端到端 | ΔGSCPI +3.7，归因可解释 |
| `backtest/evaluate.py` | naive/drift/ARIMA 基准 + RMSE/MAE/方向准确率 | 排行榜输出正常 |

运行：`pip install -r requirements.txt && python selftest.py`；
演示：`python -m sim.orchestrator --demo red_sea`；沙盘：`streamlit run app/streamlit_app.py`。

## 五、已知局限与下一步

1. **真实数据接入**：PMI 分项为付费数据（Haver/Bloomberg/S&P Global），按
   `data/raw/SOURCES.md` 指引获取或采用代理序列并披露口径差异；官方 GSCPI 发布值
   用 `python data/download_data.py` 下载，作为对标锚（M4）。
2. **事件回测**：2021 疫情峰值、2023Q4 红海危机窗口，与三个 baseline 对比（M4）。
3. **LLM 随机性**：正式实验 temperature=0、固定 prompt、N 次运行取中位数并报离散度。
4. **方向统一**：真实 PMI 分项进入管道前需统一方向（交货时间数值越低越顺畅）。

## 六、参考文献

1. Benigno, di Giovanni, Groen & Noble (2022). *The GSCPI: A New Barometer of Global Supply Chain Pressures.* NY Fed Staff Report 1017.
2. Benigno, di Giovanni, Groen & Noble (2022). *A New Barometer of Global Supply Chain Pressures.* Liberty Street Economics.
3. Benigno, Clark, Cross-Bermingham & Nourbash (2023). *GSCPI: The China Factor.* Liberty Street Economics.（贡献分解范式）
4. Stock & Watson (2002). *Forecasting Using Principal Components from a Large Number of Predictors.* JASA 97(460). / *Macroeconomic Forecasting Using Diffusion Indexes.* JBES 20(2).
5. Bernanke, Boivin & Eliasz (2005). *Measuring the Effects of Monetary Policy: A FAVAR Approach.* QJE 120(1).
6. Bryan & Cecchetti (1994). *Measuring Core Inflation.* NBER Macro Annual.
7. Ishrak et al. (2025). *Predicting the GSCPI from China's Sectoral Stock Returns.* IEOM Bangladesh Proceedings.
8. Long et al. (2026). 串行供应链生成式智能体 "agent bullwhip" 研究（arXiv:2607.19967 引述）。

## 七、免责声明

学术原型。合成数据演示仅验证工程链路；一切结论以接入真实数据后的回测为准。
底层分项多为商业数据，使用须遵守各数据商许可。
