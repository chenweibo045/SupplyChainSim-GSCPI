# 数据来源说明

## data/gscpi_official.csv

- **来源**：纽约联邦储备银行 Global Supply Chain Pressure Index (GSCPI) 官方发布页
  <https://www.newyorkfed.org/research/policy/gscpi>
- **范围**：1998-01 至 2026-10，共 346 期月度值（当前 vintage，官方会对历史值回溯修订）
- **格式**：`Date,GSCPI` 两列 CSV，日期为月末
- **注意**：GSCPI 是公开数据，但官方每月修订（空运价格指数回溯修订最多 12 个月，
  缺失填补值会被真实值替换）。做严肃回测应使用 vintage 快照；本仓库用最新版数据，
  结论以"当前可得口径"为准。

## 27 个子指标序列（尚未接入）

- 运输成本 6 项：BDI、Harpex、4 个 BLS 空运价格指数（部分公开可得）
- 制造业 PMI 三分项 × 7 经济体（中国、美国、欧元区、日本、韩国、中国台湾、英国）：
  供应商交货时间、积压订单、采购库存 —— 需 Haver / Bloomberg / S&P Global 订阅，
  或用代理序列并披露口径差异
