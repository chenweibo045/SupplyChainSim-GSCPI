# 27 个子指标获取指引

| 组 | 指标 | 来源 | 可得性 |
|---|---|---|---|
| 海运 | BDI（波罗的海干散货） | Baltic Exchange / Investing.com / 部分数据商 | 公开可查 |
| 海运 | Harpex（集装箱租船） | Harper Petersen（官网发布） | 公开可查 |
| 空运 | 4 个 BLS 进出口空运价格指数 | 美国劳工统计局 BLS | 公开免费 |
| PMI | 7 经济体 × (交货时间/积压订单/采购库存) | S&P Global / ISM 制造业 PMI 分项 | 付费（Haver/Bloomberg） |

付费分项的降级方案（MVP 用，需披露）：
  - 用官方 GSCPI 发布值作为验证锚（download_data.py 可下载）；
  - 分项用 FRED 等免费库中的货运/库存/交付类代理序列替代；
  - 中国分项可用财新/官方制造业 PMI 扩散指数近似。
