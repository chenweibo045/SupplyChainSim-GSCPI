"""历史冲击事件库（用于回测与 Demo）。

每个事件是一条**结构化冲击**（人工编码，可审计），不是新闻原文。
这与系统在线运行时 Step 1（LLM 把新闻解析成 Shock）的输出格式完全一致，
因此离线回测与在线推理走同一条代码路径。

编码纪律：
  * 只使用事件发生时点可获得的信息（不看后验）；
  * severity / capacity_delta 是分析者判断，需在附录披露编码规则并做敏感性分析；
  * 事件月份 = 冲击首次被公开报道的月份（t），预测目标为 t+1..t+H。
"""

from __future__ import annotations

from dataclasses import dataclass

from .dynamics import Shock


@dataclass
class HistoricalEvent:
    key: str
    month: str            # 事件月（月末）
    label: str
    shock: Shock
    note: str = ""


EVENTS: list[HistoricalEvent] = [
    HistoricalEvent("2011-03", "2011-03-31", "东日本大地震",
                    Shock("EastAsia", "input_supply", 0.80, -0.35, 4,
                          ("intra_asia", "transpacific"), "汽车/电子零部件断供")),
    HistoricalEvent("2011-10", "2011-10-31", "泰国洪灾",
                    Shock("EastAsia", "input_supply", 0.75, -0.40, 4,
                          ("intra_asia",), "硬盘/汽车零部件产能受损")),
    HistoricalEvent("2018-07", "2018-07-31", "中美关税第一轮",
                    Shock("Generic", "input_supply", 0.45, -0.15, 8,
                          (), "采购链路重构，慢变量")),
    HistoricalEvent("2020-03", "2020-03-31", "新冠疫情全球爆发",
                    Shock("EastAsia", "input_supply", 1.00, -0.55, 8,
                          ("intra_asia", "transpacific", "asia_europe"), "多国封控")),
    HistoricalEvent("2021-03", "2021-03-31", "长赐号堵塞苏伊士",
                    Shock("Suez/Red Sea", "canal_transit", 0.70, -0.40, 2,
                          ("asia_europe",), "6 天断航 + 排队")),
    HistoricalEvent("2021-10", "2021-10-31", "疫情冬季波 + 能源紧张",
                    Shock("EastAsia", "input_supply", 0.70, -0.30, 3,
                          ("intra_asia", "asia_europe"), "")),
    HistoricalEvent("2022-02", "2022-02-28", "俄乌冲突",
                    Shock("Europe", "input_supply", 0.65, -0.30, 4,
                          ("asia_europe",), "能源/粮食/金属")),
    HistoricalEvent("2022-04", "2022-04-30", "上海封控",
                    Shock("EastAsia", "input_supply", 0.85, -0.45, 2,
                          ("intra_asia", "transpacific", "asia_europe"), "")),
    HistoricalEvent("2023-12", "2023-12-31", "红海袭船 / 绕行好望角",
                    Shock("Suez/Red Sea", "canal_transit", 0.85, -0.40, 3,
                          ("asia_europe",), "本项目原 demo 的主案例")),
    HistoricalEvent("2024-03", "2024-03-31", "巴尔的摩大桥坍塌",
                    Shock("USWC", "port_capacity", 0.50, -0.25, 2,
                          ("transpacific",), "局部港口中断")),
    HistoricalEvent("2025-04", "2025-04-30", "对等关税冲击",
                    Shock("Generic", "input_supply", 0.60, -0.20, 6,
                          (), "贸易政策不确定性")),
    HistoricalEvent("2026-04", "2026-04-30", "关税第二轮升级",
                    Shock("Generic", "input_supply", 0.70, -0.25, 4,
                          (), "样本外最近事件")),
]


def by_key(key: str) -> HistoricalEvent:
    for e in EVENTS:
        if e.key == key:
            return e
    raise KeyError(key)
