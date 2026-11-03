"""Step 1 — 环境信号感知：非结构化新闻 → 结构化冲击向量 Shock。

设计要点：
  - LLM 可用时走 JSON Schema 约束输出（杜绝自由发挥）；
  - LLM 不可用/解析失败时回退到规则引擎，保证系统离线可运行、可自测；
  - 输出是「约束四元组」：位置 / 事件 / 传导渠道 / 严重度（+ 量化参数），
    后续物理引擎只消费结构化字段，不让 LLM 直接报数字。

自测：python -m agents.parser --selftest
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass
class Shock:
    """结构化冲击向量（Step 1 输出，全系统统一语言）。"""
    target_region: str        # "Suez/Red Sea" / "USWC" / "EastAsia" ...
    event: str                # "naval_attack" / "labor_strike" / "tariff" ...
    channel: str              # 传导渠道: "canal_transit" / "port_capacity" / "input_supply"
    severity: float           # 0~1
    capacity_delta_pct: float # 受影响环节运力/供给变化，如 -40 = 运力收缩 40%
    horizon_days: int         # 预计持续天数
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# LLM 通道（可选注入）
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATE = """你是全球供应链风险分析师。将以下新闻解析为结构化冲击向量，严格输出 JSON：
{{"target_region": str, "event": str, "channel": str,
  "severity": 0~1 的浮点数, "capacity_delta_pct": 浮点数(负=收缩), "horizon_days": 整数, "summary": str}}
可选 channel: canal_transit / port_capacity / input_supply / air_freight / demand
新闻：{news}"""


def llm_parse(news: str, llm_call: Callable[[str], str]) -> Shock:
    """llm_call: 接受 prompt 返回字符串的函数（OpenAI 兼容接口均可）。
    JSON 解析或校验失败时抛异常，由调用方决定是否回退规则引擎。"""
    raw = llm_call(PROMPT_TEMPLATE.format(news=news))
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"LLM 输出无 JSON: {raw[:200]}")
    d = json.loads(m.group(0))
    return Shock(
        target_region=str(d["target_region"]),
        event=str(d["event"]),
        channel=str(d["channel"]),
        severity=float(np_clip(d["severity"], 0.0, 1.0)),
        capacity_delta_pct=float(d["capacity_delta_pct"]),
        horizon_days=int(d["horizon_days"]),
        summary=str(d.get("summary", "")),
    )


def np_clip(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# 规则回退通道
# --------------------------------------------------------------------------- #

_RULES: list[tuple[str, Shock]] = [
    (r"red\s*sea|suez|houthi|胡塞|红海|苏伊士",
     Shock("Suez/Red Sea", "naval_attack", "canal_transit", 0.85, -40.0, 90,
           "红海袭击导致苏伊士运河通行量锐减，班轮绕航好望角")),
    (r"good\s*hope|cape of good",
     Shock("Suez/Red Sea", "reroute", "canal_transit", 0.60, -25.0, 60,
           "航线改道好望角，有效运力下降")),
    (r"strike|罢工",
     Shock("Generic", "labor_strike", "port_capacity", 0.70, -30.0, 30,
           "港口/码头劳工罢工，作业能力下降")),
    (r"lockdown|疫情|covid|shutdown|封控",
     Shock("EastAsia", "lockdown", "input_supply", 0.80, -35.0, 120,
           "区域封控造成零部件供应延迟")),
    (r"tariff|关税|sanction|制裁",
     Shock("Generic", "trade_policy", "input_supply", 0.50, -15.0, 180,
           "贸易政策变化，采购链路重构")),
    (r"typhoon|hurricane|earthquake|台风|地震|洪水|flood",
     Shock("Generic", "natural_disaster", "port_capacity", 0.75, -45.0, 21,
           "自然灾害冲击港口与产区")),
    (r"drought|巴拿马|panama",
     Shock("Panama", "canal_drought", "canal_transit", 0.65, -30.0, 150,
           "巴拿马运河干旱限行，通行能力下降")),
]


def rule_parse(news: str) -> Shock:
    text = news.lower()
    for pat, shock in _RULES:
        if re.search(pat, text):
            return shock
    return Shock("Generic", "noise", "none", 0.10, 0.0, 0, "未识别显著冲击，按噪声处理")


def parse_news(news: str, llm_call: Callable[[str], str] | None = None,
               prefer_llm: bool = True) -> Shock:
    """统一入口：默认先尝试 LLM，失败回退规则；也可强制规则模式（离线/可复现）。"""
    if llm_call is not None and prefer_llm:
        try:
            return llm_parse(news, llm_call)
        except Exception as e:  # noqa: BLE001
            print(f"[parser] LLM 解析失败，回退规则引擎: {e}")
    return rule_parse(news)


def selftest() -> int:
    s1 = parse_news("红海多艘货轮遭武装袭击，苏伊士运河通行量锐减", prefer_llm=False)
    assert s1.channel == "canal_transit" and s1.severity >= 0.8 and s1.capacity_delta_pct < 0
    s2 = parse_news("洛杉矶港码头工人宣布大罢工", prefer_llm=False)
    assert s2.event == "labor_strike" and s2.channel == "port_capacity"
    s3 = parse_news("某央行发布例行利率决议", prefer_llm=False)
    assert s3.channel == "none" and s3.capacity_delta_pct == 0.0
    # 伪 LLM 通道
    fake = lambda prompt: '{"target_region":"Panama","event":"drought","channel":"canal_transit","severity":0.7,"capacity_delta_pct":-30,"horizon_days":150,"summary":"巴拿马运河限行"}'
    s4 = parse_news("巴拿马运河水位下降", llm_call=fake)
    assert s4.target_region == "Panama" and s4.severity == 0.7
    # LLM 输出垃圾时自动回退
    s5 = parse_news("红海遇袭", llm_call=lambda p: "对不起我无法回答")
    assert s5.channel == "canal_transit"
    print(f"[selftest] 红海案例解析: {s1.to_dict()}")
    print("[selftest] PASS ✔  Step1 信号解析（LLM + 规则回退）通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--news", type=str, default=None)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.news:
        print(json.dumps(parse_news(args.news, prefer_llm=False).to_dict(),
                         ensure_ascii=False, indent=2))
