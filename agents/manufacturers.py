"""制造类 Agent（7 大经济体）：感知供应延迟 → 恐慌采购 / 去库存。

状态：库存天数、积压订单感知、交货延迟感知、恐慌度（0~1 定性档位）。
决策：只看冲击与消息，输出定性意图，不算数字。
"""

from __future__ import annotations

from agents.base import Agent, Intent, Message, Strength
from agents.parser import Shock

# 各经济体对苏伊士/欧亚航线的敞口（定性：high/mid/low）
SUEZ_EXPOSURE = {"ea": "high", "uk": "high", "china": "mid", "kr": "mid",
                 "tw": "mid", "us": "low", "jp": "low"}
INPUT_IMPORT_DEPENDENCE = {"china": "mid", "us": "high", "ea": "high", "jp": "high",
                           "kr": "high", "tw": "high", "uk": "high"}


class ManufacturerAgent(Agent):
    def __init__(self, economy: str):
        super().__init__(
            f"mfr_{economy}",
            economy=economy,
            inventory_days="mid",       # low / mid / high
            backlog_pressure="low",     # 积压订单感知
            delay_perception="none",    # none / mild / severe
            panic=0.0,
        )
        self.economy = economy

    # -- 感知 -------------------------------------------------------------- #
    def perceive(self, shock: Shock) -> None:
        super().perceive(shock)
        if shock.severity < 0.3:
            return
        # 渠道 1：运河/航运冲击 → 对欧亚敞口高的经济体感知交付延迟
        if shock.channel == "canal_transit":
            exp = SUEZ_EXPOSURE.get(self.economy, "low")
            bump = {"high": 0.5, "mid": 0.3, "low": 0.1}[exp]
            self.state["panic"] = min(1.0, self.state["panic"] + shock.severity * bump)
            if shock.severity >= 0.5 and exp != "low":
                self.state["delay_perception"] = "severe" if exp == "high" else "mild"
        # 渠道 2：投入品供应冲击（封控、关税、自然灾害）
        if shock.channel == "input_supply":
            dep = INPUT_IMPORT_DEPENDENCE.get(self.economy, "mid")
            bump = {"high": 0.45, "mid": 0.3}[dep]
            self.state["panic"] = min(1.0, self.state["panic"] + shock.severity * bump)

    def receive(self, msg: Message) -> None:
        super().receive(msg)
        # 航运 Agent 宣布绕航/提价 → 制造商恐慌度上行（牛鞭效应传导）
        if msg.action in ("reroute", "gri") and msg.strength != "low":
            cur = self.state["panic"]
            add = {"mid": 0.2, "high": 0.35}[msg.strength]
            self.state["panic"] = min(1.0, cur + add)
            if self.state["panic"] > 0.6:
                self.state["delay_perception"] = "severe"

    # -- 决策 -------------------------------------------------------------- #
    def decide(self, round_no: int = 0) -> Intent:
        p = self.state["panic"]
        if p >= 0.6:
            return Intent(self.agent_id, "panic_buy", "high",
                          f"恐慌度 {p:.1f}：预期断供，启动超额下单并提高安全库存", round_no)
        if p >= 0.35:
            return Intent(self.agent_id, "panic_buy", "mid",
                          f"恐慌度 {p:.1f}：适度增加采购与安全库存", round_no)
        if p <= 0.1 and self.state["backlog_pressure"] == "high":
            return Intent(self.agent_id, "destock", "low", "积压高企、恐慌消退，温和去库存", round_no)
        return Intent(self.agent_id, "normal", "low", "供应链感知平稳，维持现状", round_no)

    def reset(self) -> None:
        self.state.update(panic=0.0, delay_perception="none", last_shock=None)
