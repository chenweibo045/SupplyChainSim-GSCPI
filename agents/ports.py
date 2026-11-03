"""港口类 Agent（美西 / 鹿特丹 / 东亚枢纽）：评估集中到港 → 排队 / 加班模式。

状态：锚地等待感知、泊位利用率档位、作业模式。
"""

from __future__ import annotations

from agents.base import Agent, Intent, Message
from agents.parser import Shock


class PortAgent(Agent):
    def __init__(self, name: str, region: str):
        super().__init__(
            f"port_{name}",
            region=region,
            anchorage="clear",          # clear / building / jam
            berth_util="normal",        # normal / high / over
            mode="normal",              # normal / queue_mode / surge_shift
        )
        self.region = region

    def perceive(self, shock: Shock) -> None:
        super().perceive(shock)
        if shock.severity < 0.3:
            return
        if shock.channel in ("canal_transit", "port_capacity", "input_supply"):
            lvl = "severe" if shock.severity >= 0.7 else "moderate"
            if lvl == "severe":
                self.state["anchorage"] = "building"
            self.state["berth_util"] = "high" if lvl == "moderate" else "over"

    def receive(self, msg: Message) -> None:
        super().receive(msg)
        # 绕航班轮集中到港 + 制造商恐慌下单 → 港口拥堵加剧
        if msg.action == "reroute":
            self.state["anchorage"] = "building" if self.state["anchorage"] == "clear" else "jam"
        if msg.action == "panic_buy" and msg.strength == "high":
            self.state["berth_util"] = "over"

    def decide(self, round_no: int = 0) -> Intent:
        if self.state["anchorage"] == "jam" or self.state["berth_util"] == "over":
            self.state["mode"] = "queue_mode"
            return Intent(self.agent_id, "queue_mode", "high",
                          "集装箱集中到港：触发外海排队等待模式", round_no)
        if self.state["anchorage"] == "building":
            self.state["mode"] = "queue_mode"
            return Intent(self.agent_id, "queue_mode", "mid",
                          "锚地等待船只增加：进入排队模式", round_no)
        if self.state["mode"] == "queue_mode":
            return Intent(self.agent_id, "surge_shift", "mid",
                          "拥堵缓解：启动加班作业清堆积", round_no)
        return Intent(self.agent_id, "normal", "low", "港口作业正常", round_no)
