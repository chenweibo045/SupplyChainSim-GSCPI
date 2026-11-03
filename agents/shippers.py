"""航运类 Agent（3 大全球承运人）：评估安全风险与供需 → 绕航 / 提价 / 降速。

状态：受影响航线列表、安全风险感知、运力缺口感知。
"""

from __future__ import annotations

from agents.base import Agent, Intent, Message
from agents.parser import Shock


class ShippingAgent(Agent):
    def __init__(self, name: str, routes: list[str]):
        super().__init__(
            f"ship_{name}",
            routes=list(routes),
            risk_perception="none",     # none / elevated / severe
            capacity_gap="none",        # none / tight / severe
            bunker_cost="normal",       # normal / elevated
            rerouting=False,
        )
        self.routes = list(routes)

    def perceive(self, shock: Shock) -> None:
        super().perceive(shock)
        if shock.severity < 0.3:
            return
        if shock.channel == "canal_transit":
            if "asia_europe" in self.routes or "med" in self.routes:
                lvl = "severe" if shock.severity >= 0.7 else "elevated"
                self.state["risk_perception"] = lvl
                self.state["capacity_gap"] = "severe" if shock.capacity_delta_pct <= -35 else "tight"
        elif shock.channel == "port_capacity":
            self.state["capacity_gap"] = "tight"
        elif shock.channel == "air_freight":
            self.state["capacity_gap"] = "tight"

    def receive(self, msg: Message) -> None:
        super().receive(msg)
        # 同行绕航 → 自己也面临码头拥堵与运力紧张
        if msg.action == "reroute" and msg.sender != self.agent_id:
            if self.state["capacity_gap"] == "none":
                self.state["capacity_gap"] = "tight"

    def decide(self, round_no: int = 0) -> Intent:
        risk = self.state["risk_perception"]
        gap = self.state["capacity_gap"]
        if risk == "severe" and not self.state["rerouting"]:
            self.state["rerouting"] = True
            return Intent(self.agent_id, "reroute", "high",
                          "航线安全风险严重：放弃苏伊士，改道好望角", round_no)
        if risk == "elevated" and not self.state["rerouting"]:
            self.state["rerouting"] = True
            return Intent(self.agent_id, "reroute", "mid",
                          "安全风险上行：部分班次绕航", round_no)
        if gap in ("tight", "severe"):
            return Intent(self.agent_id, "gri", "high" if gap == "severe" else "mid",
                          f"运力供需紧张（{gap}）：开征燃油附加费 / GRI", round_no)
        if self.state["rerouting"]:
            return Intent(self.agent_id, "slow_steaming", "low",
                          "已绕航：降速航行吸收运力波动", round_no)
        return Intent(self.agent_id, "normal", "low", "市场平稳，维持班期", round_no)
