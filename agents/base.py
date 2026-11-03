"""Agent 基类与定性意图约定。

设计纪律（防 LLM 数值幻觉 + 防 agent bullwhip）：
  1. Agent 只输出定性意图 Intent（动作枚举 + 强度档位 + 理由），绝不输出数字；
  2. 所有定量结算发生在 physics/ 模块（查表 + 弹性系数，纯 Python）；
  3. Agent 之间通过消息黑板通信，支持多轮迭代（真正的「博弈」而非并行独白）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from agents.parser import Shock

Action = Literal[
    "normal",        # 维持现状
    "panic_buy",     # 恐慌性超额下单（牛鞭效应）
    "destock",       # 去库存 / 收缩采购
    "reroute",       # 绕航（航运）
    "gri",           # 征收综合费率上涨（General Rate Increase，航运）
    "slow_steaming", # 降速航行吸收运力
    "queue_mode",    # 港口排队模式
    "surge_shift",   # 港口加班作业
]

Strength = Literal["low", "mid", "high"]
STRENGTH_MULT = {"low": 0.5, "mid": 1.0, "high": 1.5}


@dataclass
class Intent:
    """Step 2 输出：定性意图。"""
    agent_id: str
    action: Action
    strength: Strength
    rationale: str = ""
    round_no: int = 0


@dataclass
class Message:
    """黑板消息：Agent 决策广播给其他 Agent，驱动多轮博弈。"""
    sender: str
    action: Action
    strength: Strength
    payload: dict = field(default_factory=dict)


class Agent(ABC):
    """所有智能体的基类。"""

    def __init__(self, agent_id: str, **state):
        self.agent_id = agent_id
        self.state: dict = dict(state)
        self.inbox: list[Message] = []

    def perceive(self, shock: Shock) -> None:
        """感知环境冲击，更新自身状态（定性更新，不做算术）。"""
        self.state["last_shock"] = shock.event

    def receive(self, msg: Message) -> None:
        self.inbox.append(msg)

    @abstractmethod
    def decide(self, round_no: int = 0) -> Intent:
        """根据自身状态 + 收件箱，输出定性意图。"""
        raise NotImplementedError

    def reset_round(self) -> None:
        self.inbox.clear()


def build_default_agents() -> list[Agent]:
    """按文档方案实例化 13 个代表性集聚智能体：7 制造 + 3 航运 + 3 港口。"""
    from agents.manufacturers import ManufacturerAgent
    from agents.shippers import ShippingAgent
    from agents.ports import PortAgent

    economies = ["china", "us", "ea", "jp", "kr", "tw", "uk"]
    mfrs = [ManufacturerAgent(e) for e in economies]
    shippers = [
        ShippingAgent("maersk_like", routes=["asia_europe", "transpacific"]),
        ShippingAgent("msc_like", routes=["asia_europe", "med"]),
        ShippingAgent("cosco_like", routes=["transpacific", "intraasia"]),
    ]
    ports = [
        PortAgent("uswc", region="us"),
        PortAgent("rotterdam", region="ea"),
        PortAgent("eastasia_hub", region="eastasia"),
    ]
    return mfrs + shippers + ports
