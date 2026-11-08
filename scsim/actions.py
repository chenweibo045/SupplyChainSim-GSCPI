"""Action Space（动作空间）定义（v2）。

三类智能体的动作空间都是**有限离散 + 强度档位**：
    a = (action_type, k),  k ∈ {0, 1, 2, 3}  → 强度系数 u(k) = k / 3 ∈ [0, 1]

为什么不用连续动作？
  1) LLM 只能可靠地做定性选择，连续数值交给结算层（原设计纪律）；
  2) 离散动作使状态转移可审计、可复现（同一 seed + 同一动作 → 同一轨迹）；
  3) 归因可以精确到"某个 agent 的某个动作档位"。

每个动作有：
  preconditions  前置条件（硬约束，不满足则该动作不可行）
  effects        对**中间量**的弹性（不是对 GSCPI 的直接弹性）
  lag            生效滞后（月），用于数据同化与多期结算
  channels       该动作影响的指标族（freight / pmi / port）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionType = Literal[
    # 制造商
    "hold", "raise_order", "cut_order", "expedite_air", "dual_source",
    # 承运人
    "reroute", "slow_steam", "gri", "blank_sailing",
    # 港口
    "queue_mode", "surge_shift", "divert",
]

INTENSITY_LEVELS = (0, 1, 2, 3)
INTENSITY_LABEL = {0: "维持", 1: "轻度", 2: "中度", 3: "强力"}


def u(k: int) -> float:
    """强度档位 → 连续系数。"""
    if k not in INTENSITY_LEVELS:
        raise ValueError(f"非法强度档位: {k}")
    return k / 3.0


@dataclass(frozen=True)
class ActionSpec:
    name: ActionType
    label: str            # 中文名（报告/图里用）
    owners: tuple[str, ...]
    precondition: str     # 人类可读的前置条件（代码里在 policy 中实现）
    effects: dict[str, float]   # 中间量 → 每单位强度的弹性（方向+量级，月度口径）
    lag_months: int = 0
    channels: tuple[str, ...] = ()
    note: str = ""


# --------------------------------------------------------------------------- #
# 中间量命名（结算层的"物理变量"）
# --------------------------------------------------------------------------- #
# 制造商侧：order_rate / backlog_idx / delivery_idx / purchases_idx / air_demand
# 承运人侧：eff_capacity / spot_rate / transit_days / blank_share
# 港口侧：  queue_days / berth_util / throughput
# --------------------------------------------------------------------------- #

MANUFACTURER_ACTIONS: dict[str, ActionSpec] = {
    "hold": ActionSpec(
        "hold", "维持现状", ("manufacturer",),
        "无（默认动作）", {},
        note="不产生增量，作为基线动作",
    ),
    "raise_order": ActionSpec(
        "raise_order", "超额下单/补安全库存", ("manufacturer",),
        "panic ≥ θ 或 库存 < 安全线",
        {"order_rate": 28.0},
        channels=("pmi",),
        note="牛鞭效应的行为源头；order_rate 单位=指数点/月",
    ),
    "cut_order": ActionSpec(
        "cut_order", "去库存/收缩采购", ("manufacturer",),
        "panic ≤ θ_low 且 backlog_idx > 55",
        {"order_rate": -22.0},
        channels=("pmi",),
    ),
    "expedite_air": ActionSpec(
        "expedite_air", "高值货转空运", ("manufacturer",),
        "air_substitution > 0 且 感知延迟 > 15 天",
        {},
        lag_months=1,
        channels=("freight", "pmi"),
        note="空运需求 → BLS 空运价格指数；同时缓解自身交货延迟",
    ),
    "dual_source": ActionSpec(
        "dual_source", "启用备用供应源", ("manufacturer",),
        "感知延迟 > 20 天 且 panic ≥ θ_high",
        {},
        lag_months=2,
        channels=("pmi",),
        note="结构性缓解，但带来额外采购（库存分项上升）",
    ),
}

CARRIER_ACTIONS: dict[str, ActionSpec] = {
    "reroute": ActionSpec(
        "reroute", "绕航（放弃苏伊士/红海）", ("carrier",),
        "risk ≥ θ_r 且 该承运人覆盖受影响航线",
        # 绕航 → 单航次周转天数上升 → 有效运力下降 → 运价上升
        {"transit_days": 10.5, "eff_capacity": -0.16, "rate_pressure": 0.10},
        channels=("freight", "pmi"),
        note="10.5 天 = 3500nm @16kn + 15% 港口/班期缓冲（查表，非 LLM 生成）",
    ),
    "slow_steam": ActionSpec(
        "slow_steam", "降速航行", ("carrier",),
        "已绕航 且 舱位利用率 < 0.92",
        {"transit_days": 2.5, "eff_capacity": -0.04, "rate_pressure": 0.03},
        channels=("freight",),
        note="吸收运力波动、降低燃油成本，对运价有支撑",
    ),
    "gri": ActionSpec(
        "gri", "综合费率上涨（GRI/BAF）", ("carrier",),
        "舱位利用率 ≥ 0.88",
        {"rate_pressure": 0.08},
        channels=("freight",),
        note="利用率触发的定价行为，不是成本加成",
    ),
    "blank_sailing": ActionSpec(
        "blank_sailing", "停航空班", ("carrier",),
        "舱位利用率 < 0.72（需求不足时保价）",
        {"eff_capacity": -0.05, "rate_pressure": 0.03},
        channels=("freight",),
        note="需求侧冲击下的运力管理；会推高下期运价",
    ),
}

PORT_ACTIONS: dict[str, ActionSpec] = {
    "queue_mode": ActionSpec(
        "queue_mode", "外海排队模式", ("port",),
        "queue_days ≥ θ_q 或 berth_util ≥ 0.93",
        {"queue_days": 2.0, "berth_util": 0.04, "throughput": -0.06},
        channels=("port", "pmi"),
        note="排队 → 到货延迟 → 制造端交货分项上升",
    ),
    "surge_shift": ActionSpec(
        "surge_shift", "加班/提效清积压", ("port",),
        "labor_slack ≥ 0.08 且 queue_days ≥ θ_q",
        {"queue_days": -1.6, "berth_util": -0.03, "throughput": 0.08},
        channels=("port",),
    ),
    "divert": ActionSpec(
        "divert", "分流至邻近港口", ("port",),
        "berth_util ≥ 0.95 且 labor_slack < 0.05",
        {"queue_days": -1.2, "throughput": -0.03},
        channels=("port",),
        note="零和：分流会抬高邻港队列（在结算层做转移）",
    ),
}

ALL_ACTIONS: dict[str, ActionSpec] = {
    **MANUFACTURER_ACTIONS, **CARRIER_ACTIONS, **PORT_ACTIONS,
}


def feasible(agent_kind: str, action: str) -> bool:
    return action in ALL_ACTIONS and agent_kind in ALL_ACTIONS[action].owners


def action_space(agent_kind: str) -> list[str]:
    return [k for k, v in ALL_ACTIONS.items() if agent_kind in v.owners]


@dataclass
class Intent:
    """一次动作声明：谁、做什么、多强、为什么。"""
    agent_id: str
    action: ActionType
    intensity: int
    rationale: str = ""
    round_no: int = 0

    @property
    def strength(self) -> float:
        return u(self.intensity)

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "action": self.action,
                "intensity": self.intensity, "strength": round(self.strength, 3),
                "rationale": self.rationale, "round": self.round_no}
