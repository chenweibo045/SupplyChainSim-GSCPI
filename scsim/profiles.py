"""智能体 Profile 定义（v2）。

三层 Profile：
  L1 身份/角色  identity   —— 我是谁：类型、覆盖区域、对应哪几个 GSCPI 指标
  L2 禀赋/暴露  endowment  —— 我有什么、我暴露在哪些通道
  L3 行为参数   behavior   —— 我如何把感知变成决策（阈值、惯性、从众、传导）

设计约束（与文档 §5.2 一致）：
  * Profile 只含**可解释、可校准**的参数，不塞自由文本人设；
  * 每个参数都标注来源：D=数据估计 / C=行业查表 / P=先验（需敏感性分析）；
  * 三类型智能体共享同一套字段命名，便于统一的状态更新算子。

单位约定（重要，旧版 demo 在这里出错）：
  * 库存 = 覆盖天数（天）
  * 积压/交货/采购 = PMI 扩散指数（50 为荣枯线，单位=指数点）
  * 运力 = 标准化指数（基期 100）
  * 运价 = 指数（基期 100）
  * 港口队列 = 等泊天数（天）
  * 所有"增量"都是**月度增量**，不是累计量。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

AgentType = Literal["manufacturer", "carrier", "port"]

# --------------------------------------------------------------------------- #
# 暴露矩阵：7 个经济体对三条主干航线的敞口（0~1）
# 口径：该经济体进口集装箱货值中经该航线承运的份额（先验，可用 Comtrade/海关数据校准）
# --------------------------------------------------------------------------- #
ROUTE_EXPOSURE: dict[str, dict[str, float]] = {
    #            asia_europe  transpacific  intra_asia
    "ea":    {"asia_europe": 0.78, "transpacific": 0.14, "intra_asia": 0.08},
    "uk":    {"asia_europe": 0.72, "transpacific": 0.18, "intra_asia": 0.10},
    "us":    {"asia_europe": 0.22, "transpacific": 0.66, "intra_asia": 0.12},
    "china": {"asia_europe": 0.46, "transpacific": 0.34, "intra_asia": 0.20},
    "jp":    {"asia_europe": 0.30, "transpacific": 0.38, "intra_asia": 0.32},
    "kr":    {"asia_europe": 0.34, "transpacific": 0.36, "intra_asia": 0.30},
    "tw":    {"asia_europe": 0.32, "transpacific": 0.40, "intra_asia": 0.28},
}

# 空运可替代份额：高价值/时效敏感货物占比（决定"改空运"动作的上限）
AIR_SUBSTITUTION: dict[str, float] = {
    "ea": 0.26, "uk": 0.24, "us": 0.31, "china": 0.22, "jp": 0.36, "kr": 0.41, "tw": 0.52,
}

# 投入品进口依赖度：决定"供应冲击 → 生产端交货延迟"的传导强度
INPUT_DEPENDENCE: dict[str, float] = {
    "ea": 0.62, "uk": 0.58, "us": 0.55, "china": 0.35, "jp": 0.66, "kr": 0.72, "tw": 0.74,
}


@dataclass
class BehaviorParams:
    """L3 行为参数（全部可解释、可做敏感性分析）。"""
    inertia: float = 0.55          # ρ：信念惯性（0=每轮完全更新，1=完全不动）
    attention: float = 0.8         # α：对自身冲击的注意力权重
    herding: float = 0.25          # κ：从众/模仿权重（看到同行动作后加多少）
    risk_aversion: float = 1.0     # β：风险厌恶，抬高动作触发阈值
    threshold: float = 0.45        # θ：动作触发阈值（信念标量超过才行动）
    hysteresis: float = 0.10       # h：滞回带宽，防止在阈值附近抖动
    adjustment: float = 0.6        # λ：状态向目标调整的速度


@dataclass
class ManufacturerProfile:
    """制造类智能体：对应 GSCPI 的 3 个 PMI 分项（7 个经济体各一个）。"""
    agent_id: str
    kind: AgentType
    economy: str
    # L2 禀赋
    inventory_days: float = 42.0        # 库存覆盖天数
    safety_target: float = 32.0         # 目标安全库存天数
    order_rate: float = 100.0           # 采购速率指数（基期 100）
    backlog_idx: float = 50.0           # 积压订单扩散指数
    delivery_idx: float = 50.0          # 供应商交货时间扩散指数（>50 = 更慢 = 压力更大）
    purchases_idx: float = 50.0         # 采购库存扩散指数
    route_exposure: dict[str, float] = field(default_factory=dict)
    air_substitution: float = 0.3
    input_dependence: float = 0.55
    capacity: float = 100.0             # 产出能力指数（限制接单上限）
    behavior: BehaviorParams = field(default_factory=BehaviorParams)
    # 运行时状态
    panic: float = 0.0                  # 0~1 紧迫度（文档叫"恐慌度/牛鞭势能"）
    perceived_delay: float = 0.0        # 感知到的到货延迟（天）
    source: str = "C"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CarrierProfile:
    """航运类智能体：对应 Harpex / BDI 与部分空运溢出。"""
    agent_id: str
    kind: AgentType
    name: str
    routes: list[str] = field(default_factory=list)
    capacity: float = 100.0             # 名义运力指数（基期 100）
    weight: float = 1.0                 # 在行业加总中的权重（运力份额）
    eff_capacity: float = 100.0         # 有效运力指数（周转天数变化后）
    transit_days: float = 30.0          # 单航次周转天数
    transit_extra: float = 0.0          # 外部拥堵/排队带来的额外周转天数
    utilization: float = 0.82           # 舱位利用率
    spot_rate: float = 100.0            # 即期运价指数
    bunker_cost: float = 100.0          # 燃油成本指数
    reroute_share: float = 0.0          # 绕航班次占比 0~1
    blank_share: float = 0.0            # 停航空班占比 0~1
    risk: float = 0.0                   # 航线安全风险感知 0~1
    behavior: BehaviorParams = field(default_factory=BehaviorParams)
    source: str = "C"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortProfile:
    """港口类智能体：对应港口拥堵 → 交货延迟与运价附加。"""
    agent_id: str
    kind: AgentType
    name: str
    region: str
    berth_capacity: float = 100.0       # 泊位通过能力指数
    queue_days: float = 1.5             # 当前等泊天数
    berth_util: float = 0.75            # 泊位利用率
    labor_slack: float = 0.15           # 可加班余量（约束 surge_shift）
    diversion_in: float = 0.0           # 从邻近港口分流的到港量
    behavior: BehaviorParams = field(default_factory=BehaviorParams)
    source: str = "C"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 13 个代表性集聚智能体（7 制造 + 3 航运 + 3 港口）
# --------------------------------------------------------------------------- #
ECONOMIES = ["china", "us", "ea", "jp", "kr", "tw", "uk"]

ECONOMY_LABEL = {
    "china": "中国", "us": "美国", "ea": "欧元区", "jp": "日本",
    "kr": "韩国", "tw": "中国台湾", "uk": "英国",
}


def build_manufacturers() -> list[ManufacturerProfile]:
    out = []
    for e in ECONOMIES:
        # 亚洲经济体库存更薄、对海运更敏感（先验，需用 PMI 分项校准）
        inv = {"china": 38.0, "us": 45.0, "ea": 44.0, "jp": 40.0,
               "kr": 34.0, "tw": 31.0, "uk": 42.0}[e]
        out.append(ManufacturerProfile(
            agent_id=f"mfr_{e}", kind="manufacturer", economy=e,
            inventory_days=inv, safety_target=inv * 0.75,
            route_exposure=dict(ROUTE_EXPOSURE[e]),
            air_substitution=AIR_SUBSTITUTION[e],
            input_dependence=INPUT_DEPENDENCE[e],
        ))
    return out


def build_carriers() -> list[CarrierProfile]:
    return [
        CarrierProfile("carrier_aea", "carrier", "亚欧干线承运人",
                       ["asia_europe"], weight=0.45, utilization=0.86),
        CarrierProfile("carrier_tpx", "carrier", "跨太平洋承运人",
                       ["transpacific"], weight=0.35, utilization=0.80),
        CarrierProfile("carrier_mix", "carrier", "多航线承运人",
                       ["asia_europe", "intra_asia"], weight=0.20, utilization=0.84),
    ]


def build_ports() -> list[PortProfile]:
    return [
        PortProfile("port_uswc", "port", "美西枢纽港", "us", queue_days=2.0),
        PortProfile("port_rtm", "port", "鹿特丹枢纽港", "ea", queue_days=1.2),
        PortProfile("port_eas", "port", "东亚枢纽港", "asia", queue_days=0.8),
    ]


def build_all():
    return build_manufacturers() + build_carriers() + build_ports()
