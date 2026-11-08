"""状态更新算子（v2）——本文件是"多智能体推理模块"的可执行规范。

月度时间步长。每个智能体有**水平量**（level）与**信念量**（belief），
更新遵循统一的四段式：

    perceive  →  belief  →  decide  →  transition

一、感知（perception）
    制造企业 m：
        D̂_m = D0 + Σ_r χ_{m,r} · s · Δ_r + α_c · Δtransit + α_p · queue_avg
        （D0=常态到货天数；χ=航线敞口；s=冲击严重度；Δ_r=该航线绕航增量天数）
    承运人 c：
        risk_c = max( decay(risk_c), s · 1{该航线受影响} · exposure_c )
    港口 p：
        inflow_p = base_p · (1 + s · κ_region) + Σ divert_in

二、信念（belief）
        b_m = clip( w_d·(D̂/D* - 1) + w_h·peer + w_f·rate_shock , 0, 1 )
        panic_m ← (1-ρ_m)·panic_m + ρ_m·b_m
    承运人：utilization 由需求/有效运力内生决定；risk 直接来自感知。
    港口：berth_util 由到港量/泊位能力内生决定。

三、决策（decide）—— 阈值 + 滞回（satisficing）
        if panic > θ + h        : raise_order, k=3 if panic>θ2 else 2
        elif panic < θ_low - h  : cut_order  (仅当 backlog>55)
        else                    : hold
        + 次级动作（expedite_air / dual_source），每个智能体每月至多 2 个动作。
    滞回 h 的作用：避免在阈值附近逐轮抖动（LLM 智能体尤其容易）。

四、状态转移（transition）—— 部分调整 + 饱和
        x_{t+1} = x_t + adj·(x*_t - x_t) + Σ_a e_a·u(k)
        u(k) = k/3 ；饱和由 clip 保证（PMI 分项 ∈ [20,80]）

五、跨智能体传导（消息黑板）
    动作广播为 Message(sender, action, intensity, salient)；
    接收方按 TRANSMISSION 表施加影响。这就是"大家的 action 怎么更新彼此状态"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .actions import ALL_ACTIONS, Intent, u
from .profiles import CarrierProfile, ManufacturerProfile, PortProfile, build_all


# --------------------------------------------------------------------------- #
# 冲击（Step 1 的结构化输出）
# --------------------------------------------------------------------------- #
@dataclass
class Shock:
    region: str                 # "Suez/Red Sea" / "Panama" / "USWC" ...
    channel: str                # canal_transit / port_capacity / input_supply / demand / air_freight
    severity: float             # 0~1
    capacity_delta: float       # 受影响环节运力变化（-0.4 = 收缩 40%）
    horizon_months: int = 3
    routes_affected: tuple[str, ...] = ()
    label: str = ""


# 冲击 → 航线映射
SHOCK_ROUTES = {
    "Suez/Red Sea": ("asia_europe",),
    "Panama": ("transpacific",),
    "USWC": ("transpacific",),
    "EastAsia": ("intra_asia", "transpacific"),
    "Europe": ("asia_europe",),
}


# --------------------------------------------------------------------------- #
# 跨智能体传导系数（消息黑板）
# --------------------------------------------------------------------------- #
TRANSMISSION = {
    # (接收方类型, 发送方动作) -> (接收方状态字段, 系数)
    ("manufacturer", "reroute"):      ("peer_pressure", 0.45),
    ("manufacturer", "gri"):          ("peer_pressure", 0.30),
    ("manufacturer", "blank_sailing"): ("peer_pressure", 0.25),
    ("manufacturer", "raise_order"):  ("peer_pressure", 0.15),
    ("carrier", "raise_order"):       ("utilization", 0.015),
    ("port", "raise_order"):          ("berth_util", 0.012),
    ("port", "reroute"):              ("diversion_in", 0.08),
    ("carrier", "queue_mode"):        ("transit_extra", 0.6),
}


@dataclass
class Message:
    sender: str
    action: str
    intensity: int
    salient: dict = field(default_factory=dict)

    def to_dict(self):
        return {"sender": self.sender, "action": self.action,
                "intensity": self.intensity, **self.salient}


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
BASELINE_DELAY = 20.0      # 常态到货天数
BASELINE_QUEUE = 1.5       # 常态等泊天数
DECAY = {                  # 水平量向常态回归的速度（1 - 月度残留）
    "backlog_idx": 0.15, "delivery_idx": 0.18, "purchases_idx": 0.15,
    "order_rate": 0.30, "spot_rate": 0.25, "queue_days": 0.30,
    "utilization": 0.30, "berth_util": 0.30, "transit_days": 0.25,
}
ADJ = 0.35                 # 部分调整速度


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- #
# 仿真器
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(self):
        self.agents = build_all()
        self.mfrs = [a for a in self.agents if isinstance(a, ManufacturerProfile)]
        self.carriers = [a for a in self.agents if isinstance(a, CarrierProfile)]
        self.ports = [a for a in self.agents if isinstance(a, PortProfile)]
        self.month = 0
        self.disabled: set[str] = set()
        self.history: list[dict] = []
        self.log: list[dict] = []
        self.air_stress = 0.0
        self.air_stress_prev = 0.0
        self.air_idx = 100.0
        self.rate_pressure = 0.0
        for m in self.mfrs:
            m._peer_pressure = 0.0
            m._rate_index = 100.0

    # -- 一、感知 ---------------------------------------------------------- #
    def _perceive(self, shock: Shock):
        routes = shock.routes_affected or SHOCK_ROUTES.get(shock.region, ())
        transit_extra = 0.0
        # 承运人绕航造成的额外周转天数（按运力加权）
        for c in self.carriers:
            if c.reroute_share > 0:
                transit_extra += c.reroute_share * 10.5 * (c.capacity / 100.0)
        queue_avg = float(np.mean([p.queue_days for p in self.ports]))

        # 衰退：冲击严重度按 horizon 指数衰减
        s = shock.severity * math.exp(-self.month / max(1.0, shock.horizon_months / 2.0))

        for m in self.mfrs:
            exposure = sum(m.route_exposure.get(r, 0.0) for r in routes)
            if shock.channel == "input_supply":
                exposure = max(exposure, m.input_dependence * 0.8)
            d = (BASELINE_DELAY
                 + exposure * s * 14.0
                 + 0.55 * transit_extra
                 + 0.8 * max(0.0, queue_avg - BASELINE_QUEUE))
            m.perceived_delay = d
            m._shock_s = s
            m._exposure = exposure

        for c in self.carriers:
            hits = [r for r in c.routes if r in routes]
            base_risk = s * (1.0 if hits else 0.25)
            c.risk = max(c.risk * 0.75, base_risk)

        for p in self.ports:
            # 到港需求：受冲击区域收缩 → 改港/排队
            p._inflow = 1.0 + 0.35 * s

    # -- 二~三、信念更新 + 决策 -------------------------------------------- #
    def _deliberate(self, rounds: int):
        intents_by_round = []
        for r in range(rounds):
            acts = []
            for m in self.mfrs:
                if m.agent_id not in self.disabled:
                    acts.extend(self._mfr_policy(m, r))
            for c in self.carriers:
                if c.agent_id not in self.disabled:
                    acts.extend(self._carrier_policy(c, r))
            for p in self.ports:
                if p.agent_id not in self.disabled:
                    acts.extend(self._port_policy(p, r))
            intents_by_round.append(acts)
            # 广播 + 接收（下一轮生效）
            self._broadcast(acts)
        return intents_by_round

    def _mfr_policy(self, m: ManufacturerProfile, r: int) -> list[Intent]:
        b = m.behavior
        w_d, w_h, w_f = 0.62, b.herding, 0.18
        rate_shock = _clip((m._rate_index - 100.0) / 40.0, 0, 1) if hasattr(m, "_rate_index") else 0.0
        raw = w_d * _clip((m.perceived_delay / BASELINE_DELAY) - 1.0, 0, 1.5) \
            + w_h * m._peer_pressure + w_f * rate_shock
        target = _logistic(4.0 * (raw - 0.35))
        rho = b.inertia
        m.panic = _clip((1 - rho) * m.panic + rho * target, 0.0, 1.0)

        out = []
        th, h = b.threshold, b.hysteresis
        if m.panic > th + h:
            k = 3 if m.panic > 0.72 else 2
            out.append(Intent(m.agent_id, "raise_order", k,
                              f"紧迫度{m.panic:.2f}＞阈值{th + h:.2f}，启动超额下单"))
        elif m.panic < 0.35 - h and m.backlog_idx > 55:
            out.append(Intent(m.agent_id, "cut_order", 2,
                              f"紧迫度{m.panic:.2f}回落且积压{m.backlog_idx:.0f}，去库存"))
        else:
            out.append(Intent(m.agent_id, "hold", 0, f"紧迫度{m.panic:.2f}在容带内，维持"))
        # 次级动作：空运替代 / 备用源
        if m.perceived_delay > 26 and m.air_substitution > 0.05:
            out.append(Intent(m.agent_id, "expedite_air", 2,
                              f"到货延迟{m.perceived_delay:.0f}天，高值货转空运"))
        if m.perceived_delay > 34 and m.panic > 0.7:
            out.append(Intent(m.agent_id, "dual_source", 2, "延迟持续，启用备用供应源"))
        return out

    def _carrier_policy(self, c: CarrierProfile, r: int) -> list[Intent]:
        b = c.behavior
        out = []
        if c.risk > 0.45 + b.risk_aversion * 0.05 and c.reroute_share < 1.0:
            k = 3 if c.risk > 0.7 else 2
            out.append(Intent(c.agent_id, "reroute", k, f"航线风险{c.risk:.2f}，绕行好望角"))
        elif c.utilization > 0.88:
            out.append(Intent(c.agent_id, "gri", 2, f"舱位利用率{c.utilization:.2f}，征 GRI/BAF"))
        elif c.utilization < 0.72:
            out.append(Intent(c.agent_id, "blank_sailing", 1, "需求不足，停航空班保价"))
        else:
            out.append(Intent(c.agent_id, "hold", 0, "供需平衡，维持班期"))
        if c.reroute_share > 0 and c.utilization < 0.92:
            out.append(Intent(c.agent_id, "slow_steam", 1, "绕航后降速吸收运力波动"))
        return out

    def _port_policy(self, p: PortProfile, r: int) -> list[Intent]:
        out = []
        if p.queue_days >= 5.0 or p.berth_util >= 0.93:
            out.append(Intent(p.agent_id, "queue_mode", 3 if p.queue_days >= 8 else 2,
                              f"等泊{p.queue_days:.1f}天，触发外海排队"))
        else:
            out.append(Intent(p.agent_id, "hold", 0, "作业正常"))
        if p.queue_days >= 4.0 and p.labor_slack >= 0.08:
            out.append(Intent(p.agent_id, "surge_shift", 2, "加班提效清积压"))
        if p.berth_util >= 0.95 and p.labor_slack < 0.05:
            out.append(Intent(p.agent_id, "divert", 2, "泊位饱和，分流至邻港"))
        return out

    def _broadcast(self, acts: list[Intent]):
        """黑板：动作 → 其他 agent 的状态量（这就是"action 如何更新彼此状态"）。"""
        for it in acts:
            spec = ALL_ACTIONS[it.action]
            msg = Message(it.agent_id, it.action, it.intensity,
                          {"strength": round(u(it.intensity), 2)})
            self.log.append({**msg.to_dict(), "round": it.round_no, "month": self.month})
            for m in self.mfrs:
                key = ("manufacturer", it.action)
                if key in TRANSMISSION and it.agent_id != m.agent_id:
                    field_name, coef = TRANSMISSION[key]
                    if field_name == "peer_pressure":
                        m._peer_pressure = _clip(m._peer_pressure + coef * u(it.intensity), 0, 1)
            for c in self.carriers:
                key = ("carrier", it.action)
                if key in TRANSMISSION and it.agent_id != c.agent_id:
                    field_name, coef = TRANSMISSION[key]
                    setattr(c, field_name, getattr(c, field_name) + coef * u(it.intensity))
            for p in self.ports:
                key = ("port", it.action)
                if key in TRANSMISSION and it.agent_id != p.agent_id:
                    field_name, coef = TRANSMISSION[key]
                    setattr(p, field_name, getattr(p, field_name) + coef * u(it.intensity))

    # -- 四、状态转移 ------------------------------------------------------ #
    def _transition(self, acts_final: list[Intent]):
        """动作 → 驱动量 → 指标水平（部分调整 + 饱和）。

        刻意避免两种数值失控：
          * 乘性累加（每月 ×1.22 会指数爆炸）→ 改为"压力池 + 向目标部分调整"；
          * 指标直接等于当期动作（动作消失指标就归零）→ 指标是有记忆的状态量。
        """
        for m in self.mfrs:
            m._peer_pressure = 0.0

        # ---------- 承运人：运价压力池、有效运力、利用率 ----------
        rate_delta = 0.0
        for a in acts_final:
            rate_delta += ALL_ACTIONS[a.action].effects.get("rate_pressure", 0.0) * u(a.intensity)
        self.rate_pressure = self.rate_pressure * 0.85 + rate_delta

        for c in self.carriers:
            rr = [a for a in acts_final if a.agent_id == c.agent_id and a.action == "reroute"]
            if rr:
                c.reroute_share = _clip(c.reroute_share + 0.55 * u(rr[0].intensity), 0, 1)
            bs = [a for a in acts_final if a.agent_id == c.agent_id and a.action == "blank_sailing"]
            if bs:
                c.blank_share = _clip(c.blank_share + 0.10 * u(bs[0].intensity), 0, 0.25)
            c.blank_share *= 0.85
            # 需求：制造商紧迫度抬升补库需求
            demand = 100.0 * (1.0 + 0.010 * sum(m.panic for m in self.mfrs))
            cap_target = c.capacity * (1 - 0.16 * c.reroute_share - 0.30 * c.blank_share)
            c.eff_capacity += 0.5 * (cap_target - c.eff_capacity)
            c.utilization = _clip(demand / max(c.eff_capacity, 1e-6) * 0.82, 0.3, 1.15)
            # 运价目标：压力池 + 利用率凸性
            rate_target = 100.0 * (1.0 + self.rate_pressure) * \
                (1.0 + 2.2 * max(0.0, c.utilization - 0.85) ** 1.4)
            c.spot_rate += 0.55 * (rate_target - c.spot_rate)

        # ---------- 港口 ----------
        for p in self.ports:
            for a in acts_final:
                if a.agent_id != p.agent_id:
                    continue
                spec = ALL_ACTIONS[a.action]
                k = u(a.intensity)
                p.queue_days = _clip(p.queue_days + spec.effects.get("queue_days", 0.0) * k, 0, 30)
                p.berth_util = _clip(p.berth_util + spec.effects.get("berth_util", 0.0) * k, 0.3, 1.1)
                p.throughput = getattr(p, "throughput", 100.0) * (1 + spec.effects.get("throughput", 0.0) * k)
            inflow = getattr(p, "_inflow", 1.0)
            p.queue_days = _clip(p.queue_days * 0.70 + 0.30 * BASELINE_QUEUE * inflow, 0.2, 30)
            p.berth_util = _clip(p.berth_util * 0.75 + 0.25 * (0.78 * inflow), 0.3, 1.1)

        # ---------- 制造商 ----------
        rate_proxy = float(np.mean([c.spot_rate for c in self.carriers]))
        for m in self.mfrs:
            m._rate_index = rate_proxy
            for a in acts_final:
                if a.agent_id != m.agent_id:
                    continue
                spec = ALL_ACTIONS[a.action]
                k = u(a.intensity)
                # 动作只推"驱动量"（订单速率），其余分项由目标值内生
                m.order_rate = _clip(m.order_rate + spec.effects.get("order_rate", 0.0) * k, 40, 180)
                # 结构性缓解动作直接作用在交货分项上（查表弹性，非 LLM 生成）
                if a.action == "dual_source":
                    m.delivery_idx = _clip(m.delivery_idx - 1.8 * k, 20, 80)
                elif a.action == "expedite_air":
                    m.delivery_idx = _clip(m.delivery_idx - 0.9 * k, 20, 80)
            tgt_order = 100.0 * (1.0 + 0.30 * m.panic)
            m.order_rate += 0.45 * (tgt_order - m.order_rate)
            tgt_backlog = 50 + 0.30 * (m.order_rate - 100) + 0.25 * (m.safety_target - m.inventory_days)
            tgt_delivery = 50 + 1.00 * (m.perceived_delay - BASELINE_DELAY)
            tgt_purchases = 50 + 0.25 * (m.order_rate - 100)
            m.backlog_idx += ADJ * (tgt_backlog - m.backlog_idx)
            m.delivery_idx += ADJ * (tgt_delivery - m.delivery_idx)
            m.purchases_idx += ADJ * (tgt_purchases - m.purchases_idx)
            inflow = 1.0 / max(1.0, m.perceived_delay / BASELINE_DELAY)
            m.inventory_days = _clip(m.inventory_days + 1.1 * (inflow - 1.0)
                                     - 0.05 * (m.order_rate / 100.0 - 1.0) * 10, 5, 90)
            # 水平量缓慢回归常态（指数本身均值回复）
            m.backlog_idx = 50 + (m.backlog_idx - 50) * (1 - DECAY["backlog_idx"])
            m.delivery_idx = 50 + (m.delivery_idx - 50) * (1 - DECAY["delivery_idx"])
            m.purchases_idx = 50 + (m.purchases_idx - 50) * (1 - DECAY["purchases_idx"])
            m.panic *= 0.92

        # ---------- 空运价格指数（状态量，不是当期动作的函数） ----------
        air_now = float(np.mean([u(a.intensity) for a in acts_final if a.action == "expedite_air"] + [0.0]))
        self.air_stress_prev = self.air_stress
        self.air_stress = 0.6 * self.air_stress + 0.4 * air_now
        self.air_idx = getattr(self, "air_idx", 100.0) + 0.45 * ((100.0 * (1 + 0.30 * self.air_stress)) - getattr(self, "air_idx", 100.0))

        self.month += 1

    # -- 月度步进 ---------------------------------------------------------- #
    def step(self, shock: Shock, rounds: int = 2) -> list[Intent]:
        self._perceive(shock)
        history = self._deliberate(rounds)
        final = history[-1]
        # 结算用最后几轮中"非 hold"的动作并集，防止最后一轮恰好回到容带
        merged: dict[str, Intent] = {}
        for intent in [a for rd in history for a in rd]:
            if intent.action == "hold":
                continue
            prev = merged.get((intent.agent_id, intent.action))
            if prev is None or intent.intensity > prev.intensity:
                merged[(intent.agent_id, intent.action)] = intent
        for a in final:
            if a.action == "hold" and not any(k[0] == a.agent_id for k in merged):
                merged[(a.agent_id, a.action)] = a
        acts = list(merged.values())
        self._transition(acts)
        snap = self.snapshot()
        snap["month"] = self.month
        self.history.append(snap)
        return acts

    # -- 导出 -------------------------------------------------------------- #
    def snapshot(self) -> dict:
        return {
            "mfr": {m.economy: {
                "panic": round(m.panic, 3), "delay": round(m.perceived_delay, 2),
                "inventory": round(m.inventory_days, 1), "order": round(m.order_rate, 1),
                "backlog": round(m.backlog_idx, 1), "delivery": round(m.delivery_idx, 1),
                "purchases": round(m.purchases_idx, 1)} for m in self.mfrs},
            "carrier": {c.agent_id: {
                "risk": round(c.risk, 3), "util": round(c.utilization, 3),
                "rate": round(c.spot_rate, 1), "cap": round(c.eff_capacity, 1),
                "reroute": round(c.reroute_share, 2)} for c in self.carriers},
            "port": {p.agent_id: {
                "queue": round(p.queue_days, 2), "util": round(p.berth_util, 3)}
                for p in self.ports},
        }
