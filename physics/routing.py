"""Step 3a — 航运物理结算：定性意图 → 定量航运变化（查表，无 LLM）。

核心纪律：所有数字来自公开物理参数与行业弹性查表，LLM 不参与算术，
从机制上杜绝数值幻觉。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from agents.base import Intent, STRENGTH_MULT

# ---- 物理常数（公开可查） ---------------------------------------------------- #
VESSEL_SPEED_KNOTS = 16.0           # 班轮巡航速度（节）
NAUTICAL_MILE_KM = 1.852
DAYS_PER_NM = 1.0 / (VESSEL_SPEED_KNOTS * 24.0)   # 航行 1 海里所需天数

# ---- 航线查表 --------------------------------------------------------------- #
ROUTE_TABLE = {
    "Suez/Red Sea": {
        "detour_nm": 3500,               # 苏伊士 → 好望角绕航约 3,500 海里
        "alt": "Cape of Good Hope",
        "transit_days_base": 22.0,       # 原欧亚航线约 22 天
    },
    "Panama": {
        "detour_nm": 2100,
        "alt": "Cape Horn / US rail",
        "transit_days_base": 18.0,
    },
}

# ---- 市场弹性查表（文献与行业惯例标定，论文中需披露来源与校准过程） ------------- #
# GRI: General Rate Increase；BAF: 燃油附加费
FREIGHT_ELASTICITY = {
    "reroute":      {"harpex_pct": +25.0, "bdi_pct": +8.0,  "baf_pct": +30.0},
    "gri":          {"harpex_pct": +15.0, "bdi_pct": +3.0,  "baf_pct": +15.0},
    "slow_steaming": {"harpex_pct": +5.0,  "bdi_pct": +1.0, "baf_pct": +5.0},
}


@dataclass
class RoutingOutcome:
    detour_nm: float
    added_days: float          # 交货延迟天数
    transit_days_total: float  # 绕航后总航程天数
    freight_deltas: dict       # {"harpex_pct": ..., "bdi_pct": ..., "baf_pct": ...}
    air_spillover_pct: float   # 部分高值货转空运 → 空运价格指数上行


def resolve_shipping_intent(intent: Intent, region: str = "Suez/Red Sea") -> RoutingOutcome:
    """把航运 Agent 的定性意图结算为定量结果。

    示例（文档红海案例）：reroute(high) → +3,500 nm → 海上增量 ≈9.1 天，
    加 15% 港口/班期缓冲 ≈10.5 天（mid 档），强度微调后 high 档 ≈11.8 天。
    """
    mult = STRENGTH_MULT[intent.strength]
    route = ROUTE_TABLE.get(region, ROUTE_TABLE["Suez/Red Sea"])
    detour = route["detour_nm"] if intent.action == "reroute" else 0.0
    sea_days = detour * DAYS_PER_NM                                   # 纯海上增量 ≈9.1 天
    # 缓冲(港口作业/班期调整 15%) × 强度微调(0.5/1.0/1.5 → 0.875/1.0/1.125)
    added_days = sea_days * 1.15 * (0.75 + 0.25 * mult)
    transit_total = route["transit_days_base"] + added_days

    deltas = {"harpex_pct": 0.0, "bdi_pct": 0.0, "baf_pct": 0.0}
    table = FREIGHT_ELASTICITY.get(intent.action)
    if table:
        for k, v in table.items():
            deltas[k] += v * mult
    # 多家承运人同时行动时的叠加在 orchestrator 中处理（取 max + 边际递减）

    air_spillover = 0.0
    if intent.action == "reroute" and added_days > 7:
        air_spillover = min(20.0, added_days * 1.5) * mult   # 高值货等不及 → 转空运

    return RoutingOutcome(
        detour_nm=detour,
        added_days=round(added_days, 2),
        transit_days_total=round(transit_total, 2),
        freight_deltas={k: round(v, 2) for k, v in deltas.items()},
        air_spillover_pct=round(air_spillover, 2),
    )


def selftest() -> int:
    hi = Intent("ship_test", "reroute", "high")
    out = resolve_shipping_intent(hi)
    assert 9.0 <= out.added_days <= 12.0, f"绕航延迟异常: {out.added_days}"
    assert out.freight_deltas["harpex_pct"] == 37.5
    assert out.air_spillover_pct > 0
    lo = Intent("ship_test", "reroute", "low")
    out_lo = resolve_shipping_intent(lo)
    assert out_lo.added_days < out.added_days
    gri = resolve_shipping_intent(Intent("ship_test", "gri", "mid"))
    assert gri.freight_deltas["harpex_pct"] == 15.0 and gri.added_days == 0.0
    print(f"[selftest] 红海绕航(high): +{out.added_days} 天, Harpex {out.freight_deltas['harpex_pct']:+.1f}%, "
          f"空运溢出 {out.air_spillover_pct:+.1f}%")
    print("[selftest] PASS ✔  航运物理结算查表通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
