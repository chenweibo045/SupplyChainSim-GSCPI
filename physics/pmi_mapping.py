"""Step 3b — 制造/港口意图 → PMI 分项增量结算（查表 + 强度档位）。

对应 GSCPI 的 21 个 PMI 分项（7 经济体 × 交货时间/积压订单/采购库存）
与港口拥堵代理变量。强度档位 low/mid/high = 0.5/1.0/1.5 倍乘。
数值为文献标定的方向性弹性，论文中需做敏感性披露。
"""

from __future__ import annotations

import argparse

import pandas as pd

from agents.base import Intent, STRENGTH_MULT

# 每单位强度下，意图对受影响经济体 PMI 分项的百分比冲击（方向性标定）
INTENT_TO_PMI = {
    "panic_buy":     {"backlogs_pct": +12.0, "delivery_pct": +8.0,  "purchases_pct": +10.0},
    "destock":       {"backlogs_pct": -6.0,  "delivery_pct": -3.0,  "purchases_pct": -8.0},
    "reroute":       {"delivery_pct": +6.0},                        # 延迟经制造端感知传导
    "queue_mode":    {"delivery_pct": +4.0, "backlogs_pct": +3.0},  # 港口排队 → 交付延迟
    "surge_shift":   {"delivery_pct": -2.0, "backlogs_pct": -1.0},
    "gri":           {"purchases_pct": +2.0},                       # 涨价诱发提前采购
    "slow_steaming": {"delivery_pct": +1.5},
    "normal":        {},
}

# 港口拥堵对「运输成本类指标」的附加影响（集装箱船等待天数 → 租船价）
PORT_QUEUE_FREIGHT = {"mid": {"harpex_pct": +5.0}, "high": {"harpex_pct": +10.0}}

ECONOMY_OF_AGENT = {"mfr_china": "china", "mfr_us": "us", "mfr_ea": "ea", "mfr_jp": "jp",
                    "mfr_kr": "kr", "mfr_tw": "tw", "mfr_uk": "uk"}


def resolve_manufacturing_intent(intent: Intent, economies: list[str]) -> pd.DataFrame:
    """输出: DataFrame[index=经济体, columns=delivery_pct/backlogs_pct/purchases_pct]。"""
    table = INTENT_TO_PMI.get(intent.action, {})
    mult = STRENGTH_MULT[intent.strength]
    rows = {e: {k: table.get(k, 0.0) * mult for k in
                ("delivery_pct", "backlogs_pct", "purchases_pct")} for e in economies}
    return pd.DataFrame.from_dict(rows, orient="index")


def resolve_port_intent(intent: Intent) -> dict:
    """港口意图 → 拥堵代理 + 运输成本附加冲击。"""
    if intent.action == "queue_mode":
        return {"port_queue": intent.strength,
                **{k: v * STRENGTH_MULT[intent.strength]
                   for k, v in PORT_QUEUE_FREIGHT[intent.strength].items()}}
    return {"port_queue": "none", "harpex_pct": 0.0}


def stack_deltas(deltas: list[pd.DataFrame]) -> pd.DataFrame:
    """多 Agent 同类意图叠加：同经济体取 max + 边际递减（第二家及以后 ×0.6）。"""
    if not deltas:
        return pd.DataFrame(columns=["delivery_pct", "backlogs_pct", "purchases_pct"])
    combined: dict = {}
    for df in deltas:
        for e in df.index:
            if e not in combined:
                combined[e] = df.loc[e].copy()
            else:
                combined[e] += df.loc[e] * 0.6   # 边际递减
    return pd.DataFrame.from_dict(combined, orient="index")


def selftest() -> int:
    eco = ["china", "us", "ea", "jp", "kr", "tw", "uk"]
    df_hi = resolve_manufacturing_intent(Intent("mfr_ea", "panic_buy", "high"), eco)
    assert df_hi.loc["ea", "backlogs_pct"] == 18.0   # 12 × 1.5
    stacked = stack_deltas([df_hi, df_hi])
    assert stacked.loc["ea", "backlogs_pct"] == 18.0 + 18.0 * 0.6
    pq = resolve_port_intent(Intent("port_uswc", "queue_mode", "high"))
    assert pq["harpex_pct"] == 15.0 and pq["port_queue"] == "high"
    normal = resolve_manufacturing_intent(Intent("mfr_us", "normal", "low"), eco)
    assert (normal == 0).all().all()
    print(f"[selftest] 恐慌采购(high): 积压订单 {df_hi.loc['ea','backlogs_pct']:+.1f}%, "
          f"交货时间 {df_hi.loc['ea','delivery_pct']:+.1f}%")
    print("[selftest] PASS ✔  PMI 映射查表通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
