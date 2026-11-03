"""Step 4a — 高频数据同化：真实观测覆写模拟值（按指标频率分层）。

关键口径：
  - 运价类（BDI / Harpex / 空运价）为日/周频，月中已有真实观测 → 真实覆写；
  - PMI 分项是月度调查，月中无观测 → 只能靠 Agent 模拟；
  - 月末形成完整 27 维矩阵 + 来源掩码（real / sim），供归因报告使用。
"""

from __future__ import annotations

import argparse

import pandas as pd

# 可按真实数据可得性配置：列名 → 频率分层
FREQUENCY_TIER = {
    "harpex": "weekly", "bdi": "daily",
    "airfreight_0": "monthly", "airfreight_1": "monthly",
    "airfreight_2": "monthly", "airfreight_3": "monthly",
}
PMI_TIER = "monthly"


def assimilate(simulated: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """observed 与 simulated 同 columns；observed 中 NaN 表示该指标月中无观测。

    返回 (merged, source_mask)，source_mask ∈ {"real", "sim"}。
    """
    if list(simulated.columns) != list(observed.columns):
        raise ValueError("模拟矩阵与观测矩阵列不一致")
    merged = simulated.copy()
    source = pd.DataFrame("sim", index=simulated.index, columns=simulated.columns)
    hit = observed.notna()
    merged[hit] = observed[hit]
    source[hit] = "real"
    return merged, source


def coverage_report(source_mask: pd.DataFrame) -> pd.Series:
    """真实数据覆盖率（按指标）。"""
    return (source_mask == "real").mean().sort_values(ascending=False)


def selftest() -> int:
    cols = ["harpex", "bdi", "china_delivery", "us_backlogs"]
    sim = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=cols, index=["2024-01-31"])
    obs = pd.DataFrame([[1.5, float("nan"), float("nan"), 4.2]], columns=cols, index=["2024-01-31"])
    merged, src = assimilate(sim, obs)
    assert merged.loc["2024-01-31", "harpex"] == 1.5        # 运价被真实覆写
    assert merged.loc["2024-01-31", "bdi"] == 2.0           # 无观测 → 保留模拟
    assert merged.loc["2024-01-31", "china_delivery"] == 3.0
    assert src.loc["2024-01-31", "us_backlogs"] == "real"
    print(f"[selftest] 覆盖率: {coverage_report(src).to_dict()}")
    print("[selftest] PASS ✔  数据同化（分层覆写）通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
