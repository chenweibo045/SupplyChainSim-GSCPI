"""结算层：13 个智能体的状态 → 27 个 GSCPI 子指标 → 标准化残差增量。

这一层把"行为"翻译成"指标"，是整套系统最容易出错、也是旧版 demo 出错的地方。
三个必须遵守的口径：

  (1) 单位：PMI 分项是**扩散指数**（50 为荣枯线，单位=指数点），
      运价类是**价格指数**（基期 100）。物理弹性必须落在各自的自然单位上，
      再除以该序列的历史标准差变成 z 分。绝不能把"百分比变化"直接当 z 分。
  (2) 方向：交货时间指数越高 = 交货越慢 = 压力越大（GSCPI 正向贡献）。
  (3) 月度增量：结算产出的是**当月相对上月的增量**，不是累计冲击。

旧版 demo 的错误示范：把 `harpex_pct / 100` 当作 z 增量、把 `backlogs_pct / 100`
当作 z 增量，导致 Red Sea 情景算出 ΔGSCPI ≈ +3.7（历史最大单月变动只有 +1.40，
疫情两年累计才 +4.4）。本层用 σ 归一 + 全局增益标定修复该问题。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 指标口径表（先验；接入真实 27 序列后必须用样本矩替换）
# --------------------------------------------------------------------------- #
ECONOMIES = ["china", "us", "ea", "jp", "kr", "tw", "uk"]
PMI_PARTS = ["delivery", "backlogs", "purchases"]

TRANSPORT_COLS = ["harpex", "bdi", "air_asia_in", "air_asia_out", "air_eur_in", "air_eur_out"]

# 历史标准差（水平量）：PMI 分项 5~7 点；运价指数 9~32 点
SIGMA_LEVEL = {
    "harpex": 28.0, "bdi": 34.0,
    "air_asia_in": 9.0, "air_asia_out": 8.5, "air_eur_in": 8.0, "air_eur_out": 7.5,
    "delivery": 6.2, "backlogs": 7.0, "purchases": 5.8,
}
AIR_BASE = {"air_asia_in": 100.0, "air_asia_out": 96.0,
            "air_eur_in": 93.0, "air_eur_out": 90.0}
MEAN_LEVEL = {"harpex": 100.0, "bdi": 100.0, **AIR_BASE,
              "delivery": 50.0, "backlogs": 50.0, "purchases": 50.0}

# 因子载荷先验（接入真实数据后由官方管道估计并冻结）
LOADING_PRIOR = {
    "harpex": 0.245, "bdi": 0.190,
    "air_asia_in": 0.215, "air_asia_out": 0.205,
    "air_eur_in": 0.210, "air_eur_out": 0.195,
    "delivery": 0.215, "backlogs": 0.180, "purchases": 0.150,
}
SCALE_PRIOR = 1.0        # 主成分得分标准差（由载荷向量近似给出，见 calibrate）


def all_columns() -> list[str]:
    return TRANSPORT_COLS + [f"{e}_{p}" for e in ECONOMIES for p in PMI_PARTS]


def loadings_series() -> pd.Series:
    vals = {}
    for c in TRANSPORT_COLS:
        vals[c] = LOADING_PRIOR[c]
    for e in ECONOMIES:
        for p in PMI_PARTS:
            vals[f"{e}_{p}"] = LOADING_PRIOR[p]
    # 归一化到单位长度（与 PCA 载荷一致）
    s = pd.Series(vals)
    return s / np.sqrt((s ** 2).sum())


@dataclass
class SettlementParams:
    gain_freight: float = 0.30   # 运输成本通道增益（可校准）
    gain_pmi: float = 0.30       # PMI 通道增益（可校准）
    air_per_intensity: float = 0.30
    bdi_share_of_spot: float = 0.35


DEFAULT_PARAMS = SettlementParams()


# --------------------------------------------------------------------------- #
# 状态 → 27 个原始指标
# --------------------------------------------------------------------------- #
def indicators_from_state(sim) -> pd.Series:
    """从仿真器当前状态读出 27 个子指标的**水平量**（自然单位）。"""
    idx = all_columns()
    out = pd.Series(index=idx, dtype=float)

    # --- 运输成本：承运人即期运价（运力加权）→ Harpex / BDI / 空运价 ---
    w = np.array([c.weight for c in sim.carriers], dtype=float)
    spot = np.array([c.spot_rate for c in sim.carriers], dtype=float)
    util = np.array([c.utilization for c in sim.carriers], dtype=float)
    harpex = float(np.average(spot, weights=w))
    out["harpex"] = harpex
    # 干散货：与集装箱部分相关，弹性更低
    out["bdi"] = 100.0 * ((harpex / 100.0) ** DEFAULT_PARAMS.bdi_share_of_spot) * \
        (1.0 + 0.10 * (float(np.average(util, weights=w)) - 0.82))
    # 空运：由制造商的"转空运"压力驱动（air_stress 在 dynamics 中累计）
    air = 1.0 + DEFAULT_PARAMS.air_per_intensity * getattr(sim, "air_stress", 0.0)
    for c, base in AIR_BASE.items():
        out[c] = base * air

    # --- PMI 三分项：直接取制造企业状态 ---
    for m in sim.mfrs:
        out[f"{m.economy}_delivery"] = m.delivery_idx
        out[f"{m.economy}_backlogs"] = m.backlog_idx
        out[f"{m.economy}_purchases"] = m.purchases_idx
    return out


def baseline_indicators() -> pd.Series:
    s = pd.Series(index=all_columns(), dtype=float)
    for c in TRANSPORT_COLS:
        s[c] = MEAN_LEVEL[c]
    for e in ECONOMIES:
        for p in PMI_PARTS:
            s[f"{e}_{p}"] = MEAN_LEVEL[p]
    return s


def delta_z(prev: pd.Series, cur: pd.Series) -> pd.Series:
    """水平量增量 → 标准化增量（Δz = Δx / σ_x）。"""
    dz = pd.Series(index=cur.index, dtype=float)
    for c in cur.index:
        part = c.split("_", 1)[1] if "_" in c else c
        sigma = SIGMA_LEVEL.get(part, SIGMA_LEVEL.get(c))
        dz[c] = (cur[c] - prev[c]) / sigma
    return dz


def gscpi_delta(dz: pd.Series, params: SettlementParams = DEFAULT_PARAMS):
    """ΔGSCPI = Σ_i g_c(i) · ω_i Δz_i，返回 (ΔG, 逐指标贡献)。

    g_c(i) 按通道分组：运输成本 6 项用 gain_freight，21 个 PMI 分项用 gain_pmi。
    """
    w = loadings_series() / SCALE_PRIOR
    common = [c for c in dz.index if c in w.index]
    g = pd.Series(index=common, dtype=float)
    for c in common:
        g[c] = params.gain_freight if c in TRANSPORT_COLS else params.gain_pmi
    contrib = w[common] * dz[common].fillna(0.0) * g
    return float(contrib.sum()), contrib


def gscpi_parts(dz: pd.Series) -> tuple[float, float, pd.Series]:
    """把 ΔGSCPI 拆成两条通道（单位增益下的部分和），供线性校准使用。

    ΔG = gain_freight · F + gain_pmi · P
    返回 (F, P, 逐指标单位增益贡献)。
    """
    w = loadings_series() / SCALE_PRIOR
    common = [c for c in dz.index if c in w.index]
    unit = w[common] * dz[common].fillna(0.0)
    is_freight = pd.Series([c in TRANSPORT_COLS for c in common], index=common)
    F = float(unit[is_freight].sum())
    P = float(unit[~is_freight].sum())
    return F, P, unit
