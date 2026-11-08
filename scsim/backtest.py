"""回测与评估（v2）—— 诚实口径。

三个必须修正的口径问题（旧版 demo 全部踩中）：
  1) look-ahead：drift 基准用 `g.diff().rolling(3)` 在 t 时点包含了 g_t，实际不可得。
     修正为 `g.diff().shift(1).rolling(3)`。
  2) 方向准确率：旧版比较 sign(Δpred_t) 与 sign(Δactual_t)，两者错位一期。
     正确口径是 sign(pred_t - g_{t-1}) vs sign(g_t - g_{t-1})。
  3) 评估对象：全样本 RMSE 会掩盖事件期表现；必须单列"事件窗口"与"大变动月份"。

预测目标定义（必须写进论文）：
  h=1：以 t 月末已发布数据为信息集，预测 t+1 月 GSCPI（发布滞后 1 个月）。
  h=3：同一信息集直接预测 t+3（不做滚动路径拼接）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "gscpi_official.csv"


def load_gscpi(path: Path | str | None = None) -> pd.Series:
    p = Path(path) if path else DATA
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    return s.astype(float).sort_index()


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    n: int
    rmse: float
    mae: float
    mase: float
    dir_all: float | None
    dir_big: float | None

    def as_dict(self):
        return {"n": self.n, "rmse": round(self.rmse, 4), "mae": round(self.mae, 4),
                "mase": round(self.mase, 4),
                "dir_all": None if self.dir_all is None else round(self.dir_all, 4),
                "dir_big": None if self.dir_big is None else round(self.dir_big, 4)}


def evaluate(pred: pd.Series, actual: pd.Series, big: float = 0.3) -> Metrics:
    df = pd.concat({"p": pred, "a": actual, "last": actual.shift(1)}, axis=1).dropna()
    err = df["p"] - df["a"]
    dp = df["p"] - df["last"]
    da = df["a"] - df["last"]
    nz = dp.abs() > 1e-9
    m = da.abs() >= 0.05
    dir_all = float(((np.sign(dp[m]) == np.sign(da[m])) & nz[m]).mean()) if nz[m].any() else None
    big_m = da.abs() >= big
    dir_big = float(((np.sign(dp[big_m]) == np.sign(da[big_m])) & nz[big_m]).mean()) if nz[big_m].any() else None
    naive_mae = float((actual - actual.shift(1)).abs().mean())
    return Metrics(len(df), float(np.sqrt((err ** 2).mean())), float(err.abs().mean()),
                   float(err.abs().mean() / naive_mae), dir_all, dir_big)


# --------------------------------------------------------------------------- #
# 基准
# --------------------------------------------------------------------------- #
def baselines(g: pd.Series) -> dict[str, pd.Series]:
    out = {
        "naive": g.shift(1),
        "drift3": g.shift(1) + g.diff().shift(1).rolling(3).mean(),
        "meanrev_0.9": g.shift(1) * 0.9,
        # 旧实现，仅用于复现"drift3 最优"这个假象（含未来信息，不可上线）
        "drift3_lookahead_DO_NOT_USE": g.shift(1) + g.diff().rolling(3).mean(),
    }
    # 差分 AR(1)，扩张窗
    d = g.diff()
    ar = pd.Series(np.nan, index=g.index)
    for t in range(72, len(g)):
        y = d.iloc[:t].dropna().values
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        beta, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
        ar.iloc[t] = g.iloc[t - 1] + beta[0] + beta[1] * y[-1]
    out["ar1_diff"] = ar
    return out


def leaderboard(g: pd.Series) -> pd.DataFrame:
    rows = {k: evaluate(p, g).as_dict() for k, p in baselines(g).items()}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# 事件窗口：机制模型的部分和（线性于两个增益）
# --------------------------------------------------------------------------- #
def event_parts(events, months: int = 3, rounds: int = 2) -> dict[str, dict]:
    """对每个事件模拟 H 个月，返回 (F_m, P_m) 与"共识情景"的贡献分解。

    ΔĜ_m = gain_freight·F_m + gain_pmi·P_m   —— 对增益严格线性。
    """
    from .dynamics import Simulation
    from . import settle

    out = {}
    for ev in events:
        sim = Simulation()
        prev = settle.baseline_indicators()
        Fs, Ps, parts = [], [], []
        for _ in range(months):
            sim.step(ev.shock, rounds=rounds)
            cur = settle.indicators_from_state(sim)
            dz = settle.delta_z(prev, cur)
            F, P, unit = settle.gscpi_parts(dz)
            Fs.append(F)
            Ps.append(P)
            parts.append(unit)
            prev = cur
        out[ev.key] = {"event": ev, "F": Fs, "P": Ps,
                       "cum_F": list(np.cumsum(Fs)), "cum_P": list(np.cumsum(Ps)),
                       "unit_parts": parts,
                       "final_state": sim.snapshot()}
    return out


def event_actual(g: pd.Series, events, months: int = 3) -> dict[str, list[float]]:
    """事件月的实际累计 ΔGSCPI 路径（相对事件月水平）。"""
    out = {}
    for ev in events:
        t = pd.Timestamp(ev.month)
        if t not in g.index:
            continue
        i = g.index.get_loc(t)
        base = g.iloc[i]
        path = []
        for h in range(1, months + 1):
            path.append(float(g.iloc[i + h] - base) if i + h < len(g) else np.nan)
        out[ev.key] = path
    return out


def fit_gains(parts: dict, actual: dict, keys: list[str],
              grid=np.arange(0.0, 1.51, 0.05)) -> tuple[float, float, float]:
    """在给定事件集合上网格搜索 (gain_freight, gain_pmi)，最小化累计路径 SSE。"""
    best = (0.0, 0.0, np.inf)
    for gf in grid:
        for gp in grid:
            sse = 0.0
            for k in keys:
                if k not in actual:
                    continue
                pred = np.array(parts[k]["cum_F"]) * gf + np.array(parts[k]["cum_P"]) * gp
                act = np.array(actual[k], dtype=float)
                sse += float(np.nansum((pred - act) ** 2))
            if sse < best[2]:
                best = (float(gf), float(gp), sse)
    return best


def loo_events(parts: dict, actual: dict, grid=np.arange(0.0, 1.51, 0.05)):
    """留一事件：用其余事件校准增益，预测留出事件的 3 个月路径。"""
    rows = []
    keys = [k for k in actual if k in parts and not np.isnan(actual[k]).all()]
    for k in keys:
        train = [x for x in keys if x != k]
        gf, gp, _ = fit_gains(parts, actual, train, grid)
        pred = np.array(parts[k]["cum_F"]) * gf + np.array(parts[k]["cum_P"]) * gp
        act = np.array(actual[k], dtype=float)
        rows.append({
            "event": k, "label": parts[k]["event"].label,
            "gain_freight": round(float(gf), 3), "gain_pmi": round(float(gp), 3),
            "pred": [round(float(x), 3) for x in pred],
            "actual": [round(float(x), 3) for x in act],
            "rmse": round(float(np.sqrt(np.nanmean((pred - act) ** 2))), 3),
            "naive_rmse": round(float(np.sqrt(np.nanmean(act ** 2))), 3),
        })
    return pd.DataFrame(rows)


def hybrid_series(g: pd.Series, parts: dict, actual_keys, gf: float, gp: float,
                  months: int = 3) -> pd.Series:
    """naive + 事件叠加：无事件月 Δ=0（等价 naive），事件月用机制模型增量。"""
    delta = pd.Series(0.0, index=g.index)
    for k in actual_keys:
        t = pd.Timestamp(parts[k]["event"].month)
        if t not in g.index:
            continue
        i = g.index.get_loc(t)
        for h in range(months):
            if i + h + 1 >= len(g):
                break
            d = gf * parts[k]["F"][h] + gp * parts[k]["P"][h]
            delta.iloc[i + h + 1] += d
    return g.shift(1) + delta


def dm_test(e1: pd.Series, e2: pd.Series) -> float:
    """Diebold-Mariano（h=1，平方损失，HAC 方差）。"""
    d = (e1 ** 2 - e2 ** 2).dropna()
    n = len(d)
    mu = d.mean()
    se = np.sqrt(((d - mu) ** 2).mean() / n)
    return float(mu / se) if se > 0 else 0.0
