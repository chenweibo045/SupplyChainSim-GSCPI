"""回测评估：与 baseline 对比，验证 MAS 预测的价值。

baseline 集合（缺一不可以下结论）：
  - naive   : 下期 = 本期（朴素延续）
  - drift   : 近 3 期平均变动外推
  - arima   : ARIMA(1,1,1)（statsmodels 可用时）
指标：RMSE / MAE / 方向准确率（涨跌幅符号命中）。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    rmse: float
    mae: float
    dir_acc: float   # 方向准确率（1 = 全对，0.5 = 抛硬币）


def evaluate(pred: pd.Series, actual: pd.Series) -> Metrics:
    df = pd.concat({"p": pred, "a": actual}, axis=1).dropna()
    err = df["p"] - df["a"]
    sign_ok = (np.sign(df["p"].diff()) == np.sign(df["a"].diff())).mean()
    return Metrics(rmse=float(np.sqrt((err**2).mean())),
                   mae=float(err.abs().mean()),
                   dir_acc=float(sign_ok))


def baseline_naive(g: pd.Series) -> pd.Series:
    return g.shift(1)


def baseline_drift(g: pd.Series, window: int = 3) -> pd.Series:
    return g.shift(1) + g.diff().rolling(window).mean()


def baseline_arima(g: pd.Series, burn_in: int = 60) -> pd.Series:
    """滚动一步前瞻 ARIMA(1,1,1)：每月只用当月之前的数据预测下月（诚实的外推基准）。"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return pd.Series(np.nan, index=g.index)
    vals = g.dropna()
    if len(vals) < burn_in + 5:
        return pd.Series(np.nan, index=g.index)
    pred = pd.Series(np.nan, index=vals.index, dtype=float)
    for t in range(burn_in, len(vals)):
        try:
            model = ARIMA(vals.iloc[:t], order=(1, 1, 1)).fit()
            pred.iloc[t] = float(model.forecast(steps=1).iloc[0])
        except Exception:  # noqa: BLE001
            pred.iloc[t] = np.nan
    return pred


def run_leaderboard(gscpi: pd.Series, ours: pd.Series | None = None) -> pd.DataFrame:
    """gscpi: 官方（或复现）指数序列；ours: MAS 预测序列（可逐步/逐月对齐）。"""
    rows = {}
    for name, fn in [("naive", baseline_naive), ("drift3", baseline_drift), ("arima111", baseline_arima)]:
        m = evaluate(fn(gscpi), gscpi)
        rows[name] = asdict(m)
    if ours is not None:
        rows["ours_mas"] = asdict(evaluate(ours, gscpi))
    return pd.DataFrame(rows).T.round(4)


def selftest() -> int:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    g = pd.Series(np.cumsum(rng.normal(0, 0.5, 60)), index=idx)
    board = run_leaderboard(g)
    assert board.loc["naive", "rmse"] > 0
    assert board.loc["naive", "dir_acc"] >= 0.4
    print("[selftest] 基准排行榜（合成序列）:")
    print(board.to_string())
    print("[selftest] PASS ✔  回测评估模块通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gscpi", type=str, default=None, help="官方 GSCPI csv（含日期索引）")
    ap.add_argument("--ours", type=str, default=None, help="MAS 预测 csv")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.gscpi:
        g = pd.read_csv(args.gscpi, index_col=0, parse_dates=True).iloc[:, 0]
        ours = pd.read_csv(args.ours, index_col=0, parse_dates=True).iloc[:, 0] if args.ours else None
        print(run_leaderboard(g, ours).to_string())
    else:
        print("用法: python -m backtest.evaluate --selftest | --gscpi <官方.csv> [--ours <预测.csv>]")
