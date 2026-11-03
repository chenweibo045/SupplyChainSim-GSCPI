"""管道对标：将复现的 GSCPI 与纽约联储官方发布值对比。

验收标准（M0 里程碑）：相关系数 ≥ 0.95。

用法：
    python -m pipeline.validate --ours data/processed/gscpi_reproduced.csv \
                                --official data/raw/gscpi_official.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def load_series(path: str, col: str | None = None) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df[col] if col else df.iloc[:, 0]
    return s.astype(float).sort_index()


def compare(ours: pd.Series, official: pd.Series) -> dict:
    df = pd.concat({"ours": ours, "official": official}, axis=1).dropna()
    if len(df) < 12:
        raise ValueError(f"重叠样本仅 {len(df)} 个月，不足以对标（需要真实数据）")
    corr = float(df["ours"].corr(df["official"]))
    diff = df["ours"] - df["official"]
    return {
        "n_months": int(len(df)),
        "correlation": corr,
        "rmse": float(np.sqrt((diff**2).mean())),
        "mae": float(diff.abs().mean()),
        "mean_bias": float(diff.mean()),
        "pass_095": bool(corr >= 0.95),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--ours-col", default=None)
    ap.add_argument("--official-col", default=None)
    args = ap.parse_args()
    report = compare(load_series(args.ours, args.ours_col), load_series(args.official, args.official_col))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass_095"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
