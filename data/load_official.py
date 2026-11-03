"""官方 GSCPI 数据加载器：解析纽约联储 xlsx → 标准 CSV（Date 索引 × GSCPI 列）。

用法：
    python data/load_official.py                    # 解析 data/raw/gscpi_official.xlsx
    python data/load_official.py --xlsx <路径>       # 指定其他 xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "raw"
PROCESSED = Path(__file__).parent / "processed"


def load_official_xlsx(path: Path) -> pd.Series:
    """解析纽约联储 GSCPI xlsx 的「GSCPI Monthly Data」表。"""
    df = pd.read_excel(path, sheet_name="GSCPI Monthly Data", header=None)
    # 数据区：第 0 行为表头（Date / GSCPI），第 5 行起为数据
    body = df.iloc[5:, :2].copy()
    body.columns = ["Date", "GSCPI"]
    body = body.dropna(subset=["Date"])
    s = pd.Series(body["GSCPI"].astype(float).values,
                  index=pd.to_datetime(body["Date"]), name="GSCPI")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, default=str(RAW / "gscpi_official.xlsx"))
    args = ap.parse_args()
    src = Path(args.xlsx)
    if not src.exists():
        print(f"找不到 {src}，先运行 python data/download_data.py")
        return 1
    s = load_official_xlsx(src)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED / "gscpi_official.csv"
    s.to_csv(dest)
    print(f"官方 GSCPI: {s.index[0]:%Y-%m} ~ {s.index[-1]:%Y-%m}（{len(s)} 个月）")
    print(f"最新读数 {s.iloc[-1]:+.3f}（{s.index[-1]:%Y-%m}） | 历史峰值 {s.max():+.3f}（{s.idxmax():%Y-%m}）")
    print(f"已写出: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
