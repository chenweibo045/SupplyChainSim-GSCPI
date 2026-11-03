"""数据获取脚本：尽力下载官方 GSCPI 发布值，其余序列给出获取指引。

官方 GSCPI（纽约联储，每月第 4 个工作日发布上月值）：
  下载页 https://www.newyorkfed.org/research/policy/gscpi
  直接数据文件（xlsx）见下方 URL 候选列表，任一下载成功即可。

注意：27 个底层分项（BDI/Harpex/BLS 空运价 + 7 经济体 PMI 三分项）多为
商业数据（Haver / Bloomberg / S&P Global / ISM）。学生项目可用代理序列
并在论文中披露口径差异；本项目管道本身不依赖特定来源，只要提供
「日期索引 × 27 列」的月度表即可。
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "raw"

GSCPI_XLSX_CANDIDATES = [
    "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx",
    "https://www.newyorkfed.org/medialibrary/Interactives/gscpi/downloads/gscpi_data.xlsx",
]

SOURCES_GUIDE = """\
# 27 个子指标获取指引

| 组 | 指标 | 来源 | 可得性 |
|---|---|---|---|
| 海运 | BDI（波罗的海干散货） | Baltic Exchange / Investing.com / 部分数据商 | 公开可查 |
| 海运 | Harpex（集装箱租船） | Harper Petersen（官网发布） | 公开可查 |
| 空运 | 4 个 BLS 进出口空运价格指数 | 美国劳工统计局 BLS | 公开免费 |
| PMI | 7 经济体 × (交货时间/积压订单/采购库存) | S&P Global / ISM 制造业 PMI 分项 | 付费（Haver/Bloomberg） |

付费分项的降级方案（MVP 用，需披露）：
  - 用官方 GSCPI 发布值作为验证锚（本脚本可下载）；
  - 分项用 FRED 等免费库中的货运/库存/交付类代理序列替代；
  - 中国分项可用财新/官方制造业 PMI 扩散指数近似。
"""


def try_download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  失败 {url}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true", help="只生成获取指引")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "SOURCES.md").write_text(SOURCES_GUIDE, encoding="utf-8")
    print("已写出 data/raw/SOURCES.md（27 子指标获取指引）")
    if args.skip_download:
        return 0
    dest = RAW / "gscpi_official.xlsx"
    for url in GSCPI_XLSX_CANDIDATES:
        if try_download(url, dest):
            print(f"官方 GSCPI 已下载: {dest}")
            return 0
    print("官方 xlsx 直链不可用，请手动访问 "
          "https://www.newyorkfed.org/research/policy/gscpi 下载后放入 data/raw/")
    return 1


if __name__ == "__main__":
    sys.exit(main())
