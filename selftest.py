"""一键自测：python selftest.py

依次运行各模块 --selftest，全部通过返回 0；任一失败返回 1。
"""

from __future__ import annotations

import subprocess
import sys

MODULES = [
    ("pipeline.official_gscpi", "M0 官方管道复现（需求剥离→PCA→标准化）"),
    ("agents.parser", "M1 Step1 信号解析（LLM+规则回退）"),
    ("physics.routing", "M2a 航运物理结算查表"),
    ("physics.pmi_mapping", "M2b PMI 映射查表"),
    ("sim.orchestrator", "M3 编排器（博弈循环+结算+27维合成）"),
    ("sim.assimilation", "M4a 数据同化（分层覆写）"),
    ("backtest.evaluate", "M4b 回测评估基准"),
]


def main() -> int:
    failed = []
    for mod, desc in MODULES:
        print(f"\n{'=' * 66}\n▶ {desc}\n{'=' * 66}")
        r = subprocess.run([sys.executable, "-m", mod, "--selftest"])
        if r.returncode != 0:
            failed.append(mod)
    print(f"\n{'=' * 66}")
    if failed:
        print(f"✘ 失败模块: {failed}")
        return 1
    print("✔ 全部模块自测通过 (7/7)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
