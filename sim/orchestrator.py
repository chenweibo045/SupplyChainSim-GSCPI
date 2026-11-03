"""Step 2/3/4 编排器：多智能体消息循环（真博弈）+ 物理结算 + 矩阵合成。

流程：
  for round in 1..R:
      1. 全体 Agent 感知冲击 / 读取收件箱
      2. 并行决策（定性意图）
      3. 意图广播到黑板（下游 Agent 下一轮感知 → 多轮博弈）
  然后 physics/ 统一结算 → 生成 27 维增量 → 与基期水平合成完整矩阵

离线 demo（红海案例）在合成历史上拟合官方管道，输出预测的 GSCPI 变动
与逐项归因（口径与 NY Fed「中国因素」博文的贡献分解一致）。

运行：python -m sim.orchestrator --demo red_sea
自测：python -m sim.orchestrator --selftest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.base import Agent, Intent, Message, build_default_agents
from agents.parser import Shock, parse_news
from physics import pmi_mapping, routing
from pipeline.official_gscpi import compute_gscpi, make_synthetic_panel, project_gscpi

TRANSPORT_COLS = ["harpex", "bdi", "airfreight_0", "airfreight_1", "airfreight_2", "airfreight_3"]
ECONOMIES = ["china", "us", "ea", "jp", "kr", "tw", "uk"]
PMI_SUFFIX = ["delivery", "backlogs", "purchases"]


@dataclass
class ScenarioResult:
    shocks: list[Shock]
    intents: dict[int, list[Intent]] = field(default_factory=dict)   # round -> intents
    messages: list[Message] = field(default_factory=list)
    freight_deltas: dict = field(default_factory=dict)   # harpex_pct / bdi_pct / baf_pct
    pmi_deltas: pd.DataFrame = None                      # index=经济体, 三分列
    port_queue: str = "none"
    added_days: float = 0.0


def run_scenario(news_items: list[str], llm_call=None, rounds: int = 2,
                 verbose: bool = True) -> tuple[ScenarioResult, list[Agent]]:
    """Step 1→2（多轮博弈）：新闻 → 冲击 → 智能体迭代决策。"""
    shocks = [parse_news(n, llm_call=llm_call) for n in news_items]
    agents = build_default_agents()
    result = ScenarioResult(shocks=shocks)

    for r in range(rounds):
        intents: list[Intent] = []
        for a in agents:
            for s in shocks:
                a.perceive(s)
            intents.append(a.decide(round_no=r))
        result.intents[r] = intents
        # 广播到黑板：所有 Agent 下一轮可读（消息 = 上一轮全体决策）
        for it in intents:
            msg = Message(sender=it.agent_id, action=it.action, strength=it.strength)
            result.messages.append(msg)
            for a in agents:
                if a.agent_id != it.agent_id:
                    a.receive(msg)
        if verbose:
            for it in intents:
                print(f"  [round {r}] {it.agent_id:16s} -> {it.action}({it.strength}): {it.rationale}")
        for a in agents:
            a.reset_round()
    return result, agents


def settle(result: ScenarioResult) -> ScenarioResult:
    """Step 3：物理/运筹结算（定性意图 → 定量变化）。"""
    freight: dict[str, list[float]] = {"harpex_pct": [], "bdi_pct": [], "baf_pct": []}
    pmi_dfs, added_days = [], 0.0

    for r, intents in result.intents.items():
        for it in intents:
            if it.action in routing.FREIGHT_ELASTICITY:
                out = routing.resolve_shipping_intent(
                    it, region=result.shocks[0].target_region if result.shocks else "Suez/Red Sea")
                for k in freight:
                    freight[k].append(out.freight_deltas[k])
                added_days = max(added_days, out.added_days)
            elif it.agent_id.startswith("mfr_"):
                pmi_dfs.append(pmi_mapping.resolve_manufacturing_intent(it, ECONOMIES))
            elif it.agent_id.startswith("port_"):
                pq = pmi_mapping.resolve_port_intent(it)
                if pq["port_queue"] != "none":
                    result.port_queue = pq["port_queue"]
                    freight["harpex_pct"].append(pq["harpex_pct"])

    # 叠加规则：同一指标多家行动 → max + 边际递减 0.6（第二家起）；单次事件封顶 ±60%
    def stack(vals: list[float]) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals, reverse=True)
        total = vals[0] + sum(v * 0.6 for v in vals[1:])
        return float(np.clip(total, -60.0, 60.0))

    result.freight_deltas = {k: round(stack(v), 2) for k, v in freight.items()}
    result.pmi_deltas = pmi_mapping.stack_deltas(pmi_dfs).reindex(ECONOMIES).fillna(0.0)
    result.added_days = added_days
    return result


def synthesize_matrix(result: ScenarioResult, base: pd.Series | None = None) -> pd.Series:
    """Step 4b：将结算增量映射为 27 维标准化残差增量向量 z_delta。

    base: 上一期的 27 维 z 向量（真实历史或同化结果）。增量口径：
      运输类直接取 freight_deltas 的百分比变化（除以 100 作为 z 变化近似）；
      PMI 类取各经济体三分项增量（除以 100）。
    """
    z = pd.Series(0.0, index=base.index if base is not None else
                  TRANSPORT_COLS + [f"{e}_{v}" for e in ECONOMIES for v in PMI_SUFFIX])
    fd = result.freight_deltas
    for col in TRANSPORT_COLS:
        key = "bdi_pct" if col == "bdi" else ("harpex_pct" if col == "harpex" else "baf_pct")
        z[col] = fd.get(key, 0.0) / 100.0
    for e in ECONOMIES:
        if e in result.pmi_deltas.index:
            row = result.pmi_deltas.loc[e]
            z[f"{e}_delivery"] = row["delivery_pct"] / 100.0
            z[f"{e}_backlogs"] = row["backlogs_pct"] / 100.0
            z[f"{e}_purchases"] = row["purchases_pct"] / 100.0
    return z


@dataclass
class DemoOutput:
    gscpi_now: float
    gscpi_forecast: float
    delta: float
    contribution: pd.Series
    result: ScenarioResult


def demo_red_sea(seed: int = 7) -> DemoOutput:
    """端到端离线 demo：合成历史 → 拟合官方管道 → 红海情景 → GSCPI 预测 + 归因。"""
    panel, proxies, _ = make_synthetic_panel(n_months=300, seed=seed)
    cols = TRANSPORT_COLS + [f"{e}_{v}" for e in ECONOMIES for v in PMI_SUFFIX]
    panel = panel[cols]
    resid = pd.DataFrame({c: __import__("pipeline.official_gscpi", fromlist=["purge_demand"])
                          .purge_demand(panel[c], proxies.iloc[:, 0]) for c in cols})
    fitted = compute_gscpi(resid)
    g_now = float(fitted.gscpi.iloc[-1])

    news = ["红海多艘货轮遭武装袭击，苏伊士运河通行量锐减",
            "多家班轮公司宣布暂停红海航线并绕行好望角"]
    result, _ = run_scenario(news, llm_call=None, rounds=2, verbose=False)
    result = settle(result)
    z_delta = synthesize_matrix(result)
    g_next, contrib = project_gscpi(fitted, z_delta)

    print("=" * 70)
    print("SupplyChainSim-GSCPI 红海案例端到端演示（离线合成数据）")
    print("=" * 70)
    print(f"基期 GSCPI: {g_now:+.3f} → 预测下期: {g_next:+.3f}（Δ{g_next - g_now:+.3f}）")
    print(f"航运结算: 绕航延迟 +{result.added_days} 天, 运价变动 {result.freight_deltas}")
    print("归因（各指标对 GSCPI 变动的贡献，标准化单位）:")
    print(contrib.dropna().sort_values(key=abs, ascending=False).head(8).to_string(float_format="%.3f"))
    print("注: 以上为合成数据演示；接入真实 27 序列后口径与官方一致。")
    return DemoOutput(g_now, g_next, g_next - g_now, contrib, result)


def selftest() -> int:
    res, agents = run_scenario(["红海货轮遇袭，苏伊士通行量锐减"], llm_call=None, rounds=2, verbose=False)
    assert len(agents) == 13
    assert any(it.action == "reroute" for it in res.intents[0])
    # 第二轮消息循环后制造商应出现恐慌意图（博弈传导）
    assert any(it.action == "panic_buy" for it in res.intents[1])
    res = settle(res)
    assert res.freight_deltas["harpex_pct"] > 0
    assert (res.pmi_deltas["backlogs_pct"] > 0).any()
    z = synthesize_matrix(res)
    assert z.abs().sum() > 0 and len(z) == 27
    print(f"[selftest] 消息循环 2 轮: 航运绕航 → 制造商恐慌采购传导成功")
    print(f"[selftest] 结算: Harpex {res.freight_deltas['harpex_pct']:+.1f}%, "
          f"绕航 +{res.added_days} 天, 港口 {res.port_queue}")
    print("[selftest] PASS ✔  编排器（博弈循环 + 结算 + 27 维合成）通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", choices=["red_sea"], default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--news", nargs="*", default=None)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.demo == "red_sea":
        demo_red_sea()
    elif args.news:
        res, _ = run_scenario(args.news, rounds=args.rounds, verbose=True)
        res = settle(res)
        print(json.dumps({"freight": res.freight_deltas,
                          "pmi": res.pmi_deltas.to_dict(),
                          "added_days": res.added_days}, ensure_ascii=False, indent=2))
    else:
        print("用法: python -m sim.orchestrator --selftest | --demo red_sea | --news <新闻...>")
