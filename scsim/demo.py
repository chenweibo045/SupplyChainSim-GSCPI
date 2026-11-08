"""SupplyChainSim-GSCPI v2 —— 可复现 Demo。

用法：
    python -m scsim.demo                 # 全部
    python -m scsim.demo --selftest      # 只跑自测
    python -m scsim.demo --scenario red_sea --months 3

产出：
    results/leaderboard.csv        诚实基线排行榜（真实官方 GSCPI）
    results/event_loo.csv          留一事件评估
    results/red_sea.json           红海情景的完整轨迹、归因、消息日志
    results/calibration.json       增益标定与分布检验
    results/fig_*.png              论文用图
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate, backtest as bt, events, settle
from .actions import ALL_ACTIONS, action_space
from .dynamics import Shock, Simulation
from .profiles import ECONOMY_LABEL, build_all

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1) 官方管道自测（合成数据）
# --------------------------------------------------------------------------- #
def pipeline_check(seed: int = 7) -> dict:
    panel, proxy, f_true = aggregate.make_synthetic_panel(seed=seed)
    resid = pd.DataFrame({c: aggregate.purge_demand(panel[c], proxy) for c in panel.columns})
    res = aggregate.compute_gscpi(resid)
    corr = float(np.corrcoef(res.gscpi.values, f_true.values)[0, 1])
    return {"corr_with_true_factor": round(corr, 4),
            "explained_var": round(res.explained, 4),
            "n_months": len(panel)}


# --------------------------------------------------------------------------- #
# 2) 单情景仿真
# --------------------------------------------------------------------------- #
def run_scenario(shock: Shock, months: int = 3, rounds: int = 2,
                 gains: tuple[float, float] = (0.30, 0.30)):
    sim = Simulation()
    prev = settle.baseline_indicators()
    path, contribs, intents, snaps = [], [], [], []
    for _ in range(months):
        acts = sim.step(shock, rounds=rounds)
        snaps.append(sim.snapshot())
        cur = settle.indicators_from_state(sim)
        dz = settle.delta_z(prev, cur)
        params = settle.SettlementParams(gain_freight=gains[0], gain_pmi=gains[1])
        dg, contrib = settle.gscpi_delta(dz, params)
        path.append(dg)
        contribs.append(contrib)
        intents.append([a.to_dict() for a in acts])
        prev = cur
    return {"sim": sim, "path": path, "contribs": contribs, "intents": intents,
            "levels": settle.indicators_from_state(sim), "snapshots": snaps}


def attribution_by_agent(shock: Shock, months: int = 3, gains=(0.30, 0.30)) -> pd.DataFrame:
    """反事实归因：逐个禁用智能体，看 ΔGSCPI 变化多少（边际贡献）。

    为什么用反事实而不是解析式：动作之间有交互（绕航 → 制造商恐慌 → 港口拥堵），
    线性归因会漏掉交互项。这里报告边际贡献 + 交互残差，口径透明。
    """
    base = run_scenario(shock, months, gains=gains)
    total = float(sum(base["path"]))
    rows = []
    for a in build_all():
        sim = Simulation()
        sim.disabled = {a.agent_id}
        prev = settle.baseline_indicators()
        tot = 0.0
        for _ in range(months):
            sim.step(shock, rounds=2)
            cur = settle.indicators_from_state(sim)
            dz = settle.delta_z(prev, cur)
            tot += settle.gscpi_delta(
                dz, settle.SettlementParams(*gains))[0]
            prev = cur
        rows.append({"agent": a.agent_id, "type": a.kind,
                     "marginal": round(total - tot, 4)})
    df = pd.DataFrame(rows)
    df["share"] = (df["marginal"] / total).round(3) if total else 0.0
    residual = round(total - df["marginal"].sum(), 4)
    return df.sort_values("marginal", ascending=False), total, residual


# --------------------------------------------------------------------------- #
# 3) 事件回测
# --------------------------------------------------------------------------- #
def event_backtest() -> tuple[pd.DataFrame, dict]:
    g = bt.load_gscpi()
    parts = bt.event_parts(events.EVENTS, months=3)
    actual = bt.event_actual(g, events.EVENTS, months=3)
    gf, gp, sse = bt.fit_gains(parts, actual, list(actual))
    loo = bt.loo_events(parts, actual)
    # 全样本混合模型（无事件月 = naive）
    hyb = bt.hybrid_series(g, parts, list(actual), 0.30, 0.30)
    full = bt.leaderboard(g)
    full.loc["hybrid_mas(g=.3,.3)"] = bt.evaluate(hyb, g).as_dict()
    return loo, {"in_sample_gains": [gf, gp], "in_sample_sse": round(sse, 3),
                 "leaderboard": full}


# --------------------------------------------------------------------------- #
# 4) 增益标定：机制模型的灵敏度 vs 历史波动率
# --------------------------------------------------------------------------- #
def calibration(n: int = 240, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    regions = ["Suez/Red Sea", "EastAsia", "Europe", "USWC", "Panama", "Generic"]
    channels = ["canal_transit", "input_supply", "port_capacity"]
    Fs, Ps = [], []
    for _ in range(n):
        reg = regions[rng.integers(len(regions))]
        ch = channels[rng.integers(len(channels))]
        sev = float(rng.uniform(0.2, 1.0))
        sh = Shock(reg, ch, sev, -0.3, 3)
        sim = Simulation()
        prev = settle.baseline_indicators()
        sim.step(sh, rounds=2)
        cur = settle.indicators_from_state(sim)
        F, P, _ = settle.gscpi_parts(settle.delta_z(prev, cur))
        Fs.append(F)
        Ps.append(P)
    g = bt.load_gscpi()
    sd_hist = float(g.diff().std())
    sd_unit = float(np.std(np.array(Fs) + np.array(Ps)))
    implied = sd_hist / sd_unit if sd_unit > 0 else np.nan
    # 以单位增益的分通道波动做"灵敏度比"
    return {
        "n_sim": n,
        "hist_sd_delta": round(sd_hist, 4),
        "hist_p95_abs_delta": round(float(g.diff().abs().quantile(0.95)), 4),
        "sim_sd_unit_gain": round(sd_unit, 4),
        "implied_common_gain": round(float(implied), 4),
        "note": "implied_common_gain = 历史σ(ΔGSCPI) / 单位增益下模拟σ(ΔGSCPI)",
    }


# --------------------------------------------------------------------------- #
# 5) 主程序
# --------------------------------------------------------------------------- #
def selftest() -> int:
    chk = pipeline_check()
    assert chk["corr_with_true_factor"] > 0.95, chk
    sim = Simulation()
    acts = sim.step(Shock("Suez/Red Sea", "canal_transit", 0.85, -0.4, 3, ("asia_europe",)))
    ids = {a.agent_id for a in acts}
    assert "carrier_aea" in ids
    assert any(ALL_ACTIONS[a.action].name == "reroute" for a in acts)
    cur = settle.indicators_from_state(sim)
    assert len(cur) == 27
    dz = settle.delta_z(settle.baseline_indicators(), cur)
    dg, contrib = settle.gscpi_delta(dz)
    assert abs(dg) < 2.0, f"ΔGSCPI 量级异常: {dg}"
    df, tot, res = attribution_by_agent(
        Shock("Suez/Red Sea", "canal_transit", 0.85, -0.4, 3, ("asia_europe",)), months=1)
    assert len(df) == 13
    print(f"[selftest] 官方管道 corr={chk['corr_with_true_factor']} (合成)")
    print(f"[selftest] 单月 ΔGSCPI={dg:+.3f}，13 智能体反事实归因完成（残差 {res}）")
    print("[selftest] PASS ✔ v2 引擎自测通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scenario", default="red_sea")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    print("=" * 74)
    print("SupplyChainSim-GSCPI v2 · 端到端 Demo")
    print("=" * 74)

    chk = pipeline_check()
    print(f"\n[1] 官方管道复现（合成面板）：corr={chk['corr_with_true_factor']} "
          f"PC1 解释 {chk['explained_var']:.1%}")
    (RES / "pipeline_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=2))

    ev = events.by_key("2023-12")
    out = run_scenario(ev.shock, months=args.months, rounds=args.rounds)
    g = bt.load_gscpi()
    i = g.index.get_loc(pd.Timestamp(ev.month))
    actual = [round(float(g.iloc[i + h] - g.iloc[i]), 3) for h in range(1, args.months + 1)]

    print(f"\n[2] 情景：{ev.label}（{ev.month}）")
    print("    月份   ΔGSCPI(模型)   ΔGSCPI(实际)")
    for m, d in enumerate(out["path"]):
        print(f"     +{m+1}     {d:+.3f}        {actual[m]:+.3f}")

    print("\n[3] 末月状态（节选）")
    snap = out["sim"].snapshot()
    for e in ["ea", "uk", "china", "us"]:
        s = snap["mfr"][e]
        print(f"    {ECONOMY_LABEL[e]:<4} 紧迫度 {s['panic']:.2f}  感知延迟 {s['delay']:.1f} 天  "
              f"交货 {s['delivery']:.1f}  积压 {s['backlog']:.1f}  采购 {s['purchases']:.1f}")

    print("\n[4] 指标级归因（首月，Top 8）")
    c0 = out["contribs"][0]
    print(c0.abs().sort_values(ascending=False).head(8).round(4).to_string())

    df_agent, total, residual = attribution_by_agent(ev.shock, months=args.months,
                                                     gains=(0.30, 0.30))
    print("\n[5] 智能体级反事实归因（3 个月累计）")
    print(df_agent.head(8).to_string(index=False))
    print(f"    总计 {total:+.4f}；交互残差 {residual:+.4f}")

    loo, ev_report = event_backtest()
    parts = bt.event_parts(events.EVENTS, months=3)
    actual_ev = bt.event_actual(g, events.EVENTS, months=3)
    print("\n[6] 留一事件回测（12 个历史冲击，3 个月窗口）")
    print(loo[["event", "label", "pred", "actual", "rmse", "naive_rmse"]].to_string(index=False))
    pooled = float(np.sqrt(np.mean(loo.rmse ** 2)))
    naive = float(np.sqrt(np.mean(loo.naive_rmse ** 2)))
    print(f"    合并 RMSE：模型 {pooled:.3f} vs 不变基准 {naive:.3f}；"
          f"胜出 {int((loo.rmse < loo.naive_rmse).sum())}/{len(loo)}")
    xs, ys = [], []
    for k, path in actual_ev.items():
        for h in range(3):
            if not np.isnan(path[h]):
                xs.append(parts[k]["cum_F"][h] + parts[k]["cum_P"][h]); ys.append(path[h])
    cc = float(np.corrcoef(xs, ys)[0, 1])
    print(f"    冲量-实际相关：r={cc:+.3f}；符号一致 {np.mean(np.sign(xs)==np.sign(ys)):.1%}")

    print("\n[7] 全样本排行榜（官方 GSCPI，h=1）")
    print(ev_report["leaderboard"].to_string())

    cal = calibration()
    print("\n[8] 增益标定")
    print(json.dumps(cal, ensure_ascii=False, indent=2))

    payload = {
        "scenario": ev.label, "month": ev.month,
        "model_delta": [round(x, 4) for x in out["path"]],
        "actual_delta": actual,
        "final_state": snap,
        "intents": out["intents"],
        "messages": out["sim"].log[-40:],
        "contrib_first_month": {k: round(v, 4) for k, v in c0.items() if abs(v) > 1e-4},
        "agent_attribution": df_agent.to_dict(orient="records"),
        "interaction_residual": residual,
    }
    (RES / "red_sea.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    loo.to_csv(RES / "event_loo.csv", index=False)
    ev_report["leaderboard"].to_csv(RES / "leaderboard.csv")
    (RES / "calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2))

    if not args.no_figs:
        from . import viz
        viz.make_all(g, out, c0, loo, cal, RES, parts=parts, actual=actual_ev)
        print(f"\n[9] 图表已写入 {RES}")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
