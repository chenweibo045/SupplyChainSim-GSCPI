"""Streamlit 情景沙盘：输入新闻 → 多智能体模拟 → GSCPI 预测 + 归因报告。

运行：streamlit run app/streamlit_app.py
（离线可用：LLM 未配置时自动走规则解析；历史管道用内置合成数据拟合）
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SupplyChainSim-GSCPI", layout="wide")
st.title("SupplyChainSim-GSCPI：全球供应链压力多智能体预测沙盘")

with st.sidebar:
    st.header("设置")
    rounds = st.slider("博弈轮数", 1, 4, 2)
    use_llm = st.checkbox("启用 LLM 解析（需 OPENAI_API_KEY）", value=False)
    seed = st.number_input("合成历史随机种子", value=7, step=1)

news = st.text_area(
    "输入新闻（每行一条，中英文均可）",
    value="红海多艘货轮遭武装袭击，苏伊士运河通行量锐减\n多家班轮公司宣布绕行好望角",
    height=120,
)

if st.button("运行模拟", type="primary"):
    llm_call = None
    if use_llm:
        import os
        from openai import OpenAI
        client = OpenAI()  # 读取 OPENAI_API_KEY / OPENAI_BASE_URL
        llm_call = lambda prompt: client.chat.completions.create(
            model=os.environ.get("GSCPI_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0).choices[0].message.content

    from sim.orchestrator import demo_red_sea, run_scenario, settle, synthesize_matrix
    from pipeline.official_gscpi import compute_gscpi, make_synthetic_panel, project_gscpi
    import pandas as pd
    from sim.orchestrator import TRANSPORT_COLS, ECONOMIES, PMI_SUFFIX

    with st.spinner("智能体博弈中……"):
        items = [ln for ln in news.splitlines() if ln.strip()]
        result, _ = run_scenario(items, llm_call=llm_call, rounds=rounds, verbose=False)
        result = settle(result)

        cols = TRANSPORT_COLS + [f"{e}_{v}" for e in ECONOMIES for v in PMI_SUFFIX]
        panel, proxies, _ = make_synthetic_panel(seed=int(seed))
        panel = panel[cols]
        from pipeline.official_gscpi import purge_demand
        resid = pd.DataFrame({c: purge_demand(panel[c], proxies.iloc[:, 0]) for c in cols})
        fitted = compute_gscpi(resid)
        z_delta = synthesize_matrix(result)
        g_next, contrib = project_gscpi(fitted, z_delta)

    c1, c2, c3 = st.columns(3)
    c1.metric("基期 GSCPI（合成历史）", f"{fitted.gscpi.iloc[-1]:+.2f}")
    c2.metric("预测下期 GSCPI", f"{g_next:+.2f}", f"{g_next - fitted.gscpi.iloc[-1]:+.2f}")
    c3.metric("绕航延迟", f"+{result.added_days:.1f} 天")

    st.subheader("归因报告（各指标对 GSCPI 变动的贡献，与官方口径一致）")
    st.bar_chart(contrib.dropna().sort_values())

    st.subheader("Agent 决策轨迹")
    for r, intents in result.intents.items():
        st.markdown(f"**第 {r + 1} 轮**")
        st.table({
            "Agent": [it.agent_id for it in intents],
            "意图": [it.action for it in intents],
            "强度": [it.strength for it in intents],
            "理由": [it.rationale for it in intents],
        })

    st.subheader("PMI 分项增量（%）")
    st.dataframe(result.pmi_deltas)
    st.caption("注：当前为离线合成数据演示。接入真实 27 序列（data/raw）后，"
               "预测口径与纽约联储官方发布值可比。")
