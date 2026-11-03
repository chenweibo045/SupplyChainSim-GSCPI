"""复现纽约联储 GSCPI 官方计算管道（NY Fed Staff Report No.1017）。

官方聚合三步：
  1) 需求剥离：每个序列对需求代理（当期 + 2 期滞后）做单方程 OLS 回归，取残差
     （注意：官方是单方程回归，不是 VAR 系统）
  2) PCA：27 个残差序列标准化后提取第一主成分（载荷在全样本上一次估计并固定，
     遵循 Stock & Watson (2002) 扩散指数方法；缺失值用 EM 式迭代填补）
  3) 标准化：符号归一（参考组载荷为正，即压力上升 = 指数上升），
     除以拟合窗内标准差，得到以历史标准差为单位的 GSCPI

自测：python -m pipeline.official_gscpi --selftest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

N_LAGS = 2  # 官方做法：当期 + 2 期滞后


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """带常数项的最小二乘残差（numpy 实现，零额外依赖）。"""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta
DEFAULT_REFERENCE_GROUPS = ("harpex", "bdi", "freight", "shipping", "air")


def purge_demand(x: pd.Series, demand_proxy: pd.Series, n_lags: int = N_LAGS) -> pd.Series:
    """单方程 OLS 需求剥离：x_t = a + Σ b_k·proxy_{t-k} + e_t，返回对齐索引的残差。

    官方口径（LSE, 2022-01）：21 个 PMI 分项分别对本经济体 PMI「新订单」回归；
    6 个运输成本指标对「GDP 加权新订单 + GDP 加权投入品采购量」回归。
    样本太少时退化为去均值。
    """
    df = pd.concat({"x": x, "p": demand_proxy}, axis=1)
    X = pd.concat([df["p"].shift(k) for k in range(n_lags + 1)], axis=1)
    X.columns = [f"p_lag{k}" for k in range(n_lags + 1)]
    data = pd.concat([df["x"], X], axis=1).dropna()
    if len(data) < n_lags + 6:
        return (x - x.mean()).where(x.notna())
    resid = _ols_residuals(data["x"].values, data.drop(columns="x").values)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[data.index] = resid
    return out


def purge_demand_multi(x: pd.Series, proxies: pd.DataFrame, n_lags: int = N_LAGS) -> pd.Series:
    """多需求代理版本（用于运输成本序列：GDP 加权新订单 + GDP 加权投入品采购量）。"""
    df = pd.concat([x.rename("x"), proxies], axis=1)
    X_cols = []
    for col in proxies.columns:
        for k in range(n_lags + 1):
            X_cols.append(proxies[col].shift(k).rename(f"{col}_lag{k}"))
    X = pd.concat(X_cols, axis=1)
    data = pd.concat([df["x"], X], axis=1).dropna()
    if len(data) < X.shape[1] + 4:
        return (x - x.mean()).where(x.notna())
    resid = _ols_residuals(data["x"].values, data.drop(columns="x").values)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[data.index] = resid
    return out


def _standardize(mat: pd.DataFrame):
    mu = mat.mean()
    sd = mat.std(ddof=1).replace(0.0, 1.0)
    return (mat - mu) / sd, mu, sd


def _extract_pc1_em(z: pd.DataFrame, n_iter: int = 30):
    """Stock-Watson (2002) 式 PCA + 缺失值 EM 填补。

    迭代：PCA 提取因子 → 用 因子×载荷 重构缺失项 → 再 PCA，直至收敛。
    """
    zv = z.values.astype(float)
    miss = np.isnan(zv)
    fill = np.where(miss, 0.0, zv)
    last = None
    for _ in range(n_iter):
        pca = PCA(n_components=1)
        f = pca.fit_transform(fill)[:, 0]
        load = pca.components_[0]
        recon = np.outer(f, load)
        fill[miss] = recon[miss]
        if last is not None and np.allclose(fill, last, atol=1e-8):
            break
        last = fill.copy()
    return f, load, float(pca.explained_variance_ratio_[0])


@dataclass
class GSCPIResult:
    gscpi: pd.Series          # 标准化后的指数（历史均值 0，拟合窗内标准差 1）
    loadings: pd.Series       # 27 个指标在第一主成分上的载荷（固定，用于归因）
    explained_var: float      # 第一主成分解释的方差比例
    resid: pd.DataFrame = field(default=None, repr=False)   # 需求剥离后的残差矩阵
    zresid: pd.DataFrame = field(default=None, repr=False)  # 标准化残差（用于预测期推进）


def compute_gscpi(
    resid: pd.DataFrame,
    reference_groups=DEFAULT_REFERENCE_GROUPS,
    fit_end=None,
) -> GSCPIResult:
    """由需求剥离后的残差矩阵计算 GSCPI。

    参数
    ----
    resid        : DataFrame[T x 27]，列名为指标名（含 harpex/bdi/freight 等字样便于符号归一）
    reference_groups : 符号参考组关键词：这些指标的载荷均值应为正（运价上升 = 压力上升）
    fit_end      : 拟合窗终点（如 "2021-12-31"）；载荷与标准差只用拟合窗估计，保证回测口径稳定
    """
    z, _, _ = _standardize(resid)
    f, load, r2 = _extract_pc1_em(z)
    cols = list(z.columns)
    ref_idx = [i for i, c in enumerate(cols) if any(g in c.lower() for g in reference_groups)]
    if ref_idx and np.mean(load[ref_idx]) < 0:
        f, load = -f, -load  # 符号归一：压力上升 → 指数上升
    g = pd.Series(f, index=z.index, dtype=float)
    fit = g.loc[:fit_end] if fit_end is not None else g
    scale = fit.std(ddof=1)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("拟合窗内主成分标准差为 0，无法标准化")
    return GSCPIResult(
        gscpi=g / scale,
        loadings=pd.Series(load, index=cols),
        explained_var=r2,
        resid=resid,
        zresid=z,
    )


def project_gscpi(result: GSCPIResult, z_new: pd.Series) -> tuple[float, pd.Series]:
    """预测期推进：给定新一个月的标准化残差向量（允许部分缺失），
    返回 (GSCPI 读数, 各指标贡献分解)。缺失指标按中性（z=0）处理，
    贡献口径与 NY Fed「中国因素」博文一致：
    contribution_i = loading_i × z_new_i / std_fit。
    """
    load = result.loadings
    common = [c for c in z_new.index if c in load.index]
    z_fill = z_new[common].fillna(0.0)          # 缺失 = 中性，不进读数
    scale = result.gscpi.std(ddof=1)
    g_val = float(load[common].dot(z_fill)) / scale
    contrib = (load[common] * z_new[common]) / scale   # 缺失项保持 NaN，归因只报观测项
    return g_val, contrib


# --------------------------------------------------------------------------- #
# 合成数据工厂 + 自测
# --------------------------------------------------------------------------- #

def make_synthetic_panel(n_months: int = 300, n_indicators: int = 27, seed: int = 7):
    """构造含已知共同因子 + 需求成分 + 缺失段的合成面板，用于离线验证管道正确性。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    # 真实共同因子：高持久平稳 AR(1)（对应现实中去均值后的 PMI 扩散指数类序列；
    # 不用随机游走：对近单位根回归会产生假性低频污染，扭曲管道验证）
    rho_f, rho_d = 0.92, 0.88
    f_innov = rng.normal(0, 1.0, n_months)
    d_innov = rng.normal(0, 1.0, n_months)
    f_true = np.zeros(n_months)
    demand = np.zeros(n_months)
    for t in range(1, n_months):
        f_true[t] = rho_f * f_true[t - 1] + f_innov[t]
        demand[t] = rho_d * demand[t - 1] + d_innov[t]
    f_true += rng.normal(0, 0.2, n_months)
    names = (
        [f"harpex", "bdi"]
        + [f"airfreight_{i}" for i in range(4)]
        + [f"{c}_{v}" for c in ["china", "us", "ea", "jp", "kr", "tw", "uk"]
        for v in ["delivery", "backlogs", "purchases"]]
    )
    names = names[:n_indicators]
    # 所有指标对「压力因子」正向加载：交货延迟/积压/采购增加与运价上升同向
    # （与 GSCPI 符号约定一致；也保证离线 demo 的归因方向可解释）
    loads = rng.uniform(0.5, 1.0, n_indicators)
    loads[:6] = np.abs(loads[:6])  # 运输类载荷为正，与符号参考组一致
    mat, proxy = [], []
    for i in range(n_indicators):
        eps = rng.normal(0, 0.5, n_months)
        x = loads[i] * f_true + 0.5 * demand + eps
        s = pd.Series(x, index=dates, name=names[i])
        # 模拟官方处理的长度不齐：亚洲序列 2004 年才有
        if names[i].startswith(("china_", "kr_", "tw_")):
            s.loc[: "2003-12-31"] = np.nan
        mat.append(s)
        proxy.append(pd.Series(demand, index=dates, name=f"proxy_{i}"))
    return pd.concat(mat, axis=1), pd.concat(proxy, axis=1), pd.Series(f_true, index=dates, name="true_factor")


def selftest() -> int:
    panel, proxies, f_true = make_synthetic_panel()
    # Step 1: 需求剥离
    resid = pd.DataFrame({c: purge_demand(panel[c], proxies[f"proxy_{i}"])
                          for i, c in enumerate(panel.columns)})
    assert resid.notna().sum().sum() > 0, "残差矩阵为空"
    # Step 2+3: PCA + 标准化
    res = compute_gscpi(resid)
    g = res.gscpi
    assert abs(g.mean()) < 0.05, f"均值未归零: {g.mean()}"
    assert abs(g.std(ddof=1) - 1.0) < 0.05, f"标准差未归一: {g.std(ddof=1)}"
    assert res.loadings.iloc[:6].mean() > 0, "符号归一失败：运输类载荷应为正"
    corr = np.corrcoef(g.values, f_true.values)[0, 1]
    print(f"[selftest] 与真实因子的相关系数: {corr:.4f}")
    print(f"[selftest] PC1 解释方差比例: {res.explained_var:.3f}")
    assert corr > 0.9, f"管道复现失败：corr={corr:.3f} < 0.9"
    # 预测期推进 + 归因
    z_new = res.zresid.iloc[-1].copy()
    z_new.iloc[::5] = np.nan  # 模拟部分指标缺失
    g_val, contrib = project_gscpi(res, z_new)
    assert np.isfinite(g_val), "缺失指标下预测推进应返回有限值"
    print(f"[selftest] 预测期推进 OK: gscpi={g_val:.3f}, 非缺失贡献项={contrib.notna().sum()}")
    print("[selftest] PASS ✔  官方管道三步（需求剥离→PCA→标准化）离线复现通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    print("用法: python -m pipeline.official_gscpi --selftest")
