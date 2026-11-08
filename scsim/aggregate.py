"""官方 GSCPI 聚合管道复现 + 精确归因（v2）。

官方三步（NY Fed Staff Report 1017，§3.1 原文核对）：
  1) 需求剥离：每个 PMI 分项对**本经济体新订单**的当期值 + 2 期滞后做单方程 OLS，取残差；
     6 个运输成本序列对 **GDP 加权新订单 + GDP 加权投入品采购量**及其滞后做单方程 OLS。
     ⚠ 官方是逐序列单方程回归，不是 VAR 系统（旧版文档表述错误）。
  2) PCA：27 个残差序列标准化后提取**第一主成分**（共同成分）；缺失值按
     Stock & Watson (2002) 的做法用共同成分迭代填补，样本期 1997 起。
  3) 标准化：以历史均值为 0、历史标准差为 1（指数单位 = 标准差）。

★ PCA 的定位（本项目的关键概念澄清）：
   PCA 是**官方指数定义**的一部分（构造指标），不是"我们选的预测方法"。
   因此任何预测系统都必须复用同一条聚合管道，否则预测值与官方发布值不可比。

★ 归因的可加性（本文的一个技术要点）：
   在载荷固定（或按时点冻结）时，GSCPI_t = Σ_i l_i z_{i,t} / s_f 是 z 的**线性函数**，
   所以 ΔGSCPI 可以精确拆到 27 个指标，再经结算矩阵拆到 13 个智能体的动作，
   不需要 Shapley 采样近似。若逐期重估载荷，则用一阶近似 + 残差项并披露。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_LAGS = 2


# --------------------------------------------------------------------------- #
# 1) 需求剥离
# --------------------------------------------------------------------------- #
def _ols_resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def purge_demand(x: pd.Series, proxies: pd.DataFrame | pd.Series, n_lags: int = N_LAGS) -> pd.Series:
    """单方程 OLS 需求剥离，返回对齐索引的残差。

    x_t = a + Σ_k Σ_j b_{j,k} · proxy_{j,t-k} + e_t,  k = 0..n_lags
    """
    if isinstance(proxies, pd.Series):
        proxies = proxies.to_frame()
    cols = []
    for c in proxies.columns:
        for k in range(n_lags + 1):
            cols.append(proxies[c].shift(k).rename(f"{c}_l{k}"))
    X = pd.concat(cols, axis=1)
    data = pd.concat([x.rename("y"), X], axis=1).dropna()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if len(data) < X.shape[1] + 8:
        return (x - x.mean()).where(x.notna())
    out.loc[data.index] = _ols_resid(data["y"].values, data.drop(columns="y").values)
    return out


# --------------------------------------------------------------------------- #
# 2) PCA + EM 填补
# --------------------------------------------------------------------------- #
def standardize(mat: pd.DataFrame):
    mu = mat.mean()
    sd = mat.std(ddof=1).replace(0.0, 1.0)
    return (mat - mu) / sd, mu, sd


def pca_pc1_em(z: pd.DataFrame, n_iter: int = 50, tol: float = 1e-9):
    """第一主成分 + 缺失值 EM 填补（Stock-Watson 式）。

    返回 (score, loadings, explained_var)。
    用 SVD 实现，不依赖 sklearn。
    """
    Z = z.values.astype(float)
    miss = np.isnan(Z)
    F = np.where(miss, 0.0, Z)
    prev = None
    for _ in range(n_iter):
        Zc = F - F.mean(axis=0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        score = U[:, 0] * S[0]
        load = Vt[0]
        # 符号归一：让载荷之和为正，避免每轮翻转
        if load.sum() < 0:
            load, score = -load, -score
        recon = np.outer(score, load)
        F[miss] = recon[miss]
        if prev is not None and np.max(np.abs(F - prev)) < tol:
            break
        prev = F.copy()
    ev = float(S[0] ** 2 / (S ** 2).sum())
    return score, load, ev


# --------------------------------------------------------------------------- #
# 3) 组装
# --------------------------------------------------------------------------- #
class GSCPI:
    def __init__(self, gscpi: pd.Series, loadings: pd.Series, scale: float,
                 explained: float, zscore: pd.DataFrame):
        self.gscpi = gscpi
        self.loadings = loadings
        self.scale = scale          # 拟合窗内主成分得分的标准差
        self.explained = explained
        self.zscore = zscore        # 标准化残差（供预测期推进）

    def project(self, dz: pd.Series, base_z: pd.Series | None = None) -> tuple[float, pd.Series]:
        """给定**标准化残差的月度增量** dz，返回 (ΔGSCPI, 逐指标贡献)。

        ΔG = Σ_i l_i · dz_i / s_f
        若同时给出 base_z（上期 z 水平），也可返回本期水平读数。
        """
        idx = [c for c in dz.index if c in self.loadings.index]
        l = self.loadings[idx]
        contrib = (l * dz[idx].fillna(0.0)) / self.scale
        return float(contrib.sum()), contrib

    def level(self, z: pd.Series) -> float:
        idx = [c for c in z.index if c in self.loadings.index]
        return float((self.loadings[idx] * z[idx].fillna(0.0)).sum() / self.scale)


def compute_gscpi(resid: pd.DataFrame, reference_groups=("harpex", "bdi", "air", "freight")) -> GSCPI:
    z, _, _ = standardize(resid)
    score, load, ev = pca_pc1_em(z)
    # 符号归一：运输类载荷均值应为正（运价上升 = 压力上升）
    ref = [i for i, c in enumerate(z.columns) if any(g in c.lower() for g in reference_groups)]
    if ref and np.mean(load[ref]) < 0:
        score, load = -score, -load
    s = pd.Series(score, index=z.index)
    scale = float(np.nanstd(s.values, ddof=1))
    return GSCPI(gscpi=s / scale, loadings=pd.Series(load, index=z.columns),
                 scale=scale, explained=ev, zscore=z)


# --------------------------------------------------------------------------- #
# 合成面板（仅用于工程自测；不用于任何实证结论）
# --------------------------------------------------------------------------- #
def make_synthetic_panel(n_months: int = 300, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    f = np.zeros(n_months)
    d = np.zeros(n_months)
    for t in range(1, n_months):
        f[t] = 0.93 * f[t - 1] + rng.normal()
        d[t] = 0.90 * d[t - 1] + rng.normal()
    names = (["harpex", "bdi"] + [f"air_{i}" for i in range(4)]
             + [f"{e}_{v}" for e in ["china", "us", "ea", "jp", "kr", "tw", "uk"]
                for v in ["delivery", "backlogs", "purchases"]])
    loads = rng.uniform(0.5, 1.0, 27)
    mat = {}
    for i, nm in enumerate(names):
        x = loads[i] * f + 0.45 * d + rng.normal(0, 0.5, n_months)
        s = pd.Series(x, index=dates, name=nm)
        if nm.startswith(("china_", "kr_", "tw_")):
            s.loc[:"2003-12-31"] = np.nan
        mat[nm] = s
    proxy = pd.DataFrame({"p_new_orders": d, "p_purchases": 0.6 * d}, index=dates)
    return pd.DataFrame(mat), proxy, pd.Series(f, index=dates)
