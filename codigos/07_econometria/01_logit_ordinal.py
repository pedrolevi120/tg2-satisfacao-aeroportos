"""
T6/econometria — Logit Ordinal (proportional odds) com seleção LASSO (Belloni–Chernozhukov–Hansen).

Base: MVP (SBGR/SBSP/SBKP), série MPor 2021T1–2026T2, MESMOS splits dos modelos de ML
(dados/processados/splits). Efeitos fixos de aeroporto e de PERIODO (21 trimestres).
Estimação apenas no treino; métricas preditivas em validação e teste.

Especificações:
  (C) Completo      : todos os regressores construídos (após remoção QR de dependências).
  (L) Post-LASSO    : L1-logit com penalidade teórica (hdm::rlassologit) nos dois cortes
                      relevantes (nota>=4; nota=5); união das selecionadas; FE e PROCESSO
                      não penalizados; reestimação sem penalidade (Belloni & Chernozhukov, 2013).
  (D) Post-double-selection: TODOS os atributos mantidos (parâmetros de interesse);
                      controles = união de (i) seleção na equação do resultado (L1-logit) e
                      (ii) seleção em cada equação atributo ~ controles (rlasso linear)
                      (Belloni, Chernozhukov & Hansen, 2014; Belloni, Chernozhukov & Wei, 2016).
Saídas: resultados/logit_*.csv|json, tex/econometria_logit*.tex, figuras econometria_logit_*.pdf
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             cohen_kappa_score, log_loss, accuracy_score)
from comum import *

warnings.filterwarnings("ignore")
MIN_SHARE_ATRIB = 0.03   # atributo precisa de nota válida em >= 3% do treino


# ------------------------------------------------------------------ LASSO teórico
def lambda_bcw(n, p, c=1.1):
    """Penalidade teórica de hdm (Chernozhukov, Hansen & Spindler, 2016):
    lambda = c/2 * sqrt(n) * Phi^{-1}(1 - gamma/(2p)), gamma = 0.1/log(n).
    hdm passa lambda/(2n) ao glmnet (objetivo -LL/n + l0*|b|, X padronizado);
    na escala de -LL total (statsmodels) o peso é n*lambda/(2n) = lambda/2."""
    gamma = 0.1 / np.log(n)
    return c / 2 * np.sqrt(n) * stats.norm.ppf(1 - gamma / (2 * p))


def rlasso_logit(y, Z, penalizar):
    """L1-logit com penalidade teórica; Z padronizado; retorna colunas com coef != 0."""
    n, p = Z.shape
    lam = lambda_bcw(n, int(penalizar.sum()))
    alpha = np.r_[0.0, np.where(penalizar, lam / 2, 0.0)]
    Zc = sm.add_constant(Z.values, has_constant="add")
    r = sm.Logit(y, Zc).fit_regularized(alpha=alpha, disp=0, maxiter=2000, trim_mode="auto")
    coef = r.params[1:]
    return set(Z.columns[(np.abs(coef) > 1e-8) & penalizar])


def rlasso_linear(d, Z, penalizar, c=1.1, n_iter=5):
    """LASSO linear com penalidade teórica homocedástica iterada (hdm::rlasso):
    objetivo sklearn (1/2n)||d-Zb||^2 + a||b||_1, a = c*sigma*Phi^{-1}(1-gamma/2p)/sqrt(n).
    Colunas não penalizadas são parcializadas antes (Frisch–Waugh)."""
    from sklearn.linear_model import Lasso, LinearRegression
    n = len(d)
    U = Z.loc[:, ~penalizar].values
    P = Z.loc[:, penalizar]
    if U.shape[1]:
        lr = LinearRegression().fit(U, d)
        d_r = d - lr.predict(U)
        P_r = P.values - LinearRegression().fit(U, P.values).predict(U)
    else:
        d_r, P_r = d - d.mean(), P.values - P.values.mean(0)
    sd = P_r.std(0)
    ok = sd > 1e-12
    P_r = P_r[:, ok] / sd[ok]
    cols = np.array(P.columns)[ok]
    gamma = 0.1 / np.log(n)
    q = stats.norm.ppf(1 - gamma / (2 * P_r.shape[1]))
    sigma = d_r.std()
    sel = np.zeros(P_r.shape[1], bool)
    for _ in range(n_iter):
        a = c * sigma * q / np.sqrt(n)
        m = Lasso(alpha=a, fit_intercept=False, max_iter=20000).fit(P_r, d_r)
        sel = np.abs(m.coef_) > 1e-10
        if sel.sum():
            res = d_r - LinearRegression(fit_intercept=False).fit(P_r[:, sel], d_r).predict(P_r[:, sel])
            sigma_new = np.sqrt((res ** 2).sum() / max(n - sel.sum(), 1))
        else:
            sigma_new = d_r.std()
        if abs(sigma_new - sigma) < 1e-6:
            break
        sigma = sigma_new
    return set(cols[sel])


# ------------------------------------------------------------------ logit ordinal
def ajustar_om(y, X):
    m = OrderedModel(y, X, distr="logit")
    res = m.fit(method="bfgs", maxiter=5000, disp=0, cov_type="HC0")
    if not res.mle_retvals.get("converged", True):
        res = m.fit(method="newton", maxiter=200, disp=0, cov_type="HC0",
                    start_params=res.params)
    return res


def probs(res, X):
    return np.asarray(res.model.predict(res.params, exog=X.values))


def avaliar(y, P):
    y = np.asarray(y)
    ey = P @ np.arange(1, 6)
    yp = np.clip(np.round(ey), 1, 5).astype(int)
    return {"MAE": float(mean_absolute_error(y, ey)),
            "RMSE": float(np.sqrt(mean_squared_error(y, ey))),
            "R2": float(r2_score(y, ey)),
            "QWK": float(cohen_kappa_score(y, yp, weights="quadratic", labels=[1, 2, 3, 4, 5])),
            "Acuracia_argmax": float(accuracy_score(y, P.argmax(1) + 1)),
            "LogLoss": float(log_loss(y, P, labels=[1, 2, 3, 4, 5]))}


def ll_nulo(y_eval, y_train):
    fr = pd.Series(y_train).value_counts(normalize=True).reindex(range(1, 6)).fillna(1e-12)
    return float(np.log(fr.loc[np.asarray(y_eval)].values).sum())


def ame_p5(res, X, cols, mascaras):
    """Efeito marginal médio de x_j sobre P(y=5): mean[f(tau4 - x'b)] * b_j,
    média sobre respondentes com nota válida em j; EP pelo método delta (gradiente numérico)."""
    k = X.shape[1]
    Xv = X.values
    j_idx = [list(X.columns).index(c) for c in cols]

    def f(theta):
        b = theta[:k]
        tau = res.model.transform_threshold_params(theta)[4]
        xb = Xv @ b
        dens = stats.logistic.pdf(tau - xb)
        return np.array([dens[mascaras[c]].mean() * b[j] for c, j in zip(cols, j_idx)])

    th = res.params.values if hasattr(res.params, "values") else res.params
    val = f(th)
    G = np.zeros((len(cols), len(th)))
    for i in range(len(th)):
        h = 1e-5 * max(1.0, abs(th[i]))
        e = np.zeros_like(th); e[i] = h
        G[:, i] = (f(th + e) - f(th - e)) / (2 * h)
    V = G @ np.asarray(res.cov_params()) @ G.T
    return val, np.sqrt(np.diag(V))


def teste_brant_aprox(y, X, cols):
    """Teste tipo Brant (aproximado) de proporcionalidade das chances para os atributos:
    compara coeficientes de logits binários nos cortes y>=4 e y>=5. Ignora a covariância
    cruzada entre os dois logits (positiva) => teste conservador."""
    Xc = sm.add_constant(X.values, has_constant="add")
    out = {}
    fits = {}
    for corte in [4, 5]:
        r = sm.Logit((np.asarray(y) >= corte).astype(int), Xc).fit(disp=0, maxiter=500, cov_type="HC0")
        fits[corte] = r
    idx = [list(X.columns).index(c) + 1 for c in cols]
    b4, b5 = fits[4].params[idx], fits[5].params[idx]
    V = np.asarray(fits[4].cov_params())[np.ix_(idx, idx)] + np.asarray(fits[5].cov_params())[np.ix_(idx, idx)]
    dif = b4 - b5
    W = float(dif @ np.linalg.pinv(V) @ dif)
    out["global"] = {"W": W, "gl": len(idx), "p": float(stats.chi2.sf(W, len(idx)))}
    z = dif / np.sqrt(np.diag(V))
    out["por_atributo"] = {c: {"b_corte4": float(a), "b_corte5": float(b), "p": float(2 * stats.norm.sf(abs(zz)))}
                           for c, a, b, zz in zip(cols, b4, b5, z)}
    return out


# ------------------------------------------------------------------ main
def main():
    estilo()
    df, idx = carregar_mvp()
    tr, va, te = idx["train"], idx["val"], idx["test"]
    print(f"Base MVP: {len(df)} | treino {len(tr)} val {len(va)} teste {len(te)}")

    todos = colunas_atributos(df)
    Nall = pd.DataFrame({a: likert_num(df[a]) for a in todos})
    hier = detectar_hierarquia(Nall.loc[tr])
    ramo = ramos(Nall.loc[tr], df.loc[tr, "PROCESSO"])
    share = Nall.loc[tr].notna().mean()
    atributos = [a for a in todos if share[a] >= MIN_SHARE_ATRIB and ramo[a]]
    centros = Nall.loc[tr, atributos].mean().to_dict()
    print(f"Atributos Likert: {len(todos)} detectados, {len(atributos)} com cobertura >= {MIN_SHARE_ATRIB:.0%}")
    pd.DataFrame([{"filho": f, "pai": p, "n_filho": int(Nall[f].notna().sum()),
                   "filho_usado": f in atributos} for f, p in hier.items()]
                 ).to_csv(OUT_DIR / "hierarquia_questionario.csv", index=False)

    X0, grupos, ref_ctrl = construir_desenho(df, atributos, hier, ramo, centros)
    FE = efeitos_fixos(df)
    Xfull = pd.concat([X0, FE], axis=1)
    y = df["y"]

    # ---- diagnóstico: rank da matriz usada anteriormente (features do ML)
    Xml = pd.read_parquet(SPLITS / "X_train.parquet").astype(float)
    Zml = np.column_stack([np.ones(len(Xml)), Xml.values])
    rank_ml = int(np.linalg.matrix_rank(Zml))
    print(f"Matriz ML de treino: {Zml.shape[1]} colunas (c/ constante), posto {rank_ml} "
          f"-> deficiência {Zml.shape[1] - rank_ml}")

    Xtr_full, removidas = remover_dependentes(Xfull.loc[tr], protegidas=list(FE.columns))
    print(f"Desenho econométrico: {Xfull.shape[1]} colunas; removidas por dependência/constância: {len(removidas)}")
    for c in removidas:
        print("   removida:", c)
    cols_ok = list(Xtr_full.columns)
    fe_cols = [c for c in cols_ok if c.startswith("FE_")]
    atr_cols = [c for c in cols_ok if grupos.get(c) == "atributo"]
    ctrl_cols = [c for c in cols_ok if c not in fe_cols and c not in atr_cols]

    # padronização (só para LASSO; estimação final nas escalas originais)
    mu, sd = Xtr_full.mean(), Xtr_full.std(ddof=0)
    Ztr = (Xtr_full - mu) / sd
    nao_pen = set(fe_cols) | {"PROCESSO=Embarque"}
    pen = np.array([c not in nao_pen for c in cols_ok])

    # ---- (L) post-LASSO: seleção na equação do resultado (dois cortes)
    sel_y = set()
    for corte in [4, 5]:
        s = rlasso_logit((y.loc[tr] >= corte).astype(int).values, Ztr, pen)
        print(f"  L1-logit corte y>={corte}: {len(s)} selecionadas")
        sel_y |= s
    cols_L = [c for c in cols_ok if c in sel_y or c in nao_pen]

    # ---- (D) post-double-selection: atributos sempre; controles por dupla seleção
    pen_ctrl = np.array([c in ctrl_cols and c not in nao_pen for c in cols_ok])
    sel_d = set()
    Zc = Ztr[[c for c in cols_ok if c not in atr_cols]]
    pen_c = np.array([c in ctrl_cols and c not in nao_pen for c in Zc.columns])
    for a in atr_cols:
        sel_d |= rlasso_linear(Xtr_full.loc[:, a].values, Zc, pen_c)
    sel_y_ctrl = rlasso_logit((y.loc[tr] >= 4).astype(int).values, Ztr, pen_ctrl) | \
        rlasso_logit((y.loc[tr] >= 5).astype(int).values, Ztr, pen_ctrl)
    ctrl_D = (sel_d | sel_y_ctrl) & set(ctrl_cols)
    cols_D = [c for c in cols_ok if c in atr_cols or c in ctrl_D or c in nao_pen]
    print(f"  Post-double-selection: {len(atr_cols)} atributos + {len(ctrl_D)} controles "
          f"(eq. resultado {len(sel_y_ctrl & set(ctrl_cols))}, eqs. atributos {len(sel_d)})")

    especs = {"C": cols_ok, "L": cols_L, "D": cols_D}
    nomes = {"C": "Logit ordinal completo", "L": "Logit ordinal post-LASSO",
             "D": "Logit ordinal post-double-selection"}
    resultados, fits, metricas = {}, {}, []
    for k, cols in especs.items():
        res = ajustar_om(y.loc[tr].values, Xfull.loc[tr, cols])
        fits[k] = res
        ll0 = ll_nulo(y.loc[tr], y.loc[tr])
        r = {"espec": k, "nome": nomes[k], "k_regressores": len(cols),
             "k_atributos": sum(c in atr_cols for c in cols),
             "convergiu": bool(res.mle_retvals.get("converged", True)),
             "LL": float(res.llf), "LL0": ll0, "pseudoR2_McFadden": float(1 - res.llf / ll0),
             "AIC": float(res.aic), "BIC": float(res.bic), "n_treino": int(len(tr))}
        for nome_s, ids in [("val", va), ("test", te)]:
            P = probs(res, Xfull.loc[ids, cols])
            m = avaliar(y.loc[ids], P)
            ll_m = float(np.log(P[np.arange(len(ids)), y.loc[ids].values - 1]).sum())
            m["pseudoR2_oos"] = 1 - ll_m / ll_nulo(y.loc[ids], y.loc[tr])
            r[nome_s] = m
        resultados[k] = r
        print(f"  [{k}] k={len(cols)} conv={r['convergiu']} pR2={r['pseudoR2_McFadden']:.3f} | "
              f"teste MAE={r['test']['MAE']:.3f} RMSE={r['test']['RMSE']:.3f} "
              f"R2={r['test']['R2']:.3f} QWK={r['test']['QWK']:.3f}")
    (OUT_DIR / "logit_resultados.json").write_text(json.dumps(
        {"resultados": resultados, "removidas_QR": removidas, "rank_ml": rank_ml,
         "cols_ml": int(Zml.shape[1]), "atributos": atributos, "hierarquia": hier,
         "selecionadas_L": cols_L, "controles_D": sorted(ctrl_D)}, indent=2, ensure_ascii=False))

    # ---- coeficientes, OR, AME (espec. D = principal)
    res = fits["D"]
    Xd = Xfull.loc[tr, cols_D]
    mascaras = {c: likert_num(df.loc[tr, c]).notna().values for c in atr_cols}
    ame, ame_se = ame_p5(res, Xd, atr_cols, mascaras)
    b = res.params[:len(cols_D)]
    se = res.bse[:len(cols_D)]
    tab = pd.DataFrame({"var": cols_D, "beta": b.values, "ep": se.values})
    tab["z"] = tab.beta / tab.ep
    tab["p"] = 2 * stats.norm.sf(np.abs(tab.z))
    tab["OR"] = np.exp(tab.beta)
    tab["OR_lo"] = np.exp(tab.beta - 1.96 * tab.ep)
    tab["OR_hi"] = np.exp(tab.beta + 1.96 * tab.ep)
    tab["grupo"] = tab["var"].map(lambda c: "fe" if c.startswith("FE_") else grupos.get(c, "?"))
    a_df = pd.DataFrame({"var": atr_cols, "AME_P5": ame, "AME_P5_ep": ame_se})
    tab = tab.merge(a_df, on="var", how="left")
    tab["n_validos_treino"] = tab["var"].map(lambda c: int(mascaras[c].sum()) if c in mascaras else np.nan)
    tab["sd_obs"] = tab["var"].map(lambda c: float(likert_num(df.loc[tr, c]).std()) if c in mascaras else np.nan)
    tab["pai"] = tab["var"].map(lambda c: hier.get(c, ""))
    tab.to_csv(OUT_DIR / "logit_coeficientes_D.csv", index=False)
    # coeficientes da especificação L também
    bl = fits["L"].params[:len(cols_L)]
    pd.DataFrame({"var": cols_L, "beta": bl.values, "ep": fits["L"].bse[:len(cols_L)].values}
                 ).to_csv(OUT_DIR / "logit_coeficientes_L.csv", index=False)

    # ---- teste de proporcionalidade (aprox. Brant) na espec. D
    brant = teste_brant_aprox(y.loc[tr], Xd, atr_cols)
    (OUT_DIR / "logit_brant.json").write_text(json.dumps(brant, indent=2, ensure_ascii=False))
    print(f"  Brant aprox. (atributos): W={brant['global']['W']:.1f}, gl={brant['global']['gl']}, "
          f"p={brant['global']['p']:.3g}")

    # ---- ranking logit x SHAP
    shap = pd.read_parquet(PROCESSADOS / "shap_values_test.parquet").drop(columns="AEROPORTO").astype(float)
    imp_shap = shap.abs().mean()
    comuns = [c for c in atr_cols if c in imp_shap.index]
    beta_d = dict(zip(tab["var"], tab["beta"]))
    Xte_d = Xfull.loc[te, cols_D]
    imp_idx = {c: float(np.abs(beta_d[c] * Xte_d[c]).mean()) for c in comuns}   # análogo linear do mean|SHAP|
    imp_ame = dict(zip(a_df["var"], np.abs(a_df["AME_P5"])))
    rk = pd.DataFrame({"atributo": comuns,
                       "shap_mean_abs": [imp_shap[c] for c in comuns],
                       "logit_idx_mean_abs": [imp_idx[c] for c in comuns],
                       "logit_abs_AME_P5": [imp_ame[c] for c in comuns],
                       "beta": [beta_d[c] for c in comuns],
                       "p": [float(tab.set_index("var").loc[c, "p"]) for c in comuns]})
    # importância do post-LASSO (0 se não selecionado)
    beta_l = dict(zip(cols_L, bl.values))
    Xte_l = Xfull.loc[te, cols_L]
    rk["lasso_idx_mean_abs"] = [float(np.abs(beta_l[c] * Xte_l[c]).mean()) if c in beta_l else 0.0 for c in comuns]
    for c in ["shap_mean_abs", "logit_idx_mean_abs", "logit_abs_AME_P5", "lasso_idx_mean_abs"]:
        rk[f"rank_{c}"] = rk[c].rank(ascending=False, method="average")
    rk = rk.sort_values("rank_shap_mean_abs")
    rk.to_csv(OUT_DIR / "logit_vs_shap_ranking.csv", index=False)
    spear = {}
    for c in ["logit_idx_mean_abs", "logit_abs_AME_P5", "lasso_idx_mean_abs"]:
        rho, p = stats.spearmanr(rk["shap_mean_abs"], rk[c])
        tau, pt = stats.kendalltau(rk["shap_mean_abs"], rk[c])
        spear[c] = {"spearman": float(rho), "p": float(p), "kendall": float(tau), "p_kendall": float(pt),
                    "n": int(len(rk))}
    top10_shap = set(rk.nsmallest(10, "rank_shap_mean_abs").atributo)
    top10_log = set(rk.nsmallest(10, "rank_logit_idx_mean_abs").atributo)
    spear["overlap_top10"] = len(top10_shap & top10_log)
    top4_shap = list(rk.nsmallest(4, "rank_shap_mean_abs").atributo)
    spear["top4_shap"] = top4_shap
    spear["top4_logit"] = list(rk.nsmallest(4, "rank_logit_idx_mean_abs").atributo)
    (OUT_DIR / "logit_spearman.json").write_text(json.dumps(spear, indent=2, ensure_ascii=False))
    print(f"  Spearman SHAP x logit (índice): {spear['logit_idx_mean_abs']['spearman']:.3f} "
          f"(p={spear['logit_idx_mean_abs']['p']:.3g}, n={len(rk)}); x |AME|: "
          f"{spear['logit_abs_AME_P5']['spearman']:.3f}; top-10 em comum: {spear['overlap_top10']}")

    # ---- figura: ranking
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.scatter(rk["rank_shap_mean_abs"], rk["rank_logit_idx_mean_abs"], s=28, color="#2b6cb0", zorder=3)
    mx = len(rk) + 1
    ax.plot([0, mx], [0, mx], ls="--", color="grey", lw=0.8)
    for _, r in rk.iterrows():
        if r["rank_shap_mean_abs"] <= 8 or r["rank_logit_idx_mean_abs"] <= 8:
            ax.annotate(curto(r["atributo"]), (r["rank_shap_mean_abs"], r["rank_logit_idx_mean_abs"]),
                        fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Posição no ranking SHAP (XGBoost, teste)")
    ax.set_ylabel("Posição no ranking do logit ordinal")
    ax.set_title(f"Rankings de atributos: SHAP × logit ordinal\n"
                 f"Spearman $\\rho$ = {spear['logit_idx_mean_abs']['spearman']:.2f}".replace(".", ","))
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.invert_xaxis(); ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "econometria_logit_vs_shap.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- figura: AME P(nota 5)
    ta = tab[tab.grupo == "atributo"].sort_values("AME_P5")
    fig, ax = plt.subplots(figsize=(6.3, 0.22 * len(ta) + 1.2))
    cores = ["#c53030" if p < 0.05 else "#a0aec0" for p in ta.p]
    ax.errorbar(ta.AME_P5 * 100, range(len(ta)), xerr=1.96 * ta.AME_P5_ep * 100, fmt="none",
                ecolor="#4a5568", lw=0.8, capsize=2)
    ax.scatter(ta.AME_P5 * 100, range(len(ta)), c=cores, s=22, zorder=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(ta)))
    ax.set_yticklabels([curto(c) + (" †" if c in hier else "") for c in ta["var"]], fontsize=7.5)
    ax.set_xlabel("Efeito marginal médio sobre P(nota 5), em p.p. por ponto (IC 95%)")
    ax.set_title("Logit ordinal (post-double-selection): efeitos marginais médios")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "econometria_logit_ame.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- tabelas LaTeX
    escrever_tex_coef(tab, hier, resultados, brant, len(ctrl_D))
    escrever_tex_pred(resultados, len(te))
    print("OK")


def escrever_tex_coef(tab, hier, resultados, brant, n_ctrl):
    t = tab[tab.grupo == "atributo"].copy()
    t = t.reindex(t.AME_P5.abs().sort_values(ascending=False).index)
    ramo = tab[tab.grupo == "ramo"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Logit ordinal (\emph{post-double-selection}) da satisfação geral: coeficientes, "
         r"razões de chances e efeitos marginais médios sobre $\Pr(\text{nota}=5)$ --- base MVP 2021--2026, treino}",
         r"\label{tab:econometria-logit}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Atributo & $\hat\beta$ & EP & RC & EMM $\Pr(5)$ (p.p.) & $n$ válido \\", r"\midrule"]
    for _, r in t.iterrows():
        nome = tex_escape(curto(r["var"])) + (r"$^{\dagger}$" if r["var"] in hier else "")
        L.append(f"{nome} & {fmt(r.beta, 3)}{estrelas(r.p)} & {fmt(r.ep, 3)} & {fmt(r.OR, 2)} & "
                 f"{fmt(100 * r.AME_P5, 2)} ({fmt(100 * r.AME_P5_ep, 2)}) & {int(r.n_validos_treino)} \\\\")
    if len(ramo):
        L.append(r"\midrule")
        for _, r in ramo.iterrows():
            L.append(f"{tex_escape(curto(r['var']))} & {fmt(r.beta, 3)}{estrelas(r.p)} & {fmt(r.ep, 3)} & "
                     f"{fmt(r.OR, 2)} & -- & -- \\\\")
    rD = resultados["D"]
    L += [r"\midrule",
          f"Controles selecionados (dupla seleção) & \\multicolumn{{5}}{{r}}{{{n_ctrl}}} \\\\",
          r"Efeitos fixos de aeroporto e período (trimestre) & \multicolumn{5}{r}{Sim} \\",
          r"Indicadores de ausência entre elegíveis e PROCESSO & \multicolumn{5}{r}{Sim} \\",
          f"$N$ (treino) & \\multicolumn{{5}}{{r}}{{{rD['n_treino']}}} \\\\",
          f"Pseudo-$R^2$ de McFadden & \\multicolumn{{5}}{{r}}{{{fmt(rD['pseudoR2_McFadden'], 3)}}} \\\\",
          f"Log-verossimilhança & \\multicolumn{{5}}{{r}}{{{fmt(rD['LL'], 1)}}} \\\\",
          f"Teste tipo Brant (atributos), $\\chi^2$({brant['global']['gl']}) & "
          f"\\multicolumn{{5}}{{r}}{{{fmt(brant['global']['W'], 1)} ({'$p<0{,}001$' if brant['global']['p'] < 0.001 else '$p$ = ' + fmt(brant['global']['p'], 3)})}} \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: erros-padrão robustos (HC0). "
          r"RC = razão de chances cumulativa $\exp(\hat\beta)$. EMM = efeito marginal médio de +1 ponto "
          r"sobre $\Pr(\text{nota}=5)$, calculado entre os respondentes com nota válida no atributo "
          r"(EP pelo método delta). $^{\dagger}$Sub-item perguntado apenas quando o item-pai recebe "
          r"nota $\leq 3$ (lógica de salto do questionário). Atributos centrados na média do treino e "
          r"zerados quando não perguntados. *** $p<0{,}01$; ** $p<0{,}05$; * $p<0{,}10$.}",
          r"\end{table}"]
    (TEX_DIR / "econometria_logit.tex").write_text("\n".join(L) + "\n")


def escrever_tex_pred(resultados, n_teste):
    ml = pd.read_csv(ROOT / "codigos" / "04_modelos_ml" / "tabela_comparativa.csv")
    linhas = []
    for _, r in ml.iterrows():
        linhas.append((r["nome"], r["MAE_test"], r["RMSE_test"], r["R2_test"], r["QWK_test"], "ML"))
    for k in ["C", "L", "D"]:
        m = resultados[k]["test"]
        linhas.append((resultados[k]["nome"] + f" ($k={resultados[k]['k_regressores']}$)",
                       m["MAE"], m["RMSE"], m["R2"], m["QWK"], "Econ"))
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Desempenho preditivo no conjunto de teste (mesmo \emph{split}): modelos de "
         r"\emph{Machine Learning} e logit ordinal --- base MVP 2021--2026}",
         r"\label{tab:econometria-logit-preditivo}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Modelo & MAE & RMSE & $R^2$ & QWK \\", r"\midrule"]
    prev = "ML"
    for nome, mae, rmse, r2, qwk, g in linhas:
        if g != prev:
            L.append(r"\midrule")
            prev = g
        L.append(f"{nome} & {fmt(mae)} & {fmt(rmse)} & {fmt(r2)} & {fmt(qwk)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: $n_{\text{teste}}=" + f"{n_teste:,}".replace(",", ".") + r"$. Para o logit "
          r"ordinal, a previsão pontual é $\mathbb{E}[Y\mid X]=\sum_k k\,\hat{\Pr}(Y=k\mid X)$; o QWK usa "
          r"a previsão arredondada, como nos demais modelos. $k$ = número de regressores além dos "
          r"pontos de corte.}", r"\end{table}"]
    (TEX_DIR / "econometria_logit_preditivo.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
