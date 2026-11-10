"""
T6/econometria — Double/Debiased ML (modelo parcialmente linear) para os 4 atributos
de maior importância SHAP (XGBoost, teste, base MVP).

  Y = theta * D + g(X) + e,   D = m(X) + v      (Chernozhukov et al., 2018; DoubleML)
  Y = satisfação geral (1–5); D = nota do atributo (1–5); nuisance g, m = LightGBM;
  cross-fitting 5 folds; 5 repetições de partição (mediana, Chernozhukov et al. 2018, def. 3.3).

Desenho específico do questionário (lógica de salto):
  * a amostra de cada atributo é a dos respondentes com nota válida nele (ramo PROCESSO e,
    para sub-itens, pai <= 3). Ex.: SINALIZAÇÃO só é estimada entre quem deu LOCALIZAÇÃO <= 3.
  * DESCENDENTES do atributo (sub-itens perguntados só quando ele é <= 3) são EXCLUÍDOS de X:
    são mediadores e sua presença/ausência revela D <= 3 (violaria sobreposição).
  * demais atributos entram com NaN nativo do LightGBM (ausência estrutural preservada).

Comparações: MQO simples, MQO com controles lineares, DML e a inclinação SHAP
(regressão de phi_D em D entre observados no teste). Sensibilidade a confundidor omitido
(Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis, 2022): valores de robustez RV.
Bases: MVP (principal, comparável ao SHAP) e 20 aeroportos (extensão), série MPor 2021T1–2026T2.
Tempo: dummies de PERIODO (21 trimestres) em X (antes: TRIMESTRE numérico + dummies T1–T4 de 2023).
Na base de 20 aeroportos, se FRAC_20 < 1, o DML usa uma subamostra aleatória (estratificada por
aeroporto x período, semente fixa) para caber no tempo de execução; MQO usa a mesma subamostra.
"""
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from comum import *

warnings.filterwarnings("ignore")
N_FOLDS, N_REP = 5, 5
# fração da base de 20 aeroportos usada no DML (1.0 = base inteira); ver docstring
FRAC_20 = float(os.environ.get("DML_FRAC_20", "1.0"))
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=SEED, verbose=-1, n_jobs=4)


def descendentes(a, hier):
    out, fila = set(), [a]
    while fila:
        p = fila.pop()
        for f, pai in hier.items():
            if pai == p and f not in out:
                out.add(f)
                fila.append(f)
    return out


def montar_X(df, atributos, excluir):
    """Atributos (NaN nativo) + controles codificados + FE aeroporto/período + PROCESSO."""
    A = pd.DataFrame({a: likert_num(df[a]) for a in atributos if a not in excluir}, index=df.index)
    A.columns = [f"A{i}" for i in range(A.shape[1])]
    C = controles_limpos(df)
    Cd = pd.get_dummies(C, dummy_na=True, dtype=float)
    F = pd.get_dummies(df[["AEROPORTO"]].astype(str), dtype=float)
    F = pd.concat([F, pd.get_dummies(df["PERIODO"].astype(str), prefix="P", dtype=float)], axis=1)
    F["EMB"] = (df["PROCESSO"].astype(str) == "Embarque").astype(float)
    X = pd.concat([A, Cd, F], axis=1)
    X.columns = [f"x{i}" for i in range(X.shape[1])]
    return X


def mqo(y, D, Xlin=None):
    Z = D.to_frame("D") if Xlin is None else pd.concat([D.rename("D"), Xlin], axis=1)
    Z = sm.add_constant(Z, has_constant="add")
    r = sm.OLS(y.values, Z.values).fit(cov_type="HC1")
    return float(r.params[1]), float(r.bse[1])


def lineariza(X):
    """Para MQO com controles: NaN -> 0 + indicador de ausência."""
    out = X.copy()
    for c in X.columns:
        if X[c].isna().any():
            out[c + "_na"] = X[c].isna().astype(float)
            out[c] = X[c].fillna(X[c].mean())
    out = out.loc[:, out.std() > 0]
    return out


def dml(y, D, X, seed=SEED):
    dat = pd.concat([X, D.rename("D"), y.rename("Y")], axis=1)
    data = DoubleMLData(dat, y_col="Y", d_cols="D", x_cols=list(X.columns),
                        force_all_x_finite="allow-nan")
    np.random.seed(seed)
    m = DoubleMLPLR(data, LGBMRegressor(**LGB), LGBMRegressor(**LGB),
                    n_folds=N_FOLDS, n_rep=N_REP, score="partialling out")
    np.random.seed(seed)
    m.fit(store_predictions=True)
    ci = m.confint(level=0.95)
    theta, se = float(m.coef[0]), float(m.se[0])
    # qualidade das nuisances (RMSE fora da dobra, 1a repetição)
    rm = m.evaluate_learners()
    rmse_l = float(np.mean(rm["ml_l"])); rmse_m = float(np.mean(rm["ml_m"]))
    r2_m = 1 - rmse_m ** 2 / float(D.var(ddof=0))
    m.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    sp = m.sensitivity_params
    rv = float(np.ravel(sp["rv"])[0]); rva = float(np.ravel(sp["rva"])[0])
    return {"theta": theta, "se": se, "ci_lo": float(ci.iloc[0, 0]), "ci_hi": float(ci.iloc[0, 1]),
            "p": float(m.pval[0]), "rmse_l": rmse_l, "rmse_m": rmse_m, "r2_m": r2_m,
            "RV": rv, "RVa": rva,
            "theta_lo_cf03": float(np.ravel(sp["theta"]["lower"])[0]),
            "theta_hi_cf03": float(np.ravel(sp["theta"]["upper"])[0])}


def rodar_base(df, nome_base, tratamentos, shap_slopes=None, por_aeroporto=False):
    todos = colunas_atributos(df)
    N = pd.DataFrame({a: likert_num(df[a]) for a in todos})
    hier = detectar_hierarquia(N)
    share = N.notna().mean()
    atributos = [a for a in todos if share[a] >= 0.03]
    y_all = df["y"].astype(float)
    linhas = []
    for a in tratamentos:
        desc = descendentes(a, hier)
        amostra = N[a].notna()
        grupos = [("Todos", amostra)]
        if por_aeroporto:
            grupos += [(ap, amostra & (df["AEROPORTO"] == ap)) for ap in sorted(df["AEROPORTO"].unique())]
        for gnome, mask in grupos:
            d = df.loc[mask]
            X = montar_X(d, atributos, excluir={a} | desc)
            if gnome != "Todos":
                X = X.loc[:, X.std() > 0]
            y = y_all.loc[mask]
            D = N.loc[mask, a]
            b0, s0 = mqo(y, D)
            b1, s1 = mqo(y, D, lineariza(X))
            t0 = time.time()
            r = dml(y, D, X)
            r["tempo_s"] = time.time() - t0
            linha = {"base": nome_base, "grupo": gnome, "atributo": a, "pai": hier.get(a, ""),
                     "descendentes_excluidos": "; ".join(sorted(desc)), "n": int(mask.sum()),
                     "media_D": float(D.mean()), "sd_D": float(D.std()),
                     "mqo_simples": b0, "mqo_simples_ep": s0,
                     "mqo_controles": b1, "mqo_controles_ep": s1, **{f"dml_{k}": v for k, v in r.items()}}
            if shap_slopes is not None and gnome == "Todos":
                linha.update(shap_slopes.get(a, {}))
            linhas.append(linha)
            print(f"  [{nome_base}|{gnome}] {curto(a):28s} n={mask.sum():6d} MQO={b0:.3f} "
                  f"MQO+X={b1:.3f} DML={r['theta']:.3f} [{r['ci_lo']:.3f};{r['ci_hi']:.3f}] "
                  f"R2(m)={r['r2_m']:.2f} RV={r['RV']:.3f}", flush=True)
    return pd.DataFrame(linhas)


def slopes_shap(df, idx, tratamentos):
    """Inclinação SHAP: MQO de phi_D sobre D entre observados no teste (pontos de satisfação por ponto)."""
    shap = pd.read_parquet(PROCESSADOS / "shap_values_test.parquet")
    Xte = pd.read_parquet(SPLITS / "X_test.parquet")
    out = {}
    for a in tratamentos:
        v = Xte[a]
        ok = v > 0
        phi = shap.loc[ok, a].astype(float)
        r = sm.OLS(phi.values, sm.add_constant(v[ok].astype(float).values)).fit(cov_type="HC1")
        phi_all = shap[a].astype(float)
        out[a] = {"shap_slope": float(r.params[1]), "shap_slope_ep": float(r.bse[1]),
                  "shap_mean_abs": float(phi_all.abs().mean()),
                  "shap_share_absphi_de_ausentes": float(phi_all[~ok].abs().sum() / phi_all.abs().sum()),
                  "shap_n_obs_teste": int(ok.sum())}
    return out


def main():
    estilo()
    shap = pd.read_parquet(PROCESSADOS / "shap_values_test.parquet").drop(columns="AEROPORTO").astype(float)
    imp = shap.abs().mean().drop(labels=["TERMINAL"], errors="ignore")
    atr_ml = [c for c in imp.index if c in json.loads((SPLITS / "feature_metadata.json").read_text())["likert_cols"]]
    top4 = list(imp[atr_ml].sort_values(ascending=False).index[:4])
    print("Top-4 SHAP (atributos):", top4)

    df, idx = carregar_mvp()
    sl = slopes_shap(df, idx, top4)
    res_mvp = rodar_base(df, "MVP", top4, shap_slopes=sl, por_aeroporto=True)
    df20, nome20 = carregar_20(colunas=list(ATRIBUTOS_QUESTIONARIO) + CONTROLES_CAT +
                               ["AEROPORTO", "PERIODO", "PROCESSO"])
    n20 = len(df20)
    if FRAC_20 < 1:
        df20 = (df20.groupby(["AEROPORTO", "PERIODO"], group_keys=False)
                .sample(frac=FRAC_20, random_state=SEED).sort_index().reset_index(drop=True))
    print(f"Base 20 aeroportos: {nome20}, n={n20:,}; usada no DML: {len(df20):,} (fração {FRAC_20:g})")
    res_20 = rodar_base(df20, "20 aeroportos", top4)
    res_20["frac_base"] = FRAC_20
    res_20["n_base_completa"] = n20
    res = pd.concat([res_mvp, res_20], ignore_index=True)
    res.to_csv(OUT_DIR / "dml_resultados.csv", index=False)

    figura(res, top4)
    tabela(res, top4)
    tabela_aeroporto(res, top4)
    print("OK")


def figura(res, top4):
    import matplotlib.pyplot as plt
    r = res[res.grupo == "Todos"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    series = [("MQO simples (MVP)", "MVP", "mqo_simples", None, "#a0aec0", "o"),
              ("MQO c/ controles (MVP)", "MVP", "mqo_controles", "mqo_controles_ep", "#718096", "s"),
              ("DML-LightGBM (MVP)", "MVP", "dml_theta", "dml_se", "#2b6cb0", "D"),
              ("DML-LightGBM (20 aeroportos)", "20 aeroportos", "dml_theta", "dml_se", "#c05621", "D"),
              ("Inclinação SHAP (XGBoost, MVP)", "MVP", "shap_slope", "shap_slope_ep", "#2f855a", "^")]
    w = 0.15
    for i, (lab, base, col, ecol, cor, mk) in enumerate(series):
        rr = r[r.base == base].set_index("atributo").reindex(top4)
        x = np.arange(len(top4)) + (i - 2) * w
        ax.scatter(x, rr[col], color=cor, marker=mk, s=26, label=lab, zorder=3)
        if ecol:
            ax.errorbar(x, rr[col], yerr=1.96 * rr[ecol], fmt="none", ecolor=cor, lw=0.9, capsize=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(top4)))
    ax.set_xticklabels([curto(a) for a in top4], fontsize=8.5)
    ax.set_ylabel("Variação na satisfação geral\npor +1 ponto no atributo")
    ax.set_title("Efeito de +1 ponto no atributo: MQO, DML e SHAP (IC 95%)")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "econometria_dml.pdf", bbox_inches="tight")
    plt.close(fig)


def tabela(res, top4):
    r = res[res.grupo == "Todos"]
    fr = res.loc[res.base == "20 aeroportos", "frac_base"].dropna()
    nota_frac = "" if fr.empty or fr.iloc[0] >= 1 else (
        f"Base de 20 aeroportos: subamostra aleatória de {100 * fr.iloc[0]:.0f}\\% (estratificada por "
        f"aeroporto $\\times$ período; base completa com "
        f"{int(res.loc[res.base == '20 aeroportos', 'n_base_completa'].iloc[0]):,} respondentes). ".replace(",", "."))
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Efeito de +1 ponto no atributo sobre a satisfação geral: MQO, "
         r"\emph{Double/Debiased Machine Learning} e inclinação SHAP}",
         r"\label{tab:econometria-dml}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{Base MVP (SBGR, SBSP, SBKP)} & \multicolumn{2}{c}{20 aeroportos} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r"Atributo & MQO simples & MQO + $X$ & DML & SHAP & DML & RV (\%) \\", r"\midrule"]
    for a in top4:
        m = r[(r.base == "MVP") & (r.atributo == a)].iloc[0]
        g = r[(r.base == "20 aeroportos") & (r.atributo == a)].iloc[0]
        nome = tex_escape(curto(a)) + (r"$^{\dagger}$" if m.pai else "")
        L.append(f"{nome} & {fmt(m.mqo_simples)} & {fmt(m.mqo_controles)} & "
                 f"{fmt(m.dml_theta)}{estrelas(m.dml_p)} & {fmt(m.shap_slope)} & "
                 f"{fmt(g.dml_theta)}{estrelas(g.dml_p)} & {fmt(100 * g.dml_RV, 1)} \\\\")
        L.append(f" & ({fmt(m.mqo_simples_ep)}) & ({fmt(m.mqo_controles_ep)}) & "
                 f"[{fmt(m.dml_ci_lo)}; {fmt(m.dml_ci_hi)}] & ({fmt(m.shap_slope_ep)}) & "
                 f"[{fmt(g.dml_ci_lo)}; {fmt(g.dml_ci_hi)}] & \\\\")
        L.append(f" & \\multicolumn{{4}}{{l}}{{\\scriptsize $n$ = {m.n:,}; $R^2$ de $\\hat m(X)$ = "
                 f"{fmt(m.dml_r2_m, 2)}; RV (MVP) = {fmt(100 * m.dml_RV, 1)}\\%}}".replace(",", ".").replace("{.}", "{,}")
                 + f" & \\multicolumn{{2}}{{l}}{{\\scriptsize $n$ = {g.n:,}}}".replace(",", ".") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: variável dependente = satisfação geral (1--5). "
          r"Amostra de cada linha = respondentes com nota válida no atributo. DML: modelo parcialmente linear, "
          r"\emph{nuisances} LightGBM, \emph{cross-fitting} com 5 dobras e 5 repetições (mediana); IC 95\% entre "
          r"colchetes. MQO: EP robustos (HC1) entre parênteses; $X$ = demais atributos (com indicadores de "
          r"ausência), perfil do passageiro, efeitos fixos de aeroporto e período (trimestre) e PROCESSO. SHAP: inclinação "
          r"da regressão de $\phi_D$ sobre $D$ entre observados no teste (XGBoost). RV = valor de robustez "
          r"\parencite{chernozhukov2022omitted}: fração da variação residual de $Y$ e de $D$ que um confundidor "
          r"omitido precisaria explicar para anular a estimativa. $^{\dagger}$Sub-item de Localização e "
          r"deslocamento (perguntado se esta $\leq 3$); descendentes do atributo excluídos de $X$. "
          + nota_frac +
          r"*** $p<0{,}01$; ** $p<0{,}05$; * $p<0{,}10$.}",
          r"\end{table}"]
    (TEX_DIR / "econometria_dml.tex").write_text("\n".join(L) + "\n")


def tabela_aeroporto(res, top4):
    r = res[(res.base == "MVP") & (res.grupo != "Todos")]
    aps = sorted(r.grupo.unique())
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{DML por aeroporto (base MVP): efeito de +1 ponto no atributo sobre a satisfação geral}",
         r"\label{tab:econometria-dml-aeroporto}",
         r"\begin{tabular}{l" + "r" * len(aps) + "}", r"\toprule",
         "Atributo & " + " & ".join(aps) + r" \\", r"\midrule"]
    for a in top4:
        cel, cel2 = [], []
        for ap in aps:
            m = r[(r.atributo == a) & (r.grupo == ap)].iloc[0]
            cel.append(f"{fmt(m.dml_theta)}{estrelas(m.dml_p)}")
            cel2.append(f"[{fmt(m.dml_ci_lo)}; {fmt(m.dml_ci_hi)}]")
        L.append(tex_escape(curto(a)) + " & " + " & ".join(cel) + r" \\")
        L.append(" & " + " & ".join(cel2) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: mesma especificação da "
          r"Tabela~\ref{tab:econometria-dml}, estimada separadamente em cada aeroporto. IC 95\% entre colchetes.}",
          r"\end{table}"]
    (TEX_DIR / "econometria_dml_aeroporto.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
