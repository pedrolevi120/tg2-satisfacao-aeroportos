"""
E2 — Explainable Boosting Machine (EBM; Lou et al., 2013; Nori et al., 2019 — pacote `interpret`).

Modelo aditivo generalizado (GAM) ajustado por boosting: a predição é a soma de
uma função de forma por atributo (e, opcionalmente, de poucos pares), e cada
função pode ser lida num gráfico — interpretável por construção, sem SHAP.

Protocolo (pareado com o pipeline): mesmas features, mesmo split, um modelo por
ramo do questionário, ausentes nativos (EBM cria um "bin" de ausente). O EBM faz
early stopping numa validação interna (15% do treino); o conjunto de validação do
pipeline é usado só para reportar o MAE de validação. Duas variantes:
  * EBM-GAM: só efeitos principais (interactions=0) — o mais transparente;
  * EBM-GA2M: + 20 interações de pares (as mais fortes, detectadas pelo FAST).
Importância por DIMENSÃO: contribuições locais (eval_terms) somadas linha a linha
nos itens da dimensão (pares divididos meio a meio entre os dois atributos),
média de |·| nas linhas de teste — mesma lógica do SHAP por dimensão.

Saída: resultados/e2_ebm_metricas.csv, e2_ebm_ranking.csv, e2_ebm_top4_aeroporto.csv,
       e2_pred_test_ebm*.parquet, figuras/e2_ebm_vs_shap.png, figuras/e2_ebm_formas_top4.png
"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from interpret.glassbox import ExplainableBoostingRegressor

from exp_comum import (RES, PROC_OUT, SEED, N_JOBS, TOP4, PorRamoF, Timer, avaliar, carregar,
                       comparar_pareado, preds_atuais, grupos_dim, nome_dim, fmt_ic, estilo_slide,
                       salvar, fmt_n, AZUL, LARANJA, CINZA, TINTA2, shap_por_dimensao)

warnings.filterwarnings("ignore")
VARIANTES = {"ebm_gam": dict(interactions=0), "ebm_ga2m": dict(interactions=20)}
OUTER_BAGS = int(os.environ.get("OUTER_BAGS", "8"))


def contrib_por_feature(model, X, ramo, colunas_todas):
    """Contribuições locais por feature original (n × p); pares divididos ao meio."""
    C = pd.DataFrame(0.0, index=X.index, columns=colunas_todas)
    for r, m in model.modelos.items():
        mask = ramo == r
        if not mask.any():
            continue
        cols = model.colunas[r]
        T = m.eval_terms(X.loc[mask, cols])
        A = np.zeros((mask.sum(), len(cols)))
        for j, tf in enumerate(m.term_features_):
            for f in tf:
                A[:, f] += T[:, j] / len(tf)
        C.loc[mask, cols] = A
    return C


def main():
    d = carregar()
    X, y, ramo, tr, va, te, aero = d["X"], d["y"], d["ramo"], d["tr"], d["va"], d["te"], d["aero"]
    P = preds_atuais()
    ref = P.loc[te, "pred_xgb"].values
    _, grupos = grupos_dim()
    S = pd.read_parquet(PROC_OUT / "shap_values_20.parquet")
    Dshap = shap_por_dimensao(S[X.columns], grupos).abs()
    imp_shap = Dshap.mean().sort_values(ascending=False)

    linhas, rankings, top4_aero, modelos, contribs = [], {"shap_xgb": imp_shap}, [], {}, {}
    for nome, kw in VARIANTES.items():
        def fab(r, kw=kw):
            return ExplainableBoostingRegressor(outer_bags=OUTER_BAGS, n_jobs=N_JOBS,
                                                random_state=SEED, **kw)

        def aj(m, Xt, yt, Xv, yv):
            m.fit(Xt, yt)
            return m
        with Timer(f"{nome} (outer_bags={OUTER_BAGS})") as t:
            m = PorRamoF(fab, aj).fit(X[tr], y[tr], ramo[tr], X[va], y[va], ramo[va])
        modelos[nome] = m
        pv, pt = m.predict(X[va], ramo[va]), m.predict(X[te], ramo[te])
        pd.DataFrame({"idx": np.where(te)[0], "pred": pt}).to_parquet(RES / f"e2_pred_test_{nome}.parquet")
        r = comparar_pareado(y[te], ref, pt)
        mv = avaliar(y[va], pv)
        linhas.append({"modelo": nome, "tempo_s": t.dt, "MAE_val": mv["MAE"], "R2_val": mv["R2"], **r})
        print(f"  {nome}: teste MAE {fmt_ic(r['MAE'], r['MAE_lo'], r['MAE_hi'], 4)} "
              f"R2 {fmt_ic(r['R2'], r['R2_lo'], r['R2_hi'], 3)} QWK {r['QWK']:.3f} | "
              f"ΔMAE vs XGB {fmt_ic(r['dMAE'], r['dMAE_lo'], r['dMAE_hi'], 4, True)}", flush=True)

        # contribuições locais → dimensão (teste)
        Xte = X.loc[te]
        C = contrib_por_feature(m, Xte, ramo[te], list(X.columns))
        pred_chk = C.sum(axis=1).values + np.array([m.modelos[r].intercept_ for r in ramo[te]])
        assert np.allclose(pred_chk, pt, atol=1e-6), "contribuições não somam à predição"
        Dsig = shap_por_dimensao(C, grupos)
        contribs[nome] = Dsig
        D = Dsig.abs()
        imp = D.mean().sort_values(ascending=False)
        rankings[nome] = imp
        frac_likert = C[d["likert"]].abs().sum().sum() / C.abs().sum().sum()
        print(f"  {nome}: fração |contrib| em itens Likert = {frac_likert:.3f}; top-6: "
              + " > ".join(f"{nome_dim(k)} ({v:.3f})" for k, v in imp.head(6).items()))
        for A in sorted(np.unique(aero[te])):
            ma = aero[te] == A
            ia = D[ma].mean().sort_values(ascending=False)
            top4_aero.append({"modelo": nome, "aeroporto": A, "top4": " > ".join(nome_dim(k) for k in ia.index[:4]),
                              "top4_igual_shap_conjunto": set(ia.index[:4]) == set(TOP4)})
        if nome == "ebm_gam":
            with open(RES / "e2_ebm_gam.pkl", "wb") as fh:
                pickle.dump({"modelos": m.modelos, "colunas": m.colunas}, fh)

    met = pd.DataFrame(linhas)
    met.to_csv(RES / "e2_ebm_metricas.csv", index=False)
    R = pd.DataFrame(rankings)
    R = R.loc[imp_shap.index]
    for c in list(R.columns):
        R[f"rank_{c}"] = R[c].rank(ascending=False).astype(int)
        R[f"share_{c}"] = R[c] / R[c].sum()
    R.index.name = "dimensao"
    R.insert(0, "nome", [nome_dim(k) for k in R.index])
    R.to_csv(RES / "e2_ebm_ranking.csv")
    T4 = pd.DataFrame(top4_aero)
    T4.to_csv(RES / "e2_ebm_top4_aeroporto.csv", index=False)
    resumo = {}
    for nome in VARIANTES:
        rho = spearmanr(R["shap_xgb"], R[nome]).statistic
        resumo[nome] = {"spearman_vs_shap": float(rho),
                        "top4_global": [nome_dim(k) for k in rankings[nome].index[:4]],
                        "top4_conjunto_igual": set(rankings[nome].index[:4]) == set(TOP4),
                        "aeroportos_com_mesmo_top4": int(T4[T4.modelo == nome].top4_igual_shap_conjunto.sum())}
    (RES / "e2_ebm_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=1, ensure_ascii=False))
    print(R[["nome"] + [c for c in R.columns if c.startswith("rank_")]].to_string())

    # ---------- figura 1: participação na importância, EBM-GAM vs SHAP-XGBoost ----------
    plt = estilo_slide()
    top = R.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_axisbelow(True)
    yy = np.arange(len(top))
    ax.hlines(yy, top["share_shap_xgb"] * 100, top["share_ebm_gam"] * 100, color="#cfcfca", lw=2.5, zorder=1)
    ax.scatter(top["share_shap_xgb"] * 100, yy, s=90, color=AZUL, zorder=3, label="XGBoost + SHAP (atual)",
               edgecolor="white", linewidth=2)
    ax.scatter(top["share_ebm_gam"] * 100, yy, s=90, color=LARANJA, zorder=3, label="EBM (aditivo, interpretável)",
               edgecolor="white", linewidth=2, marker="D")
    ax.set_yticks(yy)
    ax.set_yticklabels(top["nome"])
    ax.set_xlabel("Participação na importância total das dimensões (%)")
    rho = resumo["ebm_gam"]["spearman_vs_shap"]
    ax.set_title("Dois métodos diferentes, a mesma hierarquia")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, f"Teste 2021–2026 (n = {fmt_n(te.sum())}). Importância da dimensão = média de |contribuição| do item-pai + subitens. "
             f"Correlação de Spearman entre os rankings das {len(R)} dimensões: {rho:.2f}".replace("0.", "0,") + ".",
             fontsize=9.5, color=TINTA2, ha="left", va="top")
    fig.tight_layout()
    salvar(fig, "e2_ebm_vs_shap.png")

    # ---------- figura 2: contribuição média da DIMENSÃO por nota do item-pai (EBM-GAM vs SHAP) ----------
    Dg = contribs["ebm_gam"]
    Ds = shap_por_dimensao(S[X.columns], grupos)
    Xte = X.loc[te].reset_index(drop=True)
    Dg, Ds = Dg.reset_index(drop=True), Ds.reset_index(drop=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    curvas = []
    for ax, dim in zip(axes, TOP4):
        nota = Xte[dim]
        for lab, DD, cor, mk in [("EBM (aditivo)", Dg, LARANJA, "D"), ("XGBoost + SHAP", Ds, AZUL, "o")]:
            v = [DD.loc[nota == k, dim].mean() if (nota == k).sum() >= 20 else np.nan for k in [1, 2, 3, 4, 5]]
            curvas += [{"dimensao": dim, "modelo": lab, "nota": k, "contrib_media": vv,
                        "n": int((nota == k).sum())} for k, vv in zip([1, 2, 3, 4, 5], v)]
            ax.plot([1, 2, 3, 4, 5], v, marker=mk, color=cor, lw=2, ms=8, label=lab,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.axhline(0, color=CINZA, lw=1)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(nome_dim(dim), fontsize=13)
        ax.set_xlabel("nota do item-pai")
    pd.DataFrame(curvas).to_csv(RES / "e2_curvas_dimensao.csv", index=False)
    axes[0].set_ylabel("contribuição média da dimensão\n(pontos da escala 1–5)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10.5)
    fig.suptitle("Efeito de cada nota na satisfação prevista (teste; notas com n < 20 omitidas)",
                 x=0.01, ha="left", fontweight="bold", fontsize=15)
    fig.tight_layout()
    salvar(fig, "e2_ebm_formas_top4.png")


if __name__ == "__main__":
    main()
