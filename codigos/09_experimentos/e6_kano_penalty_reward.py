"""
E6 — Impacto assimétrico das dimensões (Kano / penalty-reward contrast analysis).

Pergunta: uma nota RUIM numa dimensão tira mais satisfação do que uma nota ÓTIMA
acrescenta? (Brandt, 1987; Matzler & Sauerwein, 2002; Mikulić & Prebežac, 2008)

Modelo (MQO, todas as respostas válidas da série 2021T1–2026T2, pois é análise
inferencial e não preditiva): satisfação geral (1–5) ~ para cada uma das dimensões
(nota do ITEM-PAI): dummies de faixa {1–2, 4, 5, não respondido}, referência = nota 3
+ efeitos fixos de aeroporto e período (PERIODO, ano-trimestre) + ramo (PROCESSO) +
perfil do passageiro (mesmas dummies do pipeline). Erros-padrão agrupados por
aeroporto × período.
Subitens ficam de fora (só existem quando o pai ≤ 3; o pai resume a dimensão).

Penalidade P = −β(1–2)  (quanto a satisfação cai com nota 1–2 em vez de 3)
Recompensa R = β(5)     (quanto sobe com nota 5 em vez de 3)
Índice de assimetria IA = (R − P)/(R + P) ∈ [−1, 1] (Mikulić & Prebežac, 2008),
IC 95% por simulação da normal assintótica de (β(1–2), β(5)) com a covariância
agrupada. Classificação:
  básico ("só pune")         : P significativo e (R não significativo ou IA < −0,2)
  encantamento ("só premia") : R significativo e (P não significativo ou IA > +0,2)
  desempenho (simétrico)     : P e R significativos e |IA| ≤ 0,2
Sensibilidade: (i) efeito por ponto de escala (P/(3 − nota média do grupo 1–2),
R/2); (ii) curva SHAP do XGBoost atual (SHAP médio da dimensão por faixa de nota
do pai, linhas de teste) — versão não paramétrica e multivariada do mesmo contraste.

Saída: resultados/e6_kano.csv, e6_kano_coeficientes.csv, figuras/e6_kano_penalty_reward.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from exp_comum import (RES, PROC_OUT, PIPE_RES, SEED, carregar, grupos_dim, nome_dim, estilo_slide, salvar, fmt_n,
                       AZUL, LARANJA, CINZA, TINTA2)

warnings.filterwarnings("ignore")
FAIXAS = ["1-2", "4", "5", "NA"]  # referência = 3


def faixa(s):
    f = pd.Series("NA", index=s.index, dtype=object)
    f[s.isin([1, 2])] = "1-2"
    f[s == 3] = "3"
    f[s == 4] = "4"
    f[s == 5] = "5"
    return f


def sem_colinearidade(Z, prioridade):
    """Mantém colunas na ordem de prioridade enquanto aumentam o posto (Gram)."""
    G = Z.T @ Z
    keep = []
    rank = 0
    for j in prioridade:
        cand = keep + [j]
        r = np.linalg.matrix_rank(G[np.ix_(cand, cand)], tol=1e-8 * G[j, j] if G[j, j] > 0 else None)
        if r > rank:
            keep, rank = cand, r
    return keep


def main():
    d = carregar()
    X, meta, y = d["X"], d["meta"], d["y"]
    pais, grupos = grupos_dim()
    dims = list(grupos)  # 19 itens-pai

    # ----- matriz do modelo -----
    blocos, nomes_dim = [], []
    for dm in dims:
        f = faixa(X[dm])
        for k in FAIXAS:
            nomes_dim.append(f"{dm}||{k}")
            blocos.append((f == k).astype(float).values)
    Zd = np.column_stack(blocos)
    # controles: dummies de aeroporto, período, perfil (as do pipeline) + PROCESSO
    cat_cols = [c for c in X.columns if any(c.startswith(p + "_") for p in
                ["AEROPORTO", "PERIODO", "GÊNERO", "ESCOLARIDADE", "RENDA FAMILIAR", "MOTIVO DA VIAGEM",
                 "TIPO DE VOO", "NACIONALIDADE", "VIAJANDO SOZINHO", "IDADE", "CONEXÃO"])]
    # uma categoria de referência por variável (a mais frequente)
    refs = set()
    for p in ["AEROPORTO", "PERIODO", "GÊNERO", "ESCOLARIDADE", "RENDA FAMILIAR", "MOTIVO DA VIAGEM",
              "TIPO DE VOO", "NACIONALIDADE", "VIAJANDO SOZINHO", "IDADE", "CONEXÃO"]:
        cs = [c for c in cat_cols if c.startswith(p + "_")]
        refs.add(X[cs].sum().idxmax())
    ctrl = [c for c in cat_cols if c not in refs]
    Zc = np.column_stack([X[ctrl].values.astype(float), (meta.PROCESSO == "Embarque").astype(float).values])
    nomes_c = ctrl + ["PROCESSO_Embarque"]
    Z = np.column_stack([np.ones(len(X)), Zd, Zc])
    nomes = ["const"] + nomes_dim + nomes_c
    keep = sem_colinearidade(Z, list(range(Z.shape[1])))
    caidas = [nomes[j] for j in range(len(nomes)) if j not in keep]
    print(f"  {Z.shape[1]} colunas; removidas por colinearidade exata: {caidas}")
    Z, nomes = Z[:, keep], [nomes[j] for j in keep]

    grupo = (meta.AEROPORTO + "_" + meta.PERIODO.astype(str)).values

    def estimar(mask):
        res = sm.OLS(y[mask], Z[mask]).fit(cov_type="cluster",
                                           cov_kwds={"groups": pd.factorize(grupo[mask])[0]})
        b = pd.Series(res.params, index=nomes)
        V = pd.DataFrame(res.cov_params(), index=nomes, columns=nomes)
        se = pd.Series(res.bse, index=nomes)
        return res, b, V, pd.DataFrame({"coef": b, "ep": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se})

    res, b, V, coef = estimar(np.ones(len(y), bool))
    print(f"  MQO: n={int(res.nobs)}, R²={res.rsquared:.3f}, clusters={len(set(grupo))}")
    coef.to_csv(RES / "e6_kano_coeficientes.csv")
    # Sensibilidade: sem respondentes "retilíneos" (>= 4 itens-pai respondidos, todos com a mesma nota)
    Pa = X[dims]
    retil = ((Pa.notna().sum(axis=1) >= 4) & (Pa.nunique(axis=1) == 1)).values
    print(f"  respondentes retilíneos (mesma nota em >= 4 itens-pai): {retil.sum()} ({retil.mean():.1%})")
    res2, b2, V2, coef2 = estimar(~retil)
    print(f"  MQO sem retilíneos: n={int(res2.nobs)}, R²={res2.rsquared:.3f}")

    # ----- curva SHAP (XGBoost atual, teste) -----
    S = pd.read_parquet(PROC_OUT / "shap_values_20.parquet")
    te = d["te"]
    Xte = X.loc[te]
    assert (S.index == Xte.index).all()

    rng = np.random.default_rng(SEED)
    linhas = []
    for dm in dims:
        k_lo, k_hi = f"{dm}||1-2", f"{dm}||5"
        if k_lo not in b.index or k_hi not in b.index:
            continue
        fx = faixa(X[dm])
        n = {k: int((fx == k).sum()) for k in ["1-2", "3", "4", "5"]}
        bl, bh = b[k_lo], b[k_hi]
        P, R = -bl, bh
        cov = V.loc[[k_lo, k_hi], [k_lo, k_hi]].values
        sim = rng.multivariate_normal([bl, bh], cov, 20000)
        Ps, Rs = -sim[:, 0], sim[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            IAs = (Rs - Ps) / (np.abs(Rs) + np.abs(Ps))
        IA = (R - P) / (abs(R) + abs(P))
        media_baixa = X.loc[fx == "1-2", dm].mean()
        Pp, Rp = P / (3 - media_baixa), R / 2
        IAp = (Rp - Pp) / (abs(Rp) + abs(Pp))
        sigP, sigR = coef.loc[k_lo, "hi"] < 0, coef.loc[k_hi, "lo"] > 0
        ia_lo, ia_hi = np.quantile(IAs, [0.025, 0.975])

        def classe(ia, sp, sr):
            if not sp and not sr:
                return "sem efeito claro"
            if sp and (not sr or ia < -0.2):
                return "básico"
            if sr and (not sp or ia > 0.2):
                return "encantamento"
            return "desempenho"

        # SHAP: média do SHAP da dimensão por faixa do pai (teste)
        cols = [c for c in grupos[dm] if c in S.columns]
        sd = S[cols].sum(axis=1)
        fte = faixa(Xte[dm])
        ms = {k: sd[fte.values == k].mean() if (fte == k).sum() >= 20 else np.nan for k in ["1-2", "3", "4", "5"]}
        P2, R2 = -b2[k_lo], b2[k_hi]
        IA2 = (R2 - P2) / (abs(R2) + abs(P2))
        sigP2, sigR2 = coef2.loc[k_lo, "hi"] < 0, coef2.loc[k_hi, "lo"] > 0
        linhas.append({
            "dimensao": dm, "nome": nome_dim(dm), **{f"n_{k}": v for k, v in n.items()},
            "beta_1_2": bl, "beta_1_2_lo": coef.loc[k_lo, "lo"], "beta_1_2_hi": coef.loc[k_lo, "hi"],
            "beta_4": b.get(f"{dm}||4", np.nan),
            "beta_5": bh, "beta_5_lo": coef.loc[k_hi, "lo"], "beta_5_hi": coef.loc[k_hi, "hi"],
            "penalidade": P, "recompensa": R, "IA": IA, "IA_lo": ia_lo, "IA_hi": ia_hi,
            "classe": classe(IA, sigP, sigR),
            "nota_media_grupo_1_2": media_baixa, "penalidade_por_ponto": Pp, "recompensa_por_ponto": Rp,
            "IA_por_ponto": IAp, "classe_por_ponto": classe(IAp, sigP, sigR),
            "penalidade_sem_retilineos": P2, "recompensa_sem_retilineos": R2, "IA_sem_retilineos": IA2,
            "classe_sem_retilineos": classe(IA2, sigP2, sigR2),
            "shap_1_2": ms["1-2"], "shap_3": ms["3"], "shap_4": ms["4"], "shap_5": ms["5"],
            "shap_penalidade": ms["3"] - ms["1-2"], "shap_recompensa": ms["5"] - ms["3"],
        })
    K = pd.DataFrame(linhas)
    def robusta(r):
        cs = [r.classe, r.classe_por_ponto, r.classe_sem_retilineos]
        if len(set(cs)) == 1:
            return cs[0]
        alt = [c for c in dict.fromkeys(cs) if c != r.classe]
        return f"{r.classe} ~ {alt[0]}"
    K["classe_robusta"] = K.apply(robusta, axis=1)
    K["IA_shap"] = (K.shap_recompensa - K.shap_penalidade) / (K.shap_recompensa.abs() + K.shap_penalidade.abs())
    imp = pd.read_csv(PIPE_RES / "ranking_global_20.csv")
    K = K.merge(imp[["dimensao", "mean_abs_shap"]], on="dimensao", how="left").sort_values("mean_abs_shap", ascending=False)
    K.to_csv(RES / "e6_kano.csv", index=False)
    pd.set_option("display.width", 250)
    print(K[["nome", "n_1-2", "n_3", "n_5", "penalidade", "recompensa", "IA", "IA_lo", "IA_hi", "classe",
             "IA_por_ponto", "classe_por_ponto", "IA_sem_retilineos", "classe_sem_retilineos", "classe_robusta", "shap_penalidade", "shap_recompensa", "IA_shap"]].round(3).to_string())

    # ----- figura: penalidade × recompensa (top-10 dimensões por importância SHAP) -----
    plt = estilo_slide()
    Kf = K[(K["n_1-2"] >= 100) & (K["n_5"] >= 100)].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_axisbelow(True)
    yy = np.arange(len(Kf))
    h = 0.36
    ax.barh(yy, -Kf.penalidade, height=0.62, color=LARANJA, label="Penalidade: nota 1–2 em vez de 3")
    ax.barh(yy, Kf.recompensa, height=0.62, color=AZUL, label="Recompensa: nota 5 em vez de 3")
    ax.errorbar(Kf.beta_1_2, yy, xerr=[Kf.beta_1_2 - Kf.beta_1_2_lo, Kf.beta_1_2_hi - Kf.beta_1_2],
                fmt="none", ecolor=TINTA2, capsize=3, lw=1)
    ax.errorbar(Kf.beta_5, yy, xerr=[Kf.beta_5 - Kf.beta_5_lo, Kf.beta_5_hi - Kf.beta_5],
                fmt="none", ecolor=TINTA2, capsize=3, lw=1)
    ax.axvline(0, color=CINZA, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(Kf.nome)
    lim = max(Kf.beta_5_hi.max(), (-Kf.beta_1_2_lo).max()) * 1.15
    ax.set_xlim(-lim, lim * 2.1)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    for i, (_, r) in enumerate(Kf.iterrows()):
        ax.text(lim * 2.08, i, r.classe_robusta, ha="right", va="center", fontsize=11.5, color=TINTA2,
                fontweight="bold" if r.classe_robusta != "desempenho" else "normal")
    ax.set_xlabel("Efeito na satisfação geral (pontos da escala 1–5), com IC 95%")
    ax.set_title("Nas dimensões principais, nota ruim tira e nota ótima soma (fatores de desempenho)"
                 if (K.head(4).classe_robusta == "desempenho").all()
                 else "Penalidade e recompensa das dimensões principais (Kano)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.47, 0.0), frameon=False, fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, f"MQO com {len(dims)} dimensões (faixas de nota do item-pai; ref. = nota 3), efeitos fixos de aeroporto e "
             f"período, ramo e perfil; EP agrupado por aeroporto×período; n = {fmt_n(res.nobs)} (2021–2026).\nTop-10 dimensões por importância SHAP. "
             "Classe pelo índice de assimetria IA = (R − P)/(R + P): básico < −0,2 ≤ desempenho ≤ +0,2 < encantamento.\n"
             "“A ~ B”: a classe vira B em ao menos uma sensibilidade (efeito por ponto de escala; sem quem dá a mesma nota a tudo).",
             fontsize=9.5, color=TINTA2, ha="left", va="top")
    fig.tight_layout()
    salvar(fig, "e6_kano_penalty_reward.png")


if __name__ == "__main__":
    main()
