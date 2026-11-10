"""
E5 — Estabilidade (grau de certeza) do ranking de dimensões.

Três fontes de incerteza, da mais estreita à mais ampla:
 A) AMOSTRA DE TESTE: 1.000 reamostragens bootstrap das linhas de teste sobre os
    SHAP do XGBoost atual (global e por aeroporto, ~620 linhas cada).
 B) AMOSTRA DE TREINO: R re-treinos do XGBoost (mesmos hiperparâmetros) em
    reamostragens bootstrap do treino (early stopping na validação), SHAP no
    teste inteiro; e S re-treinos que só trocam a semente (aleatoriedade do
    algoritmo). Combinação B×A: cada re-treino × 33 reamostragens do teste.
    Os re-treinos também medem quanto o MAE varia "por acaso" — régua para
    julgar se um ganho de MAE é relevante.
 C) MODELO (multiplicidade/"Rashomon"): o ranking por dimensão segundo modelos
    de famílias diferentes e empatados em precisão — XGBoost, LightGBM,
    CatBoost (atuais e ajustados no E1), Random Forest (amostra de 1.000 linhas,
    pelo custo do TreeSHAP), Ridge (SHAP linear exato) e EBM (contribuições
    aditivas do E2).
Saídas: % de vezes que cada dimensão fica no top-4, posição média e intervalo de
posições, P(top-4 = conjunto top-4 global do pipeline, lido de ranking_global_20.csv).

Uso: R_BOOT=30 S_SEEDS=10 python e5_estabilidade_ranking.py
Saída: resultados/e5_*.csv|json, figuras/e5_*.png
"""
import json
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.stats import spearmanr, kendalltau

from exp_comum import (RES, PIPE_MODELS, PROC_OUT, SEED, N_JOBS, TOP4, NOME_AERO, PorRamoF, Timer, carregar,
                       preds_atuais, grupos_dim, nome_dim, shap_por_dimensao, shap_dim_tree, avaliar,
                       estilo_slide, salvar, AZUL, LARANJA, CINZA, TINTA2, CINZA_CLARO)

warnings.filterwarnings("ignore")
R_BOOT = int(os.environ.get("R_BOOT", "30"))
S_SEEDS = int(os.environ.get("S_SEEDS", "10"))
B_ROWS = 1000


def ranks_de(imp_mat, cols):
    """imp_mat: (B × k) importâncias → (B × k) posições (1 = mais importante)."""
    order = np.argsort(-imp_mat, axis=1)
    rk = np.empty_like(order)
    rows = np.arange(imp_mat.shape[0])[:, None]
    rk[rows, order] = np.arange(1, imp_mat.shape[1] + 1)
    return pd.DataFrame(rk, columns=cols)


def resumo_ranks(RK, rotulo):
    top4_idx = [RK.columns.get_loc(c) for c in TOP4]
    conj = (RK.iloc[:, top4_idx] <= 4).all(axis=1)
    out = pd.DataFrame({"pct_top4": (RK <= 4).mean() * 100, "pos_media": RK.mean(),
                        "pos_p2_5": RK.quantile(0.025), "pos_p97_5": RK.quantile(0.975),
                        "pct_top1": (RK == 1).mean() * 100})
    out["fonte"] = rotulo
    return out, float(conj.mean() * 100)


def boot_linhas(Dabs, B, rng):
    V = Dabs.values
    n = len(V)
    M = np.empty((B, V.shape[1]))
    for b in range(B):
        M[b] = V[rng.integers(0, n, n)].mean(0)
    return M


def xgb_fab(seed):
    return lambda r: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS,
                                      random_state=seed, early_stopping_rounds=30, verbosity=0)


def aj_xgb(m, Xt, yt, Xv, yv):
    m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    return m


def ridge_contrib(model, X, ramo):
    """SHAP exato do Ridge (atributos independentes): w·z, com z padronizado no treino
    (média 0); indicadores de ausência somados ao atributo de origem."""
    C = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    for r, pipe in model.modelos.items():
        mask = ramo == r
        cols = model.colunas[r]
        imp, sc, rg = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
        Z = sc.transform(imp.transform(X.loc[mask, cols]))
        phi = Z * rg.coef_
        A = phi[:, :len(cols)].copy()
        if imp.indicator_ is not None:
            for k, j in enumerate(imp.indicator_.features_):
                A[:, j] += phi[:, len(cols) + k]
        C.loc[mask, cols] = A
    return C


def main():
    d = carregar()
    X, y, ramo, tr, va, te, aero = d["X"], d["y"], d["ramo"], d["tr"], d["va"], d["te"], d["aero"]
    _, grupos = grupos_dim()
    dims = list(grupos)
    rng = np.random.default_rng(SEED)
    Xte, rte, ate, yte = X.loc[te], ramo[te], aero[te], y[te]
    airports = sorted(np.unique(ate))

    S = pd.read_parquet(PROC_OUT / "shap_values_20.parquet")
    D0 = shap_por_dimensao(S[X.columns], grupos)[dims].abs().reset_index(drop=True)
    resumos, conjuntos = [], {}

    # ---------------- A) bootstrap das linhas de teste ----------------
    with Timer("A) bootstrap das linhas de teste (global + 20 aeroportos)"):
        RK = ranks_de(boot_linhas(D0, B_ROWS, rng), dims)
        r, c = resumo_ranks(RK, "A_teste_global")
        resumos.append(r)
        conjuntos["A_teste_global"] = c
        por_aero_A = {}
        for A in airports:
            m = ate == A
            RKa = ranks_de(boot_linhas(D0[m], B_ROWS, rng), dims)
            ra, ca = resumo_ranks(RKa, f"A_teste_{A}")
            por_aero_A[A] = (ra, ca)

    # ---------------- B) re-treinos ----------------
    reg_B, Dlist_boot, Dlist_seed = [], [], []
    cache = RES / "_cache_e5_retreinos.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Dlist_boot = [pd.DataFrame(a, columns=dims) for a in z["boot"]]
        Dlist_seed = [pd.DataFrame(a, columns=dims) for a in z["seed"]]
        reg_B = list(z["reg"])
        print(f"  B) re-treinos carregados do cache ({len(Dlist_boot)} + {len(Dlist_seed)})")
    with Timer(f"B) {R_BOOT} re-treinos bootstrap + {S_SEEDS} re-treinos de semente"):
      if not cache.exists():
          idx_tr = np.where(tr)[0]
          for k in range(R_BOOT + S_SEEDS):
              tipo = "bootstrap_treino" if k < R_BOOT else "semente"
              if tipo == "bootstrap_treino":
                  rs = np.random.default_rng(1000 + k)
                  sel = np.sort(rs.choice(idx_tr, len(idx_tr), replace=True))
                  seed = SEED
              else:
                  sel, seed = idx_tr, 2000 + k
              Xb = X.iloc[sel]
              m = PorRamoF(xgb_fab(seed), aj_xgb).fit(Xb, y[sel], ramo[sel], X[va], y[va], ramo[va])
              pt = m.predict(Xte, rte)
              Dk, _ = shap_dim_tree(m, Xte, rte, grupos)
              Dk = Dk[dims].abs().reset_index(drop=True)
              (Dlist_boot if tipo == "bootstrap_treino" else Dlist_seed).append(Dk)
              met = avaliar(yte, pt)
              g = Dk.mean().sort_values(ascending=False)
              reg_B.append({"k": k, "tipo": tipo, **met, "top4": " > ".join(nome_dim(x) for x in g.index[:4]),
                            "top4_conjunto_igual": set(g.index[:4]) == set(TOP4)})
              print(f"    re-treino {k:02d} ({tipo}): MAE={met['MAE']:.4f} R2={met['R2']:.4f} top4 ok={reg_B[-1]['top4_conjunto_igual']}",
                    flush=True)
    if not cache.exists():
        np.savez_compressed(cache, boot=np.array([D.values.astype(np.float32) for D in Dlist_boot]),
                            seed=np.array([D.values.astype(np.float32) for D in Dlist_seed]),
                            reg=np.array(reg_B, dtype=object))
    RB = pd.DataFrame(reg_B)
    RB.to_csv(RES / "e5_retreinos.csv", index=False)
    var_mae = RB.groupby("tipo").MAE.agg(["mean", "std", "min", "max"])
    print(var_mae)

    for rot, Dl in [("B_retreino_bootstrap", Dlist_boot), ("B_retreino_semente", Dlist_seed)]:
        if not Dl:
            continue
        M = np.vstack([Dk.mean().values for Dk in Dl])
        r, c = resumo_ranks(ranks_de(M, dims), rot + "_global")
        resumos.append(r)
        conjuntos[rot + "_global"] = c
    # B×A: cada re-treino bootstrap × reamostragens das linhas de teste (≈ 1.000 combinações)
    per = max(1, B_ROWS // max(1, len(Dlist_boot)))
    if Dlist_boot:
        with Timer("B×A combinado"):
            M = np.vstack([boot_linhas(Dk, per, rng) for Dk in Dlist_boot])
            RK_bxa = ranks_de(M, dims)
            pos_dist = pd.DataFrame({dm: RK_bxa[dm].value_counts(normalize=True).reindex(range(1, 9), fill_value=0) * 100
                                     for dm in dims}).T
            pos_dist.to_csv(RES / "e5_distribuicao_posicoes_global.csv")
            r, c = resumo_ranks(RK_bxa, "BxA_global")
            resumos.append(r)
            conjuntos["BxA_global"] = c
            linhas_aero = []
            for A in airports:
                mA = ate == A
                MA = np.vstack([boot_linhas(Dk[mA], per, rng) for Dk in Dlist_boot])
                ra, ca = resumo_ranks(ranks_de(MA, dims), f"BxA_{A}")
                i_r, i_c = dims.index(TOP4[2]), dims.index(TOP4[3])
                p_rc = float((MA[:, i_r] > MA[:, i_c]).mean() * 100)
                pt_rc = bool(D0[mA].mean()[TOP4[2]] > D0[mA].mean()[TOP4[3]])
                rA, cA = por_aero_A[A]
                for dim in dims:
                    linhas_aero.append({"aeroporto": A, "nome_aeroporto": NOME_AERO[A], "dimensao": dim,
                                        "nome": nome_dim(dim), "pct_top4_A": rA.loc[dim, "pct_top4"],
                                        "pct_top4_BxA": ra.loc[dim, "pct_top4"],
                                        "pos_media_BxA": ra.loc[dim, "pos_media"],
                                        "pos_p2_5_BxA": ra.loc[dim, "pos_p2_5"], "pos_p97_5_BxA": ra.loc[dim, "pos_p97_5"],
                                        "pct_conjunto_top4_A": cA, "pct_conjunto_top4_BxA": ca,
                                        "pct_restituicao_a_frente_conforto_BxA": p_rc,
                                        "restituicao_a_frente_conforto_pontual": pt_rc})
        PA = pd.DataFrame(linhas_aero)
        PA.to_csv(RES / "e5_estabilidade_aeroporto.csv", index=False)
        conj_aero = PA.groupby("aeroporto")[["pct_conjunto_top4_A", "pct_conjunto_top4_BxA"]].first()
        print(conj_aero.round(1).to_string())
        print(conj_aero.describe().round(1))

    RS = pd.concat(resumos)
    RS.index.name = "dimensao"
    RS.insert(0, "nome", [nome_dim(k) for k in RS.index])
    RS.to_csv(RES / "e5_estabilidade_global.csv")
    print(RS[RS.pct_top4 > 0].round(2).to_string())
    print(json.dumps(conjuntos, indent=1))

    # ---------------- C) multiplicidade de modelos ----------------
    rankings = {"XGBoost (atual)": D0.mean()}
    with Timer("C) SHAP de LightGBM/CatBoost atuais"):
        for nome, arq in [("LightGBM (atual)", "lgbm"), ("CatBoost (atual)", "cat")]:
            m = joblib.load(PIPE_MODELS / f"{arq}_20.joblib")
            Dk, _ = shap_dim_tree(m, Xte, rte, grupos)
            rankings[nome] = Dk[dims].abs().mean()
    with Timer("C) Ridge (SHAP linear exato)"):
        m = joblib.load(PIPE_MODELS / "ridge_20.joblib")
        C = ridge_contrib(m, Xte, rte)
        rankings["Ridge"] = shap_por_dimensao(C, grupos)[dims].abs().mean()
    with Timer("C) Random Forest (1.000 linhas de teste)"):
        m = joblib.load(PIPE_MODELS / "rf_20.joblib")
        for mm in m.modelos.values():
            mm.set_params(n_jobs=N_JOBS)
        sub = np.sort(np.random.default_rng(SEED).choice(len(Xte), 1000, replace=False))
        Dk, _ = shap_dim_tree(m, Xte.iloc[sub], rte[sub], grupos)
        rankings["Random Forest*"] = Dk[dims].abs().mean()
    with Timer("C) Ordinal cumulativo do E3 (SHAP intervencional em probabilidade, 1.000 linhas)"):
        import shap
        Cord = pd.DataFrame(0.0, index=Xte.index[sub], columns=X.columns)
        pred_ord = np.ones(len(sub))
        base_ord = np.ones(len(sub))
        Xs, rs_ = Xte.iloc[sub], rte[sub]
        for kk in [1, 2, 3, 4]:
            def fab_b(r):
                return xgb.XGBClassifier(n_estimators=1000, objective="binary:logistic", eval_metric="logloss",
                                         early_stopping_rounds=30, max_depth=6, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
                                         verbosity=0)

            def ajb(m, Xt, ytt, Xv, yvv, kk=kk):
                m.fit(Xt, (ytt > kk).astype(int), eval_set=[(Xv, (yvv > kk).astype(int))], verbose=False)
                return m
            mb = PorRamoF(fab_b, ajb).fit(X[tr], y[tr], ramo[tr], X[va], y[va], ramo[va])
            pred_ord += mb.predict(Xs, rs_, metodo="predict_proba")[:, 1]
            for r, mm in mb.modelos.items():
                mask = rs_ == r
                cols = mb.colunas[r]
                bg = X.loc[tr & (ramo == r), cols].sample(100, random_state=SEED)
                ex = shap.TreeExplainer(mm, data=bg, model_output="probability",
                                        feature_perturbation="interventional")
                Cord.loc[Xs.index[mask], cols] += ex.shap_values(Xs.loc[mask, cols])
                base_ord[mask] += float(np.atleast_1d(ex.expected_value)[0])
        dev = np.abs(Cord.sum(axis=1).values + base_ord - pred_ord).max()
        print(f"    aditividade (soma dos SHAP + base vs E[y]): max|Δ| = {dev:.2e}")
        rankings["Ordinal (E3)*"] = shap_por_dimensao(Cord, grupos)[dims].abs().mean()
    ebm_pkl = RES / "e2_ebm_gam.pkl"
    if ebm_pkl.exists():
        with Timer("C) EBM"):
            from e2_ebm import contrib_por_feature
            ob = pickle.load(open(ebm_pkl, "rb"))
            mE = PorRamoF(None, None)
            mE.modelos, mE.colunas = ob["modelos"], ob["colunas"]
            C = contrib_por_feature(mE, Xte, rte, list(X.columns))
            rankings["EBM (aditivo)"] = shap_por_dimensao(C, grupos)[dims].abs().mean()
    # modelos ajustados no E1
    from e1_ajuste_hiperparametros import fabrica as fab_e1, ajuste as aj_e1
    for arq, nome in [("xgb", "XGBoost ajustado"), ("lgbm", "LightGBM ajustado"), ("cat", "CatBoost ajustado")]:
        pj = RES / f"e1_params_{arq}.json"
        if not pj.exists():
            continue
        with Timer(f"C) {nome} (refit com params do E1)"):
            best = json.loads(pj.read_text())["melhores_params"]
            m = PorRamoF(fab_e1(arq, best), aj_e1(arq)).fit(X[tr], y[tr], ramo[tr], X[va], y[va], ramo[va])
            pt = m.predict(Xte, rte)
            ref = pd.read_parquet(RES / f"e1_pred_test_{arq}.parquet")["pred"].values
            print(f"    reprodução das predições do E1: max|Δ| = {np.abs(pt - ref).max():.2e}")
            Dk, _ = shap_dim_tree(m, Xte, rte, grupos)
            rankings[nome] = Dk[dims].abs().mean()
    RM = pd.DataFrame(rankings)
    RM = RM.loc[RM["XGBoost (atual)"].sort_values(ascending=False).index]
    RKm = RM.rank(ascending=False).astype(int)
    concord = []
    for c in RM.columns:
        concord.append({"modelo": c, "top4": " > ".join(nome_dim(x) for x in RM[c].sort_values(ascending=False).index[:4]),
                        "top4_conjunto_igual": set(RM[c].sort_values(ascending=False).index[:4]) == set(TOP4),
                        "spearman_vs_xgb": spearmanr(RM[c], RM["XGBoost (atual)"]).statistic,
                        "kendall_vs_xgb": kendalltau(RM[c], RM["XGBoost (atual)"]).statistic})
    CC = pd.DataFrame(concord)
    CC.to_csv(RES / "e5_multiplicidade_concordancia.csv", index=False)
    out = RKm.copy()
    out.insert(0, "nome", [nome_dim(k) for k in out.index])
    out.index.name = "dimensao"
    out.to_csv(RES / "e5_multiplicidade_ranks.csv")
    RM.to_csv(RES / "e5_multiplicidade_importancias.csv")
    print(out.head(10).to_string())
    print(CC.round(3).to_string())
    (RES / "e5_resumo.json").write_text(json.dumps({
        "pct_conjunto_top4": conjuntos, "variacao_MAE_retreinos": var_mae.to_dict(),
        "multiplicidade": CC.to_dict(orient="records")}, indent=2, ensure_ascii=False, default=float))

    # ================= Figuras =================
    plt = estilo_slide()
    # (1) global: distribuição das posições; por aeroporto: P(3ª dimensão do top-4 à frente da 4ª)
    n3, n4 = nome_dim(TOP4[2]), nome_dim(TOP4[3])
    if Dlist_boot:
        dims_fig = list(D0.mean().sort_values(ascending=False).index[:6])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})
        ramp = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#e9e9e6", "#e9e9e6", "#e9e9e6"]
        yy = np.arange(len(dims_fig))[::-1]
        for i, dm in enumerate(dims_fig):
            left = 0
            for pos in range(1, 9):
                v = pos_dist.loc[dm, pos]
                if v <= 0:
                    continue
                ax1.barh(yy[i], v, left=left, color=ramp[pos - 1] if pos <= 5 else "#d9d8d3", height=0.62,
                         edgecolor="white", linewidth=2)
                if v >= 9:
                    ax1.text(left + v / 2, yy[i], f"{pos}º\n{v:.0f}%", ha="center", va="center", fontsize=10.5,
                             color="white" if pos <= 3 else TINTA2)
                left += v
        ax1.set_yticks(yy)
        ax1.set_yticklabels([nome_dim(x) for x in dims_fig])
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("% das reamostragens em cada posição")
        ax1.grid(False)
        ax1.set_title("Global: posição de cada dimensão", fontsize=14)
        G = PA[PA.dimensao == TOP4[0]].set_index("aeroporto")
        G = G.sort_values("pct_restituicao_a_frente_conforto_BxA")
        y2 = np.arange(len(G))
        ax2.axvspan(10, 90, color="#f3f3f1", zorder=0)
        ax2.axvline(50, color=CINZA, lw=1)
        cores = [AZUL if v >= 50 else LARANJA for v in G.pct_restituicao_a_frente_conforto_BxA]
        ax2.hlines(y2, 50, G.pct_restituicao_a_frente_conforto_BxA, color=cores, lw=2)
        ax2.scatter(G.pct_restituicao_a_frente_conforto_BxA, y2, color=cores, s=55, zorder=3,
                    edgecolor="white", linewidth=1.5)
        ax2.set_yticks(y2)
        ax2.set_yticklabels([NOME_AERO[a] for a in G.index], fontsize=11)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel(f"% das reamostragens com {n3} à frente")
        ax2.set_title(f"Por aeroporto: {n3} × {n4}", fontsize=14)
        ax2.grid(axis="y", visible=False)
        ax2.text(97, -0.9, f"{n3} 3º", ha="right", va="center", fontsize=10, color=AZUL)
        ax2.text(3, -0.9, f"{n4} 3º", ha="left", va="center", fontsize=10, color=LARANJA)
        ax2.set_ylim(-1.5, len(G) - 0.5)
        gl = conjuntos.get("BxA_global", np.nan)
        n100 = int((PA.groupby("aeroporto").pct_conjunto_top4_BxA.first() >= 99.5).sum())
        pmin = PA.groupby("aeroporto").pct_conjunto_top4_BxA.first().min()
        faixa_aero = "100%" if pmin >= 99.95 else f"≥ {pmin:.1f}%".replace(".", ",")
        fig.suptitle(f"O top-4 é o mesmo conjunto em {gl:.0f}% das reamostragens (global) e em {faixa_aero} "
                     f"em cada um dos {len(airports)} aeroportos",
                     x=0.01, ha="left", fontweight="bold", fontsize=15)
        fig.text(0.01, -0.01, f"{len(Dlist_boot)} re-treinos do XGBoost em reamostragens bootstrap do treino × {per} reamostragens "
                 "das linhas de teste (≈1.000 combinações; por aeroporto, só as linhas do aeroporto). "
                 "Faixa cinza: ordem incerta (10–90%).", fontsize=9.5, color=TINTA2, ha="left", va="top")
        fig.tight_layout()
        salvar(fig, "e5_estabilidade_top4.png")

    # (2) multiplicidade: posição de cada dimensão segundo cada modelo
    dims_fig = list(RM.index[:8])
    cols = [c for c in ["Ridge", "EBM (aditivo)", "Random Forest*", "XGBoost (atual)", "LightGBM (atual)",
                        "CatBoost (atual)", "XGBoost ajustado", "LightGBM ajustado", "CatBoost ajustado",
                        "Ordinal (E3)*"] if c in RKm.columns]
    Hm = RKm.loc[dims_fig, cols]
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.grid(False)
    cor = np.where(Hm.values <= 4, 1.0, 0.0)
    from matplotlib.colors import ListedColormap
    ax.imshow(cor, cmap=ListedColormap(["#f3f3f1", "#b7d3f6"]), aspect="auto", vmin=0, vmax=1)
    for i in range(Hm.shape[0]):
        for j in range(Hm.shape[1]):
            v = Hm.values[i, j]
            ax.text(j, i, f"{v}º", ha="center", va="center", fontsize=13,
                    fontweight="bold" if v <= 4 else "normal", color="#0d366b" if v <= 4 else TINTA2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace(" (", "\n(").replace(" ajustado", "\najustado") for c in cols], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(dims_fig)))
    ax.set_yticklabels([nome_dim(x) for x in dims_fig])
    ax.set_title("Posição de cada dimensão no ranking, segundo cada modelo", pad=58)
    n_ok = int(CC.top4_conjunto_igual.sum())
    fig.text(0.01, 0.0, f"Azul = top-4. O mesmo conjunto top-4 em {n_ok} de {len(CC)} modelos (linear, aditivo, bagging, boosting e ordinal). "
             "Importância = média de |SHAP| da dimensão no teste\n(Ridge: SHAP linear exato; EBM: contribuições aditivas). "
             "*Random Forest e ordinal: 1.000 linhas de teste, pelo custo do TreeSHAP.",
             fontsize=9.5, color=TINTA2, ha="left", va="top")
    fig.tight_layout()
    salvar(fig, "e5_multiplicidade_modelos.png")


if __name__ == "__main__":
    main()
