"""
Série histórica — validação temporal (experimento novo, separado do split principal).

Pergunta: o modelo treinado no passado prevê o futuro tão bem quanto o split
aleatório sugere, e as prioridades que ele aponta se mantêm?

Protocolo (mesmas features, ramos e configuração do modelo selecionado em
06_modelagem_20/04_comparar_modelos_20.py, lido de modelo_escolhido_20.json):
  * TREINO: linhas de 2021–2024 que no split aleatório são treino ou teste;
  * EARLY STOPPING: linhas de 2021–2024 que no split aleatório são validação
    (o mesmo papel que a validação tem no pipeline);
  * HOLDOUT: todas as linhas de 2025 e 2026 (até 2026T2), nunca vistas.
Comparação com o split aleatório (modelo oficial do pipeline):
  (a) teste aleatório completo;  (b) teste aleatório só de 2025–26;
  (c) holdout temporal completo; (d) holdout temporal nas MESMAS linhas de (b)
      — comparação pareada (ΔMAE com IC bootstrap).
Métricas: MAE, RMSE, R², QWK com IC 95% bootstrap (1.000 reamostragens).
Prioridades: SHAP (TreeExplainer) por dimensão no holdout 2025–26; top-4
comparado ao top-4 global do pipeline e ao top-4 do modelo oficial nas linhas de
teste de 2025–26 (shap_values_20.parquet).

Saída: codigos/10_serie_historica/resultados/validacao_temporal.csv / .json
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1] / "06_modelagem_20"
sys.path.insert(0, str(PIPE))
from comum import (SPLITS, PROC_OUT, OUT_DIR as PIPE_OUT, SAIDA_TESTE, avaliar, bootstrap_ic,  # noqa: E402
                   bootstrap_medias, curto, shap_por_dimensao)
from modelos import NOMES, PorRamo  # noqa: E402

warnings.filterwarnings("ignore")
OUT = (Path(SAIDA_TESTE) / "serie") if SAIDA_TESTE else Path(__file__).resolve().parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)
ANO_CORTE = 2024  # treino até este ano; holdout depois


def shap_dim(model, X, ramo, grupos):
    import shap
    sv = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    for r, m in model.modelos.items():
        mask = ramo == r
        if not mask.any():
            continue
        cols = model.colunas[r]
        sv.loc[mask, cols] = shap.TreeExplainer(m).shap_values(X.loc[mask, cols])
    return shap_por_dimensao(sv, grupos).abs()


def main():
    t0 = time.time()
    X = pd.read_parquet(SPLITS / "X.parquet")
    meta = pd.read_parquet(SPLITS / "meta.parquet")
    melhor = json.loads((PIPE_OUT / "modelo_escolhido_20.json").read_text())["modelo"]
    grupos = json.loads((PIPE_OUT / "hierarquia_20.json").read_text())["dimensoes"]
    P = pd.read_parquet(PIPE_OUT / "predicoes_20.parquet")
    y, s, ramo = meta.y.values, meta.split.values, meta.PROCESSO.values
    ano = meta.ANO.values.astype(int)
    passado, futuro = ano <= ANO_CORTE, ano > ANO_CORTE
    tr = passado & (s != "val")
    es = passado & (s == "val")
    ho = futuro
    te_rand = s == "test"
    te_rand_fut = te_rand & futuro
    print(f"  modelo: {melhor}; treino 2021–{ANO_CORTE}={tr.sum():,} | early stopping={es.sum():,} | "
          f"holdout {ANO_CORTE + 1}–2026T2={ho.sum():,} | teste aleatório={te_rand.sum():,} "
          f"(dos quais {ANO_CORTE + 1}–26: {te_rand_fut.sum():,})")

    m = PorRamo(melhor).fit(X[tr], y[tr], ramo[tr], X[es], y[es], ramo[es])
    p_ho = m.predict(X[ho], ramo[ho])
    print(f"  ajuste: {time.time() - t0:.0f}s")
    p_rand = P[f"pred_{melhor}"].values

    linhas = []

    def reg(cenario, desc, yt, pt, extra=None):
        ic, _ = bootstrap_ic(yt, pt)
        linhas.append({"cenario": cenario, "descricao": desc, "n": int(len(yt)), "media_y": float(np.mean(yt)),
                       "vies_medio": float(np.mean(pt - yt)),
                       **{k: v[0] for k, v in ic.items()}, **{f"{k}_lo": v[1] for k, v in ic.items()},
                       **{f"{k}_hi": v[2] for k, v in ic.items()}, **(extra or {})})
        r = linhas[-1]
        print(f"  {cenario:28s} n={r['n']:>6} MAE={r['MAE']:.4f} [{r['MAE_lo']:.4f};{r['MAE_hi']:.4f}] "
              f"RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} QWK={r['QWK']:.4f} viés={r['vies_medio']:+.4f}", flush=True)

    reg("a_aleatorio_teste", "Split aleatório (pipeline), teste completo 2021–2026", y[te_rand], p_rand[te_rand])
    reg("b_aleatorio_teste_2025_26", "Split aleatório (pipeline), linhas de teste de 2025–26",
        y[te_rand_fut], p_rand[te_rand_fut])
    reg("c_temporal_holdout", "Treino 2021–2024, holdout 2025–2026T2 (todas as linhas)", y[ho], p_ho)
    # (d) mesmas linhas de (b), pareado
    idx_ho = np.where(ho)[0]
    pos = pd.Series(np.arange(len(idx_ho)), index=idx_ho)
    p_d = p_ho[pos.loc[np.where(te_rand_fut)[0]].values]
    yd = y[te_rand_fut]
    dif = np.abs(yd - p_d) - np.abs(yd - p_rand[te_rand_fut])
    dd = bootstrap_medias(dif)
    reg("d_temporal_mesmas_linhas_b", "Treino 2021–2024, avaliado nas linhas de (b)", yd, p_d,
        {"dMAE_vs_b": float(dif.mean()), "dMAE_vs_b_lo": float(np.quantile(dd, 0.025)),
         "dMAE_vs_b_hi": float(np.quantile(dd, 0.975))})
    for a in sorted(np.unique(ano[ho])):
        mm = ano[ho] == a
        reg(f"c_temporal_holdout_{a}", f"Treino 2021–2024, holdout só {a}", y[ho][mm], p_ho[mm])
    df = pd.DataFrame(linhas)
    df.to_csv(OUT / "validacao_temporal.csv", index=False)

    # ---------- prioridades no holdout ----------
    t1 = time.time()
    D_ho = shap_dim(m, X[ho], ramo[ho], grupos)
    imp_ho = D_ho.mean().sort_values(ascending=False)
    print(f"  SHAP holdout: {time.time() - t1:.0f}s")
    g = pd.read_csv(PIPE_OUT / "ranking_global_20.csv").set_index("dimensao")["mean_abs_shap"]
    S = pd.read_parquet(PROC_OUT / "shap_values_20.parquet")
    S = S[S.ANO.astype(int) > ANO_CORTE]
    imp_rand_fut = shap_por_dimensao(S[[c for c in X.columns if c in S.columns]], grupos).abs().mean() \
        .sort_values(ascending=False)
    aero_ho = meta.AEROPORTO.values[ho]
    top_aero = []
    for A in sorted(np.unique(aero_ho)):
        ia = D_ho[aero_ho == A].mean().sort_values(ascending=False)
        top_aero.append({"aeroporto": A, "top4": list(ia.index[:4]),
                         "top1_igual_global": ia.index[0] == g.index[0],
                         "top2_igual_global": ia.index[1] == g.index[1],
                         "top4_conjunto_igual_global": set(ia.index[:4]) == set(g.index[:4])})
    TA = pd.DataFrame(top_aero)
    rk = pd.DataFrame({"temporal_holdout": imp_ho, "aleatorio_teste_2025_26": imp_rand_fut,
                       "aleatorio_global": g})
    rho = rk.corr(method="spearman")
    res = {
        "modelo": melhor, "nome_modelo": NOMES[melhor], "ano_corte_treino": ANO_CORTE,
        "n_treino": int(tr.sum()), "n_early_stopping": int(es.sum()), "n_holdout": int(ho.sum()),
        "n_holdout_por_ano": {int(a): int((ano[ho] == a).sum()) for a in np.unique(ano[ho])},
        "metricas": df.set_index("cenario").to_dict(orient="index"),
        "top4_holdout": list(imp_ho.index[:4]),
        "top4_global_pipeline": list(g.index[:4]),
        "top4_aleatorio_2025_26": list(imp_rand_fut.index[:4]),
        "top4_conjunto_igual_global": set(imp_ho.index[:4]) == set(g.index[:4]),
        "top4_ordem_igual_global": list(imp_ho.index[:4]) == list(g.index[:4]),
        "top4_conjunto_igual_aleatorio_2025_26": set(imp_ho.index[:4]) == set(imp_rand_fut.index[:4]),
        "importancia_holdout_top8": {d: float(v) for d, v in imp_ho.head(8).items()},
        "importancia_aleatorio_2025_26_top8": {d: float(v) for d, v in imp_rand_fut.head(8).items()},
        "spearman_holdout_vs_global": float(rho.loc["temporal_holdout", "aleatorio_global"]),
        "spearman_holdout_vs_aleatorio_2025_26": float(rho.loc["temporal_holdout", "aleatorio_teste_2025_26"]),
        "aeroportos_top1_igual_global": int(TA.top1_igual_global.sum()),
        "aeroportos_top2_igual_global": int(TA.top2_igual_global.sum()),
        "aeroportos_top4_conjunto_igual_global": int(TA.top4_conjunto_igual_global.sum()),
        "top4_por_aeroporto_holdout": {r.aeroporto: r.top4 for r in TA.itertuples()},
        "iteracoes_early_stopping": {r: getattr(mm, "best_iteration", None) for r, mm in m.modelos.items()},
        "tempo_s": time.time() - t0,
    }
    (OUT / "validacao_temporal.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("  top-4 holdout:   " + " > ".join(curto(d) for d in imp_ho.index[:4]))
    print("  top-4 global:    " + " > ".join(curto(d) for d in g.index[:4]))
    print("  top-4 aleat.2025–26: " + " > ".join(curto(d) for d in imp_rand_fut.index[:4]))
    print(f"  ρ holdout×global={res['spearman_holdout_vs_global']:.3f}; aeroportos com top-1/top-2/top-4 iguais: "
          f"{res['aeroportos_top1_igual_global']}/{res['aeroportos_top2_igual_global']}/"
          f"{res['aeroportos_top4_conjunto_igual_global']} de {len(TA)}")
    print(f"  → {OUT / 'validacao_temporal.csv'}, .json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
