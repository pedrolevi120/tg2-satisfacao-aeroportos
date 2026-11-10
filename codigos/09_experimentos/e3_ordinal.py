"""
E3 — Enquadramento ordinal da satisfação geral (1–5).

O pipeline trata o alvo como regressão (predição contínua; QWK sobre a predição
arredondada). Aqui testamos se respeitar explicitamente a natureza ORDINAL e
DISCRETA do alvo melhora MAE/QWK, com as mesmas features, split, semente e
hiperparâmetros do XGBoost atual (um modelo por ramo):

 (a) Regressão atual + CORTES otimizados: 4 limiares escolhidos na VALIDAÇÃO para
     maximizar o QWK (no lugar de arredondar em 1,5/2,5/3,5/4,5); aplicados no teste.
 (b) Classificação MULTICLASSE (XGBoost softmax, 5 classes), early stopping na
     validação (log-loss). Saídas: valor esperado E[y]=Σ k·p_k (contínua),
     mediana da distribuição prevista (inteira; ótima para MAE), classe mais
     provável (argmax) e E[y] com cortes otimizados para QWK.
 (c) ORDINAL por limiares cumulativos (Frank & Hall, 2001): 4 classificadores
     binários P(y > k), k = 1..4; E[y] = 1 + Σ_k P(y > k).
Atenção à comparação: MAE de predição INTEIRA e MAE de predição CONTÍNUA medem
coisas diferentes (a mediana inteira minimiza o MAE, mas perde R²); por isso
reportamos MAE, R² e QWK juntos, todos pareados contra o XGBoost atual.

Saída: resultados/e3_ordinal.csv, e3_cortes.json, e3_pred_test.parquet
"""
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from exp_comum import (RES, SEED, N_JOBS, PorRamoF, Timer, _qwk_fast, avaliar, carregar,
                       comparar_pareado, preds_atuais, fmt_ic)

warnings.filterwarnings("ignore")
HP = dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
          tree_method="hist", n_jobs=N_JOBS, random_state=SEED, verbosity=0)


def aplicar_cortes(p, c):
    return 1 + np.searchsorted(np.sort(c), p, side="right")


def otimizar_cortes(y, p, ini=(1.5, 2.5, 3.5, 4.5), n_iter=4):
    """Busca coordenada a coordenada (grade fina) dos 4 limiares que maximizam o QWK."""
    c = np.array(ini, float)
    yi = y.astype(int)
    melhor = _qwk_fast(yi, aplicar_cortes(p, c))
    for _ in range(n_iter):
        for j in range(4):
            lo = c[j - 1] + 0.01 if j > 0 else 1.0
            hi = c[j + 1] - 0.01 if j < 3 else 5.0
            for v in np.linspace(lo, hi, 120):
                cc = c.copy()
                cc[j] = v
                q = _qwk_fast(yi, aplicar_cortes(p, cc))
                if q > melhor:
                    melhor, c = q, cc
    return c, melhor


def main():
    d = carregar()
    X, y, ramo, tr, va, te = d["X"], d["y"], d["ramo"], d["tr"], d["va"], d["te"]
    P = preds_atuais()
    ref_t, ref_v = P.loc[te, "pred_xgb"].values, P.loc[va, "pred_xgb"].values
    yt, yv = y[te], y[va]
    linhas, preds, cortes = [], {}, {}

    def registrar(nome, desc, pt, q_int=None, inteira=False, pv=None):
        r = comparar_pareado(yt, ref_t, pt, qwk_new=q_int)
        cls = np.asarray(q_int if q_int is not None else np.clip(np.round(pt), 1, 5)).astype(int)
        extra = {"acuracia_nota_exata": float((cls == yt).mean()),
                 "acuracia_ate_1_ponto": float((np.abs(cls - yt) <= 1).mean())}
        if pv is not None:
            mv = avaliar(yv, pv)
            extra.update({"MAE_val": mv["MAE"], "R2_val": mv["R2"]})
        linhas.append({"variante": nome, "descricao": desc, "predicao_inteira": inteira, **r, **extra})
        preds[nome] = pt
        print(f"  {nome:28s} MAE {fmt_ic(r['MAE'], r['MAE_lo'], r['MAE_hi'], 4)}  R2 {r['R2']:.3f}  "
              f"QWK {fmt_ic(r['QWK'], r['QWK_lo'], r['QWK_hi'], 3)} | ΔMAE {fmt_ic(r['dMAE'], r['dMAE_lo'], r['dMAE_hi'], 4, True)} "
              f"ΔQWK {fmt_ic(r['dQWK'], r['dQWK_lo'], r['dQWK_hi'], 3, True)} | RMSE {r['RMSE']:.4f} "
              f"acerto exato {extra['acuracia_nota_exata']:.3f} | val MAE {extra.get('MAE_val', float('nan')):.4f}", flush=True)

    # ---------- referência e (a) ----------
    registrar("regressao_atual", "XGBoost regressão (atual), QWK com arredondamento", ref_t, pv=ref_v)
    registrar("regressao_arredondada", "XGBoost regressão, predição arredondada (inteira)",
              np.clip(np.round(ref_t), 1, 5), inteira=True)
    c_reg, q_val = otimizar_cortes(yv, ref_v)
    cortes["regressao"] = {"cortes": c_reg.tolist(), "QWK_val": q_val,
                           "QWK_val_arredondado": _qwk_fast(yv.astype(int), np.clip(np.round(ref_v), 1, 5).astype(int))}
    registrar("a_regressao_cortes_QWK", "(a) regressão + cortes que maximizam QWK (validação)",
              ref_t, q_int=aplicar_cortes(ref_t, c_reg))
    registrar("a_regressao_cortes_QWK_inteira", "(a) idem, MAE/R² da classe prevista",
              aplicar_cortes(ref_t, c_reg).astype(float), inteira=True)

    # ---------- (b) multiclasse ----------
    def fab_mc(r):
        return xgb.XGBClassifier(n_estimators=1000, objective="multi:softprob", eval_metric="mlogloss",
                                 early_stopping_rounds=30, **HP)

    def aj(m, Xt, ytt, Xv, yvv):
        m.fit(Xt, ytt - 1, eval_set=[(Xv, yvv - 1)], verbose=False)
        return m
    with Timer("multiclasse XGBoost por ramo"):
        mc = PorRamoF(fab_mc, aj).fit(X[tr], y[tr], ramo[tr], X[va], y[va], ramo[va])
    Pt = mc.predict(X[te], ramo[te], metodo="predict_proba")
    Pv = mc.predict(X[va], ramo[va], metodo="predict_proba")
    k = np.arange(1, 6)
    ev_t, ev_v = Pt @ k, Pv @ k
    med_t = 1 + (np.cumsum(Pt, 1) < 0.5).sum(1)
    arg_t = 1 + Pt.argmax(1)
    ll = -np.log(np.clip(Pt[np.arange(len(yt)), yt.astype(int) - 1], 1e-12, 1)).mean()
    print(f"  multiclasse: log-loss teste={ll:.4f}; iterações={[m.best_iteration for m in mc.modelos.values()]}")
    registrar("b_multiclasse_valor_esperado", "(b) multiclasse, E[y] = Σ k·p_k", ev_t, pv=ev_v)
    registrar("b_multiclasse_mediana", "(b) multiclasse, mediana da distribuição (inteira)", med_t.astype(float), inteira=True)
    registrar("b_multiclasse_argmax", "(b) multiclasse, classe mais provável (inteira)", arg_t.astype(float), inteira=True)
    c_mc, q_val_mc = otimizar_cortes(yv, ev_v)
    cortes["multiclasse_EV"] = {"cortes": c_mc.tolist(), "QWK_val": q_val_mc}
    registrar("b_multiclasse_EV_cortes_QWK", "(b) multiclasse, E[y] + cortes que maximizam QWK", ev_t,
              q_int=aplicar_cortes(ev_t, c_mc))

    # ---------- (c) ordinal cumulativo (Frank & Hall) ----------
    exced_t, exced_v = np.zeros((te.sum(), 4)), np.zeros((va.sum(), 4))
    with Timer("ordinal cumulativo (4 limiares × 2 ramos)"):
        for j, kk in enumerate([1, 2, 3, 4]):
            def fab_b(r):
                return xgb.XGBClassifier(n_estimators=1000, objective="binary:logistic", eval_metric="logloss",
                                         early_stopping_rounds=30, **HP)

            def ajb(m, Xt, ytt, Xv, yvv, kk=kk):
                m.fit(Xt, (ytt > kk).astype(int), eval_set=[(Xv, (yvv > kk).astype(int))], verbose=False)
                return m
            mb = PorRamoF(fab_b, ajb).fit(X[tr], y[tr], ramo[tr], X[va], y[va], ramo[va])
            exced_t[:, j] = mb.predict(X[te], ramo[te], metodo="predict_proba")[:, 1]
            exced_v[:, j] = mb.predict(X[va], ramo[va], metodo="predict_proba")[:, 1]
    # monotonicidade P(y>1) >= P(y>2) >= ... (corrige cruzamentos raros)
    exced_t = np.minimum.accumulate(exced_t, axis=1)
    exced_v = np.minimum.accumulate(exced_v, axis=1)
    ord_t, ord_v = 1 + exced_t.sum(1), 1 + exced_v.sum(1)
    registrar("c_ordinal_cumulativo_EV", "(c) ordinal cumulativo, E[y] = 1 + Σ P(y>k)", ord_t, pv=ord_v)
    c_or, q_val_or = otimizar_cortes(yv, ord_v)
    cortes["ordinal_EV"] = {"cortes": c_or.tolist(), "QWK_val": q_val_or}
    registrar("c_ordinal_cumulativo_cortes_QWK", "(c) ordinal cumulativo, E[y] + cortes que maximizam QWK", ord_t,
              q_int=aplicar_cortes(ord_t, c_or))

    # ---------- (d) média regressão + ordinal (combinação simples) ----------
    registrar("d_media_regressao_ordinal", "(d) média simples: regressão atual e ordinal cumulativo",
              (ref_t + ord_t) / 2, pv=(ref_v + ord_v) / 2)

    df = pd.DataFrame(linhas)
    df.to_csv(RES / "e3_ordinal.csv", index=False)
    (RES / "e3_cortes.json").write_text(json.dumps(cortes, indent=2, default=float))
    pd.DataFrame({"idx": np.where(te)[0], **preds}).to_parquet(RES / "e3_pred_test.parquet")
    print(json.dumps(cortes, indent=1, default=float))
    # distribuição das classes previstas vs reais (teste)
    dist = pd.DataFrame({"real": pd.Series(yt.astype(int)).value_counts(normalize=True).sort_index(),
                         "arredondado": pd.Series(np.clip(np.round(ref_t), 1, 5).astype(int)).value_counts(normalize=True).sort_index(),
                         "cortes_QWK": pd.Series(aplicar_cortes(ref_t, c_reg)).value_counts(normalize=True).sort_index(),
                         "mediana_mc": pd.Series(med_t).value_counts(normalize=True).sort_index()}).fillna(0)
    dist.to_csv(RES / "e3_distribuicao_classes.csv")
    print(dist.round(3).to_string())


if __name__ == "__main__":
    main()
