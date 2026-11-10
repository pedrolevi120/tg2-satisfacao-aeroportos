"""
E1 — Ajuste de hiperparâmetros (Optuna/TPE) de XGBoost, LightGBM e CatBoost.

Protocolo (pareado com o pipeline 06_modelagem_20):
  * mesmas features (X da série 2021–2026), mesmo split, mesma semente (42), mesma
    especificação "um modelo por ramo do questionário";
  * busca: Optuna TPE, N_TRIALS tentativas por modelo; a 1ª tentativa é a
    configuração atual (do MVP), de modo que o ajuste nunca fica "abaixo" dela em CV;
  * critério: MAE médio em validação cruzada K=3 dentro do TREINO (folds
    estratificados por aeroporto × nota, como o split). Em cada fold o early
    stopping usa o conjunto de VALIDAÇÃO do pipeline (o mesmo papel que ele tem
    no pipeline), e o MAE é medido no fold deixado de fora — sem otimismo;
  * modelo final: melhores hiperparâmetros, treino completo, early stopping na
    validação (idêntico ao pipeline), avaliação no MESMO teste;
  * comparação: diferença pareada contra o XGBoost atual (predições salvas) com
    IC 95% bootstrap (1.000 reamostragens, mesma semente).
Extra: média simples (ensemble) dos três modelos atuais e dos três ajustados.

Uso: python e1_ajuste_hiperparametros.py [xgb lgbm cat]   (N_TRIALS, N_JOBS via env)
Série 2021–2026 (08/11/2026): rodado com N_TRIALS=25 (padrão do script: 50) para caber
no tempo — com ~5,5× mais linhas cada tentativa custa ~5,5× mais.
Saída: resultados/e1_*.csv|json|parquet
"""
import json
import os
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

from exp_comum import (RES, SEED, N_JOBS, PorRamoF, Timer, avaliar, carregar, comparar_pareado,
                       preds_atuais, fmt_ic)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(os.environ.get("N_TRIALS", "50"))
K = 3
ES = 50

ATUAL = {  # = modelos.fabrica (MVP); defaults das bibliotecas explicitados
    "xgb": dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1.0, reg_lambda=1.0, reg_alpha=1e-3),
    # no pipeline o bagging do LightGBM está inativo (subsample sem subsample_freq) → subsample=1
    "lgbm": dict(num_leaves=63, learning_rate=0.05, subsample=1.0, colsample_bytree=0.8,
                 min_child_samples=20, reg_lambda=1e-3, reg_alpha=1e-3),
    "cat": dict(depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_strength=1.0,
                bagging_temperature=1.0),
}


def fabrica(nome, p):
    """Fábrica com até 3.000 árvores + early stopping (ES rodadas) na validação."""
    p = dict(p)
    if nome == "xgb":
        return lambda r: xgb.XGBRegressor(n_estimators=3000, tree_method="hist", n_jobs=N_JOBS,
                                          random_state=SEED, early_stopping_rounds=ES,
                                          verbosity=0, **p)
    if nome == "lgbm":
        # subsample_freq=1 liga o bagging (no pipeline, subsample=0,8 sem freq não tem efeito)
        return lambda r: lgb.LGBMRegressor(n_estimators=3000, subsample_freq=1, n_jobs=N_JOBS,
                                           random_state=SEED, verbosity=-1, **p)
    if nome == "cat":
        return lambda r: CatBoostRegressor(iterations=3000, loss_function="RMSE", random_seed=SEED,
                                           od_type="Iter", od_wait=ES, verbose=0,
                                           thread_count=N_JOBS, **p)
    raise ValueError(nome)


def ajuste(nome, es=ES):
    if nome == "xgb":
        def f(m, Xt, yt, Xv, yv):
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            return m
    elif nome == "lgbm":
        def f(m, Xt, yt, Xv, yv):
            m.fit(Xt, yt, eval_set=[(Xv, yv)],
                  callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
            return m
    else:
        def f(m, Xt, yt, Xv, yv):
            m.fit(Xt, yt, eval_set=(Xv, yv), use_best_model=True)
            return m
    return f


def espaco(nome, t):
    if nome == "xgb":
        return dict(learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 10),
                    min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-2, 100, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
    if nome == "lgbm":
        return dict(learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    num_leaves=t.suggest_int("num_leaves", 8, 255, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 300, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
    return dict(learning_rate=t.suggest_float("learning_rate", 0.03, 0.2, log=True),
                depth=t.suggest_int("depth", 4, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 50, log=True),
                random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
                bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 2.0))


def main():
    nomes = sys.argv[1:] or ["xgb", "lgbm", "cat"]
    d = carregar()
    X, y, ramo, tr, va, te = d["X"], d["y"], d["ramo"], d["tr"], d["va"], d["te"]
    meta = d["meta"]
    P = preds_atuais()
    ref = P.loc[te, "pred_xgb"].values
    Xtr, ytr, rtr = X[tr].reset_index(drop=True), y[tr], ramo[tr]
    Xva, yva, rva = X[va], y[va], ramo[va]
    strat = (meta.loc[tr, "AEROPORTO"] + "_" + meta.loc[tr, "y"].clip(lower=2).round().astype(int).astype(str)).values
    folds = list(StratifiedKFold(K, shuffle=True, random_state=SEED).split(Xtr, strat))

    for nome in nomes:
        def objetivo(t):
            p = espaco(nome, t)
            maes = []
            for k, (i_fit, i_out) in enumerate(folds):
                m = PorRamoF(fabrica(nome, p), ajuste(nome)).fit(
                    Xtr.iloc[i_fit], ytr[i_fit], rtr[i_fit], Xva, yva, rva)
                pr = m.predict(Xtr.iloc[i_out], rtr[i_out])
                maes.append(np.abs(ytr[i_out] - pr).mean())
                t.report(float(np.mean(maes)), k)
                if t.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(maes))

        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                 pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
        st.enqueue_trial(ATUAL[nome])  # tentativa 0 = configuração atual (com mais árvores + early stopping)
        with Timer(f"Optuna {nome} ({N_TRIALS} tentativas, CV K={K})"):
            st.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=False,
                        callbacks=[lambda s, t: print(f"    [{nome}] #{t.number:02d} {t.state.name[:4]} "
                                                      f"cv_MAE={t.value if t.value else float('nan'):.4f} "
                                                      f"melhor={s.best_value:.4f}", flush=True)])
        st.trials_dataframe().to_csv(RES / f"e1_optuna_{nome}.csv", index=False)
        cv_atual = st.trials[0].value
        best = st.best_params
        print(f"  {nome}: CV MAE atual={cv_atual:.4f} → ajustado={st.best_value:.4f}; {best}")

        # modelo final: treino completo, early stopping na validação (como no pipeline)
        with Timer(f"refit final {nome}"):
            m = PorRamoF(fabrica(nome, best), ajuste(nome)).fit(
                X[tr], y[tr], ramo[tr], Xva, yva, rva)
        pv, pt = m.predict(Xva, rva), m.predict(X[te], ramo[te])
        it = {r: (getattr(mm, "best_iteration", None) if nome == "xgb" else
                  getattr(mm, "best_iteration_", None) if nome == "lgbm" else mm.get_best_iteration())
              for r, mm in m.modelos.items()}
        out = {"modelo": nome, "cv_mae_atual": cv_atual, "cv_mae_ajustado": st.best_value,
               "melhores_params": best, "iteracoes": it, "n_trials": len(st.trials),
               "n_podadas": sum(t.state.name == "PRUNED" for t in st.trials),
               "val": avaliar(yva, pv)}
        (RES / f"e1_params_{nome}.json").write_text(json.dumps(out, indent=2, default=float))
        pd.DataFrame({"idx": np.where(te)[0], "pred": pt}).to_parquet(RES / f"e1_pred_test_{nome}.parquet")
        pd.DataFrame({"idx": np.where(va)[0], "pred": pv}).to_parquet(RES / f"e1_pred_val_{nome}.parquet")
        r = comparar_pareado(y[te], ref, pt)
        print(f"  {nome} ajustado no teste: MAE {fmt_ic(r['MAE'], r['MAE_lo'], r['MAE_hi'], 4)}; "
              f"ΔMAE vs XGB atual {fmt_ic(r['dMAE'], r['dMAE_lo'], r['dMAE_hi'], 4, True)}", flush=True)


if __name__ == "__main__":
    main()
