"""
SO2.2 — Baselines: OLS, Logit Ordinal, KNN, Decision Tree.

Entrada: dados/processados/splits/{X,y}_{train,val,test}.parquet
Saída:   codigos/03_modelos_baseline/resultados_baselines.json
         modelos salvos em codigos/03_modelos_baseline/models/
"""
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "dados" / "processados" / "splits"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path(__file__).resolve().parent / "resultados_baselines.json"


def carregar():
    Xtr = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    Xv = pd.read_parquet(SPLITS_DIR / "X_val.parquet")
    Xte = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    ytr = pd.read_parquet(SPLITS_DIR / "y_train.parquet")["y"].values
    yv = pd.read_parquet(SPLITS_DIR / "y_val.parquet")["y"].values
    yte = pd.read_parquet(SPLITS_DIR / "y_test.parquet")["y"].values
    return Xtr, Xv, Xte, ytr, yv, yte


def avaliar(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE, R², QWK (arredondando preds para inteiros)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    y_pred_int = np.clip(np.round(y_pred), 1, 5).astype(int)
    y_true_int = np.clip(np.round(y_true), 1, 5).astype(int)
    qwk = float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic",
                                    labels=[1, 2, 3, 4, 5]))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "QWK": qwk}


def treinar_avaliar(nome, modelo, Xtr, ytr, Xv, yv, Xte, yte):
    t0 = time.time()
    modelo.fit(Xtr, ytr)
    t_fit = time.time() - t0
    val_metrics = avaliar(yv, modelo.predict(Xv))
    test_metrics = avaliar(yte, modelo.predict(Xte))
    joblib.dump(modelo, MODELS_DIR / f"{nome}.joblib")
    return {
        "modelo": nome,
        "tempo_fit_s": round(t_fit, 2),
        "val": val_metrics,
        "test": test_metrics,
    }


def baseline_chute_4(yv, yte):
    """Baseline trivial: chuta sempre 4 (moda)."""
    return {
        "modelo": "Baseline-chute-4",
        "tempo_fit_s": 0.0,
        "val": avaliar(yv, np.full(len(yv), 4.0)),
        "test": avaliar(yte, np.full(len(yte), 4.0)),
    }


def main():
    print("=== Carregando splits ===")
    Xtr, Xv, Xte, ytr, yv, yte = carregar()
    print(f"  treino: {Xtr.shape}, val: {Xv.shape}, teste: {Xte.shape}")

    resultados = []

    print("\n=== Baseline trivial (chute 4) ===")
    r = baseline_chute_4(yv, yte)
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  R²={r['val']['R2']:.4f}")

    print("\n=== OLS / Ridge (baseline linear) ===")
    r = treinar_avaliar("ols_ridge", Ridge(alpha=1.0), Xtr, ytr, Xv, yv, Xte, yte)
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}")

    print("\n=== KNN (k=10) ===")
    r = treinar_avaliar("knn_k10", KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
                        Xtr, ytr, Xv, yv, Xte, yte)
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}")

    print("\n=== Decision Tree (max_depth=10) ===")
    r = treinar_avaliar("decision_tree_d10",
                        DecisionTreeRegressor(max_depth=10, random_state=42),
                        Xtr, ytr, Xv, yv, Xte, yte)
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}")

    # Logit Ordinal só se a dimensão for tratável (com 243 features + 10K linhas
    # OrderedModel costuma falhar/ser muito lento; usaremos Ridge como
    # representante linear e citaremos OrderedModel como referência teórica.)
    print("\n=== Logit Ordinal (statsmodels) — feature pruning ===")
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        # Para viabilidade computacional, usa apenas as 30 features mais
        # correlacionadas (em magnitude) com o target. Remove features
        # constantes (variância zero) — OrderedModel rejeita constantes.
        Xtr_var = Xtr.loc[:, Xtr.std() > 1e-6]
        corr = pd.DataFrame(Xtr_var).apply(lambda c: np.corrcoef(c, ytr)[0, 1])
        top = corr.abs().sort_values(ascending=False).head(30).index
        ytr_int = np.clip(np.round(ytr), 1, 5).astype(int)
        t0 = time.time()
        mod = OrderedModel(ytr_int, Xtr[top].astype(float),
                           distr="logit").fit(disp=False, maxiter=50)
        t_fit = time.time() - t0
        # Prediction → probabilidades por classe → esperança
        yv_proba = mod.predict(Xv[top].astype(float))
        yte_proba = mod.predict(Xte[top].astype(float))
        classes = np.array([1, 2, 3, 4, 5])
        yv_pred = (yv_proba.values * classes).sum(axis=1) if hasattr(yv_proba, "values") else (yv_proba * classes).sum(axis=1)
        yte_pred = (yte_proba.values * classes).sum(axis=1) if hasattr(yte_proba, "values") else (yte_proba * classes).sum(axis=1)
        r = {
            "modelo": "logit_ordinal_top30",
            "tempo_fit_s": round(t_fit, 2),
            "val": avaliar(yv, yv_pred),
            "test": avaliar(yte, yte_pred),
        }
        joblib.dump(mod, MODELS_DIR / "logit_ordinal_top30.joblib")
        resultados.append(r)
        print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}")
    except Exception as e:
        print(f"  Logit Ordinal falhou: {e}")
        resultados.append({"modelo": "logit_ordinal_top30", "erro": str(e)})

    print("\n=== Salvando resultados ===")
    RESULTS_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"  → {RESULTS_PATH.relative_to(ROOT)}")

    print("\n=== Resumo (val) ===")
    print(f"{'Modelo':<30}{'MAE':>10}{'RMSE':>10}{'R²':>10}{'QWK':>10}{'Tempo(s)':>10}")
    print("-" * 80)
    for r in resultados:
        if "val" in r:
            v = r["val"]
            t = r.get("tempo_fit_s", 0)
            print(f"{r['modelo']:<30}{v['MAE']:>10.4f}{v['RMSE']:>10.4f}{v['R2']:>10.4f}{v['QWK']:>10.4f}{t:>10.2f}")


if __name__ == "__main__":
    main()
