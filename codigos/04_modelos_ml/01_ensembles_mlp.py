"""
SO2.3 — Ensembles + MLP: Random Forest, XGBoost, LightGBM, MLP.

Entrada: dados/processados/splits/{X,y}_{train,val,test}.parquet
Saída:   codigos/04_modelos_ml/resultados_ensembles.json
         modelos salvos em codigos/04_modelos_ml/models/
"""
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "dados" / "processados" / "splits"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path(__file__).resolve().parent / "resultados_ensembles.json"


def carregar():
    Xtr = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    Xv = pd.read_parquet(SPLITS_DIR / "X_val.parquet")
    Xte = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    ytr = pd.read_parquet(SPLITS_DIR / "y_train.parquet")["y"].values
    yv = pd.read_parquet(SPLITS_DIR / "y_val.parquet")["y"].values
    yte = pd.read_parquet(SPLITS_DIR / "y_test.parquet")["y"].values
    return Xtr, Xv, Xte, ytr, yv, yte


def avaliar(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    y_pred_int = np.clip(np.round(y_pred), 1, 5).astype(int)
    y_true_int = np.clip(np.round(y_true), 1, 5).astype(int)
    qwk = float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic",
                                    labels=[1, 2, 3, 4, 5]))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "QWK": qwk}


def treinar_avaliar(nome, modelo, Xtr, ytr, Xv, yv, Xte, yte, save=True):
    t0 = time.time()
    modelo.fit(Xtr, ytr)
    t_fit = time.time() - t0
    val_metrics = avaliar(yv, modelo.predict(Xv))
    test_metrics = avaliar(yte, modelo.predict(Xte))
    if save:
        joblib.dump(modelo, MODELS_DIR / f"{nome}.joblib")
    return {
        "modelo": nome,
        "tempo_fit_s": round(t_fit, 2),
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    print("=== Carregando splits ===")
    Xtr, Xv, Xte, ytr, yv, yte = carregar()
    print(f"  treino: {Xtr.shape}, val: {Xv.shape}, teste: {Xte.shape}")

    resultados = []

    print("\n=== Random Forest (n=300, depth=15) ===")
    rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                                min_samples_leaf=5, n_jobs=-1, random_state=42)
    r = treinar_avaliar("random_forest_n300_d15", rf, Xtr, ytr, Xv, yv, Xte, yte)
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}  [{r['tempo_fit_s']:.1f}s]")

    print("\n=== XGBoost (n=500, depth=6, lr=0.05) ===")
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   tree_method="hist", n_jobs=-1, random_state=42,
                                   early_stopping_rounds=30, verbosity=0)
    t0 = time.time()
    xgb_model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    t_fit = time.time() - t0
    r = {
        "modelo": "xgboost_n500_d6",
        "tempo_fit_s": round(t_fit, 2),
        "val": avaliar(yv, xgb_model.predict(Xv)),
        "test": avaliar(yte, xgb_model.predict(Xte)),
    }
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_n500_d6.joblib")
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}  [{r['tempo_fit_s']:.1f}s]")

    print("\n=== LightGBM (n=500, num_leaves=63, lr=0.05) ===")
    lgb_model = lgb.LGBMRegressor(n_estimators=500, num_leaves=63,
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, n_jobs=-1,
                                   random_state=42, verbosity=-1)
    t0 = time.time()
    lgb_model.fit(Xtr, ytr, eval_set=[(Xv, yv)],
                   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    t_fit = time.time() - t0
    r = {
        "modelo": "lightgbm_n500_l63",
        "tempo_fit_s": round(t_fit, 2),
        "val": avaliar(yv, lgb_model.predict(Xv)),
        "test": avaliar(yte, lgb_model.predict(Xte)),
    }
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm_n500_l63.joblib")
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}  [{r['tempo_fit_s']:.1f}s]")

    print("\n=== MLP (128-64, ReLU, Adam) — com normalização ===")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xv_s = scaler.transform(Xv)
    Xte_s = scaler.transform(Xte)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                        solver="adam", alpha=1e-3, learning_rate_init=1e-3,
                        max_iter=200, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=15, random_state=42)
    t0 = time.time()
    mlp.fit(Xtr_s, ytr)
    t_fit = time.time() - t0
    r = {
        "modelo": "mlp_128_64",
        "tempo_fit_s": round(t_fit, 2),
        "val": avaliar(yv, mlp.predict(Xv_s)),
        "test": avaliar(yte, mlp.predict(Xte_s)),
    }
    joblib.dump((scaler, mlp), MODELS_DIR / "mlp_128_64.joblib")
    resultados.append(r)
    print(f"  val  MAE={r['val']['MAE']:.4f}  RMSE={r['val']['RMSE']:.4f}  R²={r['val']['R2']:.4f}  QWK={r['val']['QWK']:.4f}  [{r['tempo_fit_s']:.1f}s]")

    print("\n=== Salvando resultados ===")
    RESULTS_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"  → {RESULTS_PATH.relative_to(ROOT)}")

    print("\n=== Resumo (val) ===")
    print(f"{'Modelo':<30}{'MAE':>10}{'RMSE':>10}{'R²':>10}{'QWK':>10}{'Tempo(s)':>10}")
    print("-" * 80)
    for r in resultados:
        v = r["val"]
        print(f"{r['modelo']:<30}{v['MAE']:>10.4f}{v['RMSE']:>10.4f}{v['R2']:>10.4f}{v['QWK']:>10.4f}{r['tempo_fit_s']:>10.2f}")


if __name__ == "__main__":
    main()
