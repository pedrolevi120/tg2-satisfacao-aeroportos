"""Fábricas de modelos e o wrapper 'um modelo por ramo do questionário' (spec. principal)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from comum import SEED

# Hiperparâmetros idênticos aos do MVP (03_modelos_baseline, 04_modelos_ml) + CatBoost
NOMES = {"baseline": "Baseline (moda)", "ridge": "Ridge", "knn": "KNN ($k{=}10$)",
         "dt": "Árvore de decisão", "mlp": "MLP", "rf": "Random Forest",
         "lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
ORDEM = list(NOMES)
NATIVO_NAN = {"dt", "rf", "xgb", "lgbm", "cat"}
EARLY = {"xgb", "lgbm", "cat"}


def _imp():
    # modelos sem suporte nativo a NaN: média do treino + indicador de ausência
    return SimpleImputer(strategy="mean", add_indicator=True)


def fabrica(nome):
    if nome == "baseline":
        return DummyRegressor(strategy="constant", constant=4.0)  # moda (chuta 4)
    if nome == "ridge":
        return make_pipeline(_imp(), StandardScaler(), Ridge(alpha=1.0))
    if nome == "knn":
        return make_pipeline(_imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=10, n_jobs=-1))
    if nome == "dt":
        return DecisionTreeRegressor(max_depth=10, random_state=SEED)
    if nome == "mlp":
        return make_pipeline(_imp(), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-3,
            learning_rate_init=1e-3, max_iter=200, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15, random_state=SEED))
    if nome == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=5,
                                     n_jobs=-1, random_state=SEED)
    if nome == "xgb":
        return xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                n_jobs=-1, random_state=SEED, early_stopping_rounds=30,
                                verbosity=0)
    if nome == "lgbm":
        return lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                 random_state=SEED, verbosity=-1)
    if nome == "cat":
        return CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05,
                                 loss_function="RMSE", random_seed=SEED, od_type="Iter",
                                 od_wait=30, verbose=0, thread_count=-1)
    raise ValueError(nome)


def ajustar(nome, m, Xtr, ytr, Xv, yv):
    if nome == "xgb":
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    elif nome == "lgbm":
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    elif nome == "cat":
        m.fit(Xtr, ytr, eval_set=(Xv, yv), use_best_model=True)
    else:
        m.fit(Xtr, ytr)
    return m


class PorRamo:
    """Um modelo por PROCESSO (Embarque/Desembarque). Em cada ramo usam-se apenas as
    colunas observadas (não inteiramente ausentes e não constantes) no treino do ramo."""
    RAMOS = ("Embarque", "Desembarque")

    def __init__(self, nome):
        self.nome = nome
        self.modelos, self.colunas = {}, {}

    def fit(self, X, y, ramo, Xv, yv, ramo_v):
        for r in self.RAMOS:
            tr, va = (ramo == r), (ramo_v == r)
            Xr = X.loc[tr]
            cols = [c for c in X.columns if Xr[c].notna().any() and Xr[c].nunique(dropna=False) > 1]
            self.colunas[r] = cols
            self.modelos[r] = ajustar(self.nome, fabrica(self.nome), Xr[cols], y[tr],
                                      Xv.loc[va, cols], yv[va])
        return self

    def predict(self, X, ramo):
        out = np.full(len(X), np.nan)
        for r in self.RAMOS:
            m = (ramo == r)
            if m.any():
                out[m] = self.modelos[r].predict(X.loc[m, self.colunas[r]])
        return out
