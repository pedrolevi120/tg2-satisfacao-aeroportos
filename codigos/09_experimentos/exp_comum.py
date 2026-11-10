"""Utilitários dos experimentos (09_experimentos).

Reaproveita o pré-processamento, as features, o split e a semente do pipeline
06_modelagem_20 (importados de lá, sem alterá-lo). Tudo aqui é só leitura dos
artefatos do pipeline: splits_20/{X,meta}.parquet, predicoes_20.parquet,
hierarquia_20.json, shap_values_20.parquet.

Série 2021T1–2026T2 (08/11/2026): tamanhos do split e o TOP4 deixam de ser fixos
(TOP4 = quatro primeiras dimensões de ranking_global_20.csv do pipeline); período
(PERIODO) substitui o trimestre. SAIDA_TESTE (ver 06_modelagem_20/comum.py) também
redireciona os resultados dos experimentos.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
PIPE = AQUI.parent / "06_modelagem_20"
sys.path.insert(0, str(PIPE))

from comum import (ROOT, PROCESSADOS, PROC_OUT, SPLITS, SAIDA_TESTE, OUT_DIR as PIPE_RES,  # noqa: E402
                   SEED, TARGET_COL, avaliar, bootstrap_ic, _qwk_fast,
                   curto, hierarquia, grupos_dimensao, shap_por_dimensao)

PIPE_MODELS = PIPE_RES.parent / "models"
RES = (Path(SAIDA_TESTE).resolve() / "exp") if SAIDA_TESTE else AQUI / "resultados"
FIG = RES / "figuras"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)
N_JOBS = int(os.environ.get("N_JOBS", "4"))  # máquina compartilhada: <= 4 threads
B_BOOT = 1000

NOME_AERO = {"SBBE": "Belém", "SBBR": "Brasília", "SBCF": "Confins", "SBCT": "Curitiba",
             "SBCY": "Cuiabá", "SBEG": "Manaus", "SBFL": "Florianópolis", "SBFZ": "Fortaleza",
             "SBGL": "Galeão", "SBGO": "Goiânia", "SBGR": "Guarulhos", "SBKP": "Viracopos",
             "SBMO": "Maceió", "SBPA": "Porto Alegre", "SBRF": "Recife", "SBRJ": "Santos Dumont",
             "SBSG": "Natal", "SBSP": "Congonhas", "SBSV": "Salvador", "SBVT": "Vitória"}

# top-4 global do pipeline (06_shap_20.py), na ordem
TOP4 = pd.read_csv(PIPE_RES / "ranking_global_20.csv")["dimensao"].head(4).tolist()
# modelo selecionado no pipeline (referência pareada dos experimentos)
MODELO_REF = json.loads((PIPE_RES / "modelo_escolhido_20.json").read_text())["modelo"]
if MODELO_REF != "xgb":
    print(f"  [ATENÇÃO] modelo selecionado no pipeline = {MODELO_REF}; os experimentos usam o XGBoost "
          "como referência (pred_xgb), como no TG.")

NOME_DIM = {"LOCALIZAÇÃO E DESLOCAMENTO": "Localização e deslocamento",
            "LIMPEZA GERAL DO AEROPORTO": "Limpeza geral",
            "PROCESSO DE RESTITUIÇÃO DE BAGAGENS": "Restituição de bagagens",
            "CONFORTO DA SALA DE EMBARQUE": "Conforto da sala de embarque",
            "SANITÁRIOS": "Sanitários", "DISPONIBILIDADE DE TOMADAS": "Tomadas",
            "ESTABELECIMENTOS DE ALIMENTAÇÃO": "Alimentação",
            "FACILIDADE DE DESEMBARQUE NO MEIO-FIO": "Desembarque no meio-fio",
            "INTERNET DISPONIBILIZADA PELO AEROPORTO": "Internet",
            "ESTABELECIMENTOS COMERCIAIS": "Comércio",
            "PROCESSO DE INSPEÇÃO DE SEGURANÇA": "Inspeção de segurança",
            "OPÇÕES DE TRANSPORTE ATÉ O AEROPORTO": "Opções de transporte",
            "PROCESSO DE CHECK IN": "Check-in",
            "PROCESSO DE AQUISIÇÃO DA PASSAGEM": "Aquisição da passagem",
            "ESTACIONAMENTO": "Estacionamento", "CONTROLE MIGRATÓRIO": "Controle migratório",
            "CONTROLE ADUANEIRO": "Controle aduaneiro",
            "DISPONIBILIDADE DE ASSENTOS RESERVADOS": "Assentos reservados",
            "AVALIAÇÃO DO MÉTODO DE DESEMBARQUE": "Método de desembarque"}


def nome_dim(d):
    return NOME_DIM.get(d, curto(d))


class Timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.time()
        print(f"  >> {self.msg} ...", flush=True)
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
        print(f"  << {self.msg}: {self.dt:.1f}s", flush=True)


def carregar():
    """X, meta, máscaras e vetores — idênticos ao pipeline (série 2021–2026)."""
    X = pd.read_parquet(SPLITS / "X.parquet")
    meta = pd.read_parquet(SPLITS / "meta.parquet")
    fm = json.loads((SPLITS / "feature_metadata.json").read_text())
    s = meta["split"].values
    d = dict(X=X, meta=meta, fm=fm, likert=fm["likert_cols"],
             tr=s == "train", va=s == "val", te=s == "test",
             y=meta["y"].values.astype(float), ramo=meta["PROCESSO"].values,
             aero=meta["AEROPORTO"].values)
    assert len(X) == len(meta) == fm["n_total"]
    assert {k: int(v) for k, v in meta["split"].value_counts().items()} == fm["n_split"]
    print(f"  dados: X {X.shape}; treino={d['tr'].sum():,} val={d['va'].sum():,} teste={d['te'].sum():,}", flush=True)
    return d


def preds_atuais():
    """Predições salvas do pipeline (val e teste) — referência pareada."""
    return pd.read_parquet(PIPE_RES / "predicoes_20.parquet")


def grupos_dim():
    h = json.loads((PIPE_RES / "hierarquia_20.json").read_text())
    return h["subitem->pai"], h["dimensoes"]


def base_unificada_alinhada():
    """Base completa (todas as colunas brutas) filtrada como no 03_base_features_20.py,
    na mesma ordem de linhas de X/meta."""
    df = pd.read_parquet(PROC_OUT / "base_unificada_20.parquet")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Comparação pareada no teste (mesmas reamostragens para os dois modelos)
# --------------------------------------------------------------------------
def _metricas_lote(yt, yp, idx):
    """MAE, R2, QWK para cada linha de índices bootstrap (B × n)."""
    Yt, Yp = yt[idx], yp[idx]
    err = Yt - Yp
    mae = np.abs(err).mean(1)
    r2 = 1 - (err ** 2).sum(1) / ((Yt - Yt.mean(1, keepdims=True)) ** 2).sum(1)
    yti = np.clip(np.round(yt), 1, 5).astype(int)
    ypi = np.clip(np.round(yp), 1, 5).astype(int)
    qwk = np.array([_qwk_fast(yti[i], ypi[i]) for i in idx])
    return {"MAE": mae, "R2": r2, "QWK": qwk}


def comparar_pareado(y, p_ref, p_new, B=B_BOOT, seed=SEED, qwk_new=None, qwk_ref=None):
    """Métricas de p_new no teste (IC 95% percentil) e diferenças pareadas
    p_new − p_ref (mesmas B reamostragens). qwk_new/qwk_ref: predições inteiras
    alternativas para o QWK (p.ex. cortes otimizados); por padrão, arredondamento."""
    y = np.asarray(y, float)
    p_ref, p_new = np.asarray(p_ref, float), np.asarray(p_new, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {}
    pt_new, pt_ref = avaliar(y, p_new), avaliar(y, p_ref)
    if qwk_new is not None:
        pt_new["QWK"] = float(_qwk_fast(y.astype(int), np.asarray(qwk_new, int)))
    if qwk_ref is not None:
        pt_ref["QWK"] = float(_qwk_fast(y.astype(int), np.asarray(qwk_ref, int)))
    acc = {k: {"new": [], "ref": []} for k in ["MAE", "R2", "QWK"]}
    yti = y.astype(int)
    qn = np.asarray(qwk_new, int) if qwk_new is not None else np.clip(np.round(p_new), 1, 5).astype(int)
    qr = np.asarray(qwk_ref, int) if qwk_ref is not None else np.clip(np.round(p_ref), 1, 5).astype(int)
    for _ in range(B // 100):
        idx = rng.integers(0, n, (100, n))
        for nome, p, q in [("new", p_new, qn), ("ref", p_ref, qr)]:
            err = y[idx] - p[idx]
            acc["MAE"][nome].append(np.abs(err).mean(1))
            Yt = y[idx]
            acc["R2"][nome].append(1 - (err ** 2).sum(1) / ((Yt - Yt.mean(1, keepdims=True)) ** 2).sum(1))
            acc["QWK"][nome].append(np.array([_qwk_fast(yti[i], q[i]) for i in idx]))
    for k in ["MAE", "R2", "QWK"]:
        a_new, a_ref = np.concatenate(acc[k]["new"]), np.concatenate(acc[k]["ref"])
        d = a_new - a_ref
        out[k] = pt_new[k]
        out[f"{k}_lo"], out[f"{k}_hi"] = np.quantile(a_new, [0.025, 0.975])
        out[f"d{k}"] = pt_new[k] - pt_ref[k]
        out[f"d{k}_lo"], out[f"d{k}_hi"] = np.quantile(d, [0.025, 0.975])
    out["RMSE"] = pt_new["RMSE"]
    return out


def fmt_ic(v, lo, hi, nd=3, sinal=False):
    f = (lambda x: f"{x:+.{nd}f}") if sinal else (lambda x: f"{x:.{nd}f}")
    return f"{f(v)} [{f(lo)}; {f(hi)}]".replace(".", ",")


# --------------------------------------------------------------------------
# Modelos por ramo com fábrica arbitrária (mesma lógica do modelos.PorRamo)
# --------------------------------------------------------------------------
class PorRamoF:
    """Igual a modelos.PorRamo, mas recebe fábrica/ajuste arbitrários
    (hiperparâmetros ajustados, n_jobs limitado, EBM, classificadores)."""
    RAMOS = ("Embarque", "Desembarque")

    def __init__(self, fabrica, ajustar):
        self.fabrica, self.ajustar = fabrica, ajustar
        self.modelos, self.colunas = {}, {}

    @staticmethod
    def cols_ramo(X, mask):
        Xr = X.loc[mask]
        return [c for c in X.columns if Xr[c].notna().any() and Xr[c].nunique(dropna=False) > 1]

    def fit(self, X, y, ramo, Xv, yv, ramo_v):
        for r in self.RAMOS:
            tr, va = (ramo == r), (ramo_v == r)
            cols = self.cols_ramo(X, tr)
            self.colunas[r] = cols
            self.modelos[r] = self.ajustar(self.fabrica(r), X.loc[tr, cols], y[tr],
                                           Xv.loc[va, cols], yv[va])
        return self

    def predict(self, X, ramo, metodo="predict"):
        out = None
        for r in self.RAMOS:
            m = ramo == r
            if m.any():
                p = getattr(self.modelos[r], metodo)(X.loc[m, self.colunas[r]])
                if out is None:
                    out = np.full((len(X),) + np.shape(p)[1:], np.nan)
                out[m] = p
        return out


def shap_dim_tree(model, X, ramo, grupos):
    """TreeSHAP linha a linha (cada linha pelo modelo do seu ramo), somado por dimensão.
    Retorna (D com sinal por dimensão, SHAP por feature)."""
    import shap
    sv = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    for r, m in model.modelos.items():
        mask = ramo == r
        if not mask.any():
            continue
        cols = model.colunas[r]
        ex = shap.TreeExplainer(m)
        v = ex.shap_values(X.loc[mask, cols])
        sv.loc[mask, cols] = v
    return shap_por_dimensao(sv, grupos), sv


# --------------------------------------------------------------------------
# Estilo de figura para slide
# --------------------------------------------------------------------------
AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CINZA, CINZA_CLARO, TINTA, TINTA2 = "#8c8b86", "#d9d8d3", "#0b0b0b", "#52514e"


def estilo_slide():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15,
        "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300, "axes.edgecolor": CINZA_CLARO,
        "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": TINTA2,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
        "axes.titleweight": "bold", "axes.titlelocation": "left"})
    return plt


def salvar(fig, nome):
    p = FIG / nome
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  figura → {p}")


def fmt_n(n):
    """Inteiro com separador de milhar pt-BR: 69240 -> '69.240'."""
    return f"{int(n):,}".replace(",", ".")
