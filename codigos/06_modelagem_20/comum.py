"""Utilitários compartilhados do pacote modelos20 (T5).

Série 2021T1–2026T2 (08/11/2026): a base passou a ser mpor_2021_2026_20.parquet.
Rodadas de teste: SAIDA_TESTE=<dir> redireciona TODAS as saídas (splits, base
unificada, SHAP, modelos, resultados, figuras, tabelas) para <dir>, sem tocar nos
artefatos usados pelos outros pacotes; as entradas continuam em dados/processados.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             cohen_kappa_score)

ROOT = Path(__file__).resolve().parents[2]
PROCESSADOS = ROOT / "dados" / "processados"          # entradas (base MPor, contexto)
BASE_MPOR = PROCESSADOS / "mpor_2021_2026_20.parquet"  # série 2021T1–2026T2, 20 aeroportos
CONTEXTO_SERIE = PROCESSADOS / "contexto_aeroportos_20_serie.csv"
SAIDA_TESTE = os.environ.get("SAIDA_TESTE")
if SAIDA_TESTE:
    _T = Path(SAIDA_TESTE).resolve()
    PROC_OUT = _T / "processados"
    FIG_DIR = _T / "figuras"
    TEX_DIR = _T / "tex"
    OUT_DIR = _T / "resultados"
    PNG_DIR = _T / "img_png"
    INTERP_DIR = _T / "interpretacao"
else:
    PROC_OUT = PROCESSADOS                              # saídas (splits_20, base_unificada_20, shap_values_20)
    FIG_DIR = ROOT / "midias" / "figuras"
    TEX_DIR = ROOT / "texto" / "_paralelo" / "tex"
    OUT_DIR = Path(__file__).resolve().parent / "resultados"
    PNG_DIR = ROOT / "apresentacoes" / "img_png"
    INTERP_DIR = ROOT / "codigos" / "05_interpretacao"
for _d in (PROC_OUT, FIG_DIR, TEX_DIR, OUT_DIR, PNG_DIR, INTERP_DIR):
    _d.mkdir(parents=True, exist_ok=True)
SPLITS = PROC_OUT / "splits_20"
SEED = 42
TARGET_COL = "SATISFAÇÃO GERAL"


def estilo():
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 110,
                         "savefig.dpi": 200, "pdf.fonttype": 42})


def avaliar(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    yp = np.clip(np.round(y_pred), 1, 5).astype(int)
    yt = np.clip(np.round(y_true), 1, 5).astype(int)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
            "QWK": float(cohen_kappa_score(yt, yp, weights="quadratic",
                                           labels=[1, 2, 3, 4, 5]))}


def _qwk_fast(yt, yp, k=5):
    """QWK vetorizado (rótulos 1..k) para bootstrap."""
    O = np.zeros((k, k))
    np.add.at(O, (yt - 1, yp - 1), 1)
    W = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def bootstrap_ic(y_true, y_pred, n_boot=1000, seed=SEED, alpha=0.05):
    """IC percentil (1-alpha) das métricas por reamostragem das linhas do teste.
    Retorna dict metrica -> (ponto, lo, hi) e as amostras bootstrap (para testes pareados)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt_i = np.clip(np.round(y_true), 1, 5).astype(int)
    yp_i = np.clip(np.round(y_pred), 1, 5).astype(int)
    amostras = {m: np.empty(n_boot) for m in ["MAE", "RMSE", "R2", "QWK"]}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        err = yt - yp
        amostras["MAE"][b] = np.abs(err).mean()
        amostras["RMSE"][b] = np.sqrt((err ** 2).mean())
        amostras["R2"][b] = 1 - (err ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        amostras["QWK"][b] = _qwk_fast(yt_i[idx], yp_i[idx])
    ponto = avaliar(y_true, y_pred)
    ic = {m: (ponto[m], float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2)))
          for m, a in amostras.items()}
    return ic, amostras


def bootstrap_medias(d, n_boot=1000, seed=SEED, bloco=100):
    """Médias bootstrap de um vetor d (p.ex. diferença pareada de erros), geradas em
    blocos de `bloco` reamostragens para limitar a memória (com ~69 mil linhas de
    teste, a matriz 1.000 × n inteira ocuparia ~0,5 GB). Mesma sequência de índices
    de rng.integers(0, n, (n_boot, n)) gerada de uma vez."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    out = []
    for ini in range(0, n_boot, bloco):
        idx = rng.integers(0, n, (min(bloco, n_boot - ini), n))
        out.append(d[idx].mean(axis=1))
    return np.concatenate(out)


def fmt(x, nd=3):
    """Número em pt-BR para LaTeX: 0{,}451."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{nd}f}"
    return s.replace(".", "{,}")


def tex_escape(s: str) -> str:
    return (str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


# Nomes curtos dos atributos Likert (para figuras/tabelas)
NOMES_CURTOS = {
    "FACILIDADE DE DESEMBARQUE NO MEIO-FIO": "Desembarque no meio-fio",
    "OPÇÕES DE TRANSPORTE ATÉ O AEROPORTO": "Opções de transporte",
    "PROCESSO DE CHECK IN": "Check-in",
    "TEMPO DE ESPERA NA FILA": "Espera fila check-in",
    "ORGANIZAÇÃO DAS FILAS": "Organização filas check-in",
    "QUANTIDADE DE TOTENS AA": "Qtd. totens autoatend.",
    "QUANTIDADE DE BALCÕES": "Qtd. balcões",
    "CORDIALIDADE DOS FUNCIONÁRIOS": "Cordialidade funcionários",
    "TEMPO DE ATENDIMENTO": "Tempo de atendimento",
    "PROCESSO DE AQUISIÇÃO DA PASSAGEM": "Aquisição da passagem",
    "ATENDIMENTO DA CIA. AÉREA": "Atendimento cia. aérea",
    "PROCESSO DE INSPEÇÃO DE SEGURANÇA": "Inspeção de segurança",
    "TEMPO DE ESPERA EM FILA": "Espera fila inspeção",
    "ORGANIZAÇÃO DAS FILAS2": "Organização filas inspeção",
    "ATENDIMENTO DOS FUNCIONÁRIOS": "Atendimento inspeção",
    "CONTROLE MIGRATÓRIO": "Controle migratório",
    "TEMPO DE ESPERA EM FILA3": "Espera fila migração",
    "ORGANIZAÇÃO DAS FILAS4": "Organização filas migração",
    "ATENDIMENTO DOS FUNCIONÁRIOS5": "Atendimento migração",
    "QUANTIDADE DE GUICHÊS": "Qtd. guichês",
    "CONTROLE ADUANEIRO": "Controle aduaneiro",
    "TEMPO DE ESPERA EM FILA2": "Espera fila aduana",
    "ORGANIZAÇÃO DAS FILAS3": "Organização filas aduana",
    "ATENDIMENTO DOS FUNCIONÁRIOS4": "Atendimento aduana",
    "ESTABELECIMENTOS DE ALIMENTAÇÃO": "Alimentação",
    "QUANTIDADE DE ESTABELECIMENTOS DE ALIMENTAÇÃO": "Qtd. alimentação",
    "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS DE ALIMENTAÇÃO": "Variedade alimentação",
    "RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS DE ALIMENTAÇÃO": "Preço×qualidade alimentação",
    "ESTABELECIMENTOS COMERCIAIS": "Comércio",
    "QUANTIDADE DE ESTABELECIMENTOS COMERCIAIS": "Qtd. comércio",
    "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS COMERCIAIS": "Variedade comércio",
    "RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS COMERCIAIS": "Preço×qualidade comércio",
    "ESTACIONAMENTO": "Estacionamento",
    "QUALIDADE DAS INSTALAÇÕES DE ESTACIONAMENTO": "Instalações estacionamento",
    "FACILIDADE PARA ENCONTRAR VAGAS": "Encontrar vagas",
    "FACILIDADE DE ACESSO AO TERMINAL": "Acesso ao terminal",
    "RELAÇÃO CUSTO X BENEFÍCIO": "Custo×benefício estacion.",
    "LOCALIZAÇÃO E DESLOCAMENTO": "Localização e deslocamento",
    "SINALIZAÇÃO": "Sinalização",
    "DISPONIBILIDADE DE PAINÉIS DE INFORMAÇÕES DE VOO": "Painéis de voo",
    "ACESSIBILIDADE DO TERMINAL": "Acessibilidade",
    "CONFORTO DA SALA DE EMBARQUE": "Conforto sala embarque",
    "CONFORTO TÉRMICO": "Conforto térmico",
    "CONFORTO ACÚSTICO": "Conforto acústico",
    "DISPONIBILIDADE DE ASSENTOS": "Assentos",
    "DISPONIBILIDADE DE ASSENTOS RESERVADOS": "Assentos reservados",
    "DISPONIBILIDADE DE TOMADAS": "Tomadas",
    "INTERNET DISPONIBILIZADA PELO AEROPORTO": "Internet",
    "VELOCIDADE DE CONEXÃO": "Velocidade internet",
    "FACILIDADE DE ACESSO À REDE": "Acesso à rede",
    "SANITÁRIOS": "Sanitários",
    "QUANTIDADE DE BANHEIROS": "Qtd. banheiros",
    "LIMPEZA DOS BANHEIROS": "Limpeza banheiros",
    "MANUTENÇÃO GERAL DOS SANITÁRIOS": "Manutenção sanitários",
    "LIMPEZA GERAL DO AEROPORTO": "Limpeza geral",
    "PROCESSO DE RESTITUIÇÃO DE BAGAGENS": "Restituição de bagagens",
    "FACILIDADE DE IDENTIFICAÇÃO DA ESTEIRA DE RESTITUIÇÃO": "Identificação da esteira",
    "TEMPO DE RESTITUIÇÃO": "Tempo de restituição",
    "INTEGRIDADE DA BAGAGEM": "Integridade da bagagem",
    "AVALIAÇÃO DO MÉTODO DE DESEMBARQUE": "Método de desembarque",
    "ATENDIMENTO DA CIA. AÉREA2": "Atendimento cia. (desemb.)",
    "TERMINAL": "Terminal (id)",
}


def curto(c: str) -> str:
    return NOMES_CURTOS.get(c, c.title()[:28])


def hierarquia(X: pd.DataFrame, likert: list, prec_min=0.97) -> dict:
    """Detecta a lógica de salto do questionário: subitem c é perguntado somente
    quando o item-pai p recebe nota <= 3. Retorna {subitem: pai}.
    Critério: >= prec_min das respostas de c ocorrem com p em {1,2,3}; entre os
    candidatos, o de maior cobertura de p<=3 (recall)."""
    O = X[likert].notna()
    pais = {}
    for c in likert:
        if O[c].sum() == 0:
            continue
        melhor = None
        for p in likert:
            if p == c:
                continue
            low = X[p].isin([1, 2, 3])
            both = (O[c] & low).sum()
            prec, rec = both / O[c].sum(), both / max(low.sum(), 1)
            if prec >= prec_min and (melhor is None or rec > melhor[1]):
                melhor = (p, rec)
        if melhor is not None:
            pais[c] = melhor[0]
    # itens-pai não podem ser subitens (resolve cadeias raras)
    return {c: p for c, p in pais.items() if p not in pais}


def grupos_dimensao(likert: list, pais: dict) -> dict:
    """{dimensão (nome do item-pai): [pai, subitens...]}"""
    g = {c: [c] for c in likert if c not in pais}
    for c, p in pais.items():
        g.setdefault(p, [p]).append(c)
    return g


def shap_por_dimensao(sv: pd.DataFrame, grupos: dict) -> pd.DataFrame:
    """Soma os SHAP de cada dimensão (aditividade) linha a linha."""
    return pd.DataFrame({d: sv[[c for c in cols if c in sv.columns]].sum(axis=1)
                         for d, cols in grupos.items()}, index=sv.index)
