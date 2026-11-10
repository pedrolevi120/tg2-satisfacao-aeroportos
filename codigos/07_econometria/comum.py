"""Utilitários compartilhados do pacote econometria (T6), série MPor 2021T1–2026T2.

  * Tempo: efeitos fixos por PERIODO ("2021T1"…"2026T2"; 21 níveis, 2021T2 não existe).
  * Atributos: selecionados pelo NOME (lista do questionário), não pela posição das colunas.

Pontos centrais de desenho (ver texto/_paralelo/econometria.md):
  * Atributos Likert são lidos de forma robusta ('4', '4.0', 4 -> 4; 'NS/NR' -> ausente).
  * O questionário MPor tem duas fontes de ausência ESTRUTURAL:
      (i)  ramo PROCESSO (Embarque/Desembarque) define quais blocos são perguntados;
      (ii) lógica de salto (skip logic): sub-itens só são perguntados quando o item-pai
           recebe nota <= 3 (ex.: SINALIZAÇÃO só aparece se LOCALIZAÇÃO E DESLOCAMENTO <= 3).
    A hierarquia é detectada empiricamente (regra determinística) e verificada em cada base.
  * Codificação para modelos paramétricos: valor centrado (0 quando não observado)
    + indicador de ausência APENAS entre elegíveis (perguntado mas sem nota válida)
    + indicador de "ramo de aprofundamento" 1[pai <= 3] para cada pai com filhos
    + dummy de PROCESSO. Assim a ausência estrutural é absorvida por PROCESSO / nota do pai
    e não gera colinearidade perfeita (causa da rank deficiency anterior).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROCESSADOS = ROOT / "dados" / "processados"
SPLITS = PROCESSADOS / "splits"
FIG_DIR = ROOT / "midias" / "figuras"
TEX_DIR = ROOT / "texto" / "_paralelo" / "tex"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)
TEX_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = "SATISFAÇÃO GERAL"

# Colunas que parecem Likert mas não são atributos avaliativos
NAO_ATRIBUTOS = {"TERMINAL", "PORTÃO", "TRIMESTRE", "ANO", "PERIODO", TARGET}

# Bloco avaliativo do questionário MPor (metodologia atual, idêntica nos 21 períodos), na ordem
# do arquivo do portal. Itens não-Likert (p.ex. FORMA DE CHECK-IN) caem no filtro de colunas_atributos.
ATRIBUTOS_QUESTIONARIO = (
    "AVALIAÇÃO DO MÉTODO DE DESEMBARQUE", "UTILIZOU O ESTACIONAMENTO",
    "FACILIDADE DE DESEMBARQUE NO MEIO-FIO", "OPÇÕES DE TRANSPORTE ATÉ O AEROPORTO", "FORMA DE CHECK-IN",
    "PROCESSO DE CHECK IN", "TEMPO DE ESPERA NA FILA", "ORGANIZAÇÃO DAS FILAS", "QUANTIDADE DE TOTENS AA",
    "QUANTIDADE DE BALCÕES", "CORDIALIDADE DOS FUNCIONÁRIOS", "TEMPO DE ATENDIMENTO",
    "PROCESSO DE AQUISIÇÃO DA PASSAGEM", "ATENDIMENTO DA CIA. AÉREA", "PROCESSO DE INSPEÇÃO DE SEGURANÇA",
    "TEMPO DE ESPERA EM FILA", "ORGANIZAÇÃO DAS FILAS2", "ATENDIMENTO DOS FUNCIONÁRIOS", "CONTROLE MIGRATÓRIO",
    "TEMPO DE ESPERA EM FILA3", "ORGANIZAÇÃO DAS FILAS4", "ATENDIMENTO DOS FUNCIONÁRIOS5", "QUANTIDADE DE GUICHÊS",
    "CONTROLE ADUANEIRO", "TEMPO DE ESPERA EM FILA2", "ORGANIZAÇÃO DAS FILAS3", "ATENDIMENTO DOS FUNCIONÁRIOS4",
    "ESTABELECIMENTOS DE ALIMENTAÇÃO", "QUANTIDADE DE ESTABELECIMENTOS DE ALIMENTAÇÃO",
    "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS DE ALIMENTAÇÃO",
    "RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS DE ALIMENTAÇÃO", "ESTABELECIMENTOS COMERCIAIS",
    "QUANTIDADE DE ESTABELECIMENTOS COMERCIAIS", "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS COMERCIAIS",
    "RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS COMERCIAIS", "ESTACIONAMENTO",
    "QUALIDADE DAS INSTALAÇÕES DE ESTACIONAMENTO", "FACILIDADE PARA ENCONTRAR VAGAS",
    "FACILIDADE DE ACESSO AO TERMINAL", "RELAÇÃO CUSTO X BENEFÍCIO", "LOCALIZAÇÃO E DESLOCAMENTO", "SINALIZAÇÃO",
    "DISPONIBILIDADE DE PAINÉIS DE INFORMAÇÕES DE VOO", "ACESSIBILIDADE DO TERMINAL",
    "CONFORTO DA SALA DE EMBARQUE", "CONFORTO TÉRMICO", "CONFORTO ACÚSTICO", "DISPONIBILIDADE DE ASSENTOS",
    "DISPONIBILIDADE DE ASSENTOS RESERVADOS", "DISPONIBILIDADE DE TOMADAS",
    "INTERNET DISPONIBILIZADA PELO AEROPORTO", "VELOCIDADE DE CONEXÃO", "FACILIDADE DE ACESSO À REDE",
    "SANITÁRIOS", "QUANTIDADE DE BANHEIROS", "LIMPEZA DOS BANHEIROS", "MANUTENÇÃO GERAL DOS SANITÁRIOS",
    "LIMPEZA GERAL DO AEROPORTO", "PROCESSO DE RESTITUIÇÃO DE BAGAGENS",
    "FACILIDADE DE IDENTIFICAÇÃO DA ESTEIRA DE RESTITUIÇÃO", "TEMPO DE RESTITUIÇÃO", "INTEGRIDADE DA BAGAGEM",
    "ATENDIMENTO DA CIA. AÉREA2",
)


def estilo():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "figure.dpi": 110, "savefig.dpi": 200,
        "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def fmt(x, nd=3, sinal=False):
    """Número em pt-BR para LaTeX: 0{,}451."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{nd}f}" if sinal else f"{x:.{nd}f}"
    return s.replace(".", "{,}").replace("-", "$-$") if not sinal else \
        s.replace(".", "{,}").replace("-", "$-$").replace("+", "$+$")


def estrelas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def tex_escape(s: str) -> str:
    return (str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("×", "$\\times$"))


def likert_num(s: pd.Series) -> pd.Series:
    """Converte nota Likert (str/num misto) em float 1..5; demais -> NaN."""
    v = pd.to_numeric(s.astype(str).str.strip().str.replace(",", "."), errors="coerce")
    return v.where(v.isin([1, 2, 3, 4, 5]))


def colunas_atributos(df: pd.DataFrame) -> list:
    """Atributos Likert avaliativos do questionário, selecionados pelo nome (ATRIBUTOS_QUESTIONARIO)
    e mantidos se >= 60% das respostas não nulas forem notas 1–5."""
    out = []
    for c in ATRIBUTOS_QUESTIONARIO:
        if c not in df.columns:
            continue
        if c in NAO_ATRIBUTOS:
            continue
        s = df[c].dropna().astype(str).str.strip()
        if len(s) and likert_num(s).notna().mean() >= 0.6:
            out.append(c)
    return out


def detectar_hierarquia(N: pd.DataFrame, min_n=30, tol=0.005) -> dict:
    """Filho -> pai, pela regra (quase) determinística da lógica de salto:
    filho observado  => pai observado e pai <= 3   (>= 1 - tol dos casos)
    pai observado e >= 4 => filho não observado    (violações <= tol dos filhos observados)
    e o pai com maior cobertura do filho entre pais <= 3. Não-observado inclui NS/NR.
    A tolerância absorve raros erros de digitação/NS-NR do pai (ex.: 1 caso em 84 mil
    na base de 20 aeroportos); com tol=0 a hierarquia do MVP é idêntica."""
    O = N.notna()
    hier = {}
    for c in N.columns:
        if O[c].sum() < min_n:
            continue
        best = None
        for p in N.columns:
            if p == c or O[p].sum() <= O[c].sum():
                continue
            m = O[c]
            if not (O[p] & (N[p] <= 3))[m].mean() >= 1 - tol:
                continue
            if (O[c] & O[p] & (N[p] >= 4)).sum() > tol * m.sum():
                continue
            low = O[p] & (N[p] <= 3)
            cov = O[c][low].mean()
            if best is None or cov > best[0]:
                best = (cov, p)
        if best is not None and best[0] >= 0.5:
            hier[c] = best[1]
    return hier


def ramos(N: pd.DataFrame, processo: pd.Series, min_share=0.01) -> dict:
    """Atributo -> conjunto de valores de PROCESSO em que é perguntado."""
    O = N.notna()
    out = {}
    for c in N.columns:
        sh = O[c].groupby(processo).mean()
        out[c] = set(sh[sh >= min_share].index)
    return out


# ---------------------------------------------------------------- controles
def _mapa_motivo(s):
    if s is None or pd.isna(s):
        return np.nan
    t = str(s).lower()
    if "lazer" in t or "turismo" in t:
        return "Lazer"
    if "trabalho" in t:
        return "Trabalho"
    if "fam" in t or "visit" in t or "casa" in t:
        return "Família/visita"
    if "estud" in t or "congress" in t or "acad" in t or "curso" in t:
        return "Estudos/eventos"
    if "saúde" in t or "saude" in t or "médic" in t or "trat" in t:
        return "Saúde"
    return "Outros"


def _mapa_renda(s):
    if s is None or pd.isna(s):
        return np.nan
    t = str(s)
    for chave, rot in [("Até 1", "Até 1 SM"), ("Up to $504", "Até 1 SM"),
                       ("1 e 2", "1–2 SM"), ("1 and 2", "1–2 SM"),
                       ("2 e 4", "2–4 SM"), ("2 and 4", "2–4 SM"),
                       ("4 e 10", "4–10 SM"), ("4 and 10", "4–10 SM"),
                       ("10 e 20", "10–20 SM"), ("10 and 20", "10–20 SM"),
                       ("Mais de 20", ">20 SM"), ("More than 20", ">20 SM")]:
        if chave in t:
            return rot
    return "Não informou/outros"


def _mapa_escol(s):
    if s is None or pd.isna(s):
        return np.nan
    t = str(s).lower()
    if "mestr" in t or "master" in t or "douto" in t or "phd" in t:
        return "Pós stricto sensu"
    if "especializ" in t:
        return "Especialização"
    if "superior" in t or "graduat" in t or "bachelor" in t:
        return "Superior"
    if "médio" in t or "high" in t:
        return "Médio"
    return "Fundamental ou menos"


CONTROLES_CAT = ["GÊNERO", "IDADE", "ESCOLARIDADE", "RENDA FAMILIAR", "MOTIVO DA VIAGEM",
                 "TIPO DE VOO", "NACIONALIDADE", "VIAJANDO SOZINHO", "CONEXÃO",
                 "QUANTIDADE DE VIAGENS NOS ÚLTIMOS 12 MESES",
                 "JÁ EMBARCOU/DESEMBARCOU ANTES NO AEROPORTO"]


def controles_limpos(df: pd.DataFrame, min_share=0.01) -> pd.DataFrame:
    """Categóricas sociodemográficas/viagem harmonizadas (níveis raros -> 'Outros')."""
    out = pd.DataFrame(index=df.index)
    for c in CONTROLES_CAT:
        if c not in df.columns:
            continue
        s = df[c].where(df[c].notna(), None)
        if c == "MOTIVO DA VIAGEM":
            s = s.map(_mapa_motivo)
        elif c == "RENDA FAMILIAR":
            s = s.map(_mapa_renda)
        elif c == "ESCOLARIDADE":
            s = s.map(_mapa_escol)
        else:
            s = s.map(lambda v: np.nan if v is None or pd.isna(v) or str(v) in ("None", "nan", "NS/NR")
                      else str(v).strip())
        vc = s.value_counts(normalize=True)
        raros = set(vc[vc < min_share].index)
        s = s.map(lambda v: "Outros" if v in raros else v)
        out[c] = s
    return out


def dummies_controles(C: pd.DataFrame, ref: dict | None = None):
    """Dummies drop-first (referência = nível mais frequente, fixado em `ref` para
    reuso em outras amostras). Ausente => todas zero (absorvido por PROCESSO/indicadores)."""
    ref = {} if ref is None else ref
    blocos = []
    for c in C.columns:
        s = C[c]
        if c not in ref:
            ref[c] = s.value_counts().index[0]
        niveis = [v for v in sorted(s.dropna().unique()) if v != ref[c]]
        for v in niveis:
            blocos.append(pd.Series((s == v).astype(float), index=C.index, name=f"{c}={v}"))
    D = pd.concat(blocos, axis=1) if blocos else pd.DataFrame(index=C.index)
    return D, ref


# ------------------------------------------------------------ desenho principal
def construir_desenho(df: pd.DataFrame, atributos: list, hier: dict, ramo: dict,
                      centros: dict, ref_ctrl: dict | None = None):
    """Matriz de regressores (sem FE) a partir do DataFrame bruto de respondentes.

    Retorna X (DataFrame float), grupos (coluna -> tipo) e ref_ctrl."""
    N = pd.DataFrame({a: likert_num(df[a]) for a in atributos}, index=df.index)
    proc = df["PROCESSO"].astype(str)
    cols, grupos = {}, {}
    for a in atributos:
        valido = N[a].notna()
        elegivel = proc.isin(ramo[a])
        if a in hier:
            p = hier[a]
            Np = likert_num(df[p])
            elegivel &= Np.notna() & (Np <= 3)
        cols[a] = (N[a] - centros[a]).where(valido, 0.0)
        grupos[a] = "atributo"
        na = (elegivel & ~valido).astype(float)
        if na.sum() >= 20:
            cols[f"NA|{a}"] = na
            grupos[f"NA|{a}"] = "ausente"
    for p in sorted(set(hier.values())):
        if p in atributos or p in df.columns:
            Np = likert_num(df[p])
            cols[f"RAMO|{p}<=3"] = (Np.notna() & (Np <= 3)).astype(float)
            grupos[f"RAMO|{p}<=3"] = "ramo"
    cols["PROCESSO=Embarque"] = (proc == "Embarque").astype(float)
    grupos["PROCESSO=Embarque"] = "processo"
    X = pd.DataFrame(cols, index=df.index)
    D, ref_ctrl = dummies_controles(controles_limpos(df), ref_ctrl)
    for c in D.columns:
        grupos[c] = "controle"
    X = pd.concat([X, D], axis=1)
    return X, grupos, ref_ctrl


def efeitos_fixos(df: pd.DataFrame, aeroportos_ref=None, per_ref=None):
    """Dummies de aeroporto e de período ("2021T1"…"2026T2"), drop-first com referência fixa
    (primeiro aeroporto e primeiro período em ordem alfabética = cronológica)."""
    a = df["AEROPORTO"].astype(str)
    t = df["PERIODO"].astype(str)
    aeroportos_ref = aeroportos_ref or sorted(a.unique())
    per_ref = per_ref or sorted(t.unique())
    F = {}
    for v in aeroportos_ref[1:]:
        F[f"FE_aero={v}"] = (a == v).astype(float)
    for v in per_ref[1:]:
        F[f"FE_per={v}"] = (t == v).astype(float)
    return pd.DataFrame(F, index=df.index)


def remover_dependentes(X: pd.DataFrame, protegidas=(), tol=1e-9):
    """Remove colunas constantes, duplicadas e linearmente dependentes (QR com pivotamento).
    Colunas protegidas (ex.: FE) são postas primeiro para terem prioridade."""
    from scipy.linalg import qr
    X = X.loc[:, X.std() > 0]
    X = X.T.drop_duplicates().T
    ordem = [c for c in X.columns if c in protegidas] + [c for c in X.columns if c not in protegidas]
    X = X[ordem]
    Z = (X - X.mean()) / X.std()
    Z = np.column_stack([np.ones(len(Z)), Z.values])
    _, R, piv = qr(Z, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    rank = int((d > tol * d[0]).sum())
    manter = sorted(piv[:rank])
    manter = [i - 1 for i in manter if i > 0]
    removidas = [c for i, c in enumerate(X.columns) if i not in manter]
    return X.iloc[:, manter], removidas


# ------------------------------------------------------------------ nomes curtos
NOMES_CURTOS = {
    "FACILIDADE DE DESEMBARQUE NO MEIO-FIO": "Desembarque no meio-fio",
    "OPÇÕES DE TRANSPORTE ATÉ O AEROPORTO": "Opções de transporte",
    "PROCESSO DE CHECK IN": "Check-in",
    "PROCESSO DE AQUISIÇÃO DA PASSAGEM": "Aquisição da passagem",
    "PROCESSO DE INSPEÇÃO DE SEGURANÇA": "Inspeção de segurança",
    "CONTROLE MIGRATÓRIO": "Controle migratório",
    "CONTROLE ADUANEIRO": "Controle aduaneiro",
    "ESTABELECIMENTOS DE ALIMENTAÇÃO": "Alimentação",
    "QUANTIDADE DE ESTABELECIMENTOS DE ALIMENTAÇÃO": "Qtd. alimentação",
    "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS DE ALIMENTAÇÃO": "Variedade alimentação",
    "RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS DE ALIMENTAÇÃO": "Preço×qualidade alim.",
    "ESTABELECIMENTOS COMERCIAIS": "Comércio",
    "ESTACIONAMENTO": "Estacionamento",
    "LOCALIZAÇÃO E DESLOCAMENTO": "Localização e deslocamento",
    "SINALIZAÇÃO": "Sinalização",
    "DISPONIBILIDADE DE PAINÉIS DE INFORMAÇÕES DE VOO": "Painéis de voo",
    "CONFORTO DA SALA DE EMBARQUE": "Conforto sala embarque",
    "CONFORTO TÉRMICO": "Conforto térmico",
    "CONFORTO ACÚSTICO": "Conforto acústico",
    "DISPONIBILIDADE DE ASSENTOS": "Assentos",
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
    "TEMPO DE ESPERA EM FILA": "Espera fila inspeção",
    "ORGANIZAÇÃO DAS FILAS2": "Organização filas inspeção",
    "ATENDIMENTO DOS FUNCIONÁRIOS": "Atendimento inspeção",
}


def curto(c: str) -> str:
    if c.startswith("NA|"):
        return "Ausente: " + curto(c[3:])
    if c.startswith("RAMO|"):
        return "Ramo " + curto(c[5:].replace("<=3", "")) + " $\\leq$ 3"
    return NOMES_CURTOS.get(c, c[:1] + c[1:].lower()[:32])


# ------------------------------------------------------------------ bases
def carregar_mvp():
    """Base MVP com o MESMO filtro/índice dos splits de ML (target válido, reset_index)."""
    df = pd.read_parquet(PROCESSADOS / "base_unificada_mvp.parquet")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True)
    df["y"] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    idx = {k: pd.read_parquet(SPLITS / f"X_{k}.parquet", columns=[]).index
           for k in ["train", "val", "test"]}
    return df, idx


def carregar_20(colunas=None):
    """Base dos 20 aeroportos, série 2021T1–2026T2: a construída por 00_base20.py (com regime
    na data da entrevista) se existir; senão a série bruta mpor_2021_2026_20.parquet.
    `colunas`: subconjunto opcional de colunas (economiza memória)."""
    for nome in ["econometria_base20.parquet", "mpor_2021_2026_20.parquet"]:
        p = PROCESSADOS / nome
        if p.exists():
            if colunas is not None:
                import pyarrow.parquet as pq
                disp = set(pq.read_schema(p).names)
                colunas = [c for c in dict.fromkeys(list(colunas) + [TARGET]) if c in disp]
            df = pd.read_parquet(p, columns=colunas)
            df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
            df["y"] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
            return df, nome
    raise FileNotFoundError("Base da série não encontrada (rode 10_serie_historica/01 e 07_econometria/00).")
