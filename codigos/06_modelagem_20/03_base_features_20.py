"""
T5/modelos20 — Tarefa 2: base unificada dos 20 aeroportos + features + split.

Diferenças deliberadas em relação ao MVP (03_features_e_split.py), justificadas
pela auditoria:
  * detecção de Likert normaliza "4.0" -> "4" (no MVP, colunas gravadas com
    inteiros e floats misturados — p.ex. CONFORTO TÉRMICO, LIMPEZA DOS BANHEIROS,
    ESTABELECIMENTOS COMERCIAIS — não eram reconhecidas como Likert);
  * TERMINAL (identificador 1–4) deixa de ser tratado como atributo Likert;
  * ausentes/NS/NR ficam como NaN (tratamento nativo), e PROCESSO é mantido
    (especificação escolhida na Tarefa 1: um modelo por ramo do questionário);
  * categorias com frequência < 0,5% são agrupadas em "Outros" (MOTIVO DA
    VIAGEM tem >600 grafias livres).
Série 2021T1–2026T2 (08/11/2026): a base é a série inteira (462.616 linhas antes
do filtro do alvo). O período (PERIODO, "2021T1"…"2026T2") substitui o trimestre
como categoria, e o contexto ANAC/IBGE é integrado por aeroporto × período
(dados/processados/contexto_aeroportos_20_serie.csv): regime de gestão e fração de
dias sob concessão no trimestre, passageiros e decolagens do trimestre, pistas e
socioeconômicas do polo servido. Split, lógica de ausência estrutural, detecção de
Likert e codificações são as mesmas de antes.
Teste rápido: AMOSTRA_FRAC=0.1 (subamostra aleatória das linhas, semente 42) junto
com SAIDA_TESTE=<dir> (ver comum.py), para não sobrescrever os artefatos oficiais.

Entrada:  dados/processados/mpor_2021_2026_20.parquet  (01_carregar_mpor_20.py)
          dados/processados/contexto_aeroportos_20_serie.csv
Saída:    dados/processados/base_unificada_20.parquet
          dados/processados/splits_20/{X.parquet, meta.parquet, feature_metadata.json}
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from comum import ROOT, PROC_OUT, SPLITS, SEED, TARGET_COL, BASE_MPOR, CONTEXTO_SERIE

SPLITS.mkdir(parents=True, exist_ok=True)
CONTEXTO = CONTEXTO_SERIE
AMOSTRA_FRAC = float(os.environ.get("AMOSTRA_FRAC", "1"))

# Geografia (estática) — mesma tabela de 06_visualizacoes/01_panorama_agregado.py
REGIAO = {"SBBE": "Norte", "SBEG": "Norte",
          "SBBR": "Centro-Oeste", "SBCY": "Centro-Oeste", "SBGO": "Centro-Oeste",
          "SBFZ": "Nordeste", "SBMO": "Nordeste", "SBRF": "Nordeste", "SBSG": "Nordeste", "SBSV": "Nordeste",
          "SBCF": "Sudeste", "SBGL": "Sudeste", "SBGR": "Sudeste", "SBKP": "Sudeste",
          "SBRJ": "Sudeste", "SBSP": "Sudeste", "SBVT": "Sudeste",
          "SBCT": "Sul", "SBFL": "Sul", "SBPA": "Sul"}

NAO_LIKERT = {TARGET_COL, "TERMINAL", "PORTÃO", "TRIMESTRE", "ANO", "CHAVE", "VOO",
              "PERIODO", "CHAVE_SERIE", "DATA"}
CAT_COLS = ["AEROPORTO", "PERIODO", "GÊNERO", "ESCOLARIDADE", "RENDA FAMILIAR",
            "MOTIVO DA VIAGEM", "TIPO DE VOO", "NACIONALIDADE", "VIAJANDO SOZINHO",
            "IDADE", "CONEXÃO"]
LIKERT_OK = {"1", "2", "3", "4", "5"}
LIKERT_NA = {"NS/NR", "Não Existe", "nan", "None", "Não utilizou", "Não se aplica",
             "NA/NR/NS", "NA/NR/NA"}  # variantes raras de NS/NR em outros anos da série
# Cobertura mínima para manter um atributo Likert. Em 2023 era 0,5%; na série os 4 subitens
# de ESTACIONAMENTO caem para 0,46–0,47% (~2,1 mil respostas cada, contra ~500 em 2023),
# e os 4 subitens do CONTROLE MIGRATÓRIO (excluídos em 2023, 0,47%) ficam em 0,43%.
# 0,45% preserva exatamente os mesmos 50 atributos de 2023 (o critério só existe para
# descartar blocos quase vazios).
COB_MIN = 0.0045


def norm_likert(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return t


def is_likert(s: pd.Series) -> bool:
    v = norm_likert(s.dropna()).unique()
    if len(v) == 0:
        return False
    share = np.mean([x in LIKERT_OK | LIKERT_NA for x in v])
    return share >= 0.8 and len(set(v) & LIKERT_OK) >= 3


def integrar_contexto(df):
    """Contexto por aeroporto × período (chaves icao + periodo)."""
    if not CONTEXTO.exists():
        print("  [contexto ausente] base sem features de contexto")
        return df, []
    ctx = pd.read_csv(CONTEXTO)
    chaves = ["icao", "ano", "trimestre", "periodo"]
    assert not ctx.duplicated(["icao", "periodo"]).any(), "contexto duplicado por aeroporto × período"
    ren = {c: f"CTX_{c}" for c in ctx.columns if c not in chaves}
    ctx = ctx.rename(columns=ren).drop(columns=["ano", "trimestre"])
    n0 = len(df)
    df = df.merge(ctx, left_on=["AEROPORTO", "PERIODO"], right_on=["icao", "periodo"], how="left",
                  validate="many_to_one").drop(columns=["icao", "periodo"])
    assert len(df) == n0
    sem = df["CTX_regime_gestao"].isna()
    if sem.any():
        print(f"  [contexto] {sem.sum()} linhas sem contexto: "
              f"{df.loc[sem, ['AEROPORTO', 'PERIODO']].value_counts().head(5).to_dict()}")
    print(f"  contexto integrado (aeroporto × período): {list(ren.values())}")
    return df, list(ren.values())


def main():
    df = pd.read_parquet(BASE_MPOR)
    if AMOSTRA_FRAC < 1:
        df = df.sample(frac=AMOSTRA_FRAC, random_state=SEED).sort_values("CHAVE_SERIE").reset_index(drop=True)
        print(f"  [AMOSTRA_FRAC={AMOSTRA_FRAC}] {len(df):,} linhas")
    sem_aero = df["AEROPORTO"].isna()
    if sem_aero.any():  # linhas vazias dos arquivos do portal (sem aeroporto nem nota)
        print(f"  {sem_aero.sum()} linhas sem AEROPORTO descartadas "
              f"(com nota válida: {pd.to_numeric(df.loc[sem_aero, TARGET_COL], errors='coerce').notna().sum()})")
        df = df.loc[~sem_aero].reset_index(drop=True)
    df["REGIAO"] = df["AEROPORTO"].map(REGIAO)
    assert df["REGIAO"].notna().all()
    df, ctx_cols = integrar_contexto(df)
    out = PROC_OUT / "base_unificada_20.parquet"
    tmp = df.copy()
    for col in tmp.select_dtypes(include=["object", "string"]).columns:
        tmp[col] = tmp[col].astype(str).where(tmp[col].notna(), None)
    tmp.to_parquet(out, index=False)
    print(f"  → {out}: {df.shape}")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    ok = y.notna()
    df, y = df.loc[ok].reset_index(drop=True), y.loc[ok].reset_index(drop=True)
    print(f"  target válido: {len(df):,} ({ok.mean():.2%})")

    likert = [c for c in df.columns if c not in NAO_LIKERT and c not in CAT_COLS
              and not c.startswith("CTX_") and is_likert(df[c])]
    Xl = df[likert].apply(lambda s: pd.to_numeric(norm_likert(s), errors="coerce")).astype(float)
    Xl = Xl.where(Xl.isin([1, 2, 3, 4, 5]))
    cobertura = Xl.notna().mean()
    likert = [c for c in likert if cobertura[c] >= COB_MIN]  # descarta blocos quase vazios
    Xl = Xl[likert]
    print(f"  {len(likert)} atributos Likert (cobertura ≥ {COB_MIN:.2%})")
    if AMOSTRA_FRAC == 1 and len(likert) != 50:
        print(f"  [ATENÇÃO] {len(likert)} atributos Likert (eram 50 em 2023)")

    cats = df[CAT_COLS].astype(str).where(df[CAT_COLS].notna(), None)
    for c in CAT_COLS:
        freq = cats[c].value_counts(normalize=True)
        raros = freq[freq < 0.005].index
        cats.loc[cats[c].isin(raros), c] = "Outros"
    Xc = pd.get_dummies(cats, prefix=CAT_COLS, dummy_na=False).astype(np.uint8)
    # contexto: numéricas entram direto; regime de gestão (por aeroporto × período) como dummies;
    # operador é descartado (identificador, redundante com AEROPORTO)
    # Socioeconômicas do município-SEDE são descartadas em favor das do POLO servido
    # (recomendação do T2: sedes pequenas como Confins geram outliers artificiais).
    SEDE = {"CTX_populacao_2022_mi", "CTX_pib_per_capita_k", "CTX_idh_2010", "CTX_renda_per_capita_2010"}
    ctx_num = [c for c in ctx_cols if pd.api.types.is_numeric_dtype(df[c]) and c not in SEDE]
    Xn = df[ctx_num].astype(float) if ctx_num else pd.DataFrame(index=df.index)
    if "CTX_regime_gestao" in df.columns:
        Xn = pd.concat([Xn, pd.get_dummies(df["CTX_regime_gestao"], prefix="CTX_regime").astype(np.uint8)], axis=1)
    ctx_cols = list(Xn.columns)
    X = pd.concat([Xl, Xc, Xn], axis=1)
    X.columns = X.columns.astype(str)
    print(f"  X: {X.shape} (likert={Xl.shape[1]}, dummies={Xc.shape[1]}, contexto={Xn.shape[1]})")

    # Split 70/15/15 estratificado por aeroporto × target (classes 1 e 2 fundidas: raras)
    strat = df["AEROPORTO"] + "_" + y.clip(lower=2).round().astype(int).astype(str)
    idx = np.arange(len(df))
    i_tmp, i_te = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=strat)
    i_tr, i_va = train_test_split(i_tmp, test_size=0.15 / 0.85, random_state=SEED,
                                  stratify=strat.iloc[i_tmp])
    split = np.empty(len(df), dtype=object)
    split[i_tr], split[i_va], split[i_te] = "train", "val", "test"
    meta = pd.DataFrame({"y": y.values, "AEROPORTO": df["AEROPORTO"].values,
                         "PROCESSO": df["PROCESSO"].values, "REGIAO": df["REGIAO"].values,
                         "ANO": df["ANO"].values, "TRIMESTRE": df["TRIMESTRE"].values,
                         "PERIODO": df["PERIODO"].astype(str).values,
                         "CHAVE_SERIE": df["CHAVE_SERIE"].values, "split": split})
    for extra in ["CTX_regime_gestao", "CTX_regime", "CTX_operador"]:
        if extra in df.columns:
            meta[extra] = df[extra].values
    X.to_parquet(SPLITS / "X.parquet")
    meta.to_parquet(SPLITS / "meta.parquet")
    (SPLITS / "feature_metadata.json").write_text(json.dumps({
        "target": TARGET_COL, "seed": SEED, "n_total": len(df),
        "n_split": meta["split"].value_counts().to_dict(),
        "likert_cols": likert, "categoricas": CAT_COLS, "contexto": ctx_cols,
        "cobertura_likert": {c: float(cobertura[c]) for c in likert},
        "contexto_T2_integrado": bool(ctx_cols), "amostra_frac": AMOSTRA_FRAC,
        "n_por_ano": meta["ANO"].value_counts().sort_index().astype(int).to_dict(),
        "n_por_periodo": meta["PERIODO"].value_counts().sort_index().astype(int).to_dict()},
        indent=2, ensure_ascii=False, default=int))
    print(meta.groupby(["split"]).size())
    print(meta.groupby(["ANO", "split"]).size().unstack().to_string())
    print(meta.groupby(["AEROPORTO", "split"]).size().unstack().head(3))


if __name__ == "__main__":
    main()
