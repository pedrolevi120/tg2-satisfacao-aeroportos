"""
SO2.1 — Preparação de features + split train/val/test (MVP, série 2021T1–2026T2).

Split aleatório 70/15/15 estratificado por aeroporto x nota (seed 42), agora sobre a série inteira.
Categóricas de tempo/regime: PERIODO (21 níveis) e REGIME_NO_PERIODO (regime na data da entrevista);
contexto numérico por aeroporto x período (passageiros e decolagens do trimestre) + fixos da cidade.

Entrada: dados/processados/base_unificada_mvp.parquet
Saída:   dados/processados/splits/{X,y}_{train,val,test}.parquet
         dados/processados/splits/feature_metadata.json
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSADOS = ROOT / "dados" / "processados"
SPLITS_DIR = PROCESSADOS / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "SATISFAÇÃO GERAL"
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # do total; do conjunto remaining seria 0.1765
NA_FILL_LIKERT = -1
COLUNAS_DESCARTAR = [
    "CHAVE", "CHAVE_SERIE", "PROCESSO", "DATA", "MÊS", "INÍCIO COLETA", "FIM COLETA",
    "VOO", "CIA AÉREA", "COMENTÁRIOS ADICIONAIS",
    # contexto que não entra como feature (data de assunção; regime do trimestre, substituído
    # pelo regime na data da entrevista; fração de dias de concessão, redundante com ele)
    "AERO_data_assuncao", "AERO_regime_gestao", "AERO_frac_dias_concessao",
]
# Colunas de contexto/tempo nunca são Likert (p.ex. AERO_pistas e TRIMESTRE têm valores em 1–5)
PREFIXOS_CONTEXTO = ("AERO_", "IBGE_")


def is_likert(s: pd.Series) -> bool:
    """True se a coluna tem >80% valores numéricos 1-5 ou NS/NR."""
    if s.dtype.kind in "biufc":
        # já numérica; checar se valores estão entre 1-5
        v = s.dropna().unique()
        return len(v) > 0 and set(v).issubset({1, 2, 3, 4, 5})
    # "4.0" e "4" são a mesma nota (a série mistura os dois formatos em alguns trimestres;
    # sem normalizar, a coluna deixava de ser detectada como Likert)
    valores = s.dropna().astype(str).str.strip().str.replace(r"\.0$", "", regex=True).unique()
    likert_set = {"1", "2", "3", "4", "5"}
    likert_extras = likert_set | {"NS/NR", "Não Existe", "nan", "None"}
    if len(valores) == 0:
        return False
    share = sum(v in likert_extras for v in valores) / len(valores)
    return share >= 0.8 and len(set(valores) & likert_set) >= 3


def limpar_nomes(cols) -> list:
    """LightGBM não aceita caracteres especiais de JSON em nomes de features; na série,
    MOTIVO DA VIAGEM traz respostas abertas ("Outros - ...") com vírgulas e dois-pontos."""
    return [re.sub(r'[",:\[\]{}]', " ", str(c)) for c in cols]


def main():
    print("=== Carregando base unificada ===")
    df = pd.read_parquet(PROCESSADOS / "base_unificada_mvp.parquet")
    print(f"  {len(df):,} respondentes, {df.shape[1]} colunas")

    # Target → numérico
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid = y.notna()
    print(f"\n  Respondentes com target válido: {valid.sum():,} ({100*valid.mean():.2f}%)")
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True).astype(float)

    # Descartar colunas óbvias
    df = df.drop(columns=[c for c in COLUNAS_DESCARTAR if c in df.columns], errors="ignore")

    # Identificar Likert
    # TERMINAL e PORTÃO são identificadores (não avaliações), embora tenham valores 1–5
    cols_excluir_likert = {TARGET_COL, "AEROPORTO", "TRIMESTRE", "ANO", "PERIODO",
                           "REGIME_NO_PERIODO", "TERMINAL", "PORTÃO"}
    likert_cols = [c for c in df.columns if c not in cols_excluir_likert
                   and not c.startswith(PREFIXOS_CONTEXTO) and is_likert(df[c])]
    print(f"\n  {len(likert_cols)} colunas Likert detectadas")

    # Converter Likert para numérico + NaN como -1
    X_likert = df[likert_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).fillna(NA_FILL_LIKERT).astype(int)

    # Features categóricas → one-hot
    cat_cols = ["AEROPORTO", "PERIODO", "REGIME_NO_PERIODO",
                "GÊNERO", "ESCOLARIDADE", "RENDA FAMILIAR",
                "MOTIVO DA VIAGEM", "TIPO DE VOO", "NACIONALIDADE",
                "VIAJANDO SOZINHO", "IDADE", "CONEXÃO",
                "AERO_operador"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    X_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols,
                            drop_first=False, dummy_na=False)
    X_cat.columns = limpar_nomes(X_cat.columns)

    # Numéricas contextuais
    num_cols = ["AERO_pax_trim_mi", "AERO_decolagens_trim_k", "AERO_pistas",
                "IBGE_populacao_2022_mi", "IBGE_pib_per_capita_k", "IBGE_idh_2010"]
    num_cols = [c for c in num_cols if c in df.columns]
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Concat
    X = pd.concat([X_likert, X_cat, X_num], axis=1)
    print(f"  X final: {X.shape}")

    # Estratificação por (aeroporto, target_inteiro)
    strat = df["AEROPORTO"].astype(str) + "_" + y.round().astype(int).astype(str)

    # Split: 70 / 15 / 15
    X_temp, X_test, y_temp, y_test, strat_temp, _ = train_test_split(
        X, y, strat, test_size=TEST_SIZE, random_state=SEED, stratify=strat
    )
    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, random_state=SEED, stratify=strat_temp
    )

    print(f"\n=== Splits ===")
    print(f"  Treino: {len(X_train):>6} ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Val:    {len(X_val):>6} ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Teste:  {len(X_test):>6} ({100*len(X_test)/len(X):.1f}%)")

    print(f"\n=== Verificação estratificação por aeroporto ===")
    aero = df.loc[X.index, "AEROPORTO"]
    for nome, idx in [("treino", X_train.index), ("val", X_val.index), ("teste", X_test.index)]:
        prop = aero.loc[idx].value_counts(normalize=True)
        print(f"  {nome}: " + ", ".join(f"{k}={v:.2f}" for k, v in prop.items()))

    per = df.loc[X.index, "PERIODO"]
    print("\n=== Teste por período ===")
    print(per.loc[X_test.index].value_counts().sort_index().to_string())

    # Salvar
    print(f"\n=== Salvando em {SPLITS_DIR.relative_to(ROOT)} ===")
    # Renomear colunas: pyarrow não gosta de dtypes mistos; garantir str
    X.columns = X.columns.astype(str)
    X_train.columns = X_train.columns.astype(str)
    X_val.columns = X_val.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_train.to_parquet(SPLITS_DIR / "X_train.parquet")
    X_val.to_parquet(SPLITS_DIR / "X_val.parquet")
    X_test.to_parquet(SPLITS_DIR / "X_test.parquet")
    pd.DataFrame({"y": y_train}).to_parquet(SPLITS_DIR / "y_train.parquet")
    pd.DataFrame({"y": y_val}).to_parquet(SPLITS_DIR / "y_val.parquet")
    pd.DataFrame({"y": y_test}).to_parquet(SPLITS_DIR / "y_test.parquet")

    metadata = {
        "target": TARGET_COL,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_features_total": int(X.shape[1]),
        "n_likert": len(likert_cols),
        "n_categoricas_dummies": X_cat.shape[1],
        "n_numericas": X_num.shape[1],
        "na_fill_likert": NA_FILL_LIKERT,
        "seed": SEED,
        "base": "MPor 2021T1–2026T2 (série histórica), GRU/CGH/VCP",
        "periodos": sorted(df["PERIODO"].unique().tolist()),
        "likert_cols": likert_cols,
        "categoricas_original": cat_cols,
        "numericas": num_cols,
    }
    (SPLITS_DIR / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print("  → splits + feature_metadata.json")


if __name__ == "__main__":
    main()
