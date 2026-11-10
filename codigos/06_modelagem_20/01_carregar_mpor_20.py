"""
T5/modelos20 — Base MPor dos 20 aeroportos para a modelagem.

Série 2021T1–2026T2 (08/11/2026): a base da modelagem passou a ser
dados/processados/mpor_2021_2026_20.parquet, gerada por
codigos/10_serie_historica/01_carregar_serie.py (todas as bases trimestrais do
portal com a metodologia atual). Este script é um invólucro fino:
  * se a base da série já existe, só confere e resume (linhas por ano/período);
  * se não existe (ou RECARREGAR=1), roda o carregador da série. O carregador usa
    mpor_2023_20.parquet como referência de colunas e tipos; se ela também faltar,
    é recriada antes a partir dos Excels de 2023 (lógica antiga, mantida abaixo).

Entrada:  dados/brutos/mpor/portal/... (série) e dados/brutos/mpor/2023/*.xlsx (referência)
Saída:    dados/processados/mpor_2021_2026_20.parquet (+ mpor_2023_20.parquet, se faltava)
"""
import os
import runpy
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRUTOS = ROOT / "dados" / "brutos" / "mpor" / "2023"
PROCESSADOS = ROOT / "dados" / "processados"
SERIE = PROCESSADOS / "mpor_2021_2026_20.parquet"
REF_2023 = PROCESSADOS / "mpor_2023_20.parquet"
CARREGADOR = ROOT / "codigos" / "10_serie_historica" / "01_carregar_serie.py"
TRIMESTRES = [1, 2, 3, 4]


def carregar_trimestre(t: int) -> pd.DataFrame:
    path = BRUTOS / f"pesquisa_satisfacao_2023_T{t}.xlsx"
    df = pd.read_excel(path, sheet_name=0, header=1)
    df["TRIMESTRE"] = t
    df["ANO"] = 2023
    return df


def gerar_referencia_2023():
    """Base de 2023 (referência de leiaute para o carregador da série)."""
    dfs = []
    for t in TRIMESTRES:
        df = carregar_trimestre(t)
        print(f"  2023 T{t}: {len(df):>6} respondentes, {df.shape[1]} colunas")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str).where(df[col].notna(), None)
    df.to_parquet(REF_2023, index=False)
    print(f"  → {REF_2023.relative_to(ROOT)} ({len(df):,} linhas)")


def main():
    if os.environ.get("RECARREGAR") == "1" or not SERIE.exists():
        if not REF_2023.exists():
            gerar_referencia_2023()
        print(f"  rodando {CARREGADOR.relative_to(ROOT)}")
        runpy.run_path(str(CARREGADOR), run_name="__main__")
    df = pd.read_parquet(SERIE, columns=["AEROPORTO", "ANO", "PERIODO", "PROCESSO"])
    print(f"Série: {len(df):,} respondentes, {df['AEROPORTO'].nunique()} aeroportos, "
          f"{df['PERIODO'].nunique()} períodos ({df['PERIODO'].min()} a {df['PERIODO'].max()})")
    print(df.groupby("ANO").size().to_string())
    print(df.groupby(["AEROPORTO", "ANO"]).size().unstack(fill_value=0).to_string())
    print(f"  base da modelagem: {SERIE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
