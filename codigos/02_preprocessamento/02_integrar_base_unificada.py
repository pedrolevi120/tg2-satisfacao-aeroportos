"""
SO1.5 — Integra a base MPor do MVP (série 2021T1–2026T2) com o contexto de cada aeroporto
NO PERÍODO (ANAC: regime, operador, passageiros e decolagens do trimestre; IBGE: cidade)
num único Parquet pronto pra modelagem.

Entrada:  dados/processados/mpor_2023_mvp.parquet   (série 2021–2026 filtrada no MVP)
          dados/processados/contexto_aeroportos_20_serie.csv  (aeroporto x período)
Saída:    dados/processados/base_unificada_mvp.parquet

Colunas de contexto (prefixos mantidos do MVP original):
  AERO_regime_gestao, AERO_operador, AERO_frac_dias_concessao, AERO_pax_trim_mi,
  AERO_decolagens_trim_k, AERO_pistas, IBGE_populacao_2022_mi, IBGE_pib_per_capita_k,
  IBGE_idh_2010 (as três últimas constantes por aeroporto).
REGIME_NO_PERIODO: regime na DATA da entrevista (substitui REGIME_NO_TRIMESTRE, que fixava
TRIMESTRE <= 3 de 2023 para SBSP). Infraero antes da data de assunção do concessionário
(SBSP: 17/10/2023), concessionada a partir dela.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSADOS = ROOT / "dados" / "processados"

COLS_AERO = {"regime_gestao": "AERO_regime_gestao", "operador": "AERO_operador",
             "frac_dias_concessao": "AERO_frac_dias_concessao", "pax_trim_mi": "AERO_pax_trim_mi",
             "decolagens_trim_k": "AERO_decolagens_trim_k", "pistas": "AERO_pistas",
             "data_assuncao": "AERO_data_assuncao",
             "populacao_2022_mi": "IBGE_populacao_2022_mi", "pib_per_capita_k": "IBGE_pib_per_capita_k",
             "idh_2010": "IBGE_idh_2010"}


def regime_na_data(df: pd.DataFrame) -> pd.Series:
    """Regime na data da entrevista. Em período 'misto' (troca de operador dentro do trimestre),
    decide pela data de assunção; nos demais, usa o regime do período."""
    data = pd.to_datetime(df["DATA"], errors="coerce")
    assun = pd.to_datetime(df["AERO_data_assuncao"], format="%d/%m/%Y", errors="coerce")
    reg = df["AERO_regime_gestao"].astype(str).copy()
    misto = reg.eq("misto")
    reg.loc[misto] = (data[misto] >= assun[misto]).map({True: "concessionada", False: "infraero"})
    return reg.map({"infraero": "Infraero", "concessionada": "concessionada"}).fillna(reg)


def main():
    print("=== Carregando ===")
    mpor = pd.read_parquet(PROCESSADOS / "mpor_2023_mvp.parquet")
    ctx = pd.read_csv(PROCESSADOS / "contexto_aeroportos_20_serie.csv")
    print(f"  MPor:     {len(mpor):,} respondentes, {mpor.shape[1]} colunas")
    print(f"  Contexto: {len(ctx)} aeroporto x período, {ctx.shape[1]} colunas")

    print("\n=== Merge (aeroporto x período) ===")
    ctx = ctx[["icao", "periodo"] + list(COLS_AERO)].rename(columns=COLS_AERO)
    df = mpor.merge(ctx, left_on=["AEROPORTO", "PERIODO"], right_on=["icao", "periodo"],
                    how="left", validate="many_to_one")
    df = df.drop(columns=["icao", "periodo"])
    assert df["AERO_regime_gestao"].notna().all(), "período sem contexto"
    assert len(df) == len(mpor)

    df["REGIME_NO_PERIODO"] = regime_na_data(df)
    print(f"  Resultado: {len(df):,} respondentes, {df.shape[1]} colunas")

    out = PROCESSADOS / "base_unificada_mvp.parquet"
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), None)
    df.to_parquet(out, index=False)
    print(f"\n=== Salvo em {out.name} ({out.stat().st_size / 1024:.1f} KB) ===")

    print("\n=== Quebra regime x período (SBSP) ===")
    print(df[df.AEROPORTO == "SBSP"].groupby(["PERIODO", "REGIME_NO_PERIODO"]).size().unstack(fill_value=0))
    print(df.groupby(["AEROPORTO", "REGIME_NO_PERIODO"]).size().to_frame("n"))


if __name__ == "__main__":
    main()
