"""
T6/econometria — Base de respondentes dos 20 aeroportos, série MPor 2021T1–2026T2.

Parte da série já harmonizada (codigos/10_serie_historica/01_carregar_serie.py) e acrescenta o
contexto de gestão de cada aeroporto no período (contexto_aeroportos_20_serie.csv):
  * REGIME_PERIODO, OPERADOR_PERIODO, FRAC_DIAS_CONCESSAO (do trimestre);
  * REGIME_NA_DATA: regime na data da entrevista (decide os trimestres 'mistos' pela data de assunção);
  * GRUPO_DID: papel no DiD (regime_operador_serie.csv): tratado (SBSP), tratado_secundario (SBBE),
    controle (sem troca de operador em 2021T1–2026T2) ou excluido (SBEG, SBGO, SBCT, SBSG).
  * DATA corrigida: em 2024T1–T2 e em parte de 2025T3–T4 a série traz a data como número serial do
    Excel (lido como nanossegundos após 1970); converte-se por 30/12/1899 + serial dias. Em 2022T4
    (outubro) 7.000 linhas não têm DATA; o mês é recuperado de MÊS. MES_NUM (1–12) vem sempre de MÊS.
Também confere a consistência das datas de troca entre os dois arquivos e de DATA com ANO/MÊS.

Entrada: dados/processados/mpor_2021_2026_20.parquet, dados/processados/contexto_aeroportos_20_serie.csv,
         codigos/07_econometria/regime_operador_serie.csv
Saída:   dados/processados/econometria_base20.parquet, resultados/base20_n_aeroporto_periodo.csv
"""
import pandas as pd
from comum import ROOT, PROCESSADOS, OUT_DIR, TARGET
from pathlib import Path


MESES = {"JANEIRO": 1, "FEVEREIRO": 2, "MARÇO": 3, "ABRIL": 4, "MAIO": 5, "JUNHO": 6, "JULHO": 7,
         "AGOSTO": 8, "SETEMBRO": 9, "OUTUBRO": 10, "NOVEMBRO": 11, "DEZEMBRO": 12}


def corrigir_data(s: pd.Series) -> pd.Series:
    """Datas gravadas como serial do Excel (aparecem como 1970-01-01 + n ns) -> data correta."""
    d = pd.to_datetime(s, errors="coerce")
    serial = d.notna() & (d.dt.year < 2000)
    n = d[serial].astype("int64")
    d.loc[serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(n, unit="D")
    print(f"DATA: {serial.sum()} seriais do Excel convertidos")
    return d


def main():
    df = pd.read_parquet(PROCESSADOS / "mpor_2021_2026_20.parquet")
    ctx = pd.read_csv(PROCESSADOS / "contexto_aeroportos_20_serie.csv")
    reg = pd.read_csv(Path(__file__).resolve().parent / "regime_operador_serie.csv")
    print(f"Série: {len(df):,} respondentes, {df['AEROPORTO'].nunique()} aeroportos, "
          f"{df['PERIODO'].nunique()} períodos")

    # datas de troca: arquivo DiD x contexto
    da = ctx.groupby("icao")["data_assuncao"].first()
    for _, r in reg[reg.data_troca.notna()].iterrows():
        if r.tipo_troca == "publico->concessao":
            assert pd.Timestamp(r.data_troca) == pd.to_datetime(da[r.icao], dayfirst=True), r.icao

    vazias = df["AEROPORTO"].isna()
    print(f"Linhas sem AEROPORTO (vazias, sem nota; 2022T2): {vazias.sum()} -> descartadas")
    df = df.loc[~vazias].reset_index(drop=True)
    c = ctx[["icao", "periodo", "regime_gestao", "operador", "frac_dias_concessao", "data_assuncao"]].rename(
        columns={"regime_gestao": "REGIME_PERIODO", "operador": "OPERADOR_PERIODO",
                 "frac_dias_concessao": "FRAC_DIAS_CONCESSAO", "data_assuncao": "DATA_ASSUNCAO"})
    n0 = len(df)
    df = df.merge(c, left_on=["AEROPORTO", "PERIODO"], right_on=["icao", "periodo"], how="left",
                  validate="many_to_one").drop(columns=["icao", "periodo"])
    assert len(df) == n0 and df["REGIME_PERIODO"].notna().all()
    df["DATA"] = corrigir_data(df["DATA"])
    df["MES_NUM"] = df["MÊS"].astype(str).str.strip().str.upper().map(MESES).astype("Int64")
    assert df["MES_NUM"].notna().all()
    ok = df["DATA"].notna()
    assert (df.loc[ok, "DATA"].dt.month == df.loc[ok, "MES_NUM"]).all()
    assert (df.loc[ok, "DATA"].dt.year == df.loc[ok, "ANO"]).all()
    print(f"DATA ausente após correção: {(~ok).sum()} linhas "
          f"({df.loc[~ok, 'PERIODO'].value_counts().to_dict()})")
    data = df["DATA"]
    assun = pd.to_datetime(df["DATA_ASSUNCAO"], format="%d/%m/%Y", errors="coerce")
    rna = df["REGIME_PERIODO"].astype(str).copy()
    m = rna.eq("misto")
    rna.loc[m] = (data[m] >= assun[m]).map({True: "concessionada", False: "infraero"})
    df["REGIME_NA_DATA"] = rna
    df = df.merge(reg[["icao", "grupo_did"]].rename(columns={"grupo_did": "GRUPO_DID"}),
                  left_on="AEROPORTO", right_on="icao", how="left").drop(columns="icao")
    assert df["GRUPO_DID"].notna().all()

    out = PROCESSADOS / "econometria_base20.parquet"
    df.to_parquet(out, index=False)
    nv = pd.to_numeric(df[TARGET], errors="coerce").notna()
    print(f"-> {out.name}: {len(df):,} linhas ({nv.sum():,} com satisfação geral válida)")
    tab = df[nv].groupby(["AEROPORTO", "PERIODO"]).size().unstack(fill_value=0)
    tab.to_csv(OUT_DIR / "base20_n_aeroporto_periodo.csv")
    print(tab.to_string())
    print(df.groupby(["GRUPO_DID"])["AEROPORTO"].unique())
    print(pd.crosstab(df["AEROPORTO"], df["REGIME_NA_DATA"]))


if __name__ == "__main__":
    main()
