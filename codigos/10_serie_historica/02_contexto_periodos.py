"""
Série histórica — contexto por aeroporto x período (2021T1–2026T2) para a modelagem.

- Regime de gestão e operador em cada trimestre, a partir das datas de assunção da tabela curada
  (dados/brutos/anac/regime_gestao_2023_curado.csv): SBEG 11/01/2022, SBGO 24/03/2022, SBCT 31/03/2022,
  SBBE 04/09/2023, SBSP 17/10/2023; SBRJ Infraero o tempo todo; SBSG troca de concessionária
  (Inframerica -> Zurich) em 19/02/2024, sem mudar o regime. Eventos depois de 2024 não levantados.
- Passageiros (pagos + grátis, embarcados + desembarcados) e decolagens NO TRIMESTRE, ANAC Dados Estatísticos.
- Variáveis da cidade (Censo 2022, PIB, IDHM 2010) e pistas: constantes, copiadas do contexto de 2023.

Saída: dados/processados/contexto_aeroportos_20_serie.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "dados" / "processados"
ANAC = ROOT / "dados" / "brutos" / "anac"

TROCA_OPERADOR = {  # icao: (data, operador antes, operador depois) — troca sem mudar o regime
    "SBSG": ("19/02/2024", "Inframerica", "Zurich Airport Brasil"),
}


def periodos():
    out = []
    for ano in range(2021, 2027):
        for t in range(1, 5):
            if (ano, t) > (2026, 2):
                break
            ini = pd.Timestamp(ano, 3 * t - 2, 1)
            fim = (ini + pd.offsets.QuarterEnd(0))
            out.append((ano, t, f"{ano}T{t}", ini, fim))
    return out


def movimento():
    cols = ["ANO", "MES", "AEROPORTO_DE_ORIGEM_SIGLA", "AEROPORTO_DE_DESTINO_SIGLA",
            "PASSAGEIROS_PAGOS", "PASSAGEIROS_GRATIS", "DECOLAGENS"]
    ctx = pd.read_csv(PROC / "contexto_aeroportos_20.csv")
    icaos = set(ctx["icao"])
    partes = []
    for ch in pd.read_csv(ANAC / "Dados_Estatisticos.csv", sep=";", skiprows=1, usecols=cols,
                          chunksize=500_000, encoding="utf-8-sig", low_memory=False):
        ch = ch[(ch["ANO"] >= 2021) & (ch["AEROPORTO_DE_ORIGEM_SIGLA"].isin(icaos)
                                       | ch["AEROPORTO_DE_DESTINO_SIGLA"].isin(icaos))]
        partes.append(ch)
    df = pd.concat(partes, ignore_index=True)
    for c in ["PASSAGEIROS_PAGOS", "PASSAGEIROS_GRATIS", "DECOLAGENS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["PAX"] = df["PASSAGEIROS_PAGOS"] + df["PASSAGEIROS_GRATIS"]
    df["TRI"] = (df["MES"] - 1) // 3 + 1
    emb = df.groupby(["AEROPORTO_DE_ORIGEM_SIGLA", "ANO", "TRI"]).agg(pax_e=("PAX", "sum"), dec=("DECOLAGENS", "sum"))
    des = df.groupby(["AEROPORTO_DE_DESTINO_SIGLA", "ANO", "TRI"]).agg(pax_d=("PAX", "sum"))
    emb.index.names = des.index.names = ["icao", "ano", "trimestre"]
    m = emb.join(des, how="outer").fillna(0)
    m["pax_trim_mi"] = ((m["pax_e"] + m["pax_d"]) / 1e6).round(3)
    m["decolagens_trim_k"] = (m["dec"] / 1e3).round(2)
    return m[["pax_trim_mi", "decolagens_trim_k"]].reset_index()


def main():
    reg = pd.read_csv(ANAC / "regime_gestao_2023_curado.csv", dtype=str, comment="#").set_index("icao")
    ctx = pd.read_csv(PROC / "contexto_aeroportos_20.csv")
    fixos = ctx[ctx["trimestre"] == 4].set_index("icao")[
        ["pistas", "populacao_2022_mi", "pib_per_capita_k", "idh_2010", "renda_per_capita_2010",
         "populacao_2022_polo_mi", "pib_per_capita_polo_k", "idh_2010_polo"]]
    linhas = []
    for icao, r in reg.iterrows():
        assuncao = pd.to_datetime(r["data_assuncao"], dayfirst=True) if isinstance(r["data_assuncao"], str) else None
        for ano, t, per, ini, fim in periodos():
            dias = (fim - ini).days + 1
            if r["regime_gestao"] == "infraero":
                frac = 0.0
            elif assuncao is None or assuncao <= ini:
                frac = 1.0
            else:
                frac = min(max((fim - assuncao).days + 1, 0), dias) / dias
            regime = "concessionada" if frac == 1 else ("infraero" if frac == 0 else "misto")
            if frac == 0:
                operador = "Infraero"
            else:
                operador = {"SBSP": "Aena Brasil (BOAB)", "SBBE": "NOA – Norte da Amazônia Airports"}.get(
                    icao, r["concessionaria"])
            if icao in TROCA_OPERADOR:
                d, antes, depois = TROCA_OPERADOR[icao]
                operador = depois if pd.Timestamp(fim) >= pd.to_datetime(d, dayfirst=True) else antes
            linhas.append({"icao": icao, "ano": ano, "trimestre": t, "periodo": per,
                           "regime_gestao": regime, "frac_dias_concessao": round(frac, 3),
                           "operador": operador, "data_assuncao": r["data_assuncao"]})
    out = pd.DataFrame(linhas)
    out = out.merge(movimento(), on=["icao", "ano", "trimestre"], how="left")
    out = out.join(fixos, on="icao")
    out.to_csv(PROC / "contexto_aeroportos_20_serie.csv", index=False)
    print(out.groupby(["periodo", "regime_gestao"]).size().unstack(fill_value=0))
    print(out[out.icao.isin(["SBSP", "SBBE", "SBEG", "SBGO", "SBCT"])].pivot(
        index="periodo", columns="icao", values="frac_dias_concessao").head(12))
    print(out[out.icao == "SBPA"][["periodo", "pax_trim_mi"]].T)


if __name__ == "__main__":
    main()
