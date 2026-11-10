"""
Contexto externo dos 20 aeroportos para a camada LLM e o app (TG-2; série 2021–2026 em 08/11/2026).

Dados públicos oficiais baixados da internet. O app e o prompt usam o ÚLTIMO ANO COMPLETO da série da
pesquisa (ANO_REF = 2025; a pesquisa MPor vai de 2021T1 a 2026T2). As colunas de 2023 (versão anterior
do TG) continuam no arquivo, com o mesmo nome de antes, para comparação:
  - ANAC, Voo Regular Ativo (VRA): pontualidade e cancelamento das partidas
      2023: https://siros.anac.gov.br/siros/registros/diversos/vra/2023/VRA_2023_MM.csv  (dados/brutos/anac/vra_2023)
      2025: https://siros.anac.gov.br/siros/registros/diversos/vra/2025/VRA_2025_MM.csv  (dados/brutos/vra_2025)
  - ANAC, Dados Estatísticos do Transporte Aéreo: passageiros e decolagens no ano, % de passageiros em voos
    internacionais, destinos diretos, empresas aéreas e variação do movimento desde 2019 (arquivo já
    baixado pelo script 00; cobre até 2026/08)
  - Já coletados pelo script 00 (ANAC/IBGE/Atlas): pistas, rodada e ano da concessão, grupo controlador,
    população, PIB per capita e IDHM do polo.
  - Regime e operador ATUAIS (2026T2): dados/processados/contexto_aeroportos_20_serie.csv
    (codigos/10_serie_historica/02_contexto_periodos.py), mais a troca de concessionária de Natal (SBSG,
    Inframerica -> Zurich em 19/02/2024), que a tabela curada de 2023 não registra.

Saída: dados/processados/contexto_externo_20.csv (uma linha por aeroporto; colunas `*_2023` e `*_2025`;
`pos_pontualidade`/`pos_cancelamento` = posição entre os 20 no ANO_REF, 1 = melhor).

Uso: python codigos/02_preprocessamento/04_contexto_externo.py [--ano 2025]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANAC = ROOT / "dados" / "brutos" / "anac"
ESTAT = ANAC / "Dados_Estatisticos.csv"
CARACT = HERE / "aeroportos_20_caracteristicas.csv"
MUNIC = HERE / "municipios_20_ibge.csv"
SERIE = ROOT / "dados" / "processados" / "contexto_aeroportos_20_serie.csv"
SAIDA = ROOT / "dados" / "processados" / "contexto_externo_20.csv"

ANO_REF = 2025            # último ano completo da série MPor 2021T1–2026T2
ANO_LEGADO = 2023         # versão anterior do TG (colunas mantidas)
ANO_BASE_VAR = 2019       # pré-pandemia, para a recuperação do movimento
PERIODO_ATUAL = "2026T2"  # último período da série: regime e operador "atuais"
VRA_DIR = {2023: ANAC / "vra_2023", 2025: ROOT / "dados" / "brutos" / "vra_2025"}

URL_VRA = "https://siros.anac.gov.br/siros/registros/diversos/vra/{ano}/VRA_{ano}_MM.csv"
URL_ESTAT = ("https://sistemas.anac.gov.br/dadosabertos/Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/"
             "Dados%20Estat%C3%ADsticos%20do%20Transporte%20A%C3%A9reo/Dados_Estatisticos.csv")

# Troca de concessionária sem troca de regime, posterior à tabela curada de 2023
# (mesma data usada em codigos/10_serie_historica/02_contexto_periodos.py).
ATUALIZACAO_CONCESSAO = {
    "SBSG": {"data_assuncao": "19/02/2024", "grupo_controlador": "Zurich Airport",
             "obs_concessao": "concessão relicitada em 2023 (leilão vencido pela Zurich); até 18/02/2024 "
                              "a operação era da Inframerica"},
}


def pontualidade(aeros, ano):
    """% de partidas com atraso > 30 min (entre as realizadas) e % de partidas canceladas (entre as
    previstas), por aeroporto de origem, VRA do ano."""
    cols = ["Sigla ICAO Aeroporto Origem", "Partida Prevista", "Partida Real", "Situação Voo"]
    arqs = sorted(VRA_DIR[ano].glob(f"VRA_{ano}_*.csv"))
    assert len(arqs) == 12, f"VRA {ano}: {len(arqs)} arquivos mensais em {VRA_DIR[ano]}"
    v = pd.concat(pd.read_csv(p, sep=";", usecols=cols, dtype=str) for p in arqs)
    n_registros = len(v)
    v = v[v["Sigla ICAO Aeroporto Origem"].isin(aeros)]
    sit = v["Situação Voo"].str.upper()
    v = v[sit.isin(["REALIZADO", "CANCELADO"])].assign(cancelado=sit == "CANCELADO")
    real = v[~v["cancelado"]].copy()
    fmt = "%d/%m/%Y %H:%M"
    atraso = (pd.to_datetime(real["Partida Real"], format=fmt, errors="coerce")
              - pd.to_datetime(real["Partida Prevista"], format=fmt, errors="coerce"))
    real["atraso30"] = atraso.dt.total_seconds() / 60 > 30
    real = real[atraso.notna()]
    g = v.groupby("Sigla ICAO Aeroporto Origem")
    out = pd.DataFrame({
        f"partidas_previstas_{ano}": g.size(),
        f"pct_canceladas_{ano}": 100 * g["cancelado"].mean(),
        f"pct_atraso30_{ano}": 100 * real.groupby("Sigla ICAO Aeroporto Origem")["atraso30"].mean(),
    })
    out.attrs["n_registros"] = n_registros
    return out


def movimento(aeros, anos):
    """Por ano: passageiros (embarcados + desembarcados) e decolagens, % de passageiros em voos
    internacionais, destinos diretos regulares, empresas aéreas e variação do movimento desde 2019
    (ANAC Dados Estatísticos)."""
    cols = ["ANO", "EMPRESA_SIGLA", "AEROPORTO_DE_ORIGEM_SIGLA", "AEROPORTO_DE_DESTINO_SIGLA",
            "NATUREZA", "GRUPO_DE_VOO", "PASSAGEIROS_PAGOS", "PASSAGEIROS_GRATIS", "DECOLAGENS"]
    manter = [str(a) for a in (ANO_BASE_VAR, *anos)]
    partes = []
    for ch in pd.read_csv(ESTAT, sep=";", skiprows=1, usecols=cols, dtype=str, chunksize=500_000):
        partes.append(ch[ch["ANO"].isin(manter)])
    e = pd.concat(partes)
    for c in ("PASSAGEIROS_PAGOS", "PASSAGEIROS_GRATIS", "DECOLAGENS"):
        e[c] = pd.to_numeric(e[c], errors="coerce").fillna(0)
    e["pax"] = e["PASSAGEIROS_PAGOS"] + e["PASSAGEIROS_GRATIS"]
    # passageiros do aeroporto = embarcados (origem) + desembarcados (destino)
    lado = pd.concat([
        e.rename(columns={"AEROPORTO_DE_ORIGEM_SIGLA": "icao"})[["ANO", "icao", "NATUREZA", "pax"]],
        e.rename(columns={"AEROPORTO_DE_DESTINO_SIGLA": "icao"})[["ANO", "icao", "NATUREZA", "pax"]]])
    lado = lado[lado["icao"].isin(aeros)]
    pax = lado.groupby(["icao", "ANO"])["pax"].sum().unstack()
    dec = e[e["AEROPORTO_DE_ORIGEM_SIGLA"].isin(aeros)].groupby(
        ["AEROPORTO_DE_ORIGEM_SIGLA", "ANO"])["DECOLAGENS"].sum().unstack()
    out = {}
    for ano in map(str, anos):
        intl = lado[(lado["ANO"] == ano) & (lado["NATUREZA"] == "INTERNACIONAL")].groupby("icao")["pax"].sum()
        reg = e[(e["ANO"] == ano) & (e["GRUPO_DE_VOO"] == "REGULAR") & (e["DECOLAGENS"] > 0)
                & e["AEROPORTO_DE_ORIGEM_SIGLA"].isin(aeros)]
        g = reg.groupby("AEROPORTO_DE_ORIGEM_SIGLA")
        out.update({
            f"pax_{ano}_mi": pax[ano] / 1e6,
            f"decolagens_{ano}": dec[ano],
            f"pct_pax_internacional_{ano}": 100 * intl.reindex(pax.index).fillna(0) / pax[ano],
            f"var_pax_{ANO_BASE_VAR}_{ano}": 100 * (pax[ano] / pax[str(ANO_BASE_VAR)] - 1),
            f"destinos_diretos_{ano}": g["AEROPORTO_DE_DESTINO_SIGLA"].nunique(),
            f"empresas_aereas_{ano}": g["EMPRESA_SIGLA"].nunique(),
        })
    return pd.DataFrame(out)


def regime_atual():
    """Regime e operador no último período da série (2026T2)."""
    s = pd.read_csv(SERIE)
    s = s[s["periodo"] == PERIODO_ATUAL].set_index("icao")
    return s[["regime_gestao", "operador"]].rename(
        columns={"regime_gestao": "regime_atual", "operador": "operador_atual"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, default=ANO_REF, help="ano de referência do app/LLM (padrão 2025)")
    ano_ref = ap.parse_args().ano
    anos = sorted({ANO_LEGADO, ano_ref})
    car = pd.read_csv(CARACT).set_index("icao")
    aeros = car.index.tolist()
    mun = pd.read_csv(MUNIC).set_index("icao")
    pont = {a: pontualidade(aeros, a) for a in anos}
    df = pd.concat(pont.values(), axis=1).join(movimento(aeros, anos), how="outer")
    df = df.join(car[["pistas", "rodada", "ano_concessao", "data_assuncao", "grupo_controlador"]])
    df["obs_concessao"] = pd.NA
    for icao, upd in ATUALIZACAO_CONCESSAO.items():
        for k, v in upd.items():
            df.loc[icao, k] = v
    df = df.join(regime_atual())
    df["periodo_regime_atual"] = PERIODO_ATUAL
    df["municipio_polo"] = mun["municipio_polo"]
    df["populacao_polo_2022"] = mun["populacao_2022_polo"]
    col_pib = [c for c in mun.columns if c.startswith("pib_per_capita_") and c.endswith("_brl_polo")][0]
    df["pib_per_capita_polo_2023"] = mun[col_pib]   # ano mais recente do PIB dos Municípios (IBGE)
    df["idhm_polo_2010"] = mun["idh_2010_polo"]
    # posição entre os 20 (1 = melhor) nas métricas de voo; sem sufixo = ano de referência
    for a in anos:
        df[f"pos_pontualidade_{a}"] = df[f"pct_atraso30_{a}"].rank(method="min").astype(int)
        df[f"pos_cancelamento_{a}"] = df[f"pct_canceladas_{a}"].rank(method="min").astype(int)
    df["pos_pontualidade"] = df[f"pos_pontualidade_{ano_ref}"]
    df["pos_cancelamento"] = df[f"pos_cancelamento_{ano_ref}"]
    df["ano_ref"] = ano_ref
    df["n_registros_vra_ano_ref"] = pont[ano_ref].attrs["n_registros"]
    df["fonte_voos"] = URL_VRA.format(ano=ano_ref)
    df["fonte_movimento"] = URL_ESTAT
    df.index.name = "icao"
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(SAIDA)
    metr = [f"pct_atraso30_{ano_ref}", f"pct_canceladas_{ano_ref}", f"pax_{ano_ref}_mi",
            f"decolagens_{ano_ref}", f"pct_pax_internacional_{ano_ref}", f"var_pax_{ANO_BASE_VAR}_{ano_ref}",
            f"destinos_diretos_{ano_ref}", f"empresas_aereas_{ano_ref}"]
    print(f"VRA {ano_ref}: {pont[ano_ref].attrs['n_registros']:,} registros")
    print(df[metr + ["pos_pontualidade", "pos_cancelamento", "regime_atual"]].round(1).to_string())
    print(f"\nMédia dos 20 ({ano_ref}):\n", df[metr].mean().round(2).to_string())
    leg = [c.replace(str(ano_ref), str(ANO_LEGADO)) for c in metr]
    print(f"\nMédia dos 20 ({ANO_LEGADO}):\n", df[leg].mean().round(2).to_string())


if __name__ == "__main__":
    main()
