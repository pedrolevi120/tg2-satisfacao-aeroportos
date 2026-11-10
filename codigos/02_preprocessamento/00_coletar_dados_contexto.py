"""
TG-2 — Coleta programática dos dados de contexto (ANAC, IBGE, Atlas/PNUD)
para os 20 aeroportos da Pesquisa MPor 2023.

Fontes (todas oficiais, baixadas por este script e guardadas em dados/brutos/):
  ANAC  — Dados Estatísticos do Transporte Aéreo (pax pagos+grátis por etapa,
          decolagens), Movimentações Aeroportuárias 2023 (dado do operador:
          pax local+conexão, pousos/decolagens) e cadastro de aeródromos
          públicos (coordenadas, pistas).
  IBGE  — API de agregados (Censo 2022: população e área, tabela 4714) e
          API de pesquisas (PIB dos Municípios: PIB e PIB per capita).
  PNUD/IPEA/FJP — Atlas do Desenvolvimento Humano (IDHM 2010 e renda per
          capita 2010), via API OData do Ipeadata (séries ADH_IDHM, ADH_RDPC).
  Regime de gestão — tabela curada a partir das páginas oficiais da ANAC
          (dados/brutos/anac/regime_gestao_2023_curado.csv, com URL por linha).

Saídas:
  codigos/02_preprocessamento/aeroportos_20_caracteristicas.csv
  codigos/02_preprocessamento/municipios_20_ibge.csv
  codigos/02_preprocessamento/aeroportos_mvp_caracteristicas.csv  (subconjunto)
  codigos/02_preprocessamento/municipios_mvp_ibge.csv             (subconjunto)
  codigos/02_preprocessamento/checagem_contexto20.csv  (ANAC estatístico vs operador, 2023)
  codigos/02_preprocessamento/resumo_contexto20.csv    (participação dos 20 no país; uma linha por ano)
  dados/processados/contexto_aeroportos_20.csv         (aeroporto x trimestre de 2023; a série
                                                        2021T1–2026T2 é feita por 10_serie_historica/02)
  texto/_paralelo/tex/dados_contexto20.tex             (só com --tex; tabela de 2023)

Anos (série 2021T1–2026T2, 08/11/2026): as colunas de 2023 (ANO_LEGADO) continuam com os mesmos nomes,
porque o contexto trimestral de 2023 e a tabela do texto dependem delas; o movimento do ano de referência
do app/LLM (ANO_REF, padrão 2025 = último ano completo da série) entra como colunas extras
(pax_2025, pax_2025_mi, decolagens_2025, voos_2025_k, movimentos_2025). Regime e operador continuam os
da tabela curada de 2023; regime por período e o "atual" (2026T2) estão em
dados/processados/contexto_aeroportos_20_serie.csv.

Uso:  python codigos/02_preprocessamento/00_coletar_dados_contexto.py [--ano 2025] [--force] [--tex]
      (--force rebaixa os brutos; sem ele, reaproveita o que já está em disco)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PREPROC = ROOT / "codigos" / "02_preprocessamento"
BRUTOS_ANAC = ROOT / "dados" / "brutos" / "anac"
BRUTOS_IBGE = ROOT / "dados" / "brutos" / "ibge"
FORCE = "--force" in sys.argv
ESCREVER_TEX = "--tex" in sys.argv

ANO = ANO_LEGADO = 2023     # contexto trimestral, checagem com o operador e tabela do texto
ANO_REF = int(sys.argv[sys.argv.index("--ano") + 1]) if "--ano" in sys.argv else 2025
MVP = ["SBGR", "SBSP", "SBKP"]

# --------------------------------------------------------------------------
# URLs oficiais
# --------------------------------------------------------------------------
ANAC_DA = "https://sistemas.anac.gov.br/dadosabertos/"
URL_ANAC_ESTAT = ANAC_DA + quote(
    "Voos e operações aéreas/Dados Estatísticos do Transporte Aéreo/Dados_Estatisticos.csv")
URL_ANAC_MOV = ANAC_DA + quote(
    "Operador Aeroportuário/Dados de Movimentação Aeroportuárias/2023/") + \
    "Movimentacoes_Aeroportuarias_2023{mes:02d}.csv"
URL_ANAC_AERODROMOS = ANAC_DA + quote(
    "Aerodromos/Aeródromos Públicos/Lista de aeródromos públicos/AerodromosPublicos.csv")
URL_ANAC_CARACT = ANAC_DA + quote(
    "Aerodromos/Aeródromos Públicos/Características Gerais/"
    "pda_aerodromos_publicos_caracteristicas_gerais.csv")

IBGE = "https://servicodados.ibge.gov.br/api"
URL_IBGE_MUN = IBGE + "/v1/localidades/municipios/{cod}"
URL_IBGE_CENSO = IBGE + "/v3/agregados/4714/periodos/2022/variaveis/93|6318?localidades=N6[{cods}]"
URL_IBGE_PIB = IBGE + "/v1/pesquisas/38/indicadores/46997/resultados/{cods}"      # PIB (mil R$)
URL_IBGE_PIBPC = IBGE + "/v1/pesquisas/38/indicadores/47001/resultados/{cods}"    # PIB per capita (R$)
URL_IPEA = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{serie}')"

# Página humana para citação (coluna de fonte)
FONTE_IBGE_CIDADES = "https://cidades.ibge.gov.br/brasil/{uf}/{slug}/panorama"

# --------------------------------------------------------------------------
# Os 20 aeroportos da Pesquisa MPor 2023 e os municípios associados.
#   mun_sede    = município onde o sítio aeroportuário está (IBGE), mesmo
#                 critério do MVP (Guarulhos para SBGR).
#   mun_polo    = núcleo urbano servido (capital/sede da RM), usado como
#                 contexto socioeconômico alternativo.
# Códigos IBGE de 7 dígitos; conferidos contra a API de localidades abaixo.
# --------------------------------------------------------------------------
AEROPORTOS = {
    #        nome curto,          sede (cod, nome),                       polo (cod, nome)
    "SBGR": ("Guarulhos",         3518800, "Guarulhos",                3550308, "São Paulo"),
    "SBGL": ("Galeão",            3304557, "Rio de Janeiro",           3304557, "Rio de Janeiro"),
    "SBKP": ("Viracopos",         3509502, "Campinas",                 3509502, "Campinas"),
    "SBCF": ("Confins",           3117876, "Confins",                  3106200, "Belo Horizonte"),
    "SBPA": ("Porto Alegre",      4314902, "Porto Alegre",             4314902, "Porto Alegre"),
    "SBCT": ("Curitiba",          4125506, "São José dos Pinhais",     4106902, "Curitiba"),
    "SBEG": ("Manaus",            1302603, "Manaus",                   1302603, "Manaus"),
    "SBFZ": ("Fortaleza",         2304400, "Fortaleza",                2304400, "Fortaleza"),
    "SBBR": ("Brasília",          5300108, "Brasília",                 5300108, "Brasília"),
    "SBSP": ("Congonhas",         3550308, "São Paulo",                3550308, "São Paulo"),
    "SBGO": ("Goiânia",           5208707, "Goiânia",                  5208707, "Goiânia"),
    "SBSG": ("Natal",             2412005, "São Gonçalo do Amarante",  2408102, "Natal"),
    "SBBE": ("Belém",             1501402, "Belém",                    1501402, "Belém"),
    "SBSV": ("Salvador",          2927408, "Salvador",                 2927408, "Salvador"),
    "SBMO": ("Maceió",            2707701, "Rio Largo",                2704302, "Maceió"),
    "SBRJ": ("Santos Dumont",     3304557, "Rio de Janeiro",           3304557, "Rio de Janeiro"),
    "SBVT": ("Vitória",           3205309, "Vitória",                  3205309, "Vitória"),
    "SBFL": ("Florianópolis",     4205407, "Florianópolis",            4205407, "Florianópolis"),
    "SBCY": ("Cuiabá",            5108402, "Várzea Grande",            5103403, "Cuiabá"),
    "SBRF": ("Recife",            2611606, "Recife",                   2611606, "Recife"),
}
ICAOS = list(AEROPORTOS)
REGIAO_UF = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}


# --------------------------------------------------------------------------
# Download com cache/retomada
# --------------------------------------------------------------------------
def baixar(url: str, destino: Path, tentativas: int = 8) -> Path:
    """Baixa `url` para `destino` (retoma downloads parciais com Range)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".part")
    if destino.exists() and not FORCE:
        return destino
    for i in range(tentativas):
        ini = parcial.stat().st_size if parcial.exists() else 0
        headers = {"Range": f"bytes={ini}-"} if ini else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 416:          # já completo
                    break
                r.raise_for_status()
                modo = "ab" if (ini and r.status_code == 206) else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (ini if modo == "ab" else 0)
                with open(parcial, modo) as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            if not total or parcial.stat().st_size >= total:
                break
        except requests.RequestException as e:
            print(f"    tentativa {i + 1} falhou ({e}); retomando...")
            time.sleep(3)
    parcial.rename(destino)
    print(f"    baixado: {destino.relative_to(ROOT)} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def baixar_json(url: str, destino: Path, timeout: int = 300):
    if destino.exists() and not FORCE:
        return json.loads(destino.read_text(encoding="utf-8"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            dados = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            print(f"    tentativa {i + 1} falhou ({e})")
            time.sleep(5)
    else:
        raise RuntimeError(f"falha ao baixar {url}")
    destino.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    baixado: {destino.relative_to(ROOT)}")
    return dados


# --------------------------------------------------------------------------
# ANAC
# --------------------------------------------------------------------------
def anac_estatisticos(ano: int = ANO) -> pd.DataFrame:
    """Pax (pagos+grátis) embarcados+desembarcados e decolagens no ano."""
    path = baixar(URL_ANAC_ESTAT, BRUTOS_ANAC / "Dados_Estatisticos.csv")
    cols = ["ANO", "AEROPORTO_DE_ORIGEM_SIGLA", "AEROPORTO_DE_DESTINO_SIGLA", "GRUPO_DE_VOO",
            "AEROPORTO_DE_ORIGEM_PAIS", "AEROPORTO_DE_DESTINO_PAIS",
            "PASSAGEIROS_PAGOS", "PASSAGEIROS_GRATIS", "DECOLAGENS"]
    partes, nac = [], {"pax_emb": 0.0, "pax_desemb": 0.0, "decolagens": 0.0}
    for ch in pd.read_csv(path, sep=";", skiprows=1, usecols=cols, chunksize=500_000,
                          encoding="utf-8-sig", low_memory=False):
        ch = ch[ch["ANO"] == ano]
        pax = pd.to_numeric(ch["PASSAGEIROS_PAGOS"], errors="coerce").fillna(0) + \
            pd.to_numeric(ch["PASSAGEIROS_GRATIS"], errors="coerce").fillna(0)
        o_br, d_br = ch["AEROPORTO_DE_ORIGEM_PAIS"] == "BRASIL", ch["AEROPORTO_DE_DESTINO_PAIS"] == "BRASIL"
        nac["pax_emb"] += pax[o_br].sum()
        nac["pax_desemb"] += pax[d_br].sum()
        nac["decolagens"] += pd.to_numeric(ch["DECOLAGENS"], errors="coerce").fillna(0)[o_br].sum()
        partes.append(ch[ch["AEROPORTO_DE_ORIGEM_SIGLA"].isin(ICAOS)
                         | ch["AEROPORTO_DE_DESTINO_SIGLA"].isin(ICAOS)])
    df = pd.concat(partes, ignore_index=True)
    for c in ["PASSAGEIROS_PAGOS", "PASSAGEIROS_GRATIS", "DECOLAGENS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["PAX"] = df["PASSAGEIROS_PAGOS"] + df["PASSAGEIROS_GRATIS"]
    emb = df.groupby("AEROPORTO_DE_ORIGEM_SIGLA").agg(pax_emb=("PAX", "sum"), decolagens=("DECOLAGENS", "sum"))
    des = df.groupby("AEROPORTO_DE_DESTINO_SIGLA").agg(pax_desemb=("PAX", "sum"), pousos=("DECOLAGENS", "sum"))
    out = emb.join(des, how="outer").reindex(ICAOS).fillna(0)
    out[f"pax_{ano}"] = out["pax_emb"] + out["pax_desemb"]
    print(f"  Grupos de voo incluídos ({ano}):", sorted(df["GRUPO_DE_VOO"].dropna().unique()))
    out.attrs["nacional"] = {"pax": nac["pax_emb"] + nac["pax_desemb"], "decolagens": nac["decolagens"]}
    return out


def anac_movimentacoes() -> pd.DataFrame:
    """Dado do operador aeroportuário: pax (local+conexões) e movimentos de aeronaves."""
    partes = []
    for mes in range(1, 13):
        path = baixar(URL_ANAC_MOV.format(mes=mes),
                      BRUTOS_ANAC / "movimentacoes_2023" / f"Movimentacoes_Aeroportuarias_2023{mes:02d}.csv")
        ch = pd.read_csv(path, sep=";", skiprows=1, encoding="utf-8-sig", dtype=str,
                         usecols=["MES", "NR_AEROPORTO_REFERENCIA", "NR_MOVIMENTO_TIPO",
                                  "QT_PAX_LOCAL", "QT_PAX_CONEXAO_DOMESTICO", "QT_PAX_CONEXAO_INTERNACIONAL"])
        partes.append(ch[ch["NR_AEROPORTO_REFERENCIA"].isin(ICAOS)])
    df = pd.concat(partes, ignore_index=True)
    for c in ["QT_PAX_LOCAL", "QT_PAX_CONEXAO_DOMESTICO", "QT_PAX_CONEXAO_INTERNACIONAL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["PAX"] = df["QT_PAX_LOCAL"] + df["QT_PAX_CONEXAO_DOMESTICO"] + df["QT_PAX_CONEXAO_INTERNACIONAL"]
    g = df.groupby("NR_AEROPORTO_REFERENCIA")
    out = pd.DataFrame({
        "pax_operador_2023": g["PAX"].sum(),
        "movimentos_operador_2023": g.size(),
        "decolagens_operador_2023": df[df["NR_MOVIMENTO_TIPO"] == "D"].groupby("NR_AEROPORTO_REFERENCIA").size(),
        "meses_operador_2023": g["MES"].nunique(),   # cobertura: Infraero não reportou em parte de 2023
    }).reindex(ICAOS)
    print("  Tipos de movimento:", df["NR_MOVIMENTO_TIPO"].value_counts().to_dict())
    return out


def anac_cadastro() -> pd.DataFrame:
    """Coordenadas decimais (lista de aeródromos) e nº de pistas (características gerais)."""
    p1 = baixar(URL_ANAC_AERODROMOS, BRUTOS_ANAC / "AerodromosPublicos.csv")
    p2 = baixar(URL_ANAC_CARACT, BRUTOS_ANAC / "pda_aerodromos_publicos_caracteristicas_gerais.csv")
    lista = pd.read_csv(p1, sep=";", skiprows=1, encoding="latin-1", dtype=str)
    lista = lista.drop_duplicates("Código OACI").set_index("Código OACI").reindex(ICAOS)
    cg = pd.read_csv(p2, dtype=str).drop_duplicates("Código OACI").set_index("Código OACI").reindex(ICAOS)
    pistas = cg[["Designação (Pista 1)", "Designação (Pista 2)"]].notna().sum(axis=1)
    return pd.DataFrame({
        "nome_anac": cg["Nome"],
        "municipio_anac": cg["Município"],
        "uf": cg["UF"],
        "latitude": pd.to_numeric(lista["LATGEOPOINT"].str.replace(",", "."), errors="coerce").round(4),
        "longitude": pd.to_numeric(lista["LONGEOPOINT"].str.replace(",", "."), errors="coerce").round(4),
        "pistas": pistas,
        "designacao_pistas": cg["Designação (Pista 1)"].fillna("") +
                             cg["Designação (Pista 2)"].map(lambda s: f"; {s}" if isinstance(s, str) else ""),
    })


# --------------------------------------------------------------------------
# IBGE e Atlas (IPEA/PNUD)
# --------------------------------------------------------------------------
def _ultimo_ano(res: dict) -> tuple[str, float]:
    validos = {a: v for a, v in res.items() if v not in (None, "-", "...", "")}
    ano = max(validos)
    return ano, float(validos[ano])


def ibge_municipios(cods: list[int]) -> pd.DataFrame:
    cods = sorted(set(cods))
    lista = "|".join(map(str, cods))
    # Localidades (nome oficial e UF) — confere os códigos da tabela AEROPORTOS
    meta = {}
    for c in cods:
        d = baixar_json(URL_IBGE_MUN.format(cod=c), BRUTOS_IBGE / "localidades" / f"{c}.json")
        meta[c] = {"municipio": d["nome"], "uf": d["microrregiao"]["mesorregiao"]["UF"]["sigla"]}
    censo = baixar_json(URL_IBGE_CENSO.format(cods=",".join(map(str, cods))),
                        BRUTOS_IBGE / "censo2022_tabela4714_pop_area.json")
    pib = baixar_json(URL_IBGE_PIB.format(cods=lista), BRUTOS_IBGE / "pib_municipios_46997.json")
    pibpc = baixar_json(URL_IBGE_PIBPC.format(cods=lista), BRUTOS_IBGE / "pib_per_capita_47001.json")

    linhas = {c: dict(meta[c]) for c in cods}
    for var in censo:
        chave = {"93": "populacao_2022", "6318": "area_km2"}[str(var["id"])]
        for s in var["resultados"][0]["series"]:
            linhas[int(s["localidade"]["id"])][chave] = float(s["serie"]["2022"])
    cod6 = {c // 10: c for c in cods}   # API de pesquisas usa código de 6 dígitos
    anos = set()
    for bloco, chave, fator in [(pib, "pib_bi_brl", 1e-6), (pibpc, "pib_per_capita_brl", 1.0)]:
        for r in bloco[0]["res"]:
            ano, v = _ultimo_ano(r["res"])
            anos.add(ano)
            c = cod6[int(r["localidade"])]
            linhas[c][chave] = v * fator
            linhas[c]["ano_pib"] = ano
            linhas[c][chave + "_2021"] = float(r["res"]["2021"]) * fator   # esquema legado do MVP
    assert len(anos) == 1, f"anos de PIB divergentes: {anos}"

    # Atlas do Desenvolvimento Humano (Censo 2010) via Ipeadata
    for serie, chave in [("ADH_IDHM", "idh_2010"), ("ADH_RDPC", "renda_per_capita_2010")]:
        d = baixar_json(URL_IPEA.format(serie=serie), BRUTOS_IBGE / f"ipeadata_{serie}.json", timeout=600)
        v = pd.DataFrame(d["value"])
        v = v[(v["NIVNOME"] == "Municípios") & v["VALDATA"].str.startswith("2010")]
        v = v.set_index(v["TERCODIGO"].astype(int))["VALVALOR"]
        for c in cods:
            linhas[c][chave] = float(v.get(c, float("nan")))
    df = pd.DataFrame.from_dict(linhas, orient="index")
    df.index.name = "cod_ibge"
    return df


# --------------------------------------------------------------------------
# Regime de gestão (tabela curada com URL oficial por aeroporto)
# --------------------------------------------------------------------------
def regime() -> pd.DataFrame:
    df = pd.read_csv(BRUTOS_ANAC / "regime_gestao_2023_curado.csv", dtype=str, comment="#")
    return df.set_index("icao").reindex(ICAOS)


# --------------------------------------------------------------------------
# Contexto por (aeroporto, trimestre) para a modelagem (pacote T5)
# --------------------------------------------------------------------------
TRIMESTRES = {1: ("2023-01-01", "2023-03-31"), 2: ("2023-04-01", "2023-06-30"),
              3: ("2023-07-01", "2023-09-30"), 4: ("2023-10-01", "2023-12-31")}


def contexto_trimestral(aero: pd.DataFrame, muni: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por aeroporto x trimestre. Features numéricas anuais repetidas;
    regime varia no trimestre da transição (fração de dias sob concessão)."""
    m = muni.set_index("icao")
    pibpc = [c for c in muni.columns if c.startswith("pib_per_capita_") and not c.endswith("_polo")][0]
    pibpc_polo = [c for c in muni.columns if c.startswith("pib_per_capita_") and c.endswith("_polo")][0]
    linhas = []
    for _, r in aero.iterrows():
        mud = pd.to_datetime(r["data_mudanca_2023"], dayfirst=True) if isinstance(r["data_mudanca_2023"], str) else None
        for t, (ini, fim) in TRIMESTRES.items():
            ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
            dias = (fim - ini).days + 1
            if r["regime_gestao"] == "infraero":
                frac = 0.0
            elif mud is None:
                frac = 1.0
            else:
                frac = min(max((fim - mud).days + 1, 0), dias) / dias
            regime_t = "concessionada" if frac == 1 else ("infraero" if frac == 0 else "misto")
            operador = r["concessionaria"] if frac > 0 or r["regime_gestao"] == "infraero" else "Infraero"
            if r["icao"] in ("SBSP", "SBBE") and frac > 0:
                operador = {"SBSP": "Aena Brasil (BOAB)", "SBBE": "NOA – Norte da Amazônia Airports"}[r["icao"]]
            linhas.append({
                "icao": r["icao"], "trimestre": t,
                "regime_gestao": regime_t, "frac_dias_concessao": round(frac, 3),
                "operador": operador,
                "pax_2023_mi": r["pax_2023_mi"], "decolagens_2023_k": r["voos_2023_k"],
                "pistas": r["pistas"],
                "populacao_2022_mi": round(m.loc[r["icao"], "populacao_2022"] / 1e6, 3),
                "pib_per_capita_k": round(m.loc[r["icao"], pibpc] / 1e3, 2),
                "idh_2010": m.loc[r["icao"], "idh_2010"],
                "renda_per_capita_2010": m.loc[r["icao"], "renda_media_2010"],
                "populacao_2022_polo_mi": round(m.loc[r["icao"], "populacao_2022_polo"] / 1e6, 3),
                "pib_per_capita_polo_k": round(m.loc[r["icao"], pibpc_polo] / 1e3, 2),
                "idh_2010_polo": m.loc[r["icao"], "idh_2010_polo"],
            })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Tabela LaTeX (texto/_paralelo/tex/dados_contexto20.tex)
# --------------------------------------------------------------------------
OPERADOR_CURTO = {
    "SBGR": "GRU Airport", "SBGL": "RIOgaleão", "SBKP": "ABV (Viracopos)", "SBCF": "BH Airport",
    "SBPA": "Fraport Brasil", "SBCT": "CCR Aeroportos", "SBEG": "Vinci Airports", "SBFZ": "Fraport Brasil",
    "SBBR": "Inframerica", "SBSP": "Infraero $\\to$ Aena", "SBGO": "CCR Aeroportos", "SBSG": "Inframerica",
    "SBBE": "Infraero $\\to$ NOA", "SBSV": "Vinci Airports", "SBMO": "Aena Brasil", "SBRJ": "Infraero",
    "SBVT": "Zurich Airport", "SBFL": "Zurich (Floripa Airport)", "SBCY": "Centro-Oeste Airports",
    "SBRF": "Aena Brasil",
}


def _num(x: float, casas: int) -> str:
    s = f"{x:,.{casas}f}".replace(",", "X").replace(".", "{,}").replace("X", ".")
    return s


def tabela_latex(aero: pd.DataFrame) -> str:
    t = aero.sort_values("pax_2023", ascending=False)
    linhas = []
    for _, r in t.iterrows():
        if r["regime_inicio_2023"] != r["regime_gestao"]:
            d = r["data_mudanca_2023"][:5]
            regime = f"Transição ({d}/2023)"
        else:
            regime = "Infraero" if r["regime_gestao"] == "infraero" else "Concessão"
        linhas.append(f"{r['icao']} & {r['cidade']}/{r['uf']} & {r['regiao']} & "
                      f"{_num(r['pax_2023'] / 1e6, 2)} & {_num(r['decolagens_2023'], 0)} & "
                      f"{regime} & {OPERADOR_CURTO[r['icao']]} \\\\")
    tot_pax, tot_dec = aero["pax_2023"].sum() / 1e6, aero["decolagens_2023"].sum()
    corpo = "\n".join(linhas)
    return rf"""% Gerado por codigos/02_preprocessamento/00_coletar_dados_contexto.py — não editar à mão.
\begin{{table}}[htbp]
\centering
\caption{{Aeroportos cobertos pela Pesquisa MPor 2023: localização, movimento e regime de gestão em 2023.}}
\label{{tab:dados-contexto20}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\linewidth}}{{l>{{\raggedright\arraybackslash}}p{{3.1cm}}lrr>{{\raggedright\arraybackslash}}p{{2.2cm}}>{{\raggedright\arraybackslash}}X}}
\toprule
ICAO & Município/UF & Região & Pax (mi) & Decolagens & Regime 2023 & Operador \\
\midrule
{corpo}
\midrule
Total & & & {_num(tot_pax, 2)} & {_num(tot_dec, 0)} & & \\
\bottomrule
\end{{tabularx}}
\par\smallskip
\parbox{{\linewidth}}{{\footnotesize Fonte: elaboração própria a partir de ANAC, Dados Estatísticos do
Transporte Aéreo (passageiros pagos e grátis embarcados e desembarcados; decolagens de voos regulares,
não regulares e improdutivos) e páginas de concessões da ANAC. Município = município-sede do sítio
aeroportuário (IBGE). Transição: Congonhas passou da Infraero à Aena em 17/10/2023 e Belém da Infraero
à NOA em 04/09/2023.}}
\end{{table}}
"""


# --------------------------------------------------------------------------
def slug(nome: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode().lower()
    return s.replace(" ", "-").replace("'", "")


def main():
    print(f"=== ANAC: Dados Estatísticos ({ANO}) ===")
    est = anac_estatisticos(ANO)
    est_ref = est
    if ANO_REF != ANO:
        print(f"=== ANAC: Dados Estatísticos ({ANO_REF}, ano de referência do app/LLM) ===")
        est_ref = anac_estatisticos(ANO_REF)
    print("=== ANAC: Movimentações Aeroportuárias (operador) ===")
    mov = anac_movimentacoes()
    print("=== ANAC: cadastro de aeródromos ===")
    cad = anac_cadastro()
    print("=== IBGE + Atlas ===")
    cods = [v[1] for v in AEROPORTOS.values()] + [v[3] for v in AEROPORTOS.values()]
    mun = ibge_municipios(cods)
    reg = regime()

    for icao, (_, c_sede, n_sede, c_polo, n_polo) in AEROPORTOS.items():
        assert mun.loc[c_sede, "municipio"] == n_sede, (icao, mun.loc[c_sede, "municipio"])
        assert mun.loc[c_polo, "municipio"] == n_polo, (icao, mun.loc[c_polo, "municipio"])

    # ---------------- aeroportos ----------------
    aero = pd.DataFrame(index=pd.Index(ICAOS, name="icao"))
    aero["nome"] = reg["nome"]
    aero["cidade"] = [AEROPORTOS[i][2] for i in ICAOS]
    aero["uf"] = cad["uf"]
    aero["regiao"] = aero["uf"].map(REGIAO_UF)
    aero["latitude"] = cad["latitude"]
    aero["longitude"] = cad["longitude"]
    aero["pax_2023"] = est["pax_2023"].astype(int)
    aero["pax_2023_mi"] = (est["pax_2023"] / 1e6).round(2)
    aero["decolagens_2023"] = est["decolagens"].astype(int)
    aero["voos_2023_k"] = (est["decolagens"] / 1e3).round(1)          # "voos" = decolagens
    aero["movimentos_2023"] = (est["decolagens"] + est["pousos"]).astype(int)  # pousos+decolagens
    aero["pax_operador_2023"] = mov["pax_operador_2023"].astype("Int64")
    aero["movimentos_operador_2023"] = mov["movimentos_operador_2023"].astype("Int64")
    aero["meses_operador_2023"] = mov["meses_operador_2023"].astype("Int64")
    aero["terminais"] = pd.NA          # sem fonte oficial programática (ver dados.md)
    aero["pistas"] = cad["pistas"].astype(int)
    aero["posicoes_pista"] = pd.NA     # idem (nome legado do MVP; = posições de pátio)
    for c in ["concessionaria", "grupo_controlador", "regime_gestao", "regime_inicio_2023", "regime_detalhe_2023",
              "rodada", "ano_concessao", "data_assuncao", "prazo_concessao_anos",
              "mudou_operador_2023", "data_mudanca_2023", "obs_transicao_2023"]:
        aero[c] = reg[c]
    # Esquema legado do MVP primeiro (mesmos nomes de colunas), extras depois
    legado = ["nome", "cidade", "uf", "regiao", "latitude", "longitude", "pax_2023_mi", "voos_2023_k",
              "terminais", "pistas", "posicoes_pista", "concessionaria", "regime_gestao", "ano_concessao",
              "prazo_concessao_anos", "obs_transicao_2023"]
    aero = aero[legado + [c for c in aero.columns if c not in legado]]
    if ANO_REF != ANO:   # movimento do ano de referência do app/LLM (colunas extras)
        aero[f"pax_{ANO_REF}"] = est_ref[f"pax_{ANO_REF}"].astype(int)
        aero[f"pax_{ANO_REF}_mi"] = (est_ref[f"pax_{ANO_REF}"] / 1e6).round(2)
        aero[f"decolagens_{ANO_REF}"] = est_ref["decolagens"].astype(int)
        aero[f"voos_{ANO_REF}_k"] = (est_ref["decolagens"] / 1e3).round(1)
        aero[f"movimentos_{ANO_REF}"] = (est_ref["decolagens"] + est_ref["pousos"]).astype(int)
    aero["fonte_movimento"] = URL_ANAC_ESTAT
    aero["fonte_movimento_operador"] = URL_ANAC_MOV.format(mes=1).replace("202301", "2023MM")
    aero["fonte_coordenadas_pistas"] = URL_ANAC_AERODROMOS + " ; " + URL_ANAC_CARACT
    aero["fonte_regime"] = reg["url_fonte"]
    aero = aero.reset_index()

    # ---------------- municípios ----------------
    linhas = []
    for icao, (_, c_sede, _, c_polo, _) in AEROPORTOS.items():
        s, p = mun.loc[c_sede], mun.loc[c_polo]
        linhas.append({
            # --- esquema legado do MVP (mesmos nomes) ---
            "icao": icao, "municipio": s["municipio"], "uf": s["uf"],
            "populacao_2022": int(s["populacao_2022"]), "area_km2": round(s["area_km2"], 1),
            "pib_2021_bi_brl": round(s["pib_bi_brl_2021"], 2),
            "pib_per_capita_2021_brl": round(s["pib_per_capita_brl_2021"], 2),
            "idh_2010": s["idh_2010"],
            "renda_media_2010": s["renda_per_capita_2010"],   # = renda per capita média (Atlas), R$ ago/2010
            "observacao": ("Município-sede do aeroporto" +
                           ("" if c_polo == c_sede else f"; polo servido: {p['municipio']}")),
            # --- extras ---
            "cod_ibge": c_sede,
            f"pib_{s['ano_pib']}_bi_brl": round(s["pib_bi_brl"], 2),
            f"pib_per_capita_{s['ano_pib']}_brl": round(s["pib_per_capita_brl"], 2),
            "cod_ibge_polo": c_polo, "municipio_polo": p["municipio"],
            "populacao_2022_polo": int(p["populacao_2022"]),
            f"pib_per_capita_{p['ano_pib']}_brl_polo": round(p["pib_per_capita_brl"], 2),
            "idh_2010_polo": p["idh_2010"],
            "fonte_populacao_area": URL_IBGE_CENSO.format(cods=c_sede),
            "fonte_pib": URL_IBGE_PIB.format(cods=c_sede),
            "fonte_pib_per_capita": URL_IBGE_PIBPC.format(cods=c_sede),
            "fonte_idh_renda": URL_IPEA.format(serie="ADH_IDHM") + " ; " + URL_IPEA.format(serie="ADH_RDPC"),
            "fonte_consulta_humana": FONTE_IBGE_CIDADES.format(uf=s["uf"].lower(), slug=slug(s["municipio"])),
        })
    muni = pd.DataFrame(linhas)

    aero.to_csv(PREPROC / "aeroportos_20_caracteristicas.csv", index=False)
    muni.to_csv(PREPROC / "municipios_20_ibge.csv", index=False)
    aero[aero["icao"].isin(MVP)].to_csv(PREPROC / "aeroportos_mvp_caracteristicas.csv", index=False)
    muni[muni["icao"].isin(MVP)].to_csv(PREPROC / "municipios_mvp_ibge.csv", index=False)

    contexto_trimestral(aero, muni).to_csv(ROOT / "dados" / "processados" / "contexto_aeroportos_20.csv",
                                           index=False)

    if ESCREVER_TEX:   # tabela de 2023 do texto (não reescrita por padrão)
        tex = ROOT / "texto" / "_paralelo" / "tex" / "dados_contexto20.tex"
        tex.write_text(tabela_latex(aero), encoding="utf-8")

    chk = aero[["icao", "pax_2023", "pax_operador_2023", "decolagens_2023", "movimentos_operador_2023",
                "meses_operador_2023"]].copy()
    chk["dif_pax_pct"] = ((chk["pax_2023"] / chk["pax_operador_2023"] - 1) * 100).round(2)
    chk["dif_mov_pct"] = ((2 * chk["decolagens_2023"] / chk["movimentos_operador_2023"] - 1) * 100).round(2)
    chk.to_csv(PREPROC / "checagem_contexto20.csv", index=False)
    linhas_res = []
    for ano, e in sorted({ANO: est, ANO_REF: est_ref}.items()):
        nac, pax, dec = e.attrs["nacional"], e[f"pax_{ano}"].sum(), e["decolagens"].sum()
        linhas_res.append({
            "ano": ano, "pax_20_aeroportos": int(pax), "pax_brasil": int(nac["pax"]),
            "share_pax_pct": round(100 * pax / nac["pax"], 2),
            "decolagens_20_aeroportos": int(dec), "decolagens_brasil": int(nac["decolagens"]),
            "share_decolagens_pct": round(100 * dec / nac["decolagens"], 2)})
    resumo = pd.DataFrame(linhas_res)
    resumo.to_csv(PREPROC / "resumo_contexto20.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== Aeroportos ===")
    print(aero[["icao", "cidade", "uf", "pax_2023_mi", "decolagens_2023", "voos_2023_k", "pistas",
                "concessionaria", "regime_detalhe_2023"]].to_string(index=False))
    print("\n=== Checagem ANAC estatístico vs operador ===")
    print(chk.to_string(index=False))
    print("\n=== Resumo nacional ===")
    print(resumo.to_string(index=False))
    print("\n=== Municípios ===")
    print(muni.iloc[:, :11].to_string(index=False))


if __name__ == "__main__":
    main()
