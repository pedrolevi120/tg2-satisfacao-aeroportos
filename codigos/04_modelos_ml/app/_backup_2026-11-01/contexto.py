"""
Contexto operacional dos aeroportos do MVP (fonte única para app e scripts).

Valores do MVP original (GRU/CGH/VCP): movimento ANAC 2023 e estatísticas
de satisfação da MPor 2023 (ver texto do TG, cap. de dados).
"""

MEDIA_NACIONAL = 4.38

CONTEXTO_MVP = {
    "SBGR": {"nome": "Aeroporto Internacional de Guarulhos",
             "nome_curto": "Guarulhos (GRU)",
             "cidade": "Guarulhos", "uf": "SP", "regiao": "Sudeste",
             "pax_mi": 41.3, "voos_k": 131, "regime": "Concessão privada",
             "concessionaria": "GRU Airport (Invepar/ACSA)",
             "n": 6054, "media": 4.21, "posicao": 15, "pct_4_5": 88.7},
    "SBSP": {"nome": "Aeroporto de Congonhas",
             "nome_curto": "Congonhas (CGH)",
             "cidade": "São Paulo", "uf": "SP", "regiao": "Sudeste",
             "pax_mi": 22.4, "voos_k": 93, "regime": "Concessão privada (a partir de 17/out/2023)",
             "concessionaria": "Aena Brasil (anteriormente Infraero)",
             "n": 4033, "media": 4.16, "posicao": 18, "pct_4_5": 89.5},
    "SBKP": {"nome": "Aeroporto Internacional de Viracopos",
             "nome_curto": "Viracopos (VCP)",
             "cidade": "Campinas", "uf": "SP", "regiao": "Sudeste",
             "pax_mi": 12.3, "voos_k": 62, "regime": "Concessão privada",
             "concessionaria": "Aeroportos Brasil Viracopos (ABV)",
             "n": 4132, "media": 4.47, "posicao": 7, "pct_4_5": 94.1},
}

# Prefixos de variáveis que não são atributos gerenciáveis pelo operador
# (dummies de aeroporto, perfil do respondente). Ficam fora do top-5 enviado ao LLM.
PREFIXOS_NAO_ACIONAVEIS = ("AEROPORTO_", "GÊNERO_", "GENERO_", "MOTIVO DA VIAGEM_",
                           "FAIXA", "ESCOLARIDADE", "RENDA", "IDADE", "PROCESSO_",
                           "TIPO DE VOO", "NACIONALIDADE", "MES", "MÊS", "ANO")


def top_acionaveis(ranking_df, icao, k=5, col_shap="shap_abs_medio"):
    """Top-k atributos acionáveis de um aeroporto a partir do ranking SHAP."""
    sub = ranking_df[ranking_df["aeroporto"] == icao].sort_values("rank")
    sub = sub[~sub["feature"].str.upper().str.startswith(PREFIXOS_NAO_ACIONAVEIS)]
    return [{"feature": r["feature"], "shap_abs": float(r[col_shap])}
            for _, r in sub.head(k).iterrows()]


# ---------------------------------------------------------------------------
# 20 aeroportos (TG-2)
# ---------------------------------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PANORAMA = ROOT / "dados" / "processados" / "panorama_aeroportos_2023.parquet"
RANKING_20 = ROOT / "codigos" / "05_interpretacao" / "ranking_prioridades_20.csv"
RANKING_MVP = ROOT / "codigos" / "05_interpretacao" / "ranking_prioridades_por_aeroporto.csv"


def estatisticas_satisfacao():
    """n, média, % notas 4–5 e posição de cada aeroporto (MPor 2023, 20 aeroportos)."""
    import pandas as pd
    p = pd.read_parquet(PANORAMA)
    g = (p.dropna(subset=["SATISFACAO_GERAL_NUM"])
          .groupby(["AEROPORTO", "nome_aero", "regiao", "uf"])["SATISFACAO_GERAL_NUM"]
          .agg(n="count", media="mean", pct_4_5=lambda s: 100 * (s >= 4).mean())
          .reset_index().sort_values("media", ascending=False))
    g["posicao"] = range(1, len(g) + 1)
    media_nac = p["SATISFACAO_GERAL_NUM"].mean()
    return g.set_index("AEROPORTO"), media_nac


def carregar_ranking():
    """Ranking dos 20 aeroportos (T5) se existir; senão o ranking do MVP (3 aeroportos)."""
    import pandas as pd
    if RANKING_20.exists():
        r = pd.read_csv(RANKING_20).rename(columns={"atributo": "feature",
                                                    "mean_abs_shap": "shap_abs_medio"})
        return r, "20"
    return pd.read_csv(RANKING_MVP), "mvp"


CARACT_20 = ROOT / "codigos" / "02_preprocessamento" / "aeroportos_20_caracteristicas.csv"


def _regime_curto(detalhe: str) -> str:
    if detalhe.startswith("Concessão privada"):
        return detalhe.replace(" (ano inteiro)", "").replace("(ano inteiro; ", "(")
    if detalhe.startswith("Infraero (ano inteiro)"):
        return "Administração pública (Infraero)"
    return detalhe.split(" (T1")[0]   # remove o detalhamento por trimestre


def contexto_20():
    """Contexto de todos os aeroportos para o prompt/app: dados oficiais do pacote
    `dados` (ANAC 2023; pax = embarcados + desembarcados; voos = decolagens) +
    estatísticas de satisfação da MPor 2023. Retorna {icao: dict} e a média nacional."""
    import pandas as pd
    stats, media_nac = estatisticas_satisfacao()
    car = pd.read_csv(CARACT_20).set_index("icao")
    out = {}
    for icao, s in stats.iterrows():
        c = car.loc[icao]
        out[icao] = {
            "nome": c["nome"], "nome_curto": f"{s['nome_aero']} ({icao})",
            "cidade": c["cidade"], "uf": c["uf"], "regiao": c["regiao"],
            "pax_mi": round(float(c["pax_2023_mi"]), 2),
            "voos_k": round(float(c["decolagens_2023"]) / 1000, 1),
            "regime": _regime_curto(c["regime_detalhe_2023"]),
            "concessionaria": c["concessionaria"],
            "n": int(s["n"]), "media": round(float(s["media"]), 2),
            "posicao": int(s["posicao"]), "pct_4_5": round(float(s["pct_4_5"]), 1),
        }
    return out, media_nac
