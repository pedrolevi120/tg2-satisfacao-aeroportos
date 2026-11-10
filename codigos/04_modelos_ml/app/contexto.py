"""
Contexto operacional dos aeroportos (fonte única para app e scripts).

Série 2021T1–2026T2 (08/11/2026): satisfação da MPor na série inteira (21 trimestres) e no último
ano completo (2025), com a evolução por ano/trimestre e a tendência; movimento ANAC e dados externos
de 2025; regime de gestão e operador atuais (2026T2) e trocas dentro da série.
O dicionário CONTEXTO_MVP abaixo guarda os valores do MVP original (GRU/CGH/VCP, MPor 2023).
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
# 20 aeroportos (TG-2), série 2021T1–2026T2
# ---------------------------------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PANORAMA = ROOT / "dados" / "processados" / "panorama_aeroportos_2021_2026.parquet"
CONTEXTO_SERIE = ROOT / "dados" / "processados" / "contexto_aeroportos_20_serie.csv"
RANKING_20 = ROOT / "codigos" / "05_interpretacao" / "ranking_prioridades_20.csv"
RANKING_MVP = ROOT / "codigos" / "05_interpretacao" / "ranking_prioridades_por_aeroporto.csv"

PERIODO_INI, PERIODO_FIM = "2021T1", "2026T2"
SERIE = f"{PERIODO_INI}–{PERIODO_FIM}"
ANO_RECENTE = 2025      # último ano completo da série (satisfação, movimento e dados externos)
TOL_TENDENCIA = 0.02    # |inclinação| < 0,02 ponto/ano = estável


def _tendencia(periodos):
    """Inclinação (pontos de satisfação por ano) da reta ajustada às médias trimestrais."""
    import numpy as np
    if len(periodos) < 4:
        return float("nan")
    x = [int(p[:4]) + (int(p[-1]) - 1) / 4 for p, _, _ in periodos]
    return float(np.polyfit(x, [m for _, m, _ in periodos], 1)[0])


def leitura_tendencia(t):
    """Leitura pronta da tendência (para o LLM não interpretar a inclinação)."""
    if t != t:
        return "não estimada"
    if abs(t) < TOL_TENDENCIA:
        return "estável"
    return "em alta" if t > 0 else "em queda"


def _carregar_panorama():
    import pandas as pd
    return pd.read_parquet(PANORAMA).dropna(subset=["SATISFACAO_GERAL_NUM"])


def _resumo(p):
    g = (p.groupby(["AEROPORTO", "nome_aero", "regiao", "uf"])["SATISFACAO_GERAL_NUM"]
          .agg(n="count", media="mean", pct_4_5=lambda s: 100 * (s >= 4).mean())
          .reset_index().sort_values("media", ascending=False))
    g["posicao"] = range(1, len(g) + 1)
    return g.set_index("AEROPORTO")


def estatisticas_satisfacao():
    """Por aeroporto: n, média, % notas 4–5 e posição na série 2021T1–2026T2 e em 2025 (sufixo
    `_2025`). Retorna (tabela, média nacional da série); as médias nacionais por ano estão em
    medias_nacionais()."""
    p = _carregar_panorama()
    g = _resumo(p)
    r = _resumo(p[p["ANO"] == ANO_RECENTE])[["n", "media", "pct_4_5", "posicao"]]
    g = g.join(r.add_suffix(f"_{ANO_RECENTE}"))
    return g, p["SATISFACAO_GERAL_NUM"].mean()


def medias_nacionais():
    """Média nacional (todos os respondentes dos 20 aeroportos) por ano e na série ('serie')."""
    p = _carregar_panorama()
    d = p.groupby("ANO")["SATISFACAO_GERAL_NUM"].mean().to_dict()
    d["serie"] = p["SATISFACAO_GERAL_NUM"].mean()
    return d


def carregar_ranking():
    """Ranking dos 20 aeroportos (T5) se existir; senão o ranking do MVP (3 aeroportos)."""
    import pandas as pd
    if RANKING_20.exists():
        r = pd.read_csv(RANKING_20).rename(columns={"atributo": "feature",
                                                    "mean_abs_shap": "shap_abs_medio"})
        return r, "20"
    return pd.read_csv(RANKING_MVP), "mvp"


CARACT_20 = ROOT / "codigos" / "02_preprocessamento" / "aeroportos_20_caracteristicas.csv"


def _dia_anterior(data):
    import pandas as pd
    return (pd.to_datetime(data, dayfirst=True) - pd.Timedelta(days=1)).strftime("%d/%m/%Y")


# troca de concessionária sem troca de regime (mesma data de 10_serie_historica/02_contexto_periodos.py)
TROCA_OPERADOR = {"SBSG": "19/02/2024"}


def regime_serie(icao):
    """Regime de gestão no período da pesquisa (2021T1–2026T2) e operador atual (2026T2), a partir
    de dados/processados/contexto_aeroportos_20_serie.csv. Retorna (texto do regime, operador)."""
    import pandas as pd
    s = pd.read_csv(CONTEXTO_SERIE)
    s = s[s["icao"] == icao].sort_values(["ano", "trimestre"])
    atual = s.iloc[-1]
    regimes = set(s["regime_gestao"])
    if regimes == {"infraero"}:
        return f"Administração pública (Infraero) em todo o período da pesquisa ({SERIE})", atual["operador"]
    if regimes == {"concessionada"}:
        txt = f"Concessão privada em todo o período da pesquisa ({SERIE})"
        ops = list(dict.fromkeys(s["operador"]))
        if len(ops) > 1:   # troca de concessionária sem troca de regime (SBSG)
            troca = TROCA_OPERADOR.get(icao) or s[s["operador"] == ops[-1]].iloc[0]["periodo"]
            txt += f"; troca de concessionária em {troca}: {ops[0]} → {ops[-1]}"
        return txt, atual["operador"]
    data = atual["data_assuncao"]
    return (f"Concessão privada desde {data}; antes, Infraero (até {_dia_anterior(data)}), "
            f"mudança ocorrida dentro do período da pesquisa ({SERIE})"), atual["operador"]


def contexto_20():
    """Contexto de todos os aeroportos para o prompt/app: movimento ANAC de 2025 (pax = embarcados
    + desembarcados; voos = decolagens), regime no período da pesquisa e operador atual, e
    satisfação MPor na série 2021T1–2026T2 e em 2025, com média por ano e tendência.
    Retorna {icao: dict} e a média nacional da série."""
    import pandas as pd
    stats, media_nac = estatisticas_satisfacao()
    nac = medias_nacionais()
    car = pd.read_csv(CARACT_20).set_index("icao")
    a = ANO_RECENTE
    col_pax, col_voos = (f"pax_{a}_mi", f"voos_{a}_k") if f"pax_{a}_mi" in car else ("pax_2023_mi", "voos_2023_k")
    out = {}
    for icao, s in stats.iterrows():
        c = car.loc[icao]
        regime, operador = regime_serie(icao)
        out[icao] = {
            "nome": c["nome"], "nome_curto": f"{s['nome_aero']} ({icao})",
            "cidade": c["cidade"], "uf": c["uf"], "regiao": c["regiao"],
            "ano_movimento": int(col_pax[4:8]),
            "pax_mi": round(float(c[col_pax]), 2), "voos_k": round(float(c[col_voos]), 1),
            "regime": regime, "concessionaria": operador,
            "periodo": SERIE,
            "n": int(s["n"]), "media": round(float(s["media"]), 2),
            "posicao": int(s["posicao"]), "pct_4_5": round(float(s["pct_4_5"]), 1),
            "ano_recente": a, "n_recente": int(s[f"n_{a}"]),
            "media_recente": round(float(s[f"media_{a}"]), 2),
            "posicao_recente": int(s[f"posicao_{a}"]),
            "pct_4_5_recente": round(float(s[f"pct_4_5_{a}"]), 1),
            "media_nac_recente": round(float(nac[a]), 2),
        }
    return out, media_nac


# ---------------------------------------------------------------------------
# Dados extras do aeroporto para o relatório LLM (desempenho por dimensão,
# certeza do ranking, evolução trimestral, ramos e comentários)
# ---------------------------------------------------------------------------
MPOR_20 = ROOT / "dados" / "processados" / "mpor_2021_2026_20.parquet"
ESTABILIDADE = ROOT / "codigos" / "09_experimentos" / "resultados" / "e5_estabilidade_aeroporto.csv"
NLP = ROOT / "codigos" / "08_nlp" / "saida"


def nome_legivel(atributo: str) -> str:
    """'ORGANIZAÇÃO DAS FILAS2' -> 'Organização das filas' (nomes da planilha MPor)."""
    return atributo.rstrip("0123456789").strip().capitalize()


def nota_nacional(ranking):
    """Nota média nacional de cada dimensão, ponderada pelo nº de respostas."""
    r = ranking.dropna(subset=["nota_media", "n_respostas"])
    soma = (r["nota_media"] * r["n_respostas"]).groupby(r["feature"]).sum()
    return soma / r.groupby("feature")["n_respostas"].sum()


# O que cada dimensão do top-5 mede no questionário MPor (evita que o LLM
# confunda, por exemplo, localização no terminal com acesso ao aeroporto).
DESCRICAO_DIMENSAO = {
    "LOCALIZAÇÃO E DESLOCAMENTO": "facilidade de se orientar e de se deslocar dentro do terminal "
                                  "(não é o acesso ao aeroporto, que é outro item da pesquisa)",
    "LIMPEZA GERAL DO AEROPORTO": "limpeza das áreas comuns do terminal",
    "PROCESSO DE RESTITUIÇÃO DE BAGAGENS": "entrega das bagagens despachadas no desembarque",
    "CONFORTO DA SALA DE EMBARQUE": "conforto da área de espera antes do embarque",
    "SANITÁRIOS": "banheiros do terminal",
    "DISPONIBILIDADE DE TOMADAS": "tomadas para carregar aparelhos no terminal",
}


def _temas_da_dimensao(ag, icao, itens, n=2):
    """Temas críticos dos comentários ligados aos itens (pai + subitens) da dimensão."""
    a = ag[(ag["aeroporto"] == icao) & ag["atributos_likert"].notna()
           & ~ag["cobertura"].isin(["sem_tema", "geral"]) & (ag["criticos_por_mil"] >= 0.5)
           & ~ag["tema_rotulo"].str.contains("elogio", case=False)]
    lig = a[a["atributos_likert"].map(lambda s: any(i in s.split("; ") for i in itens))]
    lig = lig.sort_values("criticos_por_mil", ascending=False).head(n)
    return [(r["tema_rotulo"], float(r["criticos_por_mil"])) for _, r in lig.iterrows()]


def prioridades_detalhadas(ranking, icao, k=5):
    """Top-k dimensões acionáveis com o que medem (descrição e subitens), desempenho
    local × nacional, leitura da matriz importância × desempenho, faixa de posição no
    ranking (IC 95% das reamostragens treino × teste do experimento E5) e temas
    críticos dos comentários ligados a cada dimensão."""
    import pandas as pd
    nac = nota_nacional(ranking)
    est = (pd.read_csv(ESTABILIDADE).query("aeroporto == @icao").set_index("dimensao")
           if ESTABILIDADE.exists() else pd.DataFrame())
    from comentarios import sem_ano
    arq_ag = NLP / "agregado_aeroporto_tema.csv"
    ag = sem_ano(pd.read_csv(arq_ag), ["aeroporto", "tema"]) if arq_ag.exists() else None
    out = []
    for a in top_acionaveis(ranking, icao, k=k):
        r = ranking[(ranking["aeroporto"] == icao) & (ranking["feature"] == a["feature"])].iloc[0]
        gap = float(r["nota_media"] - nac[a["feature"]])
        subitens = [s.strip() for s in str(r.get("subitens", "")).split("|")
                    if s.strip() and s.strip() != "nan"]
        d = {"feature": a["feature"], "nome": nome_legivel(a["feature"]),
             "descricao": DESCRICAO_DIMENSAO.get(a["feature"], ""),
             "subitens": [nome_legivel(s).lower() for s in subitens],
             "shap_abs": a["shap_abs"], "nota": float(r["nota_media"]),
             "nota_nac": float(nac[a["feature"]]), "gap": gap,
             "pct_ate_3": 100 * float(r["pct_nota_ate_3"]),
             "leitura": "CORRIGIR PRIMEIRO" if gap < 0 else "MANTER",
             "temas": (_temas_da_dimensao(ag, icao, [a["feature"]] + subitens)
                       if ag is not None else [])}
        if a["feature"] in est.index:
            e = est.loc[a["feature"]]
            d["pos_min"], d["pos_max"] = int(e["pos_p2_5_BxA"]), int(e["pos_p97_5_BxA"])
        out.append(d)
    return out


CONTEXTO_EXTERNO = ROOT / "dados" / "processados" / "contexto_externo_20.csv"
METRICAS_EXTERNAS = ["pct_atraso30", "pct_canceladas", "pct_pax_internacional", "var_pax_2019",
                     "destinos_diretos", "empresas_aereas", "pax_mi", "decolagens"]


def _col_ano(m, ano):
    """Nome da coluna de contexto_externo_20.csv para a métrica genérica no ano."""
    return {"var_pax_2019": f"var_pax_2019_{ano}", "pax_mi": f"pax_{ano}_mi"}.get(m, f"{m}_{ano}")


def contexto_externo(icao, ano=ANO_RECENTE):
    """Dados públicos oficiais do aeroporto (ANAC VRA e Dados Estatísticos do ano, padrão 2025; IBGE;
    Atlas), com a média dos 20 aeroportos nas métricas comparáveis. As métricas saem com nomes
    genéricos (sem o ano: pct_atraso30, pct_canceladas, ...) e o ano em 'ano'. Gerado por
    codigos/02_preprocessamento/04_contexto_externo.py. None se o arquivo não existir."""
    import pandas as pd
    if not CONTEXTO_EXTERNO.exists():
        return None
    c = pd.read_csv(CONTEXTO_EXTERNO).set_index("icao")
    cols = {_col_ano(m, ano): m for m in METRICAS_EXTERNAS if _col_ano(m, ano) in c}
    if f"pct_atraso30_{ano}" not in c:
        return None
    d = c.loc[icao].to_dict()
    d.update({m: d[col] for col, m in cols.items()})
    d["media_20"] = c[list(cols)].rename(columns=cols).mean().to_dict()
    d["pos_pontualidade"] = int(c[f"pct_atraso30_{ano}"].rank(method="min")[icao])
    d["pos_cancelamento"] = int(c[f"pct_canceladas_{ano}"].rank(method="min")[icao])
    d["ano"] = ano
    d["n_aeroportos"] = len(c)
    return d


def satisfacao_detalhada(icao):
    """Satisfação geral média e nº de respondentes do aeroporto por período (2021T1…2026T2), por
    ano e por ramo (embarque/desembarque), com a tendência (pontos por ano) e a sua leitura."""
    p = _carregar_panorama()
    p = p[p["AEROPORTO"] == icao]
    agg = lambda col: [(k, float(g.mean()), int(g.size)) for k, g in p.groupby(col)["SATISFACAO_GERAL_NUM"]]
    periodos = agg("PERIODO")
    t = _tendencia(periodos)
    return {"periodos": periodos, "trimestres": periodos, "anos": [(int(a), m, n) for a, m, n in agg("ANO")],
            "ramos": agg("PROCESSO"), "tendencia": t, "leitura_tendencia": leitura_tendencia(t)}


def comentarios_resumo(icao, n_temas=3):
    """Resumo dos comentários abertos (pacote NLP/T7): volume, sentimento e temas
    críticos mais citados. None se as saídas do NLP não existirem."""
    import pandas as pd
    arqs = [NLP / f for f in ("agregado_aeroporto_tema.csv", "sentimento_por_aeroporto.csv",
                              "triangulacao_por_aeroporto.csv", "descritiva_por_aeroporto.csv")]
    if not all(p.exists() for p in arqs):
        return None
    from comentarios import sem_ano
    ag, sen, tri, desc = (pd.read_csv(p) for p in arqs)
    ag = sem_ano(ag, ["aeroporto", "tema"])
    tri = sem_ano(tri, "aeroporto")
    desc = sem_ano(desc.rename(columns={"Unnamed: 0": "aeroporto"}), "aeroporto").set_index("aeroporto").loc[icao]
    sen = sem_ano(sen, "AEROPORTO").set_index("AEROPORTO")
    temas = (ag[(ag["aeroporto"] == icao) & (~ag["cobertura"].isin(["sem_tema", "geral"]))]
             .sort_values("criticos_por_mil", ascending=False).head(n_temas))
    t = tri.set_index("aeroporto").loc[icao]
    return {"n_valido": int(desc["n_valido"]), "pct_valido": float(desc["pct_valido"]),
            "pct_neg": float(sen.loc[icao, "pct_neg"]), "pct_neg_media": float(sen["pct_neg"].mean()),
            "temas": [(r["tema_rotulo"], float(r["criticos_por_mil"])) for _, r in temas.iterrows()],
            "nao_coberto": (t["top_nao_coberto"], float(t["top_nao_coberto_por_mil"]))}
