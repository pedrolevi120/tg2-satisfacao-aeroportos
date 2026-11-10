"""
Templates de prompt para a camada LLM.

Versão 3 (01/11/2026; comparações operacionais já interpretadas). Depois do lote dos 20
relatórios: risco de multa de concessão só para aeroportos concedidos. Além do contexto e do top-5 SHAP, o prompt leva:
  - por prioridade: o que a dimensão mede, nota local × nacional, leitura da matriz
    importância × desempenho, certeza da posição no ranking e comentários ligados;
  - satisfação na série inteira e no último ano completo, média por ano, tendência já
    interpretada (estável / em alta / em queda) e ramos embarque/desembarque;
  - comentários abertos (sentimento e temas críticos);
  - dados externos oficiais (ANAC VRA e Dados Estatísticos do último ano completo, IBGE, Atlas):
    pontualidade, cancelamentos, perfil do movimento, concessão e polo atendido.
Série 2021T1–2026T2 (08/11/2026): mesma estrutura e mesmas regras; os dados passaram a cobrir
21 trimestres da MPor (com as cautelas de coleta no próprio prompt) e o contexto externo é de 2025.
"""

PROMPT_RELATORIO = """Você é um consultor sênior em gestão e operação aeroportuária, redigindo um relatório executivo aprofundado para a diretoria do aeroporto e para o regulador (ANAC). Escreva em português brasileiro formal, técnico e detalhado, sempre fundamentado em dados.

CONTEXTO DO AEROPORTO
Nome: {nome_completo}
Código ICAO: {icao}
Localização: {cidade}/{uf}, região {regiao}
Movimento anual ({ano_movimento}): {pax_mi} milhões de passageiros, {voos_k} mil voos
Regime de gestão: {regime}
Operador atual: {concessionaria}

DADOS OPERACIONAIS E DO ENTORNO (fontes oficiais públicas: ANAC, Voo Regular Ativo e Dados Estatísticos {ano_externo}; IBGE; Atlas do Desenvolvimento Humano)
{externo}

DESEMPENHO GERAL (Pesquisa de Satisfação MPor, {periodo}, {n_periodos} trimestres, n={n} respondentes)
Satisfação geral média no período: {media} de 5 (a média nacional dos 20 aeroportos é {media_nac})
Posição no ranking nacional (20 aeroportos) no período: {posicao}º
Percentual de respondentes satisfeitos (notas 4-5) no período: {pct_4_5}%
Em {ano_recente}, último ano completo: média {media_recente} (média nacional {media_nac_recente}) {compara_recente}; {posicao_recente}º lugar entre os 20; {pct_4_5_recente}% de notas 4-5; n={n_recente}
Satisfação geral por ano: {anos}
Tendência ao longo do período: {tendencia} ponto por ano [{leitura_tendencia}]
Satisfação geral por etapa da entrevista: {ramos}
({cautelas})

PRIORIDADES IDENTIFICADAS PELA ANÁLISE SHAP (modelo XGBoost) — as 5 dimensões mais associadas à satisfação geral dos passageiros deste aeroporto, em ordem de importância. A importância (|SHAP| médio) indica o peso relativo da dimensão na satisfação prevista pelo modelo; é uma associação, não um efeito causal medido. A "certeza" é a posição da dimensão no ranking em 95% de cerca de 1.000 reamostragens. A LEITURA vem da matriz importância × desempenho: CORRIGIR PRIMEIRO = importante e com nota abaixo da média nacional; MANTER = importante e com nota na média nacional ou acima.
{lista_prioridades}

O QUE OS PASSAGEIROS ESCREVEM (comentários abertos da mesma pesquisa)
{comentarios}

INSTRUÇÕES PARA O RELATÓRIO (siga TODAS):
1. Use APENAS os números fornecidos acima. NUNCA invente estatísticas, percentuais, rankings ou benchmarks que não estejam nesta mensagem, nem fatos sobre o aeroporto que não estejam aqui.
2. INTEGRE SEMPRE as particularidades do aeroporto (localização, movimento, regime de gestão, concessionária e os dados operacionais e do entorno) com as prioridades do modelo. Ao comentar cada prioridade, conecte-a explicitamente ao contexto deste aeroporto específico.
3. Compare o desempenho do aeroporto com a média nacional ({media_nac}) e interprete a posição no ranking: o aeroporto está acima ou abaixo da média? O que isso significa dado o seu porte e o seu perfil (doméstico ou internacional, pontualidade)?
4. Copie a LEITURA de cada dimensão exatamente como está acima. Só trate como problema a corrigir as dimensões marcadas CORRIGIR PRIMEIRO. Uma dimensão marcada MANTER tem nota na média nacional ou acima; não a descreva como mal avaliada, e sim como ponto forte a preservar, porque continua pesando na satisfação.
5. Interprete cada dimensão pelo que ela mede, conforme a descrição e os subitens informados.
6. Os dados operacionais (atrasos, cancelamentos, perfil do movimento) não fazem parte do modelo de satisfação: use-os só como contexto para explicar as prioridades, sem afirmar que causam a satisfação. Por exemplo, mais atrasos podem significar mais tempo de espera na sala de embarque e no desembarque. A comparação de cada dado operacional com a média dos 20 já vem pronta entre colchetes (MELHOR, PIOR, acima, abaixo ou praticamente igual): reproduza-a sem inverter o sentido e chame essa referência de "média dos 20 aeroportos", não de média nacional. Não afirme que a concessão, o operador ou o regime de gestão explicam ou contribuem para a satisfação.
7. Use os comentários abertos como evidência qualitativa apenas quando o tema tiver relação com a dimensão. Não atribua aos passageiros frases que não estejam nesta mensagem.
8. Estruture o relatório em SEIS seções, cada uma desenvolvida com profundidade (parágrafos completos, não tópicos rasos):

   ## 1. Sumário Executivo
   Um parágrafo sintetizando a situação do aeroporto e a principal mensagem do relatório.

   ## 2. Diagnóstico Atual
   Analise a posição no ranking nacional, a comparação com a média, o porte e o perfil do aeroporto (movimento, voos internacionais, pontualidade e cancelamentos comparados à média dos 20) e o regime de gestão. Discuta o que o percentual de satisfeitos revela, a posição no último ano completo, a evolução entre os anos (reproduza a leitura da tendência entre colchetes, sem tratar diferenças de poucos centésimos como mudança) e se há diferença relevante entre embarque e desembarque.

   ## 3. Análise das Prioridades (a parte central)
   Para CADA uma das 5 dimensões prioritárias, escreva um parágrafo dedicado contendo: (a) o que a dimensão mede na experiência do passageiro; (b) o desempenho do aeroporto nela, comparado à média nacional, e a sua LEITURA; (c) o peso relativo indicado pela importância SHAP e a certeza da sua posição; (d) o que os comentários indicam, se houver tema relacionado; (e) se a leitura for CORRIGIR PRIMEIRO, hipóteses do que pode explicar a nota abaixo da média, à luz dos dados operacionais; se for MANTER, o que sustenta o bom desempenho.

   ## 4. Recomendações de Ação
   Comece pelas dimensões CORRIGIR PRIMEIRO. Para cada prioridade, proponha ações concretas e específicas, organizadas em curto prazo (até 6 meses) e médio prazo (6-18 meses); para as dimensões MANTER, ações para preservar o nível atual.

   ## 5. Riscos de Não Agir
   Descreva as consequências de não endereçar as prioridades ({riscos}).

   ## 6. Indicadores de Acompanhamento (KPIs)
   Liste 4 a 6 indicadores mensuráveis que o gestor deve monitorar para avaliar o progresso, vinculados às prioridades.

9. Tom: técnico, analítico, sem adjetivos vagos como "excelente" ou "péssimo". Sustente afirmações com os números.
10. Extensão total: entre 800 e 1100 palavras. Seja completo e detalhado.
11. Encerre com a frase: "Análise gerada automaticamente — revisão humana recomendada."

Gere o relatório completo:
"""


def _br(x, nd=2):
    """Número no formato pt-BR (vírgula decimal e ponto de milhar)."""
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _sinal(x, nd=2):
    return ("+" if x >= 0 else "−") + _br(abs(x), nd)


def _certeza(p):
    if "pos_min" not in p:
        return "não estimada"
    if p["pos_min"] == p["pos_max"]:
        return f"{p['pos_min']}ª posição em pelo menos 95% das reamostragens"
    return f"entre a {p['pos_min']}ª e a {p['pos_max']}ª posição em 95% das reamostragens"


def _prioridade(i, p):
    mede = p["descricao"] or p["nome"].lower()
    if p["subitens"]:
        mede += "; subitens avaliados: " + ", ".join(p["subitens"])
    temas = ("; ".join(f"{t} ({_br(v, 1)} críticos por mil respondentes)" for t, v in p["temas"])
             or "nenhum tema crítico relevante")
    return (f"  {i}. {p['nome']} — LEITURA: {p['leitura']}\n"
            f"     O que mede: {mede}\n"
            f"     Nota do aeroporto {_br(p['nota'])}; média nacional {_br(p['nota_nac'])}; "
            f"diferença {_sinal(p['gap'])}; {_br(p['pct_ate_3'], 0)}% das notas até 3\n"
            f"     Importância {_br(p['shap_abs'], 3)}; certeza: {_certeza(p)}\n"
            f"     Comentários ligados: {temas}")


def _bloco_comentarios(c):
    if c is None:
        return "Sem dados de comentários para este aeroporto."
    temas = "; ".join(f"{t} ({_br(v, 1)} por mil respondentes)" for t, v in c["temas"])
    nc, nc_v = c["nao_coberto"]
    return (f"Comentários válidos: {c['n_valido']} ({_br(c['pct_valido'], 1)}% dos respondentes); "
            f"{_br(c['pct_neg'], 1)}% deles com sentimento negativo "
            f"(média dos 20 aeroportos: {_br(c['pct_neg_media'], 1)}%).\n"
            f"Temas críticos mais citados (comentários críticos por mil respondentes): {temas}.\n"
            f"Tema crítico mais citado que o questionário NÃO cobre: {nc} "
            f"({_br(nc_v, 1)} por mil respondentes).")


def _compara(x, media, tol, menor_melhor=None):
    """Comparação já interpretada com a média dos 20, para o modelo não inverter o
    sentido (ex.: ler mais atraso como melhor pontualidade)."""
    if abs(x - media) <= tol:
        return "[praticamente igual à média dos 20]"
    acima = x > media
    if menor_melhor is None:
        return "[acima da média dos 20]" if acima else "[abaixo da média dos 20]"
    pior = acima if menor_melhor else not acima
    return "[PIOR que a média dos 20]" if pior else "[MELHOR que a média dos 20]"


def _bloco_externo(e):
    if e is None:
        return "Sem dados externos para este aeroporto."
    m, n, a = e["media_20"], e["n_aeroportos"], e["ano"]
    linhas = [
        f"Atrasos: {_br(e['pct_atraso30'], 1)}% das partidas realizadas saíram com mais de "
        f"30 minutos de atraso (média dos 20: {_br(m['pct_atraso30'], 1)}%) "
        f"{_compara(e['pct_atraso30'], m['pct_atraso30'], 0.3, True)}; é o "
        f"{e['pos_pontualidade']}º mais pontual entre os {n} (1º = menos atrasos)",
        f"Cancelamentos: {_br(e['pct_canceladas'], 1)}% das partidas previstas foram canceladas "
        f"(média dos 20: {_br(m['pct_canceladas'], 1)}%) "
        f"{_compara(e['pct_canceladas'], m['pct_canceladas'], 0.3, True)}",
        f"Perfil do movimento: {_br(e['pct_pax_internacional'], 1)}% dos passageiros em voos "
        f"internacionais (média dos 20: {_br(m['pct_pax_internacional'], 1)}%) "
        f"{_compara(e['pct_pax_internacional'], m['pct_pax_internacional'], 1)}; "
        f"{int(e['destinos_diretos'])} destinos com voos regulares diretos (média dos 20: "
        f"{_br(m['destinos_diretos'], 0)}); {int(e['empresas_aereas'])} empresas aéreas "
        f"(média dos 20: {_br(m['empresas_aereas'], 0)})",
        f"Movimento de passageiros em {a} comparado a 2019: {_sinal(e['var_pax_2019'], 1)}% "
        f"(média dos 20: {_sinal(m['var_pax_2019'], 1)}%) "
        f"{_compara(e['var_pax_2019'], m['var_pax_2019'], 1)}",
        f"Infraestrutura: {int(e['pistas'])} pista(s) de pouso e decolagem",
    ]
    if isinstance(e.get("rodada"), str) and e["rodada"] != "não concedido":
        conc = f"Concessão: {e['rodada']} rodada de concessões federais"
        if isinstance(e.get("data_assuncao"), str):
            conc += f", operação assumida pela concessionária atual em {e['data_assuncao']}"
        if isinstance(e.get("grupo_controlador"), str):
            conc += f"; controle: {e['grupo_controlador']}"
        if isinstance(e.get("obs_concessao"), str):
            conc += f"; {e['obs_concessao']}"
        linhas.append(conc)
    linhas.append(
        f"Cidade-polo atendida: {e['municipio_polo']}, com {_br(e['populacao_polo_2022'] / 1e6, 2)} "
        f"milhões de habitantes (Censo 2022), PIB per capita de R$ {_br(e['pib_per_capita_polo_2023'], 0)} "
        f"(2023) e IDHM de {_br(e['idhm_polo_2010'], 3)} (2010)")
    return "\n".join(f"- {l}" for l in linhas)


CAUTELAS = ("Cautelas da série: não houve coleta no 2º trimestre de 2021 e o 3º trimestre de 2021 cobre só "
            "agosto e setembro; o 4º trimestre de 2022 teve coleta só em outubro e o 1º de 2023 só em março, com "
            "menos entrevistas; a pesquisa foi suspensa em 16/06/2026, então 2026 cobre só janeiro a meados de "
            "junho.{extra} Diferenças de poucos centésimos entre anos não indicam tendência.")
CAUTELA_AEROPORTO = {
    "SBPA": " Neste aeroporto não houve coleta de maio a dezembro de 2024 (enchentes no Rio Grande do Sul), "
            "então 2024 cobre só janeiro a abril.",
}


def _compara_nacional(x, media, tol=0.02):
    if abs(x - media) <= tol:
        return "[praticamente igual à média nacional]"
    return "[acima da média nacional]" if x > media else "[abaixo da média nacional]"


def _ano_txt(a, ultimo):
    return f"{a} (até junho)" if a == ultimo and a == 2026 else str(a)


def montar_prompt(*, nome_completo, icao, cidade, uf, regiao, pax_mi, voos_k,
                  regime, concessionaria, n, media, posicao, pct_4_5,
                  media_nac, prioridades, satisfacao, comentarios, externo,
                  ano_movimento=2025, periodo="2021T1–2026T2", ano_recente=2025, n_recente=None,
                  media_recente=None, posicao_recente=None, pct_4_5_recente=None,
                  media_nac_recente=None):
    """Monta o prompt completo. `prioridades` vem de contexto.prioridades_detalhadas,
    `satisfacao` de contexto.satisfacao_detalhada, `comentarios` de
    contexto.comentarios_resumo e `externo` de contexto.contexto_externo (ou None).
    Os campos `*_recente` são do último ano completo (contexto.contexto_20)."""
    lista = "\n".join(_prioridade(i + 1, p) for i, p in enumerate(prioridades[:5]))
    ultimo = max(a for a, _, _ in satisfacao["anos"])
    anos = "; ".join(f"{_ano_txt(a, ultimo)} {_br(m)} (n={k})" for a, m, k in satisfacao["anos"])
    ramos = "; ".join(f"{r} {_br(m)} (n={k})" for r, m, k in satisfacao["ramos"])
    concedido = not (externo and str(externo.get("rodada", "")).startswith("não"))
    riscos = ("perda de competitividade, multas regulatórias por descumprimento de metas de concessão, "
              "deterioração da experiência" if concedido else
              "perda de competitividade, deterioração da experiência e cobrança do regulador; este "
              "aeroporto não é concedido, então não cite metas nem multas de concessão")
    return PROMPT_RELATORIO.format(riscos=riscos,
        nome_completo=nome_completo, icao=icao, cidade=cidade, uf=uf,
        regiao=regiao, pax_mi=_br(pax_mi), voos_k=_br(voos_k, 1), ano_movimento=ano_movimento,
        regime=regime, concessionaria=concessionaria, externo=_bloco_externo(externo),
        ano_externo=externo["ano"] if externo else ano_movimento,
        periodo=periodo, n_periodos=len(satisfacao["periodos"]),
        n=n, media=_br(media), media_nac=_br(media_nac), posicao=posicao,
        pct_4_5=_br(pct_4_5, 1), ano_recente=ano_recente, n_recente=n_recente,
        media_recente=_br(media_recente), media_nac_recente=_br(media_nac_recente),
        compara_recente=_compara_nacional(media_recente, media_nac_recente),
        posicao_recente=posicao_recente, pct_4_5_recente=_br(pct_4_5_recente, 1),
        anos=anos, tendencia=_sinal(satisfacao["tendencia"], 3),
        leitura_tendencia=satisfacao["leitura_tendencia"], ramos=ramos,
        cautelas=CAUTELAS.format(extra=CAUTELA_AEROPORTO.get(icao, "")),
        lista_prioridades=lista, comentarios=_bloco_comentarios(comentarios),
    )


def prompt_do_contexto(ctx, icao, media_nac, *, prioridades, satisfacao, comentarios, externo):
    """Atalho: monta o prompt a partir de um item de contexto.contexto_20()."""
    return montar_prompt(
        nome_completo=ctx["nome"], icao=icao, cidade=ctx["cidade"], uf=ctx["uf"],
        regiao=ctx["regiao"], pax_mi=ctx["pax_mi"], voos_k=ctx["voos_k"],
        ano_movimento=ctx["ano_movimento"], regime=ctx["regime"],
        concessionaria=ctx["concessionaria"], n=ctx["n"], media=ctx["media"],
        posicao=ctx["posicao"], pct_4_5=ctx["pct_4_5"], media_nac=media_nac,
        periodo=ctx["periodo"], ano_recente=ctx["ano_recente"], n_recente=ctx["n_recente"],
        media_recente=ctx["media_recente"], posicao_recente=ctx["posicao_recente"],
        pct_4_5_recente=ctx["pct_4_5_recente"], media_nac_recente=ctx["media_nac_recente"],
        prioridades=prioridades, satisfacao=satisfacao, comentarios=comentarios, externo=externo)
