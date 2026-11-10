"""
Templates de prompt para a camada LLM.
"""

PROMPT_RELATORIO = """Você é um consultor sênior em gestão e operação aeroportuária, redigindo um relatório executivo aprofundado para a diretoria do aeroporto e para o regulador (ANAC). Escreva em português brasileiro formal, técnico e detalhado, sempre fundamentado em dados.

CONTEXTO DO AEROPORTO
Nome: {nome_completo}
Código ICAO: {icao}
Localização: {cidade}/{uf}, região {regiao}
Movimento anual (2023): {pax_mi} milhões de passageiros, {voos_k} mil voos
Regime de gestão: {regime}
Concessionária: {concessionaria}

DESEMPENHO RECENTE (Pesquisa de Satisfação MPor 2023, n={n} respondentes)
Satisfação geral média: {media:.2f} de 5 (a média nacional dos 20 aeroportos é 4,38)
Posição no ranking nacional (20 aeroportos): {posicao}º
Percentual de respondentes satisfeitos (notas 4-5): {pct_4_5:.1f}%

PRIORIDADES IDENTIFICADAS PELA ANÁLISE SHAP (modelo XGBoost) — as 5 dimensões mais associadas à satisfação geral dos passageiros deste aeroporto. O valor |SHAP| indica o peso relativo de cada dimensão na satisfação prevista pelo modelo; é uma associação, não um efeito causal medido:
{lista_prioridades}

INSTRUÇÕES PARA O RELATÓRIO (siga TODAS):
1. Use APENAS os números fornecidos acima. NUNCA invente estatísticas, percentuais ou benchmarks que não estejam nesta mensagem.
2. INTEGRE SEMPRE as particularidades do aeroporto (localização, movimento, regime de gestão, concessionária) com as prioridades do modelo. Ao comentar cada prioridade, conecte-a explicitamente ao contexto deste aeroporto específico.
3. Compare o desempenho do aeroporto com a média nacional (4,38) e interprete a posição no ranking: o aeroporto está acima ou abaixo da média? O que isso significa dado o seu porte?
4. Estruture o relatório em SEIS seções, cada uma desenvolvida com profundidade (parágrafos completos, não tópicos rasos):

   ## 1. Sumário Executivo
   Um parágrafo sintetizando a situação do aeroporto e a principal mensagem do relatório.

   ## 2. Diagnóstico Atual
   Analise a posição no ranking nacional, a comparação com a média (4,38), o porte do aeroporto (movimento) e o regime de gestão. Discuta o que o percentual de satisfeitos revela.

   ## 3. Análise das Prioridades (a parte central)
   Para CADA um dos 5 atributos prioritários, escreva um parágrafo dedicado contendo: (a) o que o atributo representa na experiência do passageiro; (b) por que ele é crítico NESTE aeroporto, conectando ao contexto (movimento, localização, regime); (c) o peso relativo indicado pelo valor SHAP; (d) hipóteses do que pode estar causando a baixa avaliação.

   ## 4. Recomendações de Ação
   Para cada prioridade, proponha ações concretas e específicas, organizadas em curto prazo (até 6 meses) e médio prazo (6-18 meses).

   ## 5. Riscos de Não Agir
   Descreva as consequências de não endereçar as prioridades (perda de competitividade, multas regulatórias por descumprimento de metas de concessão, deterioração da experiência).

   ## 6. Indicadores de Acompanhamento (KPIs)
   Liste 4 a 6 indicadores mensuráveis que o gestor deve monitorar para avaliar o progresso, vinculados às prioridades.

5. Tom: técnico, analítico, sem adjetivos vagos como "excelente" ou "péssimo". Sustente afirmações com os números.
6. Extensão total: entre 700 e 1000 palavras. Seja completo e detalhado.
7. Encerre com a frase: "Análise gerada automaticamente — revisão humana recomendada."

Gere o relatório completo:
"""


def montar_prompt(*, nome_completo, icao, cidade, uf, regiao, pax_mi, voos_k,
                  regime, concessionaria, n, media, posicao, pct_4_5,
                  top_atributos):
    """Monta o prompt completo. top_atributos é lista de dicts com {feature, shap_abs}."""
    lista = "\n".join(
        f"  {i+1}. {a['feature']} (impacto médio = {a['shap_abs']:.3f})"
        for i, a in enumerate(top_atributos[:5])
    )
    return PROMPT_RELATORIO.format(
        nome_completo=nome_completo, icao=icao, cidade=cidade, uf=uf,
        regiao=regiao, pax_mi=pax_mi, voos_k=voos_k,
        regime=regime, concessionaria=concessionaria,
        n=n, media=media, posicao=posicao, pct_4_5=pct_4_5,
        lista_prioridades=lista,
    )
