# Notas — contexto externo 2025, panorama 2021–2026 e app/prompts (08/11/2026)

Escopo: trocar o "último ano completo" do contexto externo de 2023 para 2025, refazer o panorama agregado com a série 2021T1–2026T2 e deixar o app e os prompts do LLM cientes da série. O lote de 20 relatórios LLM **não** foi rodado.

## 1. Mudanças por arquivo

| Arquivo | O que mudou |
|---|---|
| `dados/brutos/vra_2025/VRA_2025_01..12.csv` | **Novo.** 12 arquivos VRA de 2025, baixados do portal SIROS/ANAC (1.005.807 registros). O VRA de 2023 continua em `dados/brutos/anac/vra_2023/`. |
| `codigos/02_preprocessamento/04_contexto_externo.py` | **Reescrito e parametrizado.** `ANO_REF=2025` (CLI `--ano`), `ANO_LEGADO=2023`, base de variação 2019. Para cada ano gera `pct_atraso30_{a}`, `pct_canceladas_{a}`, `pos_pontualidade_{a}`, `pos_cancelamento_{a}`, `pax_{a}_mi`, `decolagens_{a}`, `pct_pax_internacional_{a}`, `var_pax_2019_{a}`, `destinos_diretos_{a}` e `empresas_aereas_{a}`. As colunas sem sufixo `pos_pontualidade`/`pos_cancelamento` passam a se referir a 2025. Regime e operador atuais vêm de `contexto_aeroportos_20_serie.csv` em 2026T2 (`regime_atual`, `operador_atual`, `periodo_regime_atual`). A concessão do SBSG foi corrigida no código: Zurich desde 19/02/2024, antes Inframerica. Os valores de 2023 batem exatamente com o arquivo antigo. |
| `codigos/02_preprocessamento/00_coletar_dados_contexto.py` | Ano parametrizado: `ANO_LEGADO=2023` fixo para as saídas legadas; `--ano` (padrão 2025) define o ano novo. `aeroportos_20_caracteristicas.csv` ganhou `pax_2025`, `pax_2025_mi`, `decolagens_2025`, `voos_2025_k` e `movimentos_2025`. `resumo_contexto20.csv` agora tem uma linha por ano. O `.tex` em `texto/_paralelo/tex/` só é reescrito com `--tex`. Os arquivos `contexto_aeroportos_20.csv`, municípios e checagem saem byte-idênticos, então `10_serie_historica/02_contexto_periodos.py` não quebra. |
| `codigos/06_visualizacoes/01_panorama_agregado.py` | **Reescrito para a série.** Lê `mpor_2021_2026_20.parquet`, com o regime por período vindo de `contexto_aeroportos_20_serie.csv`. Gera o parquet `dados/processados/panorama_aeroportos_2021_2026.parquet` (com ANO, TRIMESTRE, PERIODO e regime_gestao), `panorama_resumo.csv` (série + 2025 + médias anuais + tendência + regime 2026T2), `panorama_por_periodo.csv`, `panorama_nacional_ano.csv`, `panorama_regiao_ano.csv` e `panorama_regime_periodo.csv`. Figuras em `midias/figuras/` (nomes da tese mantidos): `panorama_mapa_brasil`, `panorama_ranking` (série, com ◆ para 2025), `panorama_ranking_apres`, `panorama_por_regiao` (agora linhas região × ano), `panorama_evolucao_trimestral` (21 períodos com lacunas sombreadas) e `panorama_heatmap_atributos`. **Novas figuras:** `panorama_ranking_2025` e `panorama_regime_periodo`. Cópias PNG em `apresentacoes/img_png/<nome>-1.png`. O parquet antigo `panorama_aeroportos_2023.parquet` foi mantido. |
| `codigos/04_modelos_ml/app/contexto.py` | Usa o parquet do panorama da série e o contexto por período. `estatisticas_satisfacao()` e `medias_nacionais()` retornam série e ano. Funções novas: `_tendencia`/`leitura_tendencia` (inclinação MQO em ponto/ano; \|incl.\| < 0,02 é "estável") e `regime_serie(icao)` (texto do histórico de regime + operador 2026T2). `contexto_externo(icao, ano=2025)` devolve chaves genéricas (`pct_atraso30`, `pax_mi` etc.) e recalcula as posições. `satisfacao_detalhada()` devolve períodos, anos, ramos e tendência. |
| `codigos/04_modelos_ml/app/prompts.py` | Mantém a estrutura e as regras da v3 (comparações pré-interpretadas, "CORRIGIR PRIMEIRO/MANTER", multa só para concedidos). Cabeçalho "MPor, 2021T1–2026T2, N trimestres". Nova linha "Em 2025, último ano completo: média … (nacional …) [compara]; Nº". Nova linha de satisfação por ano (2026 como "até junho"). Nova linha de tendência com leitura entre colchetes. Os trimestres agora levam o ano (ex.: 2025T3). O aviso específico de 2023 foi trocado pelas lacunas de coleta da série, com cautela extra para o SBPA (enchentes de 2024). Bloco externo com o ano de referência e a concessionária atual. Novo `prompt_do_contexto(...)` reúne a montagem para o app e o lote. |
| `codigos/04_modelos_ml/app/comentarios.py` | `sem_ano()` aceita tabelas do NLP com colunas ANO/PERIODO: descarta BRASIL, fica com a linha total e dá erro claro se não houver. Os exemplos de comentário mostram o PERIODO. |
| `codigos/04_modelos_ml/app/app.py` | Novas métricas: média da série, média 2025 vs nacional 2025, posição na série (com a de 2025 na legenda), respondentes e % de notas 4–5. **Nova aba "Evolução da satisfação"**, com linha por trimestre vs média nacional, tabela por ano, tendência e etapa embarque/desembarque. Tabelas operacional e externa com o ano 2025 e o operador atual. Legenda "contexto ANAC 2025". |
| `codigos/04_modelos_ml/app/capturar_telas.py` | Inclui a captura da nova aba (`mvp_06_evolucao.png`). **Não foi rodado** porque sobrescreveria `midias/imagens`. |
| `codigos/05_interpretacao/04_relatorios_20.py` | `--refazer` regenera os 20 relatórios por cima dos existentes. `--dry-run` imprime os prompts sem LLM, sem gravar e sem exigir o Ollama. Usa `prompt_do_contexto`. |

Não alterados:
- `05_interpretacao/02_gerar_relatorios_mvp.py` e `03_comparar_llms.py`: já estavam quebrados com a assinatura antiga de `montar_prompt(top_atributos=…)`; são legados da comparação de LLMs.
- `app/README.md`: o texto ainda fala de 2023.

## 2. Comandos

```bash
cd tg2-satisfacao-aeroportos   # raiz do repositório
.venv/bin/python codigos/02_preprocessamento/00_coletar_dados_contexto.py            # --ano 2025 (padrão); --tex reescreve o .tex
.venv/bin/python codigos/02_preprocessamento/04_contexto_externo.py                  # --ano 2025 (padrão); exige os 12 VRA do ano
.venv/bin/python codigos/10_serie_historica/02_contexto_periodos.py                  # (inalterado) contexto por período
.venv/bin/python codigos/06_visualizacoes/01_panorama_agregado.py                    # figuras + csv + parquet + PNGs
.venv/bin/python codigos/05_interpretacao/04_relatorios_20.py --dry-run --aeroportos SBGR   # confere o prompt sem LLM
.venv/bin/python codigos/05_interpretacao/04_relatorios_20.py --refazer              # lote LLM (Ollama/gemma2:9b), depois do ranking novo
.venv/bin/streamlit run codigos/04_modelos_ml/app/app.py
```

Ordem: o 00 e o 04 vêm antes do app e do lote. O panorama vem antes do app. O lote LLM só deve rodar depois que o `06_modelagem_20` gerar o ranking SHAP novo.

## 3. Números-chave

### 3.1 Contexto externo 2025 (formato da Tabela 3.2)

Fontes:
- Pontualidade: VRA 2025. Voos regulares realizados com partida do aeroporto; atraso de mais de 30 min na partida. Cancelamentos sobre os previstos.
- Movimento: Dados Estatísticos ANAC 2025. Passageiros = pagos + grátis, embarcados + desembarcados; voos = decolagens.

Posições de 1 (melhor) a 20.

| ICAO | Atraso >30 min (%) | Pos. | Cancel. (%) | Pos. | Pax 2025 (mi) | Decolagens | % pax intl. | Destinos diretos | Empresas | Var. pax 2019→2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| SBBE | 8,0 | 15 | 1,8 | 10 | 3,9 | 15.946 | 4,5 | 38 | 10 | +9,8% |
| SBBR | 4,9 | 1 | 1,0 | 3 | 16,7 | 58.347 | 5,3 | 56 | 14 | −1,1% |
| SBCF | 8,4 | 17 | 1,7 | 8 | 13,18 | 57.949 | 4,2 | 77 | 13 | +19,7% |
| SBCT | 8,0 | 14 | 2,9 | 18 | 6,2 | 26.538 | 3,0 | 38 | 15 | −5,5% |
| SBCY | 7,0 | 8 | 1,8 | 11 | 2,52 | 10.329 | 0,0 | 25 | 4 | −13,9% |
| SBEG | 10,0 | 19 | 5,6 | 20 | 3,2 | 15.593 | 5,8 | 54 | 18 | +6,4% |
| SBFL | 8,2 | 16 | 3,2 | 19 | 5,10 | 18.207 | 24,5 | 33 | 21 | +33,6% |
| SBFZ | 6,0 | 5 | 1,3 | 4 | 6,1 | 19.931 | 7,6 | 42 | 11 | −14,7% |
| SBGL | 7,7 | 13 | 1,5 | 6 | 17,84 | 58.862 | 31,9 | 69 | 44 | +28,1% |
| SBGO | 6,3 | 6 | 1,6 | 7 | 3,9 | 13.834 | 0,0 | 18 | 4 | +22,8% |
| SBGR | 11,4 | 20 | 2,6 | 15 | 47,08 | 143.691 | 35,4 | 133 | 49 | +9,6% |
| SBKP | 9,6 | 18 | 2,7 | 16 | 13,20 | 61.134 | 8,5 | 116 | 31 | +24,4% |
| SBMO | 5,2 | 2 | 1,0 | 2 | 3,0 | 10.139 | 1,5 | 28 | 5 | +38,7% |
| SBPA | 7,1 | 9 | 2,0 | 13 | 7,45 | 27.954 | 4,7 | 33 | 16 | −10,3% |
| SBRF | 7,2 | 10 | 1,9 | 12 | 9,93 | 37.318 | 5,0 | 62 | 15 | +13,2% |
| SBRJ | 5,3 | 3 | 2,8 | 17 | 6,18 | 28.562 | 0,0 | 13 | 3 | −32,3% |
| SBSG | 7,2 | 11 | 0,9 | 1 | 2,39 | 8.286 | 4,3 | 25 | 5 | +3,1% |
| SBSP | 7,6 | 12 | 1,7 | 9 | 24,58 | 93.372 | 0,0 | 48 | 5 | +7,7% |
| SBSV | 5,8 | 4 | 1,5 | 5 | 8,0 | 28.758 | 6,9 | 43 | 12 | +7,2% |
| SBVT | 6,6 | 7 | 2,3 | 14 | 3,49 | 13.251 | 0,0 | 24 | 7 | +6,3% |
| **Média dos 20 (2025)** | **7,38** | | **2,09** | | **10,19** | **37.400** | **7,65** | **48,75** | **15,10** | **+7,63%** |
| Média dos 20 (2023, versão enviada) | 8,18 | | 3,23 | | 8,98 | 35.618 | 6,90 | 47,90 | 14,00 | −3,80% |

Totais nacionais:
- 2025: os 20 aeroportos somam 203.865.933 passageiros de 235.335.761 no Brasil (86,63%) e 748.001 de 887.719 decolagens (84,26%).
- 2023: 179.605.354 passageiros (86,14%) e 712.362 decolagens (82,79%).

Mudanças relevantes entre 2023 e 2025:
- A pontualidade e os cancelamentos melhoraram na média.
- O tráfego dos 20 aeroportos passou de −3,8% para +7,6% em relação a 2019.
- O SBGR continua o pior em atraso (11,4%, 20º).
- O SBEG é o pior em cancelamento (5,6%).
- O SBBR é o mais pontual.
- O SBSG tem a menor taxa de cancelamento.
- O SBRJ segue bem abaixo de 2019 (−32,3%).

### 3.2 Panorama da satisfação geral (1–5), 2021T1–2026T2

São 461.593 respostas válidas de satisfação geral em 21 trimestres.

**Média nacional por ano** (ponderada pelos respondentes):

| Ano | Média | % notas 4–5 | n | Trimestres |
|---|---|---|---|---|
| 2021 | 4,450 | 92,5 | 44.536 | T1, T3, T4 (sem T2; T3 só ago–set) |
| 2022 | 4,432 | 92,7 | 82.821 | T1–T4 (T4 só outubro) |
| 2023 | 4,375 | 92,1 | 84.662 | T1–T4 (T1 só março) — igual ao valor da versão enviada |
| 2024 | 4,426 | 93,7 | 99.204 | T1–T4 |
| 2025 | 4,458 | 93,8 | 103.166 | T1–T4 |
| 2026 | 4,492 | 94,8 | 47.204 | T1–T2 (suspensa em 16/06/2026) |
| **Série** | **4,434** | **93,2** | **461.593** | 21 períodos (média das médias dos aeroportos: 4,437) |

Leitura: 2023 foi o vale da série. Desde então a satisfação sobe todo ano, e 2026 é o melhor ano parcial.

**Ranking da série e de 2025.** A tendência é a inclinação das médias trimestrais, em ponto por ano.

| Pos. série | Aeroporto | Média série | Média 2025 | Pos. 2025 | Tendência | Δ 2025−2022 |
|---|---|---|---|---|---|---|
| 1 | Florianópolis (SBFL) | 4,742 | 4,748 | 1 | +0,004 | −0,03 |
| 2 | Vitória (SBVT) | 4,660 | 4,643 | 2 | +0,006 | −0,02 |
| 3 | Maceió (SBMO) | 4,600 | 4,618 | 3 | +0,001 | +0,03 |
| 4 | Porto Alegre (SBPA) | 4,559 | 4,574 | 4 | +0,006 | +0,01 |
| 5 | Curitiba (SBCT) | 4,532 | 4,519 | 8 | −0,020 | +0,01 |
| 6 | Viracopos (SBKP) | 4,506 | 4,463 | 12 | −0,031 | −0,14 |
| 7 | Confins (SBCF) | 4,499 | 4,531 | 7 | +0,018 | +0,08 |
| 8 | Fortaleza (SBFZ) | 4,468 | 4,496 | 10 | 0,000 | +0,02 |
| 9 | Manaus (SBEG) | 4,462 | 4,512 | 9 | +0,016 | +0,08 |
| 10 | Brasília (SBBR) | 4,454 | 4,544 | 6 | +0,036 | +0,10 |
| 11 | Goiânia (SBGO) | 4,453 | 4,452 | 13 | +0,002 | −0,01 |
| 12 | Natal (SBSG) | 4,428 | 4,491 | 11 | +0,011 | +0,04 |
| 13 | Galeão (SBGL) | 4,386 | 4,405 | 14 | +0,001 | 0,00 |
| 14 | Salvador (SBSV) | 4,376 | 4,365 | 16 | −0,017 | −0,04 |
| 15 | Cuiabá (SBCY) | 4,343 | 4,547 | 5 | **+0,094** | **+0,28** |
| 16 | Santos Dumont (SBRJ) | 4,313 | 4,351 | 17 | 0,000 | +0,03 |
| 17 | Guarulhos (SBGR) | 4,293 | 4,350 | 18 | +0,017 | +0,11 |
| 18 | Recife (SBRF) | 4,289 | 4,393 | 15 | +0,050 | +0,20 |
| 19 | Congonhas (SBSP) | 4,272 | 4,302 | 19 | +0,002 | +0,06 |
| 20 | Belém (SBBE) | 4,114 | 3,907 | 20 | −0,029 | **−0,29** |

O topo é estável: SBFL, SBVT, SBMO e SBPA ocupam as quatro primeiras posições tanto na série quanto em 2025. O pior é sempre SBBE, e SBSP fica em 19º nos dois recortes.

**Tendências notáveis:**
- **Cuiabá (SBCY)** teve a maior recuperação. Foi o pior aeroporto em 2023 (3,94 no ano; 3,80 em 2023T3) e chegou a 4,55 em 2025 (5º).
- **Recife (SBRF)** subiu +0,05 ponto/ano, de 18º na série para 15º em 2025.
- **Brasília (SBBR)** subiu de 10º para 6º.
- **Belém (SBBE)** caiu: 3,91 em 2025, com vale de cerca de 3,69 em 2025T2–T3, e se recuperou para cerca de 4,4 em 2026. É o único aeroporto abaixo de 4.
- **Viracopos (SBKP)** caiu de 6º para 12º (−0,14 de 2022 para 2025).
- Pelo critério de ±0,02 ponto/ano:
  - "em alta": SBCY, SBRF e SBBR;
  - "em queda": SBKP e SBBE;
  - estáveis: os demais, com SBCT (−0,020) no limite.
- O Guarulhos (MVP) está em 17º na série e 18º em 2025. Melhorou em nível (+0,11), mas é estável na tendência (+0,017).

**Regiões (média por ano):**

| Região | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| Sul | 4,631 | 4,619 | 4,599 | 4,610 | 4,612 | 4,616 |
| Centro-Oeste | 4,351 | 4,392 | 4,248 | 4,439 | 4,514 | 4,568 |
| Nordeste | 4,467 | 4,423 | 4,370 | 4,407 | 4,473 | 4,490 |
| Sudeste | 4,430 | 4,409 | 4,367 | 4,413 | 4,430 | 4,429 |
| Norte | 4,351 | 4,313 | 4,266 | 4,274 | 4,203 | 4,427 |

Leitura:
- O Sul lidera todos os anos.
- O Centro-Oeste saiu do último lugar em 2023 para o 2º em 2025–26, puxado por SBCY e SBBR.
- O Norte é o último em 2025 por causa de Belém.

**Regime de gestão por período** (regime vigente em cada trimestre; tabela completa em `codigos/06_visualizacoes/panorama_regime_periodo.csv`):

| Período | Concessionados (n aeroportos, média) | Infraero (n, média) | Misto |
|---|---|---|---|
| 2021T1 | 14 · 4,443 | 6 · 4,454 | — |
| 2022T1 | 14 · 4,453 | 3 · 4,261 | SBCT/SBEG/SBGO · 4,465 |
| 2023T3 | 17 · 4,415 | 2 · 4,132 | SBBE · 4,066 |
| 2023T4 | 18 · 4,378 | SBRJ · 4,255 | SBSP · 4,166 |
| 2024T1 | 19 · 4,418 | SBRJ · 4,330 | — |
| 2025 (T1–T4) | 19 · 4,448–4,486 | SBRJ · 4,327–4,378 | — |
| 2026T2 | 19 · 4,511 | SBRJ · 4,299 | — |

Leitura:
- Em 2021T1, com 6 aeroportos Infraero, os dois grupos estavam empatados.
- Depois das 6ª/7ª rodadas, o grupo Infraero passou a ser formado só pelos aeroportos de pior avaliação: SBBE, SBSP e SBRJ. Desde 2024 é só o SBRJ.
- A diferença entre os grupos reflete **mudança de composição**, não efeito causal da concessão. A tese deve apresentá-la como descritiva; o DiD fica com a econometria.

**Atributos com maior variação entre aeroportos** (heatmap, top 20): encontrar vagas, quantidade de opções de alimentação, de guichês e de comércio, instalações de estacionamento e tempo de restituição de bagagens, entre outros.

## 4. Validação feita

- Prompts em `--dry-run` revisados para SBGR, SBSG (operador Zurich), SBRJ (Infraero), SBPA (cautela das enchentes) e SBSP (concessão em 2023T4).
- AppTest do Streamlit com SBGR, SBPA, SBRJ, SBSG e SBBE: sem exceções e com 5 abas.
- App headless na porta 8611: health ok e capturas conferidas. O processo foi encerrado e o Ollama não foi iniciado.

## 5. Pontos para revisão

1. **Pasta do VRA 2025:** `dados/brutos/vra_2025/` segue a instrução, mas o de 2023 está em `dados/brutos/anac/vra_2023/`. O mapeamento por ano está no `VRA_DIR` do 04.
2. **SBSG:** a concessão (Zurich desde 19/02/2024) foi sobrescrita no código do 04 (`ATUALIZACAO_CONCESSAO`). Outros eventos de concessão posteriores a 2024 não foram levantados, mas o regime por período vem do arquivo da série.
3. **`00_coletar_dados_contexto.py`:** o `.tex` agora só é reescrito com `--tex`. A checagem de movimentações por operador continua só para 2023.
4. **Linhas vazias na base:** `mpor_2021_2026_20.parquet` tem **610 linhas de 2022T2 sem AEROPORTO**, 349 delas só com comentário. O panorama as descarta; vale avisar quem gerou a base.
5. **Rankings e tendência:** o ranking principal é pela média da série, com 2025 como secundário (◆ na figura e figura própria). O limiar de tendência é 0,02 ponto/ano.
6. **Figuras:** `panorama_por_regiao` deixou de ser boxplot e virou linhas região × ano. Há duas figuras novas (`panorama_ranking_2025` e `panorama_regime_periodo`) que o `texto/` ainda não referencia.
7. **Comparação por regime:** é descritiva e confundida pela composição dos grupos (ver 3.2).
8. **Pendências:**
   - os scripts 02/03 de `05_interpretacao` e o `app/README.md` seguem desatualizados;
   - o `capturar_telas.py` não foi rodado;
   - o lote de 20 relatórios LLM ainda precisa rodar com `--refazer`, depois do ranking SHAP novo.
