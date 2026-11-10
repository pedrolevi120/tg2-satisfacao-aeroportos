# Modelagem na série 2021T1–2026T2: notas (06_modelagem_20, 09_experimentos, validação temporal)

*08/11/2026. Adaptação do pipeline de ML (antes só 2023) à série inteira, seguindo `ESPEC.md`.
Os resultados antigos (só 2023) estão em `versao_enviada_prof/codigos_e_resultados_2023.tar.gz`.*

## Resumo

- **Modelo selecionado: XGBoost**, como em 2023, pelo MAE de validação.
- **Desempenho no teste:**
  - MAE 0,348 [0,345; 0,350], contra 0,352 em 2023;
  - R² 0,478 [0,470; 0,485], contra 0,482;
  - QWK 0,624.
- Os 4 *ensembles* continuam empatados: diferença de no máximo 0,0025 de MAE.
- **O top-4 é o mesmo conjunto de 2023:** Localização, Limpeza, Conforto e Restituição.
  - Na série, Conforto (0,061) passa à frente de Restituição (0,054). Em 2023 a ordem era Restituição > Conforto.
  - Isso é **efeito de composição da amostra**, não mudança de prioridade. Em 2021–22, 71–72% das entrevistas são de embarque, contra ~50% de 2023 em diante. Dentro de cada ramo, a ordem das dimensões é idêntica em todos os anos (ver "Ranking por ano").
  - Nas linhas de 2023–2026, a ordem volta a ser a de 2023 (Restituição em 3º).
- **Por aeroporto:** Localização é 1º e Limpeza é 2º nos **20/20** aeroportos. O conjunto top-4 é igual ao global nos 20.
- **Validação temporal:** com treino em 2021–2024 e teste em 2025–2026T2, o MAE piora só 0,005 [0,004; 0,005] em relação ao split aleatório nas mesmas linhas (0,345 contra 0,340). O R² vai de 0,470 para 0,464. O top-4 é o mesmo conjunto (ρ = 0,995).
- **E1–E7:** nenhuma alternativa passa a régua de 0,003 de MAE, exceto o enquadramento ordinal (−0,007, só no MAE), como em 2023. O top-4 é o mesmo conjunto em 100% das reamostragens, nos 20 aeroportos e em 10 modelos.

## NÚMEROS-CHAVE (rodada oficial de 08/11/2026)

### Amostra

- **n válido: 461.593 respondentes** com nota de satisfação geral. São 462.616 linhas na base; saem 610 linhas vazias (sem aeroporto) e 413 sem nota válida.
- **X:** 140 colunas, assim divididas:
  - 50 atributos Likert, os mesmos de 2023;
  - 80 dummies: aeroporto, 21 períodos, perfil e ramo;
  - 10 de contexto: 7 numéricas e 3 dummies de regime.
- **Split 70/15/15:** treino 323.115 / validação 69.239 / teste 69.239.

| Ano | n válido | treino | validação | teste |
|---|---|---|---|---|
| 2021 | 44.536 | 31.143 | 6.676 | 6.717 |
| 2022 | 82.821 | 57.804 | 12.628 | 12.389 |
| 2023 | 84.662 | 59.432 | 12.726 | 12.504 |
| 2024 | 99.204 | 69.507 | 14.671 | 15.026 |
| 2025 | 103.166 | 72.149 | 15.480 | 15.537 |
| 2026 (T1–T2) | 47.204 | 33.080 | 7.058 | 7.066 |

A parcela de embarque é 71,6% em 2021, 70,8% em 2022 e 49,3–50,1% de 2023 a 2026.

### Os 9 modelos (teste, n = 69.239; IC 95% *bootstrap* de 1.000 reamostragens; ΔMAE pareado)

| Modelo | MAE val | MAE teste [IC 95%] | RMSE teste [IC] | R² teste [IC] | QWK teste [IC] | ΔMAE vs XGBoost [IC] | tempo de ajuste (s) |
|---|---|---|---|---|---|---|---|
| Baseline (moda) | 0,597 | 0,597 [0,593; 0,601] | 0,794 [0,791; 0,798] | −0,425 [−0,440; −0,412] | 0,000 | +0,2489 [+0,2447; +0,2531] | 1 |
| Ridge | 0,364 | 0,365 [0,362; 0,367] | 0,489 [0,484; 0,493] | 0,460 [0,452; 0,467] | 0,615 [0,610; 0,621] | +0,0169 [+0,0162; +0,0175] | 5 |
| KNN (k=10) | 0,400 | 0,399 [0,396; 0,401] | 0,535 [0,530; 0,539] | 0,353 [0,346; 0,360] | 0,506 [0,500; 0,513] | +0,0510 [+0,0495; +0,0527] | 8 |
| Árvore de decisão | 0,363 | 0,362 [0,360; 0,365] | 0,507 [0,503; 0,512] | 0,419 [0,410; 0,427] | 0,588 [0,581; 0,594] | +0,0147 [+0,0136; +0,0157] | 14 |
| MLP | 0,363 | 0,363 [0,360; 0,366] | 0,499 [0,494; 0,504] | 0,438 [0,428; 0,448] | 0,608 [0,602; 0,614] | +0,0155 [+0,0145; +0,0166] | 103 |
| Random Forest | 0,351 | 0,350 [0,348; 0,353] | 0,489 [0,484; 0,493] | 0,460 [0,453; 0,468] | 0,609 [0,603; 0,615] | +0,0025 [+0,0018; +0,0031] | 370 |
| LightGBM | 0,349 | 0,348 [0,346; 0,351] | 0,481 [0,476; 0,485] | 0,478 [0,470; 0,485] | 0,623 [0,616; 0,629] | +0,0007 [+0,0004; +0,0010] | 11 |
| **XGBoost (selecionado)** | **0,348** | **0,348 [0,345; 0,350]** | 0,481 [0,476; 0,485] | **0,478 [0,470; 0,485]** | 0,624 [0,618; 0,630] | – | 22 |
| CatBoost | 0,350 | 0,350 [0,347; 0,352] | 0,481 [0,476; 0,485] | 0,477 [0,470; 0,485] | 0,621 [0,614; 0,627] | +0,0017 [+0,0014; +0,0021] | 21 |

- Valores com 4 casas: XGBoost MAE 0,3478, R² 0,4775 e QWK 0,6237.
- Nenhum modelo foi subamostrado. KNN e MLP usaram todas as linhas e couberam no tempo.

**XGBoost no teste, por ano** (`metricas_por_ano_20.csv`):

| Ano | n teste | média y | MAE | RMSE | R² | QWK |
|---|---|---|---|---|---|---|
| 2021 | 6.717 | 4,446 | 0,354 | 0,491 | 0,472 | 0,612 |
| 2022 | 12.389 | 4,436 | 0,354 | 0,484 | 0,479 | 0,623 |
| 2023 | 12.504 | 4,375 | 0,350 | 0,488 | 0,497 | 0,640 |
| 2024 | 15.026 | 4,423 | 0,349 | 0,477 | 0,467 | 0,616 |
| 2025 | 15.537 | 4,464 | 0,342 | 0,477 | 0,476 | 0,627 |
| 2026 | 7.066 | 4,479 | 0,336 | 0,472 | 0,454 | 0,604 |

**Comparação com 2023 (versão enviada):**
- 2023, teste n = 12.700: MAE 0,352 e R² 0,482.
- Série inteira: MAE 0,348 e R² 0,478.
- Nas linhas de teste de 2023 do modelo da série: MAE 0,350 e R² 0,497.

### LOAO (20 dobras; `loao_20.csv`)

| Métrica (média; mín–máx entre aeroportos) | XGBoost | Ridge |
|---|---|---|
| MAE fora do aeroporto (LOAO) | 0,352 (0,269–0,446) | 0,367 (0,284–0,462) |
| MAE dentro (split aleatório, mesmo aeroporto) | 0,347 (0,250–0,436) | 0,364 (0,267–0,449) |
| ΔMAE LOAO − dentro | +0,005 (−0,004 a +0,022) | +0,003 (−0,008 a +0,023) |
| R² LOAO / dentro | 0,435 / 0,447 | 0,419 / 0,429 |
| QWK LOAO / dentro | 0,593 / 0,599 | 0,589 / 0,590 |

- Com o XGBoost, o MAE fora do aeroporto é maior que o de dentro em 14 dos 20 aeroportos.
- Os aeroportos que mais perdem são Maceió (+0,022), Florianópolis (+0,020) e Salvador (+0,011).
- O maior viés de nível é em Belém: +0,14, ou seja, o modelo superestima a satisfação de Belém quando não a vê.

### SHAP (teste, XGBoost por ramo)

- Os atributos de serviço (Likert) respondem por **81,6%** do |SHAP| total. O resto vem de perfil, aeroporto, período e contexto.

| Dimensão | média de \|SHAP\| (série) | 2023 (versão enviada) |
|---|---|---|
| 1. Localização e deslocamento | 0,164 | 0,180 |
| 2. Limpeza geral | 0,087 | 0,091 |
| 3. Conforto da sala de embarque | 0,061 | 0,059 (4º) |
| 4. Restituição de bagagens | 0,054 | 0,063 (3º) |
| 5. Tomadas | 0,029 | |
| 6. Sanitários | 0,025 | |
| 7. Alimentação | 0,022 | |
| 8. Desembarque no meio-fio | 0,018 | |
| 9. Inspeção de segurança | 0,017 | |
| 10. Aquisição da passagem | 0,011 | |

**Por aeroporto** (`shap_resumo_20.json`):
- Localização é 1º em **20/20**, Limpeza é 2º em **20/20**, e o conjunto top-4 é igual ao global em **20/20**.
- A ordem 3º × 4º varia: Restituição fica em 3º em Confins, Fortaleza, Galeão, Guarulhos e Congonhas, e Conforto nos outros 15.
- Estabilidade do top-5 por aeroporto em 500 reamostragens do teste: média 98,8%, mínimo 90,3% (Viracopos).

### Ranking por ano (`ranking_por_ano_20.csv`, figura `shap_ranking_por_ano.pdf`)

| Ano | n teste | top-4 do ano | mesmo conjunto que o global? | % das reamostragens com o conjunto global | ρ vs global |
|---|---|---|---|---|---|
| 2021 | 6.717 | Loc > Limp > Conf > **Tomadas** (Rest 5º) | não | 29% | 0,993 |
| 2022 | 12.389 | Loc > Limp > Conf > **Tomadas** (Rest 5º) | não | 0% | 0,993 |
| 2023 | 12.504 | Loc > Limp > Rest > Conf | sim | 100% | 0,998 |
| 2024 | 15.026 | Loc > Limp > Rest > Conf | sim | 100% | 0,993 |
| 2025 | 15.537 | Loc > Limp > Rest > Conf | sim | 100% | 0,986 |
| 2026 | 7.066 | Loc > Limp > Rest > Conf | sim | 100% | 0,989 |

- **Localização é 1º e Limpeza é 2º em todos os anos.**
- Em 2021–22, Restituição cai para 5º e Tomadas sobe para 4º. A causa é a composição: em 2021–22, só 28–29% das entrevistas são de desembarque, contra ~50% depois. A média global de |SHAP| conta zero para as linhas do outro ramo.
- **Dentro de cada ramo, a ordem é a mesma em todos os anos:**
  - Embarque: Limpeza > Conforto > Localização em 6/6 anos;
  - Desembarque: Localização > Restituição > Controle migratório > Controle aduaneiro em 6/6 anos.
  - Os valores estão no `ranking_por_ano_20.csv`, escopo `ramo_*`, e em `ranking_por_ano_ramo` no json.
- **Mudança de nível.** Em Localização, |SHAP| sobe de 0,132 (2021–22) para 0,172–0,179 (2023+). Em Limpeza, cai de 0,106–0,109 para 0,077–0,081. Dentro dos ramos, os níveis são quase constantes: Localização no desembarque fica em 0,272–0,283, e Limpeza no embarque em 0,150–0,161. Logo, a variação é composição, não mudança de importância.
- **Aeroporto × ano** (n ≥ 200 no teste; 120 combinações):
  - Localização é 1º em 115 delas e Limpeza é 2º em 112.
  - As 5 exceções de 1º lugar estão todas em 2021–22: Cuiabá, Natal e Vitória (Limpeza 1º) e Santos Dumont (Conforto 1º) em 2021, e Manaus em 2022.
  - De 2023 a 2026, Localização é 1º em 20/20 aeroportos em todos os anos.

### Validação temporal (`10_serie_historica/resultados/validacao_temporal.csv/.json`)

- **Divisão das linhas:**
  - treino: 2021–2024, linhas de treino e teste do split, n = 264.522;
  - *early stopping*: 2021–2024, linhas de validação, n = 46.701;
  - holdout: 2025–2026T2, n = 150.370.
- Árvores no *early stopping*: 377 no embarque e 140 no desembarque.

| Cenário | n | MAE [IC 95%] | RMSE | R² [IC 95%] | QWK | viés médio |
|---|---|---|---|---|---|---|
| (a) split aleatório, teste completo | 69.239 | 0,348 [0,345; 0,350] | 0,481 | 0,478 [0,470; 0,485] | 0,624 | +0,003 |
| (b) split aleatório, teste só 2025–26 | 22.603 | 0,340 [0,336; 0,345] | 0,475 | 0,470 [0,455; 0,483] | 0,620 | +0,001 |
| (c) temporal, holdout 2025–26 completo | 150.370 | 0,342 [0,341; 0,344] | 0,474 | 0,471 [0,466; 0,476] | 0,617 | −0,010 |
| (c) só 2025 | 103.166 | 0,344 [0,342; 0,346] | 0,477 | 0,478 [0,471; 0,484] | 0,623 | −0,008 |
| (c) só 2026T1–T2 | 47.204 | 0,338 [0,335; 0,341] | 0,469 | 0,454 [0,445; 0,463] | 0,603 | −0,015 |
| (d) temporal, nas mesmas linhas de (b) | 22.603 | 0,345 [0,341; 0,349] | 0,478 | 0,464 [0,451; 0,478] | 0,611 | −0,009 |

- **ΔMAE pareado (d) − (b) = +0,0048 [+0,0040; +0,0055].** Prever o futuro custa meio centésimo de ponto de MAE. O modelo subestima levemente a satisfação de 2025–26 (viés −0,01), coerente com a média subindo de 4,43 para 4,46–4,49.
- **Top-4 no holdout:** Localização (0,174) > Limpeza (0,085) > Restituição (0,060) > Conforto (0,053).
  - É o mesmo conjunto que o global, numa ordem diferente, por composição.
  - É idêntico, na mesma ordem, ao do split aleatório nas linhas de 2025–26.
  - ρ = 0,995 contra o global e 0,998 contra o aleatório 2025–26.
- **Por aeroporto no holdout:** top-1, top-2 e conjunto top-4 iguais ao global em 20, 20 e 20 aeroportos.

### Experimentos E1–E7 (manchetes; detalhes em `09_experimentos/RELATORIO.md`)

- **E1** (Optuna, 25 tentativas por modelo, CV K = 3 no treino):
  - XGBoost ajustado: MAE 0,3453, ΔMAE −0,0025 [−0,0028; −0,0022];
  - LightGBM ajustado: ΔMAE −0,0024;
  - CatBoost ajustado: ΔMAE +0,0006;
  - média dos 3 ajustados: ΔMAE −0,0020;
  - em todos, R² e QWK ficam iguais. Tudo abaixo da régua de 0,003.
- **E2 (EBM):**
  - só efeitos principais: ΔMAE +0,0138, R² 0,464;
  - com 20 interações: ΔMAE +0,0073, R² 0,471;
  - mesmo top-4 (conjunto) global, ρ = 0,97 com o SHAP-XGBoost, 19/20 aeroportos com o mesmo top-4.
- **E3 (ordinal):**
  - ordinal cumulativo, valor esperado: MAE 0,3410, **ΔMAE −0,0067 [−0,0071; −0,0064]**, R² +0,0015, QWK +0,0009;
  - multiclasse: ΔMAE −0,0065;
  - cortes que maximizam o QWK: QWK 0,624 → 0,651, um ganho da regra de arredondamento;
  - com nota inteira, acerta a nota exata em 75,0% e erra por no máximo 1 ponto em 98,9%.
- **E4 (perfil e operacionais):**
  - ΔMAE −0,0005 (+33 variáveis) e −0,0008 (+42), ambos irrelevantes; top-8 idêntico e top-4 igual em 20/20 aeroportos.
  - A variável nova mais usada é o **mês da série** (`O_mes_serie`, |SHAP| 0,020). Ela capta a tendência temporal, mas não melhora a previsão. Em seguida vem a frequência de viagens (0,015–0,017).
- **E5 (estabilidade):**
  - top-4 = mesmo conjunto em **100%** das reamostragens de teste, dos 30 re-treinos *bootstrap*, dos 10 de semente e das ~1.000 combinações treino × teste;
  - **100% em cada um dos 20 aeroportos**; Localização 1º e Limpeza 2º em 100%; globalmente, Conforto 3º e Restituição 4º em 100%.
  - **Por aeroporto (% com Conforto à frente de Restituição):**
    - Restituição firme em 3º: Congonhas 4% e Guarulhos 5%;
    - ordem incerta (10–90%): Fortaleza 26%, Galeão 35%, Confins 44%, Goiânia 73% e Porto Alegre 80%;
    - Conforto firme em 3º (≥ 90%): os outros 13.
  - Variação do MAE: ±0,0002 por semente e ±0,0003 por *bootstrap*.
  - Multiplicidade: 10 modelos dão o mesmo conjunto top-4 (ρ ≥ 0,965 com o XGBoost).
- **E6 (Kano):**
  - MQO com n = 461.593, R² = 0,426, 418 clusters aeroporto × período. Há 8,3% de respondentes retilíneos.
  - **Desempenho:** Localização (P 0,36; R 0,51), Limpeza (0,31; 0,41) e Restituição (0,44; 0,60). Conforto fica entre desempenho e básico.
  - **Básicos:** Inspeção de segurança (IA −0,24), Controle migratório (−0,33) e Método de desembarque (−0,24).
  - Controle aduaneiro: desempenho ~ básico (IA −0,19, IC até +0,05). Em 2023 era básico.
  - **Encantamento:** Aquisição da passagem (+0,63) e Tomadas (+0,32; desempenho por ponto de escala).
  - Mudança em relação a 2023: Comércio, que era encantamento, passou a desempenho (IA +0,17).
- **E7:**
  - maior ganho: 0,007 de MAE (ordinal);
  - nenhuma variante melhora R² em mais de 0,0025 nem QWK em mais de 0,002 (sem contar a regra de cortes).

### Tempos da rodada oficial (`10_serie_historica/resultados/tempos_execucao.txt`)

A máquina estava com carga de 15 a 38, por causa de outras sessões em paralelo.

| Etapa | tempo |
|---|---|
| 01 carregar | 1 s |
| 03 base de features | 29 s |
| 04 nove modelos + IC | 16 min |
| 05 LOAO | 11 min |
| 06 SHAP | 3 min |
| validação temporal | 4 min |
| E1 (25 tentativas × 3 modelos) | 1 h 57 min |
| E2 EBM | 33 min |
| E3 | 4 min |
| E4 | 6 min |
| E5 | 1 h 23 min |
| E6 | 30 s |
| E7 | 2 min |
| **Total (11:50 → 16:30)** | **4 h 40 min** |

O 06 foi rodado de novo às 16:32 (88 s), só para acrescentar o ranking por ramo. O SHAP é determinístico, então os números não mudam.

## O que mudou, arquivo por arquivo

### `codigos/06_modelagem_20/`

- **`comum.py`**
  - Novas constantes: `BASE_MPOR` (`mpor_2021_2026_20.parquet`), `CONTEXTO_SERIE`, `SPLITS`, `PROC_OUT`, `PNG_DIR`, `INTERP_DIR`.
  - `SAIDA_TESTE=<dir>` redireciona **todas** as saídas (splits, base unificada, SHAP, modelos, resultados, figuras, `.tex`, ranking do app) para `<dir>`. Serve para testar em amostra sem sobrescrever os artefatos que outros pacotes leem.
  - Nova `bootstrap_medias()`: bootstrap de médias em blocos de 100 reamostragens (memória).
- **`01_carregar_mpor_20.py`**: virou invólucro fino. Se `mpor_2021_2026_20.parquet` existe, só confere e resume (linhas por ano e por aeroporto × ano). Se não existe (ou com `RECARREGAR=1`), roda `10_serie_historica/01_carregar_serie.py`. A base de 2023 (`mpor_2023_20.parquet`) continua sendo a referência de colunas do carregador; se faltar, é recriada dos Excels de 2023.
- **`03_base_features_20.py`**
  - Lê a série. Descarta 610 linhas sem `AEROPORTO` (linhas vazias do portal, nenhuma com nota válida).
  - Contexto: junção por `AEROPORTO` + `PERIODO` com `contexto_aeroportos_20_serie.csv` (`validate="many_to_one"`). Entram no X as 7 numéricas pedidas (`CTX_frac_dias_concessao`, `CTX_pax_trim_mi`, `CTX_decolagens_trim_k`, `CTX_pistas`, `CTX_populacao_2022_polo_mi`, `CTX_pib_per_capita_polo_k`, `CTX_idh_2010_polo`) e as 3 dummies de regime (concessionada, infraero, misto). `CTX_regime_gestao` e `CTX_operador` ficam no `meta`.
  - Categóricas: `PERIODO` no lugar de `TRIMESTRE` (21 períodos → 21 dummies).
  - `meta` ganhou `ANO`, `TRIMESTRE`, `PERIODO`, `CHAVE_SERIE`. `feature_metadata.json` ganhou `n_por_ano` e `n_por_periodo`.
  - Split idêntico (70/15/15, estratificado por aeroporto × nota, semente 42). Lógica de ausência estrutural, ramos e codificações sem mudança.
  - **Duas correções na detecção de atributos Likert (decisão a revisar):**
    1. `NA/NR/NS` e `NA/NR/NA` (15 ocorrências em outros anos) passaram a contar como NS/NR. Sem isso, `PROCESSO DE AQUISIÇÃO DA PASSAGEM` deixava de ser reconhecido como Likert.
    2. A cobertura mínima foi de 0,5% para **0,45%**. Na série, os 4 subitens de Estacionamento caem para 0,46–0,47% (cerca de 2,1 mil respostas cada, contra ~500 em 2023). Já os 4 subitens do Controle migratório, excluídos em 2023 com 0,47%, ficam em 0,43%. Com 0,45%, o X tem **exatamente os mesmos 50 atributos de 2023**.
  - `AMOSTRA_FRAC=0.1` subamostra as linhas (para testes).
- **`04_comparar_modelos_20.py`**
  - A diferença pareada de MAE (1.000 reamostragens) é calculada em blocos de 100 via `bootstrap_medias`. A matriz 1.000 × 69 mil de uma vez ocuparia ~0,5 GB.
  - `predicoes_20.parquet` ganhou `ANO` e `PERIODO`.
  - Novo `resultados/metricas_por_ano_20.csv`: métricas do modelo selecionado no teste, por ano.
  - Títulos e legendas trazem "2021–2026". Os 9 modelos, os hiperparâmetros, o IC e a seleção por MAE de validação não mudaram. Nenhum modelo foi subamostrado.
- **`05_loao_20.py`**: só caminhos e títulos. As dummies de `PERIODO` permanecem no LOAO (só as de aeroporto saem).
- **`06_shap_20.py`**
  - `shap_values_20.parquet` ganhou `ANO`, `PERIODO` e `CHAVE_SERIE`.
  - `ranking_prioridades_20.csv` (lido pelo app e pelo NLP) tem as mesmas colunas de antes.
  - **Novo:**
    - Ranking por ano em `resultados/ranking_por_ano_20.csv`, com colunas `escopo` (global/aeroporto), `aeroporto`, `ANO`, `rank`, `dimensao`, `nome`, `mean_abs_shap` e `n_teste`. Aeroporto × ano só entra com pelo menos `N_MIN_ANO` = 200 linhas de teste.
    - Resumo em `shap_resumo_20.json`, nas chaves `ranking_por_ano` e `ranking_aeroporto_ano`: top-4 de cada ano, se é o mesmo conjunto e a mesma ordem do global, % de 500 reamostragens do teste do ano com o mesmo conjunto, posições e |SHAP| das 6 primeiras e ρ de Spearman com o ranking global.
    - Também no json: `global_top4`, `n_aeroportos_top1_global`, `n_aeroportos_top2_global` e `n_aeroportos_top4_conjunto_global`.
    - **Ranking por ano dentro de cada ramo:** linhas com `escopo` = `ramo_Embarque` e `ramo_Desembarque` no mesmo CSV, e chave `ranking_por_ano_ramo` no json. Separa mudança de prioridade de mudança de composição embarque/desembarque.
    - Figura `midias/figuras/shap_ranking_por_ano.pdf` e cópia `apresentacoes/img_png/shap_ranking_por_ano-1.png`. Painel esquerdo: posição por ano. Painel direito: |SHAP| médio por ano. O rodapé traz a parcela de embarque por ano e a estabilidade dentro dos ramos.
- **`rodar_tudo.sh`**
  - Não roda mais o `02_robustez_mvp.py`, que é de outro pacote.
  - Roda a validação temporal no fim.
  - Contorna o libomp (ver Problemas).
- `modelos.py`: sem mudança.

### `codigos/09_experimentos/`

- **`exp_comum.py`**
  - Saiu o assert `(59262, 12700, 12700)`. Agora confere os tamanhos contra o `feature_metadata.json`.
  - `TOP4` é lido de `06_modelagem_20/resultados/ranking_global_20.csv` (as 4 primeiras dimensões).
  - `MODELO_REF` é lido de `modelo_escolhido_20.json`: avisa se não for o XGBoost.
  - Caminhos vêm do `comum.py` (respeita `SAIDA_TESTE`).
  - Novo `fmt_n()`.
- **`e1`**: só a docstring. Rodado com `N_TRIALS=25` (o padrão do script continua 50) para caber no tempo. O resto do protocolo é igual.
- **`e2`**: o rodapé da figura calcula n e o número de dimensões.
- **`e4`**
  - `PERIODO` no lugar de `TRIMESTRE` na lista do que já é usado.
  - O "mês" virou **mês da série** (`O_mes_serie` = (ano − 2021)·12 + mês), para não misturar anos.
- **`e5`**
  - A figura usa os nomes da 3ª e da 4ª dimensões do TOP4 calculado. O "≥ 99,8%" do título agora é calculado (mínimo entre aeroportos).
  - As colunas `pct_restituicao_a_frente_conforto_*` mantêm o nome, mas significam "3ª do top-4 à frente da 4ª". Na série, isso é **Conforto à frente de Restituição**: o nome da coluna está invertido em relação ao conteúdo.
  - Título: quando o mínimo entre aeroportos é 100%, escreve "100%" e não "≥ 100,0%". O PNG da rodada oficial foi gerado antes dessa correção. Só a faixa do título foi regravada com PIL, na mesma fonte DejaVu Sans Bold, para não rodar de novo 1 h 23 min de E5. O conteúdo da figura não mudou.
- **`e6`**: efeitos fixos de `PERIODO` (antes `TRIMESTRE`), cluster `AEROPORTO_PERIODO`, n calculado. O título só afirma "fatores de desempenho" se as 4 primeiras dimensões forem de desempenho.
- **`e7`**: o MAE de referência, o n e o maior ganho (título) são calculados. O rodapé traz ΔMAE/ΔR²/ΔQWK do ordinal.
- **`rodar_tudo.sh`**
  - Usa `N_TRIALS=25` por padrão.
  - Apaga o cache do E5 antes e depois.
  - Contorna o libomp.
- Antes da rodada completa, os resultados antigos de E1–E6 (`e1_*`…`e6_*`, `sintese_previsao.csv`) foram apagados, para nada de 2023 ser reaproveitado em silêncio. O E5 lê `e1_params_*.json` e o `e2_ebm_gam.pkl`, e o E7 lê as predições de todos.

### `codigos/10_serie_historica/03_validacao_temporal.py` (novo)

- Mesma configuração do modelo selecionado (mesmas features, um modelo por ramo).
- Linhas por papel:
  - **treino:** linhas de 2021–2024 que são treino ou teste no split aleatório;
  - ***early stopping*:** linhas de 2021–2024 que são validação no split aleatório;
  - **holdout:** todas as linhas de 2025–2026T2.
- Cenários de avaliação:
  - (a) teste aleatório completo;
  - (b) teste aleatório só de 2025–26;
  - (c) holdout temporal completo e por ano;
  - (d) holdout temporal nas mesmas linhas de (b), com ΔMAE pareado.
- MAE/RMSE/R²/QWK com IC *bootstrap* de 1.000 reamostragens.
- TreeSHAP por dimensão no holdout. O top-4 é comparado com o global do pipeline e com o do modelo oficial nas linhas de teste de 2025–26.
- Saída: `resultados/validacao_temporal.csv` e `.json`.

## Como rodar

```bash
cd tg2-satisfacao-aeroportos   # raiz do repositório
bash codigos/06_modelagem_20/rodar_tudo.sh                  # 01→03→04→05→06 + validação temporal
N_JOBS=4 N_TRIALS=25 bash codigos/09_experimentos/rodar_tudo.sh   # E1→E7
# teste rápido em 10% das linhas, sem tocar nos artefatos oficiais:
AMOSTRA_FRAC=0.1 SAIDA_TESTE=/tmp/teste_tg bash codigos/06_modelagem_20/rodar_tudo.sh
```

A rodada oficial desta sessão usou o mesmo encadeamento, num único script sequencial. Os tempos estão em `resultados/tempos_execucao.txt`.

## Problemas encontrados

- **libomp quebrado no Homebrew.** O link `/opt/homebrew/opt/libomp` sumiu: o pacote está só em `/opt/homebrew/Cellar/libomp/22.1.5`. Por isso xgboost e lightgbm não carregavam (`Library not loaded: @rpath/libomp.dylib`).
  - Os dois `rodar_tudo.sh` exportam `DYLD_LIBRARY_PATH` para o Cellar quando o link falta.
  - Rodar os scripts direto com `python` falha até alguém fazer `brew link --force libomp` ou `brew reinstall libomp`. Não mexi no Homebrew.
  - Atenção: binários do sistema (por exemplo, `/usr/bin/time`) apagam o `DYLD_LIBRARY_PATH`.
- A máquina estava com carga de 13–38 (outras sessões), então os tempos acima são piores do que seriam com a máquina livre.

## Decisões para o Pedro revisar

1. **Cobertura mínima de 0,45% e variantes NA/NR/NS e NA/NR/NA** (`03`). Isso mantém exatamente os 50 Likert de 2023. Sem as variantes, sairia a Aquisição da passagem. Com o corte de 0,5%, sairiam os 4 subitens de Estacionamento (cobertura de 0,46–0,47% na série).
2. **610 linhas sem `AEROPORTO` descartadas** (todas sem nota válida).
3. **`PERIODO` (21 dummies) no lugar de `TRIMESTRE`** no X.
   - Outros consumidores de `X.parquet` e de `base_unificada_20.parquet` (por exemplo, o 07) passam a ver `PERIODO_*`.
   - O LOAO mantém as dummies de período.
4. **Ordem 3º × 4º global mudou de Restituição > Conforto (2023) para Conforto > Restituição (série).**
   - A causa é composição: 71% de embarque em 2021–22.
   - Dentro dos ramos, a ordem é estável, e nos anos 2023–2026 a ordem é a de 2023.
   - Ao escrever o texto, vale dizer "o conjunto das 4 prioridades é o mesmo" e explicar a troca 3º × 4º. Alternativa: reportar o ranking por ramo ou ponderar por ano.
5. **Validação temporal:**
   - corte em 2024 (treino 2021–24, holdout 2025–26T2);
   - *early stopping* nas linhas de validação de 2021–24;
   - as linhas de teste aleatório de 2021–24 entram no treino temporal.
6. **E1 com 25 tentativas** (em 2023 foram 50). A melhor tentativa veio na 15ª (XGBoost) e na 8ª (LightGBM).
7. **E4 com "mês da série" no lugar de mês do ano.** Ele vira a variável nova mais usada, mas não melhora o MAE.
8. **Os experimentos usam `pred_xgb` como referência** (o modelo selecionado continua XGBoost; há aviso se mudar).
9. **As tabelas `.tex` em `texto/_paralelo/tex/` foram regeneradas pelo pipeline** (o `TEX_DIR` do `comum.py`). Elas não são usadas por `\input` nos `.tex` do texto principal; conferir antes de usar.
10. **libomp:** recomendo `brew reinstall libomp` (ou `brew link --force libomp`) para não depender do `DYLD_LIBRARY_PATH`.
