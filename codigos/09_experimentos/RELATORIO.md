# Experimentos de modelagem (09_experimentos): relatório — série 2021T1–2026T2

*08/11/2026. Os números são da série 2021–2026: 461.593 respondentes válidos, 20 aeroportos. A versão de 20/10 (só 2023, n = 84.662) está em `versao_enviada_prof/codigos_e_resultados_2023.tar.gz`. Os scripts só leem e importam de `codigos/06_modelagem_20/`; mudanças da adaptação estão em `codigos/10_serie_historica/notas_modelagem.md`.*

## Resumo em 6 linhas

1. **Previsão: nenhuma alternativa justifica trocar o modelo.** Testamos 13 alternativas novas contra o XGBoost do pipeline (MAE 0,348): ajuste fino, EBM, enquadramento ordinal, mais variáveis de perfil e médias de modelos. O melhor ajuste fino ganha 0,0025, abaixo da régua de 0,003.
2. **A exceção continua sendo o enquadramento ordinal**, com MAE 0,007 menor [0,006; 0,007]. R², RMSE e QWK quase não mudam, e o conjunto do top-4 é o mesmo. Isso pede uma frase de robustez, não uma troca de modelo. É o mesmo resultado de 2023.
3. **O top-4 é "certo".** É o mesmo conjunto em 100% de ~1.000 reamostragens (treino × teste) e em 100% em cada um dos 20 aeroportos. Localização é 1º e Limpeza é 2º em 100%. A ordem 3º × 4º é firme no global (Conforto à frente), mas incerta em 5 aeroportos.
4. **O ranking não depende do modelo.** Dez modelos de famílias diferentes dão o mesmo conjunto top-4 (ρ de Spearman ≥ 0,965 com o XGBoost).
5. **Kano:** Localização, Limpeza e Restituição são **fatores de desempenho**: nota ruim tira e nota ótima soma. Inspeção de segurança, Controle migratório e Método de desembarque são **fatores básicos**: só punem.
6. **Recomendação:** manter o XGBoost. Vale levar ao texto a estabilidade do top-4 (E5), a concordância entre modelos (E5) e o Kano (E6).

## Como ler as tabelas

- Todas as comparações são **pareadas** no **mesmo teste** do pipeline (n = 69.239), com as **mesmas 1.000 reamostragens *bootstrap*** (semente 42) de `04_comparar_modelos_20.py`. ΔMAE = MAE da variante − MAE do XGBoost atual. **ΔMAE < 0 significa que a variante é melhor.**
- **Régua de relevância:** |ΔMAE| ≤ 0,003 ponto é irrelevante, mesmo com IC fora do zero. Com n = 69 mil, os IC são estreitos (±0,0003 a ±0,0005 no ΔMAE), e quase qualquer diferença "é significativa".
- **Ruído de referência:** re-treinar o mesmo XGBoost trocando só a semente faz o MAE variar entre 0,3474 e 0,3480 (desvio-padrão 0,0002).
- **Mesmas condições do pipeline:**
  - features: X de 461.593 × 140;
  - split 70/15/15: 323.115 / 69.239 / 69.239;
  - um modelo por ramo do questionário, ausentes nativos, mesma semente.
- O `e4_perfil.py` re-treina o XGBoost atual com 4 *threads* e reproduz as predições salvas exatamente (Δ = 0). O E5 reproduz exatamente as do E1.
- O top-4 de referência é lido de `06_modelagem_20/resultados/ranking_global_20.csv`: Localização > Limpeza > Conforto > Restituição.
  - Em 2023 a ordem era Restituição > Conforto.
  - A troca vem da composição da amostra: em 2021–22, ~71% das entrevistas são de embarque, contra ~50% depois.
  - Dentro de cada ramo, a ordem é igual em todos os anos (`notas_modelagem.md`).

---

## E1. Ajuste de hiperparâmetros (`e1_ajuste_hiperparametros.py`)

**O que foi feito.** Usamos Optuna (TPE) com **25 tentativas por modelo** (`N_TRIALS=25`; em 2023 foram 50, mas a série tem 5,5 vezes mais linhas). A 1ª tentativa é a configuração atual, como ponto de partida. O critério foi o MAE médio numa validação cruzada K = 3 **dentro do treino**, com dobras estratificadas por aeroporto × nota.

- Em cada dobra, o *early stopping* usa a validação do pipeline.
- O modelo final é treinado com os melhores parâmetros no treino inteiro e avaliado uma única vez no teste.
- Tempo, com 4 *threads* e a máquina carregada: XGBoost 46 min, LightGBM 20 min, CatBoost 47 min.

| Modelo | MAE em CV (treino): atual → ajustado | Teste: MAE [IC 95%] | ΔMAE vs XGBoost atual [IC 95%] | R² teste | QWK teste |
|---|---|---|---|---|---|
| XGBoost ajustado | 0,3475 → 0,3453 | 0,3453 [0,3426; 0,3477] | −0,0025 [−0,0028; −0,0022] | 0,478 | 0,623 |
| LightGBM ajustado | 0,3482 → 0,3451 | 0,3454 [0,3427; 0,3478] | −0,0024 [−0,0028; −0,0020] | 0,477 | 0,623 |
| CatBoost ajustado | 0,3486 → 0,3478 | 0,3483 [0,3457; 0,3507] | +0,0006 [+0,0002; +0,0009] | 0,479 | 0,622 |
| Média dos 3 ajustados | – | 0,3458 [0,3432; 0,3482] | −0,0020 [−0,0023; −0,0017] | 0,480 | 0,623 |
| Média dos 3 atuais | – | 0,3481 [0,3455; 0,3505] | +0,0003 [+0,0001; +0,0005] | 0,479 | 0,623 |

Melhores parâmetros (em `resultados/e1_params_*.json`):

- **XGBoost:** profundidade 10, η 0,023, `min_child_weight` 7,9, `subsample` 0,81, α 8,7.
- **LightGBM:** 230 folhas, η 0,029, `min_child_samples` 117, *bagging* 0,97.
- **CatBoost:** profundidade 8, η 0,031, `l2_leaf_reg` 11,1.

Com mais dados, as árvores ótimas ficam mais profundas: XGBoost de 6 para 10. Mesmo assim, o ganho é de 2,5 milésimos de ponto. No LightGBM do pipeline, `subsample=0.8` continua sem efeito, porque falta `subsample_freq`.

**Conclusão.** Os hiperparâmetros do MVP seguem perto do ótimo: o maior ganho (−0,0025) fica abaixo da régua de 0,003. R² e QWK não mudam.

## E2. EBM, modelo interpretável por construção (`e2_ebm.py`)

**O que foi feito.** Usamos a *Explainable Boosting Machine* (`interpret` 0.7.8) em duas versões: só efeitos principais (GAM) e GAM + 20 interações de pares (GA2M).

- O esquema é o mesmo do pipeline: um modelo por ramo, ausentes nativos e 8 *bags*.
- A importância por dimensão segue a regra do SHAP: soma do item-pai e dos subitens, linha a linha.
- Tempo: cerca de 16 min por versão.

| Modelo | MAE teste [IC 95%] | ΔMAE vs XGBoost [IC 95%] | R² | QWK | ρ do ranking vs SHAP-XGBoost | Mesmo top-4? |
|---|---|---|---|---|---|---|
| EBM (só efeitos principais) | 0,3615 [0,3589; 0,3639] | +0,0138 [+0,0132; +0,0143] | 0,464 | 0,617 | 0,97 | sim, na mesma ordem; 19/20 aeroportos |
| EBM + 20 interações | 0,3551 [0,3525; 0,3574] | +0,0073 [+0,0068; +0,0078] | 0,471 | 0,620 | 0,97 | sim (3º e 4º trocados); 19/20 aeroportos |

- O aeroporto que difere continua sendo o **Galeão**: no EBM, Controle migratório entra em 4º, no lugar de Conforto, pelo peso dos voos internacionais.
- As **curvas** do EBM e do SHAP por nota do item-pai quase coincidem e são quase retas (`figuras/e2_ebm_formas_top4.png`).
  - Em Localização, a contribuição vai de −0,87 (nota 1) a +0,16 (nota 5): cerca de **0,26 ponto de satisfação prevista por ponto de nota**.
  - Em Restituição, vai de −0,92 a +0,17.

**Conclusão.** O EBM prevê pior que o XGBoost (+0,007 a +0,014), mas dá **a mesma hierarquia**. A relação é essencialmente aditiva.

## E3. Enquadramento ordinal (`e3_ordinal.py`)

**O que foi feito.** Testamos quatro formas de tratar a nota de 1 a 5, sempre com XGBoost e os mesmos hiperparâmetros:

- **(a)** regressão + 4 cortes otimizados na validação para o QWK;
- **(b)** multiclasse;
- **(c)** ordinal cumulativo (Frank & Hall, 2001), com E[y] = 1 + Σ P(y > k);
- **(d)** média entre regressão e ordinal.

| Variante | MAE [IC 95%] | ΔMAE vs atual [IC 95%] | R² | QWK | ΔQWK [IC 95%] |
|---|---|---|---|---|---|
| Regressão atual (QWK por arredondamento) | 0,3478 [0,3451; 0,3502] | – | 0,478 | 0,624 | – |
| (a) Regressão + cortes que maximizam o QWK | 0,3478 (igual) | 0 | 0,478 | **0,651** | **+0,027** [+0,024; +0,030] |
| (b) Multiclasse, valor esperado | 0,3412 [0,3386; 0,3436] | −0,0065 [−0,0069; −0,0061] | 0,478 | 0,621 | −0,003 [−0,005; −0,001] |
| (c) Ordinal cumulativo, valor esperado | **0,3410** [0,3384; 0,3435] | **−0,0067** [−0,0071; −0,0064] | 0,479 | 0,625 | +0,001 [−0,001; +0,003] |
| (c) + cortes que maximizam o QWK | 0,3410 | −0,0067 | 0,479 | 0,652 | +0,028 [+0,025; +0,031] |
| (d) Média regressão + ordinal | 0,3435 [0,3409; 0,3459] | −0,0043 [−0,0044; −0,0041] | 0,480 | 0,624 | 0,000 |

- O ganho do ordinal se repete na **validação**: 0,3415 contra 0,3482. O **RMSE fica igual** (0,4803 contra 0,4810). O ordinal reparte melhor os erros pequenos, mas não acerta melhor a média.
- O ordinal dá o mesmo **conjunto** top-4 (E5), na ordem Localização > Limpeza > Restituição > Conforto. Os SHAP foram calculados em probabilidade, em 1.000 linhas.
- **Armadilha de métrica.** Com a predição **arredondada para nota inteira**, o MAE cai para 0,262, mas o R² cai para 0,34. Esse MAE não é comparável ao 0,348.
- Com nota inteira, o modelo **acerta a nota exata em 75,0%** dos passageiros e erra por no máximo 1 ponto em **98,9%**.

**Conclusão.** Tratar a nota como ordinal reduz o MAE em 0,007, o dobro da régua, sem mudar R², RMSE, QWK nem o conjunto top-4. Os cortes otimizados sobem o QWK de 0,624 para 0,651, mas isso é uma melhoria da **regra de arredondamento**, não do modelo.

## E4. Perfil do passageiro e da viagem (`e4_perfil.py`)

**Inventário.** O modelo atual já usa gênero, faixa etária, escolaridade, renda, nacionalidade, motivo, tipo de voo, conexão, viajando sozinho, aeroporto e **período** (ano × trimestre). Faltavam:

- **perfil e viagem:** frequência de viagens, já esteve no aeroporto, antecedência, tempo de espera, acompanhantes, quem comprou a passagem, canal de compra, transporte, deficiência, estacionamento, *check-in*;
- **operacionais:** companhia aérea, hora da entrevista, **mês da série** e terminal.
  - O mês agora é contado desde jan/2021, `O_mes_serie` = (ano − 2021)·12 + mês. Assim, março de 2022 não se mistura com março de 2025.

| Variante (mesmo XGBoost) | nº de *features* | MAE [IC 95%] | ΔMAE vs atual [IC 95%] | R² | QWK | Mesmo top-4 (global; aeroportos) |
|---|---|---|---|---|---|---|
| Atual (re-treinado) | 140 | 0,3478 [0,3451; 0,3502] | 0 (reproduz exatamente) | 0,478 | 0,624 | sim; 20/20 |
| + perfil/viagem faltante | 173 | 0,3473 [0,3446; 0,3496] | −0,0005 [−0,0008; −0,0002] | 0,479 | 0,626 | sim; 20/20 |
| + perfil + operacionais | 182 | 0,3470 [0,3443; 0,3494] | −0,0008 [−0,0011; −0,0005] | 0,479 | 0,625 | sim; 20/20 |

- **Variáveis novas mais usadas:**
  - o **mês da série** (|SHAP| 0,020), que capta a tendência de alta da satisfação de 2023 a 2026;
  - a **frequência de viagens** (0,015–0,017), do tamanho de "Desembarque no meio-fio".
- Nenhuma das duas melhora a previsão: elas tomam crédito do perfil e do período já usados. O top-8 de dimensões fica idêntico.

**Conclusão.** Perfil e variáveis operacionais não acrescentam precisão nem mexem no ranking.

## E5. Estabilidade do ranking, o grau de certeza (`e5_estabilidade_ranking.py`)

**O que foi feito.** Medimos quatro fontes de incerteza:

- **(A) amostra de teste:** 1.000 reamostragens, global e por aeroporto (~3.460 linhas cada);
- **(B) amostra de treino:** 30 re-treinos *bootstrap* do treino e 10 re-treinos que só trocam a semente;
- **(B×A) combinação:** cada re-treino × 33 reamostragens do teste, ~1.000 combinações;
- **(C) modelo:** 10 modelos.

Tempo: 1 h 23 min.

| Fonte de incerteza | Top-4 = {Localização, Limpeza, Conforto, Restituição} | Localização em 1º | Limpeza em 2º | Conforto 3º / Restituição 4º |
|---|---|---|---|---|
| (A) teste, global | 100% | 100% | 100% | 100% |
| (B) re-treino *bootstrap* | 100% (30/30) | 100% | 100% | 100% |
| (B) re-treino de semente | 100% (10/10) | 100% | 100% | 100% |
| (B×A) combinado, global | **100%** | 100% | 100% | 100% |
| (B×A) por aeroporto | **100% em todos os 20** | – | – | ver abaixo |

**Por aeroporto, só a ordem 3º × 4º varia.** Porcentagem das combinações com Conforto à frente de Restituição:

- **Restituição firme em 3º (≤ 10%):** Congonhas (4%) e Guarulhos (5%);
- **ordem incerta (10–90%):** Fortaleza (26%), Galeão (35%), Confins (44%), Goiânia (73%) e Porto Alegre (80%);
- **Conforto firme em 3º (≥ 90%):** Salvador, Recife, Curitiba, Manaus, Santos Dumont, Viracopos, Brasília, Natal, Belém, Florianópolis, Cuiabá, Maceió e Vitória.

Sanitários fica em 6º em 99% das combinações; Tomadas é 5º.

Variação do MAE nos re-treinos:

- *bootstrap*: 0,3483–0,3497 (dp 0,0003);
- semente: 0,3474–0,3480 (dp 0,0002).

**(C) Multiplicidade de modelos.** Todos dão o mesmo conjunto top-4:

| Modelo | Top-4 (ordem) | ρ de Spearman vs XGBoost | τ de Kendall |
|---|---|---|---|
| XGBoost atual | Loc > Limp > Conf > Rest | 1,00 | 1,00 |
| LightGBM atual | Loc > Limp > Conf > Rest | 1,00 | 1,00 |
| CatBoost atual | Loc > Rest > Conf > Limp | 0,97 | 0,91 |
| Random Forest (1.000 linhas) | Loc > Limp > Conf > Rest | 0,99 | 0,94 |
| Ridge (SHAP linear exato) | Loc > Limp > Conf > Rest | 0,96 | 0,88 |
| EBM (aditivo) | Loc > Limp > Conf > Rest | 0,97 | 0,89 |
| XGBoost / LightGBM ajustados (E1) | Loc > Limp > Conf > Rest | 1,00 | 1,00 |
| CatBoost ajustado (E1) | Loc > Rest > Conf > Limp | 0,97 | 0,91 |
| Ordinal cumulativo (E3; 1.000 linhas) | Loc > Limp > Rest > Conf | 0,99 | 0,93 |

**Conclusão.** A certeza sobre as prioridades é ainda maior que em 2023: o conjunto top-4 é 100% estável global e por aeroporto (antes, ≥ 99,8%). O que é incerto é a ordem 3º × 4º em 5 aeroportos, e a posição de Limpeza no CatBoost.

## E6. Impacto assimétrico, Kano / *penalty-reward contrast* (`e6_kano_penalty_reward.py`)

**O que foi feito.** Uma regressão linear (MQO) com todas as **461.593** respostas.

- **Variável explicada:** satisfação geral.
- **Explicativas:** para cada uma das 19 dimensões, dummies da nota do item-pai {1–2, 4, 5, não respondido}, com **referência = nota 3**.
- **Controles:** efeitos fixos de aeroporto e de **período** (ano × trimestre), ramo e perfil.
- **Erros-padrão:** agrupados por aeroporto × período (**418 grupos**).

O modelo tem R² de 0,426. Com isso:

- **Penalidade** P = quanto a satisfação cai com nota 1–2 em vez de 3.
- **Recompensa** R = quanto sobe com nota 5 em vez de 3.
- **Índice de assimetria:** IA = (R − P)/(R + P), com classificação básico se IA < −0,2, encantamento se IA > +0,2 e desempenho no meio.

Sensibilidades: (i) efeito por ponto de escala; (ii) sem os **8,3% de "retilíneos"**.

| Dimensão | Penalidade (1–2 vs 3) [IC 95%] | Recompensa (5 vs 3) [IC 95%] | IA [IC 95%] | Por ponto: P / R | Classe (robusta) |
|---|---|---|---|---|---|
| Localização e deslocamento | −0,36 [−0,39; −0,34] | +0,51 [+0,49; +0,52] | +0,17 [+0,13; +0,20] | 0,29 / 0,25 | **desempenho** |
| Limpeza geral | −0,31 [−0,36; −0,27] | +0,41 [+0,39; +0,43] | +0,13 [+0,06; +0,20] | 0,26 / 0,20 | **desempenho** |
| Conforto da sala de embarque | −0,25 [−0,26; −0,23] | +0,25 [+0,24; +0,26] | +0,01 [−0,04; +0,05] | 0,20 / 0,12 | desempenho ~ básico |
| Restituição de bagagens | −0,44 [−0,47; −0,41] | +0,60 [+0,58; +0,61] | +0,15 [+0,11; +0,18] | 0,31 / 0,30 | **desempenho** |
| Tomadas | −0,08 [−0,09; −0,07] | +0,15 [+0,14; +0,16] | +0,32 [+0,25; +0,40] | 0,06 / 0,08 | encantamento ~ desempenho |
| Sanitários | −0,12 [−0,13; −0,10] | +0,17 [+0,16; +0,18] | +0,20 [+0,11; +0,29] | 0,09 / 0,09 | desempenho ~ encantamento |
| Alimentação | −0,16 [−0,18; −0,14] | +0,23 [+0,22; +0,24] | +0,19 [+0,12; +0,27] | 0,12 / 0,12 | desempenho ~ encantamento |
| Inspeção de segurança | −0,18 [−0,21; −0,15] | +0,11 [+0,10; +0,13] | −0,24 [−0,35; −0,13] | 0,14 / 0,06 | **básico** |
| Controle migratório | −0,42 [−0,51; −0,33] | +0,21 [+0,18; +0,24] | −0,33 [−0,45; −0,19] | 0,31 / 0,11 | **básico** |
| Controle aduaneiro | −0,46 [−0,61; −0,31] | +0,32 [+0,25; +0,38] | −0,19 [−0,38; +0,05] | 0,34 / 0,16 | desempenho ~ básico |
| Método de desembarque | −0,55 [−0,74; −0,35] | +0,34 [+0,25; +0,43] | −0,24 [−0,45; +0,04] | 0,39 / 0,17 | **básico** |
| Aquisição da passagem | −0,02 [−0,05; +0,00] | +0,11 [+0,10; +0,12] | +0,63 [+0,32; +1,00] | 0,02 / 0,05 | **encantamento** |
| Comércio | −0,12 [−0,17; −0,08] | +0,17 [+0,15; +0,19] | +0,17 [−0,00; +0,38] | 0,09 / 0,09 | desempenho |

"A ~ B" significa que a classe vira B em ao menos uma das sensibilidades. Tabela completa: `resultados/e6_kano.csv`.

**Conclusão.**

- **Localização, Limpeza e Restituição são fatores de desempenho**, como em 2023. Por ponto de escala, a nota ruim tira de 0,26 a 0,31 e a nota boa soma de 0,20 a 0,30. As curvas quase retas do EBM (E2) confirmam isso.
- Conforto tende a básico por ponto de escala.
- **Inspeção de segurança, Controle migratório e Método de desembarque são básicos.**
  - Controle aduaneiro, básico em 2023, agora fica entre desempenho e básico, com IC largo (n de notas 1–2 = 201).
- **Aquisição da passagem** é encantamento.
- **Comércio** passou de "tende a encantamento" em 2023 para desempenho na série.
- **Ressalvas:** é associação, não causalidade; a nota Likert é tratada como número; pode haver efeito halo.

---

## Síntese da previsão (`e7_sintese.py`, figura `sintese_ganhos_previsao.png`)

| Variante | MAE | ΔMAE vs XGBoost atual [IC 95%] | R² | Relevante (> 0,003 e IC fora do 0)? |
|---|---|---|---|---|
| XGBoost atual (pipeline) | 0,3478 [0,3451; 0,3502] | – | 0,478 | – |
| LightGBM / CatBoost / RF atuais | 0,3485 / 0,3495 / 0,3502 | +0,0007 / +0,0017 / +0,0025 | 0,478 / 0,477 / 0,460 | não |
| Ridge atual | 0,3647 | +0,0169 [+0,0162; +0,0175] | 0,460 | pior |
| Média dos 3 *boostings* atuais | 0,3481 | +0,0003 [+0,0001; +0,0005] | 0,479 | não |
| XGBoost / LightGBM / CatBoost ajustados | 0,3453 / 0,3454 / 0,3483 | −0,0025 / −0,0024 / +0,0006 | 0,478 / 0,477 / 0,479 | não |
| Média dos 3 ajustados | 0,3458 | −0,0020 [−0,0023; −0,0017] | 0,480 | não |
| EBM / EBM + interações | 0,3615 / 0,3551 | +0,0138 / +0,0073 | 0,464 / 0,471 | pior |
| Multiclasse (valor esperado) | 0,3412 | −0,0065 [−0,0069; −0,0061] | 0,478 | sim, só no MAE (QWK cai 0,003) |
| **Ordinal cumulativo (valor esperado)** | **0,3410** | **−0,0067 [−0,0071; −0,0064]** | 0,479 | **sim, só no MAE** (R² +0,0015; QWK +0,0009) |
| + perfil / + perfil + operacionais | 0,3473 / 0,3470 | −0,0005 / −0,0008 | 0,479 / 0,479 | não |

## (a) Recomendação: o ganho justifica trocar algum número ou modelo?

**Não.** O quadro é o mesmo de 2023.

1. **Manter o XGBoost.** Ajuste fino, EBM, perfil e *ensembles* ficam dentro de ±0,003. O melhor, o XGBoost ajustado, ganha 0,0025.
2. **Ordinal (−0,007): não trocar.** R², RMSE e QWK não mudam, e o conjunto top-4 é o mesmo. Cabe uma frase de robustez: "tratar a nota como ordinal (Frank & Hall, 2001) reduz o MAE em 0,007 [0,006; 0,007], sem alterar R², QWK nem o conjunto do *top*-4".
3. **QWK com cortes otimizados (0,651 contra 0,624): não trocar.** É regra de arredondamento, e serve como nota de rodapé.
4. **Vale acrescentar ao texto:**
   - a estabilidade do top-4 (E5, 100% global e por aeroporto) e a concordância entre 10 modelos;
   - o Kano (E6);
   - a validação temporal (`10_serie_historica`): com treino até 2024 e teste em 2025–26, o MAE piora só 0,005 e o top-4 é o mesmo.

## (b) Por que estes modelos

- **Escada do simples ao complexo.** O grande salto é do baseline ao linear: MAE de 0,597 para 0,365. Os *ensembles* ganham 0,017 sobre o linear.
- **Estado da arte em dados tabulares.** *Boosting* fica à frente de redes neurais (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022; McElfresh et al., 2023). O MLP ficou em 0,363, atrás dos *boostings*.
- **Ausência estrutural.** XGBoost, LightGBM e CatBoost tratam ausentes nativamente, o que é essencial com ramos e lógica de salto.
- **Explicação exata e barata.** O TreeSHAP é exato e aditivo, o que permite somar por dimensão (Lundberg et al., 2020).
- **Empate técnico e critério prévio.** Os quatro *ensembles* diferem em no máximo 0,0025 de MAE. O XGBoost foi escolhido pelo MAE de validação e pelo custo: 22 s contra 370 s do RF.
- **A escolha não é frágil (E1).** O ajuste fino ganha 0,0025, e a variação por semente é de ±0,0002.
- **Interpretáveis como controle (E2, E5).** Ridge e EBM preveem 0,007–0,017 pior, mas dão o mesmo top-4 (cf. Rudin, 2019).
- **Multiplicidade (efeito Rashomon).** Dez modelos concordam no conjunto top-4 (Breiman, 2001; Fisher, Rudin & Dominici, 2019).
- **Regressão como formulação mais simples (E3).** O ordinal muda o MAE em 0,007 sem mudar R², QWK ou o conjunto top-4.

## (c) Achados que valem um slide

1. **"O top-4 é certo"** (`figuras/e5_estabilidade_top4.png`): 100% das ~1.000 reamostragens, global e em cada aeroporto.
2. **"Dez modelos, a mesma resposta"** (`figuras/e5_multiplicidade_modelos.png`): ρ ≥ 0,965.
3. **"Nota ruim tira, nota ótima soma"** (`figuras/e6_kano_penalty_reward.png`): Localização, Limpeza e Restituição são de desempenho; inspeção e controles de fronteira são básicos.
4. **"O teto de previsão é dos dados, não do modelo"** (`figuras/sintese_ganhos_previsao.png`): nenhuma alternativa melhora o MAE em mais de 0,007, e o R² fica em ~0,48. O modelo acerta a nota exata de 3 em cada 4 passageiros.
5. **"Curvas quase retas"** (`figuras/e2_ebm_formas_top4.png`): ~0,26 ponto de satisfação por ponto de nota em Localização.

## Arquivos e reprodução

- **Scripts:** `exp_comum.py`, `e1_…py` a `e6_…py` e `e7_sintese.py`.
- **Ordem de execução:** `N_JOBS=4 N_TRIALS=25 bash rodar_tudo.sh` (E1 → E7). O E5 usa os parâmetros do E1 e o EBM do E2; o E7 lê todos.
- **Tempos da rodada de 08/11, com 4 *threads* e a máquina carregada:** E1 1 h 57 min, E2 33 min, E3 4 min, E4 6 min, E5 1 h 23 min, E6 30 s, E7 2 min.
- **Saídas:** `resultados/` (CSV/JSON/parquet + `log_e*.txt`) e `resultados/figuras/` (PNG, 300 dpi).
- **libomp:** xgboost e lightgbm exigem `DYLD_LIBRARY_PATH` apontando para o Cellar do libomp enquanto o link do Homebrew estiver quebrado. O `rodar_tudo.sh` faz isso sozinho.

## Limitações

- O E1 usou 25 tentativas, contra 50 em 2023. Com o ganho de 0,0025 e a curva já plana após ~10 tentativas (ver `log_e1.txt`), mais tentativas dificilmente passariam a régua.
- São várias comparações no mesmo teste: por isso a régua de 0,003.
- No E5, Random Forest e ordinal foram avaliados em 1.000 linhas de teste, pelo custo do SHAP.
- O título da figura `e5_estabilidade_top4.png` foi regravado após a rodada. O código antigo escrevia "≥ 100,0%", e o código já está corrigido. Os números da figura são os da rodada.
- **Kano:** escala Likert tratada como numérica, associação e não causalidade. Dimensões com poucas notas 1–2 (Controle aduaneiro, Método de desembarque, Assentos reservados) têm IC largo.

### Referências citadas (novas em relação ao `referencias.bib`)

Akiba et al. (2019) *Optuna*, KDD · Brandt (1987) · Breiman (2001) *Statistical Modeling: The Two Cultures* · Fisher, Rudin & Dominici (2019), JMLR · Frank & Hall (2001), ECML · Lou et al. (2013), KDD · Matzler & Sauerwein (2002), IJSIM · McElfresh et al. (2023), NeurIPS · Mikulić & Prebežac (2008), Managing Service Quality · Nori et al. (2019) *InterpretML* · Rudin (2019), Nat. Mach. Intell. · Shwartz-Ziv & Armon (2022), Information Fusion. Já no `.bib`: Grinsztajn et al. (2022), Lundberg et al. (2020).
