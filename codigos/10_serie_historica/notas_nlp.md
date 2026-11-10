# Comentários abertos (08_nlp) na série 2021T1–2026T2: notas

*08/11/2026. Adaptação do pacote `codigos/08_nlp/` (antes só 2023) à série inteira, seguindo `ESPEC.md`.
Saídas antigas, só 2023: `codigos/08_nlp/saida/_2023/` (CSVs, parquets, embeddings, logs, `tex/` e `fig/`) e o
tarball `versao_enviada_prof/codigos_e_resultados_2023.tar.gz`.*

## Resumo em uma linha

São 80.046 comentários válidos e 66.741 substantivos, em 462.006 respondentes. O sentimento continua validado nos
mesmos 200 comentários manuais de 2023, com métricas idênticas. Os temas da série vêm do BERTopic de 2023, sem
reajuste: 2023 mantém a atribuição original e os outros anos são atribuídos por vizinhos mais próximos (k = 5), que
reproduzem 2023 com 91% de concordância de tema. Banheiros segue em 1º lugar em todos os anos. A triangulação já
usa o ranking SHAP novo da série.

## O que mudou, arquivo por arquivo (`codigos/08_nlp/`)

- **`comum.py`**
  - Novas constantes: `BASE` (série), `BASE_COM` (saída de 01), `REF2023` (`saida/_2023/`), `CHAVE = "CHAVE_SERIE"`,
    `ANOS_ROT = "2021–2026"` (figuras) e `ANOS_TEX = "2021--2026"` (legendas LaTeX).
  - `RUIDO_EXATO` ganhou duas regras, e nenhuma delas altera 2023 (0 ocorrências naquele ano):
    - `"sim"` isolado: 2.458 linhas em 2021T3–2021T4, ao que parece a resposta a "deseja comentar?" sem o texto.
      Antes caíam como "genérico".
    - `"nada a informar"`, `"nada informar"`, `"nao informar"` e `"nada a informar sobre"`: 1.810 linhas, quase
      todas de SBCY em 2021–2022 (preenchimento padrão do entrevistador). Antes caíam como substantivas e levavam o
      tema "Informação" a 23 críticos por mil em 2022.
- **`01_carregar_comentarios.py`**
  - Lê `mpor_2021_2026_20.parquet` e mantém `CHAVE_SERIE`, `CHAVE` (Int64), `ANO`, `TRIMESTRE` e `PERIODO`.
  - Grava `saida/mpor_serie_20aero_comentarios.parquet`. O nome antigo `mpor2023_20aero_comentarios.parquet` foi
    apagado da `saida/` (cópia em `_2023/`). Nenhum consumidor fora do pacote lia esse arquivo.
  - **Descarta as 610 linhas de 2022T2 sem `AEROPORTO`**, que também não têm nenhum Likert. O
    `06_modelagem_20/03_base_features_20.py` faz o mesmo. N = 462.006.
- **`02_limpeza_descritiva.py`**
  - Novo `saida/descritiva_por_ano.csv`, em formato longo (BRASIL e aeroporto × ano), com n, válidos, substantivos e
    vazios.
  - Tabela e figura agregam a série. Legenda e título dizem "2021–2026".
- **`02b_anonimizar_ner.py`**
  - A chave passa a ser `CHAVE_SERIE`.
  - Os textos são ordenados por comprimento antes da inferência (menos *padding*), sem efeito no resultado.
  - O filtro de frequência em minúsculas usa o corpus da série.
- **`03a_amostra_validacao.py`**: **não sorteia nem grava mais nada.** Só confere que os 200 comentários de
  `amostra_validacao_200.csv` existem na série (`ANO == 2023` + `CHAVE` + `AEROPORTO`), com texto e classe idênticos.
- **`03b_sentimento.py`**
  - Inferência nos 80.046 comentários válidos, ordenados por comprimento, no MPS.
  - A validação faz o *merge* dos rótulos só com `ANO == 2023`.
  - Novo `saida/sentimento_por_ano.csv` (BRASIL e aeroporto × ano), com n, escore médio, % negativo, % positivo,
    % crítico e satisfação média.
  - O json ganhou duas correlações ecológicas: por ano e nas células aeroporto × ano (n = 120).
  - `sentimento_por_aeroporto.csv` mantém o formato do MVP (série agregada).
  - Corrigido um defeito antigo de formatação na legenda de `nlp_sentimento_satisfacao.tex`: "], $n=" saía como
    "]. $n=". Corrigi o script e apliquei a mesma troca ao `.tex` já gerado.
- **`04a_topicos_bertopic.py`**
  - Agora aponta para `saida/_2023/` (o modelo de referência).
  - **Não é rodado de novo.** A docstring explica o motivo.
- **`04a2_atribuir_temas_serie.py`** (novo): atribui os tópicos de 2023 à série (método abaixo).
- **`04b_temas_agregados.py`**
  - Usa `CHAVE_SERIE`.
  - `comentarios_topicos.parquet` ganhou `CHAVE_SERIE`, `ANO`, `PERIODO`, `origem_tema` (`bertopic_2023` ou
    `knn_ref2023`) e `tema_knn_oof` (só 2023, robustez).
  - `agregado_aeroporto_tema.csv` mantém o formato do MVP (série agregada; denominador = respondentes da série).
  - **Novas saídas:**
    - `agregado_tema_ano.csv`: aeroporto × tema × ano, inclui BRASIL, com grade completa e zeros. Colunas de críticos
      e comentários por mil, % dos válidos, % dos substantivos e posição dentro do aeroporto × ano.
    - `temas_por_ano_brasil.csv`: taxa e posição por ano, mais a coluna `2023_knn_oof`.
    - `temas_2023_robustez_metodo.csv`.
    - Figura `midias/figuras/nlp_temas_evolucao_ano.pdf`, com cópia em
      `apresentacoes/img_png/nlp_temas_evolucao_ano-1.png`.
  - Títulos e legendas dizem "2021–2026".
- **`04c_validacao_temas.py`**
  - Faz o *merge* da amostra com `ANO == 2023`, com *assert* de que os 178 têm `origem_tema == bertopic_2023`. Os
    índices `ERRADOS` continuam válidos.
  - Nova seção `knn_oof_178` no json: transferência da validação para o método dos outros anos.
- **`05_triangulacao.py`**
  - Notas Likert da série (`BASE_COM`).
  - Exclusão das colunas novas da lista de Likert.
  - Legenda "2021--2026".
- **`README.md`**: reescrito (ordem, método da atribuição, saídas novas, ressalvas).

## Comandos e tempos (rodada oficial, 08/11, 12:06–12:26, máquina com outras sessões pesadas)

```
cd codigos/08_nlp
../../.venv/bin/python 01_carregar_comentarios.py      #   2 s
../../.venv/bin/python 02_limpeza_descritiva.py        #   6 s
../../.venv/bin/python 03a_amostra_validacao.py        #   1 s  (só conferência)
../../.venv/bin/python 04a2_atribuir_temas_serie.py    # 116 s  (embeddings de 52,9 mil textos no MPS + CV do kNN)
../../.venv/bin/python 02b_anonimizar_ner.py           # 360 s  (MPS)
../../.venv/bin/python 03b_sentimento.py               # 687 s  (2 modelos no MPS + bootstrap)
../../.venv/bin/python 04b_temas_agregados.py          #   7 s
../../.venv/bin/python 04c_validacao_temas.py          #   1 s
../../.venv/bin/python 05_triangulacao.py              #   9 s
```

- Logs em `saida/_*.log` e `saida/_rodada_serie.log`.
- O 04a2 só depende de 02, então pode rodar antes de 02b e 03b.
- O cache `saida/embeddings_substantivos.npy` é da série e é indexado por `saida/embeddings_substantivos_chaves.parquet`.
- Depois da rodada, 04b, 04c e 05 foram rodados de novo (17 s) só para ajustar a legenda da figura.

## Atribuição de temas na série: método e reprodução de 2023

1. **Embeddings.** O mesmo MiniLM multilíngue, normalizado. 2023 reaproveita o cache da referência; numa
   re-codificação de 300 textos, o cosseno mínimo foi 1,000000.
2. **2023 mantém a atribuição original** do HDBSCAN e da realocação. São os documentos de ajuste: é o mesmo que o
   BERTopic faz com os dados de treino. Por isso os 178 da validação de temas continuam valendo sem alteração.
3. **Outros anos: kNN.** Cosseno com voto ponderado por 1/distância, sobre os 13.868 substantivos de 2023 rotulados
   pelo HDBSCAN, **incluindo a classe outlier −1**, que vira `sem_tema`. O k foi escolhido pela concordância em
   validação cruzada estratificada de 5 dobras dentro de 2023 (*seed* 42):

   | Método (reprodução de 2023, fora da dobra) | Concord. tópico | Concord. tema | % outlier (orig. 20,4%) | % sem_tema (orig. 23,1%) |
   |---|---|---|---|---|
   | kNN k = 3 | 0,891 | 0,902 | 16,6 | 19,4 |
   | **kNN k = 5 (escolhido)** | **0,898** | **0,909** | **16,3** | **19,1** |
   | kNN k = 7 | 0,896 | 0,908 | 16,0 | 18,9 |
   | kNN k = 10 | 0,894 | 0,905 | 15,8 | 18,7 |
   | Centróide + limiar calibrado (0,575) | 0,716 | 0,751 | 10,0 | 14,1 |

   Com k = 5, 95,0% dos não outliers mantêm o tópico e 69,6% dos outliers continuam outliers. Entre os previstos com
   tema, 91,1% acertam o tema original.
4. **Realocação de outliers (`topico_realocado`, configuração secundária).** Centróide mais próximo com cosseno ≥ 0,5.
   Reproduz **100%** o `reduce_outliers(strategy="embeddings", threshold=0.5)` de 2023.
5. **Robustez da comparação entre anos.** Como o kNN marca menos outliers (16% contra 20%), os temas dos outros anos
   ficam levemente inflados em relação a 2023. Refazendo 2023 com o mesmo kNN (fora da dobra), a razão entre as taxas
   de críticos por mil fica entre 0,96 e 1,10 na maioria dos temas. As exceções são transporte (1,23), elogio geral
   (1,13), facilidade (1,29) e `sem_tema` (0,86). O ρ de Spearman entre temas específicos, original × kNN, é 0,997.
   A figura mostra o valor kNN de 2023 como um círculo aberto.
6. **Transferência da validação manual** (178 substantivos de 2023):
   - o kNN fora da dobra concorda com o tema original em 92,7% deles;
   - entre os 140 a que o kNN atribui tema, a precisão fica entre 0,79 e 0,86 (10 têm tema diferente, não julgado);
   - entre os que concordam, a precisão é 0,85;
   - para comparar: a precisão da configuração principal em 2023 é 0,83.

## Números-chave para a reescrita do TG

### Cobertura dos comentários

| Ano | Respondentes | Válidos | % | Substantivos | % | Campo em branco |
|---|---|---|---|---|---|---|
| 2021 | 44.588 | 4.467 | 10,0 | 3.745 | 8,4 | 42,2% |
| 2022 | 82.904 | 11.792 | 14,2 | 9.736 | 11,7 | 52,2% |
| 2023 | 84.776 | 15.754 | 18,6 | 13.868 | 16,4 | 0,0% |
| 2024 | 99.275 | 18.499 | 18,6 | 15.548 | 15,7 | 0,0% |
| 2025 | 103.238 | 20.348 | 19,7 | 16.556 | 16,0 | 0,0% |
| 2026 (até jun.) | 47.225 | 9.186 | 19,5 | 7.288 | 15,4 | 0,0% |
| **Total** | **462.006** | **80.046** | **17,3** | **66.741** | **14,4** | |

- Mediana de 6 palavras nos substantivos.
- Satisfação média: 4,35 com comentário substantivo, 4,44 sem comentário válido.
- 188 comentários batem no limite de 200 caracteres.
- O NER mascarou nomes em 158 comentários.
- Por aeroporto (série), % válidos vai de 28,5% (SBFL) e 24,4% (SBBE) até 10,8% (SBMO).
- **2023 é idêntico ao anterior:** 15.754 válidos e 13.868 substantivos. A diferença em críticos por mil por
  aeroporto × tema entre o 2023 novo e o antigo é exatamente 0.

### Sentimento (XLM-R primário)

- **Validação** (mesmos 200 de 2023; resultados idênticos aos anteriores, e 100% das 15.754 classes de 2023 iguais):

  | Modelo | Acurácia | F1 macro | κ | Binário: acurácia | Binário: F1 |
  |---|---|---|---|---|---|
  | XLM-R | 0,700 | 0,568 | 0,513 | 0,875 | 0,865 |
  | BERTweet-PT | 0,600 | 0,526 | 0,423 | 0,895 | 0,883 |

- **Distribuição na série (XLM-R):** 42,1% positivo, 29,9% negativo, 28,0% neutro, ou seja, 57,9% "críticos" (não
  positivos). BERTweet-PT: 36,7% positivo, 21,1% negativo, 42,2% neutro.
- **Satisfação média por classe prevista:** negativo 4,17, neutro 4,24, positivo 4,69.
- **ρ de Spearman entre escore e satisfação geral:**
  - válidos: 0,385 (IC 95% de 0,379 a 0,391; n = 79.992);
  - substantivos: 0,337;
  - BERTweet: 0,409 e 0,363.
  - Em 2023 era 0,40 nos válidos.
- **Correlação ecológica** (20 aeroportos, série): ρ = 0,62 (p = 0,003); em 2023 era 0,63.
  - Por ano: 0,15 (2021, n.s.), 0,36 (2022, n.s.), 0,63 (2023), 0,48 (2024), 0,60 (2025) e 0,56 (2026).
  - Nas 120 células aeroporto × ano: ρ = 0,50.
- **Por ano (Brasil):**

  | Ano | % positivo | % negativo | % crítico | Escore médio |
  |---|---|---|---|---|
  | 2021 | 33,0 | 33,5 | 67,0 | −0,05 |
  | 2022 | 32,5 | 31,2 | 67,5 | −0,03 |
  | 2023 | 38,9 | 33,4 | 61,1 | +0,03 |
  | 2024 | 43,9 | 29,8 | 56,1 | +0,09 |
  | 2025 | 46,8 | 27,6 | 53,2 | +0,13 |
  | 2026 | 50,3 | 25,4 | 49,7 | +0,17 |

  O tom dos comentários melhora de forma monotônica desde 2022. Parte disso é composição: a participação de elogios
  genéricos curtos entre os válidos vai de 12,0% em 2023 para 20,7% em 2026.
- **Quedas fortes de % crítico por aeroporto, 2023 → 2026:**
  - SBBE: 79 → 43 (concessão em 09/2023);
  - SBSP: 81 → 52 (concessão em 10/2023);
  - SBCT: 63 → 42;
  - SBGO: 64 → 51.
  - SBGL é a exceção: 81 → 82.

### Temas (série agregada, Brasil)

- **Cobertura dos substantivos:** 48,7% em tema coberto pelo questionário, 9,8% parcial, 10,1% não coberto, 11,3%
  global/elogio e 20,1% `sem_tema`. Entre os substantivos críticos: 58,6% coberto, 5,1% parcial, 10,3% não coberto e
  23,6% `sem_tema`. Em 2023 era 46,6 / 9,5 / 10,3 / 10,4 / 23,1.
- **Ranking por críticos por mil respondentes** (série; entre parênteses, % de comentários não positivos):
  1. Banheiros: 8,74 (85%);
  2. Bagagem: 6,56 (92%);
  3. Oferta de alimentação: 6,05 (88%);
  4. Preços: 5,87 (95%);
  5. Sinalização: 5,62 (85%);
  6. Transporte terrestre: 4,05;
  7. **Atrasos de voo\***: 3,75 (97%);
  8. Tomadas: 3,47;
  9. Climatização: 3,01;
  10. Assentos: 2,89.

  Em 2023 a ordem era banheiros, bagagem, preços, alimentação, sinalização, transporte e atrasos (7º): os mesmos
  temas, com troca só entre 3º e 4º.
- **Temas não cobertos pelo questionário** (\*):

  | Tema | N | % dos comentários | Críticos por mil | % crítico |
  |---|---|---|---|---|
  | Atrasos de voo e conexão | 1.792 | 2,2 | 3,75 | 97 |
  | Evolução percebida ("melhorou") | 1.731 | 2,2 | 0,69 | 18 |
  | Obras, reformas e capacidade | 1.028 | 1,3 | 1,00 | 45 |
  | Oferta de voos e rotas | 903 | 1,1 | 1,80 | 92 |
  | Sala VIP | 884 | 1,1 | 1,78 | 93 |
  | Bebedouros | 422 | 0,5 | 0,87 | 95 |

  - Juntos, são 10,1% dos substantivos.
  - Atrasos é o tema não coberto mais criticado em 13 dos 20 aeroportos, com maiores taxas em SBBE (7,3), SBKP (5,9)
    e SBRF (5,8).
- **Por ano** (posição entre os temas específicos, Brasil; `temas_por_ano_brasil.csv`):
  - **Banheiros: 1º em todos os 6 anos.**
  - De 2023 a 2026 o topo é estável:
    - bagagem fica em 2º nos quatro anos;
    - alimentação, preços e sinalização se alternam entre 3º e 5º;
    - transporte é 6º, atrasos 7º e tomadas 8º;
    - climatização é 8º–9º e assentos 10º.
  - Sinalização sobe de 5º (2023) para 3º (2025–2026). A taxa vai de 5,96 para 6,96 em 2025.
  - Preços cai de 6,41 para 5,34 críticos por mil, e alimentação de 6,18 para 4,53 em 2026.
  - Climatização cai de 3,73 para 2,67.
  - Sala VIP vai de 12º em 2023 para 17º em 2026, e internet de 16º para 18º.
  - Limpeza do terminal (majoritariamente elogio) sobe de 20º para 15º.
  - 2021–2022 diferem (bagagem em 5º e 9º, preços e alimentação no topo), mas ver a ressalva abaixo.
- **Mudanças locais grandes** (críticos por mil, 2023 → 2025):
  - quedas:
    - SBBE climatização: 27,9 → 7,9;
    - SBBE banheiros: 26,2 → 14,3;
    - SBGL transporte: 9,2 → 2,4;
    - SBMO e SBSG sala VIP: 8–9 → 2–3;
  - altas:
    - SBSG sinalização: 6,1 → 19,5;
    - SBSG transporte: 11,3 → 19,9 (22,6 em 2026);
    - SBBE sinalização: 6,8 → 16,6;
    - SBGL e SBKP banheiros: 6–7 → 11–14.

### Triangulação (05) — com o ranking SHAP NOVO da série

`ranking_prioridades_20.csv` foi regravado às 12:20 de 08/11. Tem `n_teste` somando 69.239, que é o teste da série.
O 05 rodou depois disso, às 12:26, com essa versão.

| | Série | 2023 (anterior) |
|---|---|---|
| Atributos do top-5 SHAP cobertos pelo top-5 NLP (média) | 2,3/5 | 2,8/5 |
| ρ médio NLP × SHAP (dentro do aeroporto) | 0,61 | 0,60 |
| ρ médio NLP(A) × SHAP(B ≠ A) | 0,60 | 0,60 |
| ρ médio NLP × nota invertida | 0,34 | 0,37 |
| ρ médio NLP(A) × nota(B ≠ A) | 0,30 | 0,32 |
| ρ médio SHAP × nota | −0,11 | −0,01 |
| Tema 1 do NLP no top-5 SHAP | 9/20 | 12/20 |

- **Conclusão inalterada:** a concordância NLP × SHAP é estrutural. Fora da diagonal ela é quase igual à da diagonal,
  ou seja, vem de temas comuns a todos os aeroportos, não de especificidade local.
- **Entre aeroportos, por tema:**
  - a taxa de críticas acompanha a nota invertida do atributo-pai em vários temas: banheiros ρ = 0,66, informação
    0,73, internet 0,71, transporte 0,62 e check-in 0,54 (p < 0,05 em todos);
  - bagagem fica em 0,40 (p = 0,08).
- Se o pacote de modelagem regravar o ranking de novo, basta rodar o `05_triangulacao.py` outra vez (9 s).

## Ressalvas e decisões a revisar

1. **kNN em vez de centróide.** O pedido era centróide mais próximo com limiar calibrado. Ele foi implementado e
   reportado, mas reproduz o HDBSCAN de 2023 só com 72% (tópico) e 75% (tema), contra 90% e 91% do kNN. Usei o kNN como
   principal. Se preferir o centróide, a coluna `topico_centroide` já está em `bertopic_atribuicoes.parquet`.
2. **Método híbrido.** 2023 tem a atribuição original e os outros anos têm o kNN. Isso preserva a validação manual e
   os números de 2023, mas o kNN marca menos outliers, o que infla de leve os outros anos em relação a 2023 (item 5 do
   método). A figura mostra 2023 nas duas versões.
3. **2021–2022 não são comparáveis em nível com 2023–2026:**
   - o campo de comentário vem em branco em 42–52% das linhas (taxa de válidos de 10–14%);
   - a amostra tem 71% de embarque, contra 50% depois. Isso derruba temas de desembarque: bagagem cai para 3 críticos
     por mil, mas entre desembarcados de 2021 são 9,9;
   - a correlação ecológica não é significativa nesses anos.
   - Recomendo apresentar a evolução com ênfase em 2023–2026 e usar 2021–2022 só como contexto. A figura sombreia
     esses anos.
4. **Mudanças em `comum.classificar`:** "Sim" isolado e "nada a informar" passam a ser ruído. É inócuo para 2023 e
   evita dois artefatos de coleta.
5. **610 linhas sem aeroporto (2022T2)** foram descartadas. Elas continuam na base comum: quem conta 462.616 nos
   outros pacotes deve saber que o NLP usa 462.006 (o 03 da modelagem também descarta).
6. **A queda do % crítico 2023 → 2026 (61 → 50%)** vem junto com o aumento dos elogios genéricos curtos (12 → 21%
   dos válidos). Convém não ler isso só como melhora de qualidade.
7. O `.tex` de sentimento × satisfação teve um defeito antigo de vírgula corrigido à mão, com a mesma troca aplicada
   ao script. Os outros `.tex` (`texto/_paralelo/tex/nlp_*.tex`) e as figuras `midias/figuras/nlp_*.pdf` foram
   regenerados pelos scripts. Cópias antigas estão em `saida/_2023/tex` e `saida/_2023/fig`.
