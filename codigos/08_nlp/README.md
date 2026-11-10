# 08_nlp — Comentários abertos (T7)

Base: série MPor **2021T1–2026T2** (20 aeroportos, 21 trimestres, `dados/processados/mpor_2021_2026_20.parquet`,
ver `codigos/10_serie_historica/ESPEC.md`). Chave de junção: `CHAVE_SERIE`. A amostra manual de validação e o
modelo de tópicos de referência são de **2023** e ficam fixos (casados por `ANO == 2023` + `CHAVE` + `AEROPORTO`).
Saídas antigas, só 2023: `saida/_2023/` (e `versao_enviada_prof/codigos_e_resultados_2023.tar.gz`).

Ordem de execução (`source .venv/bin/activate; cd codigos/08_nlp`):

| # | Script | Faz |
|---|---|---|
| 1 | `01_carregar_comentarios.py` | lê a base da série → `saida/mpor_serie_20aero_comentarios.parquet` |
| 2 | `02_limpeza_descritiva.py` | classifica vazio/ruído/genérico/substantivo, estatística por aeroporto e por ano |
| 2b | `02b_anonimizar_ner.py` | NER local (PER → `[NOME]`) para o texto exportado |
| 3a | `03a_amostra_validacao.py` | **só confere** a amostra fixa de 200 (2023) na série — não sorteia nem sobrescreve; rótulos em `rotulos_manuais_200.csv`, rubrica em `RUBRICA_SENTIMENTO.md` |
| 3b | `03b_sentimento.py` | XLM-R (primário) e BERTweet-PT em toda a série, validação nos 200 de 2023, correlação com satisfação geral, sentimento por ano |
| 4a | `04a_topicos_bertopic.py` | BERTopic + MiniLM multilíngue nos substantivos de **2023** — modelo de referência; **não é rodado de novo** (entradas/saídas em `saida/_2023/`) |
| 4a2 | `04a2_atribuir_temas_serie.py` | atribui os tópicos de 2023 a todos os substantivos da série (ver abaixo) |
| 4b | `04b_temas_agregados.py` | aplica `temas.py` (tópico → tema → atributo Likert), gera parquet/CSV do MVP, temas por ano e figura de evolução |
| 4c | `04c_validacao_temas.py` | validação manual da atribuição de temas (178 comentários substantivos da amostra de 2023) |
| 5 | `05_triangulacao.py` | NLP × SHAP × nota Likert por aeroporto (usa `ranking_prioridades_20.csv` — deve ser o regenerado na série) |

Etapas pesadas (MPS): 02b (NER), 03b (dois classificadores) e 4a2 (embeddings). Rodar uma de cada vez.

## Atribuição de temas na série (4a2)

O BERTopic **não** é reajustado: os números de tópico mapeados à mão em `temas.py` só valem para o modelo de 2023.

1. Embeddings com o mesmo modelo (`paraphrase-multilingual-MiniLM-L12-v2`, normalizados); 2023 reaproveita o cache
   da referência (conferido: cosseno 1,000 numa re-codificação de 300 comentários).
2. **2023** mantém a atribuição original do BERTopic (são os documentos de ajuste — o mesmo que o BERTopic faz com os
   dados de treino). Assim a validação manual de temas (4c) continua valendo sem alteração.
3. **Demais anos**: k vizinhos mais próximos (cosseno, voto ponderado por 1/distância) entre os 13.868 comentários
   de 2023 rotulados pelo HDBSCAN, **incluindo a classe outlier (−1)** — que vira `sem_tema`, como em 2023. O `k` é
   escolhido pela maior concordância com o HDBSCAN em validação cruzada estratificada de 5 dobras dentro de 2023.
4. A alternativa "centróide mais próximo + limiar de outlier calibrado" é calculada e reportada, mas reproduz pior
   o HDBSCAN (os grupos do HDBSCAN, formados no espaço UMAP de 5 dimensões, não são convexos no espaço original, e um
   limiar único não reproduz a fração de outliers). Ver `saida/atribuicao_temas_serie.json`.
5. `topico_realocado` (configuração secundária): outlier → centróide mais próximo se cosseno ≥ 0,5 — reproduz 100%
   o `reduce_outliers(strategy="embeddings", threshold=0.5)` usado em 2023.

Concordâncias na reprodução de 2023 e participação de cada tema (original × kNN fora da dobra):
`saida/atribuicao_temas_serie.json`, `saida/atribuicao_2023_reproducao_por_tema.csv`,
`saida/temas_2023_robustez_metodo.csv`; transferência da validação manual: `saida/validacao_temas.json` (`knn_oof_178`).

## Saídas

Para o MVP (mesmos nomes e colunas de antes, agora agregando a série; colunas `ANO`/`PERIODO` acrescentadas onde há
uma linha por comentário): `saida/comentarios_topicos.parquet`, `saida/agregado_aeroporto_tema.csv`,
`saida/sentimento_por_aeroporto.csv`, `saida/triangulacao_por_aeroporto.csv`, `saida/descritiva_por_aeroporto.csv`.

Novas por ano: `saida/descritiva_por_ano.csv`, `saida/sentimento_por_ano.csv`, `saida/agregado_tema_ano.csv`
(aeroporto × tema × ano, inclui `BRASIL`), `saida/temas_por_ano_brasil.csv`; figura
`midias/figuras/nlp_temas_evolucao_ano.pdf` (cópia PNG em `apresentacoes/img_png/nlp_temas_evolucao_ano-1.png`).

Ressalva: em 2021–2022 o campo de comentário vem em branco com mais frequência e a taxa de comentários válidos é menor
(10–14% contra 19–20% em 2023–2026; amostra 71% embarque); em 2021T3–2021T4 o campo traz só "Sim" em 2.458 linhas (tratado como ruído).
Taxas por mil respondentes desses anos ficam subestimadas; a participação no total de comentários não.

Não usa Ollama. Tudo local; `random_state=42`.
