# codigos/

Cada pasta numerada é uma etapa do TG. Os scripts leem de `dados/` e gravam saídas na própria pasta (não versionadas).

| Pasta | O que faz |
|---|---|
| `01_exploracao_dados/` | Exploração inicial da pesquisa (MVP: Guarulhos, Congonhas, Viracopos) |
| `02_preprocessamento/` | Carga das planilhas, base unificada, *features*, divisão treino/validação/teste e contexto público (ANAC, IBGE) |
| `03_modelos_baseline/` | Modelos simples do MVP (média, linear, KNN, árvore) |
| `04_modelos_ml/` | *Ensembles* e rede neural do MVP; `app/` é a ferramenta em Streamlit |
| `05_interpretacao/` | SHAP, geração dos relatórios pelo LLM local e comparação cega dos 4 LLMs; `relatorios_20/` traz os 20 relatórios finais |
| `06_modelagem_20/` | Pipeline principal: 9 modelos nos 20 aeroportos, leave-one-airport-out e SHAP por dimensão (`rodar_tudo.sh`) |
| `06_visualizacoes/` | Panorama nacional 2021–2026 |
| `07_econometria/` | Logit ordinal, Double ML, DiD de Congonhas (com permutação) e SHAP dos ausentes |
| `08_nlp/` | Comentários abertos: limpeza, anonimização, sentimento, temas (BERTopic) e triangulação com o modelo |
| `09_experimentos/` | Alternativas testadas: ajuste fino, EBM, ordinal, perfil, estabilidade do ranking, Kano |
| `10_serie_historica/` | Carga da série 2021T1–2026T2 e validação no tempo (treina até 2024, testa em 2025–26) |
