# MVP Streamlit — Framework de Priorização Aeroportuária (TG-2: 20 aeroportos)

App interativo que integra:
- Prioridades SHAP por aeroporto: `codigos/05_interpretacao/ranking_prioridades_20.csv` (pacote modelos20). Se esse arquivo faltar, cai no ranking do MVP.
- Contexto operacional oficial: ANAC 2023 e regime de gestão, de `codigos/02_preprocessamento/aeroportos_20_caracteristicas.csv` (pacote dados).
- Estatísticas de satisfação da MPor 2023, de `dados/processados/panorama_aeroportos_2023.parquet`.
- Camada LLM local (Ollama), com seletor de modelo: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Gemma 2 9B (padrão, o melhor na comparação `03_comparar_llms.py`).
- Aba de comentários (NLP): `codigos/08_nlp/saida/` (pacote nlp). É omitida se as saídas não existirem.
- Persistência SQLite do histórico de relatórios.

## Pré-requisitos

```bash
brew install ollama
ollama pull gemma2:9b      # e/ou llama3.1:8b, mistral:7b, qwen2.5:7b
pip install streamlit ollama altair playwright   # já no .venv do projeto
```

## Como rodar

```bash
cd "/Users/pedrolevi/TG ITA"
source .venv/bin/activate
ollama serve &
streamlit run codigos/04_modelos_ml/app/app.py
```

Telas para o TG (sobe o app headless, captura e encerra):
`python codigos/04_modelos_ml/app/capturar_telas.py` → `midias/imagens/mvp_0*.png`.

## Estrutura

```
app/
├── app.py              ← interface (sidebar: aeroporto + LLM; abas Prioridades, Relatório, Histórico, Comentários)
├── contexto.py         ← fonte única de contexto (T2 + MPor) e do ranking; filtro de atributos acionáveis
├── comentarios.py      ← aba de NLP (sentimento, temas críticos, triangulação, exemplos)
├── db_init.py          ← schema SQLite + I/O
├── llm_client.py       ← wrapper Ollama (gerar_relatorio / gerar_relatorio_meta com tokens/s)
├── prompts.py          ← template do prompt
├── capturar_telas.py   ← screenshots com Playwright (1440x900, escala 2)
└── tg_aeroportos.db    ← histórico (TG-1 + 24 relatórios da comparação de LLMs)
```
