# Priorização de melhorias em aeroportos brasileiros com Machine Learning

Código do Trabalho de Graduação de **Pedro Levi Sales Cavalcante** (ITA, 2026), orientado pelo Prof. Alessandro V. M. de Oliveira.

O framework usa a **Pesquisa de Satisfação do Passageiro** do Ministério de Portos e Aeroportos (20 maiores aeroportos, 461.593 entrevistas de 2021T1 a 2026T2) para dizer, em cada aeroporto, **o que melhorar primeiro**:

1. **Dados**: o questionário tem dois ramos (embarque e desembarque) e lógica de salto (subitens só para quem deu nota ≤ 3). A célula vazia é tratada como informação, não como falta.
2. **Modelos**: 9 modelos, do mais simples aos *ensembles* de árvores. Os 4 *ensembles* empatam (MAE 0,348–0,350 no teste); o XGBoost foi escolhido por critério definido antes do teste.
3. **Importância**: SHAP somado por dimensão. Top-4 nos 20 aeroportos: Localização, Limpeza, Conforto da sala de embarque e Restituição de bagagens, iguais em 100% das 990 reamostragens.
4. **Prioridade**: importância × desempenho do aeroporto contra a média nacional (matriz importância-desempenho).
5. **Relatório**: um LLM local (Gemma 2 9B via Ollama) redige o texto a partir dos números prontos, com verificador automático de números.
6. **Eixos complementares**: econometria (Logit, Double ML, DiD de Congonhas), 80 mil comentários abertos (sentimento e temas) e panorama nacional.

Tudo roda localmente, com software livre.

## Dados

Os dados **não estão no repositório**. São públicos:

- Microdados trimestrais: portal de dados abertos do Ministério (dados.transportes.gov.br), "Pesquisa de Satisfação do Passageiro".
- Página da pesquisa: https://www.gov.br/portos-e-aeroportos/pt-br/assuntos/transporte-aereo/pesquisa-de-satisfacao
- Contexto (ANAC e IBGE): baixado por `codigos/02_preprocessamento/00_coletar_dados_contexto.py`.

Coloque as planilhas em `dados/brutos/` e rode `codigos/10_serie_historica/01_carregar_serie.py`.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pipeline principal (≈1 h)
bash codigos/06_modelagem_20/rodar_tudo.sh

# ferramenta
ollama pull gemma2:9b && ollama serve &
streamlit run codigos/04_modelos_ml/app/app.py   # http://localhost:8501
```

## Estrutura

- `codigos/`: todo o código da pesquisa, por etapa. Ver [`codigos/README.md`](codigos/README.md).
- `codigos/04_modelos_ml/app/_backup_2026-11-01/`: versão anterior do app e dos *prompts*, citada no TG.
- `codigos/10_serie_historica/notas_*.md`: notas de trabalho de cada etapa da série 2021–2026.
- `apresentacoes/capturar_defesa.py`: captura das telas do app usadas na apresentação.
- `texto/`: *scripts* de compilação do TG em LaTeX. Os 20 relatórios gerados pelo LLM estão em [`codigos/05_interpretacao/relatorios_20/`](codigos/05_interpretacao/relatorios_20/).
