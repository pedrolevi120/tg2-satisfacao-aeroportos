# Resumo da EDA inicial — Pesquisa MPor 2023 MVP

- **Dataset**: `dados/processados/mpor_2023_mvp.parquet`
- **Respondentes**: 14,250
- **Aeroportos**: 3 (SBGR, SBKP, SBSP)
- **Trimestres**: 4
- **Colunas Likert detectadas**: 41

## Distribuição da variável-alvo (`SATISFAÇÃO GERAL`)

| Nota | Respondentes | % |
|---:|---:|---:|
| 1 | 60 | 0.42% |
| 2 | 152 | 1.07% |
| 3 | 1,112 | 7.80% |
| 4 | 7,427 | 52.12% |
| 5 | 5,468 | 38.37% |
| NaN/NS | 31 | 0.22% |

## Top 10 atributos mais correlacionados com Satisfação Geral

| Atributo | Pearson r |
|---|---:|
| DISPONIBILIDADE DE ASSENTOS RESERVADOS | +0.584 |
| CONFORTO DA SALA DE EMBARQUE | +0.525 |
| LIMPEZA GERAL DO AEROPORTO | +0.523 |
| PROCESSO DE RESTITUIÇÃO DE BAGAGENS | +0.519 |
| ORGANIZAÇÃO DAS FILAS3 | +0.499 |
| LOCALIZAÇÃO E DESLOCAMENTO | +0.471 |
| SANITÁRIOS | +0.445 |
| ATENDIMENTO DOS FUNCIONÁRIOS4 | +0.414 |
| DISPONIBILIDADE DE TOMADAS | +0.409 |
| SINALIZAÇÃO | +0.402 |

## Bottom 5 (mais fracos)

| Atributo | Pearson r |
|---|---:|
| INTEGRIDADE DA BAGAGEM | +0.170 |
| VELOCIDADE DE CONEXÃO | +0.154 |
| QUANTIDADE DE BALCÕES | +0.107 |
| RELAÇÃO CUSTO X BENEFÍCIO | +0.068 |
| TERMINAL | -0.051 |

## Satisfação média por aeroporto

| Aeroporto | Média | Desvio | Mediana | N |
|---|---:|---:|---:|---:|
| Guarulhos (GRU) | 4.211 | 0.709 | 4.0 | 6,054 |
| Congonhas (CGH) | 4.159 | 0.648 | 4.0 | 4,033 |
| Viracopos (VCP) | 4.473 | 0.641 | 5.0 | 4,132 |