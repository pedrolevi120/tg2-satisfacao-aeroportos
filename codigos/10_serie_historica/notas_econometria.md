# Notas: econometria e cadeia MVP na série 2021T1–2026T2

Escopo deste pacote: cadeia MVP (GRU/CGH/VCP: `02_preprocessamento/01–03`, `06_modelagem_20/02_robustez_mvp.py`
e os scripts MVP de modelos/SHAP que a econometria consome) e `07_econometria` (00–05 + `comum.py`).
Kano, NLP, app/contexto/panorama e o restante de `06_modelagem_20`/`09_experimentos` ficam com outros agentes.
Referência de sanidade: `versao_enviada_prof/codigos_e_resultados_2023.tar.gz` (rodada só com 2023).

---

## 1. Mudanças por arquivo

### Cadeia MVP

| Arquivo | Mudança |
|---|---|
| `02_preprocessamento/01_carregar_mpor_2023.py` | Lê `dados/processados/mpor_2021_2026_20.parquet` e filtra SBGR/SBSP/SBKP. **O nome da saída foi mantido** (`mpor_2023_mvp.parquet`), mas o conteúdo agora é a série (76.690 linhas). O dicionário continua vindo da LEGENDA de 2023T2. |
| `02_integrar_base_unificada.py` | Contexto por (AEROPORTO, PERIODO) de `contexto_aeroportos_20_serie.csv`, com `validate="many_to_one"`. `REGIME_NO_TRIMESTRE` virou `REGIME_NO_PERIODO`: regime **na data da entrevista** (no período "misto", decide pela data de assunção). Em SBSP 2023T4 são 114 entrevistas Infraero e 1.096 concessionada. |
| `03_features_e_split.py` | Split mantido: 70/15/15, estratificado por aeroporto, semente 42. Também mantidos `-1` para ausente e o tratamento de ausentes. PERIODO, REGIME_NO_PERIODO e AERO_operador entram como categóricas. As 6 numéricas de contexto (pax/decolagens do trimestre, pistas, IBGE) passam a ser por período. Colunas `AERO_`/`IBGE_` saem da detecção Likert. TERMINAL foi excluído dos Likert. **Corrigido um bug de formato**: notas gravadas como "4.0" não eram reconhecidas como Likert. Por isso agora há **61 colunas Likert** (antes 42), por exemplo CONFORTO TÉRMICO, LIMPEZA DOS BANHEIROS e ESTABELECIMENTOS COMERCIAIS. Nova função `limpar_nomes()`, porque o LightGBM recusa `,:[]{}` em nomes; afeta só as dummies de MOTIVO DA VIAGEM, que é texto livre com 226 níveis. X tem 356 colunas (antes 243). |
| `06_modelagem_20/02_robustez_mvp.py` | Usa `limpar_nomes`, `n_jobs=4` e legendas com "2021–2026". O n de validação e de teste vem dos dados, não está mais fixo no código. Desenho idêntico ao anterior. |
| `03_modelos_baseline/01`, `04_modelos_ml/01–02`, `05_interpretacao/01` | **Não editados**, apenas re-rodados, porque a econometria consome `shap_values_test.parquet` e `tabela_comparativa.csv`. Efeito colateral: regeneram `ranking_prioridades_por_aeroporto.csv`, `tabela_comparativa*` e as figuras `shap_*`/`comparacao_modelos.pdf`. |

### 07_econometria

| Arquivo | Mudança |
|---|---|
| `comum.py` | `colunas_atributos` passa a selecionar **pelo nome**: tupla `ATRIBUTOS_QUESTIONARIO` (bloco do questionário na ordem do arquivo) com ≥ 60% de valores Likert. `efeitos_fixos` gera `FE_aero=` e `FE_per={PERIODO}` (drop-first). `carregar_20(colunas=None)` lê `econometria_base20.parquet` (cai para a série bruta se não existir). PERIODO entrou em `NAO_ATRIBUTOS`. |
| `00_base20.py` | Reescrito; constrói a base a partir da série. **Problemas de dados corrigidos aqui** (ver §4): descarta 610 linhas vazias (2022T2, sem AEROPORTO) e converte 33.600 datas gravadas como serial do Excel (2024T1–T2 e parte de 2025T3–T4) com base 1899-12-30. Restam 7.000 NaT, todos em 2022T4 (outubro); para essas linhas usa-se `MES_NUM`, vindo de `MÊS`. Acrescenta REGIME/OPERADOR/FRAC_DIAS por período, `REGIME_NA_DATA` e `GRUPO_DID`. Saída: `dados/processados/econometria_base20.parquet` (462.006 linhas; 461.593 com satisfação válida). |
| `regime_operador_serie.csv` | Substitui `regime_operador_2023.csv` (apagado). Uma linha por aeroporto: operador no início e no fim, data e tipo da troca, `grupo_did`. Tratado: SBSP (17/10/2023). Tratado secundário: SBBE (04/09/2023). Excluídos: SBEG, SBGO, SBCT (concessões de 2022) e SBSG (troca de concessionária em 2024). Os outros 14 são controles. |
| `01_logit_ordinal.py` | EF de PERIODO (21 trimestres) no lugar de trimestre/ano. Textos dizem "base MVP 2021–2026". O n do teste no `.tex` vem dos dados (antes 2.133 fixo). |
| `02_double_ml.py` | Dummies de PERIODO em X. Na base de 20 aeroportos, **subamostra estratificada de 25% por aeroporto × período, semente 42** (`DML_FRAC_20=0.25`; 115.405 de 461.593 respondentes). O MQO dessa base usa a mesma subamostra. A fração e o n completo vão para o CSV e para a nota da tabela. MVP com base inteira. LightGBM com `n_jobs=4`; tempo por ajuste registrado. |
| `03_did_congonhas.py` | **Redesenhado** (§3). |
| `04_shap_ausentes.py` | O "$p<0{,}001$" fixo no texto virou p calculado (`p_txt`). |
| `05_did_permutacao_mensal.py` | Reescrito para o event study mensal com meses relativos inteiros. Permutação dos leads, média pós total, curto e médio prazo. Substitui as próprias linhas no `.tex` de forma idempotente. Refaz a figura de placebos. |

Saídas regeneradas em `texto/_paralelo/tex/`:
- `econometria_logit*.tex`, `econometria_logit_shap.tex`
- `econometria_dml*.tex`
- `econometria_did.tex`, `econometria_did_eventstudy.tex`
- **novos:** `econometria_did_horizonte.tex`, `econometria_did_belem.tex`
- `modelos20_robustez_*.tex`

Figuras em `midias/figuras/`:
- `econometria_did_{tendencias,eventstudy,placebos}.pdf`
- **nova:** `econometria_did_belem.pdf`
- `econometria_logit_*`, `econometria_dml.pdf`, `modelos20_robustez*`

---

## 2. Como rodar (ordem) e tempos

Pré-requisito nesta máquina: o symlink `/opt/homebrew/opt/libomp` está quebrado, então XGBoost e LightGBM não carregam. Contorno por processo, sem mexer no sistema:
`export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/libomp/22.1.5/lib` (a correção definitiva é `brew reinstall libomp`).

```bash
cd tg2-satisfacao-aeroportos   # raiz do repositório
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/libomp/22.1.5/lib
.venv/bin/python codigos/02_preprocessamento/01_carregar_mpor_2023.py
.venv/bin/python codigos/02_preprocessamento/02_integrar_base_unificada.py
.venv/bin/python codigos/02_preprocessamento/03_features_e_split.py
.venv/bin/python codigos/03_modelos_baseline/01_baselines.py          # 26 s
.venv/bin/python codigos/04_modelos_ml/01_ensembles_mlp.py            # 89 s
.venv/bin/python codigos/04_modelos_ml/02_comparar_modelos.py         # 3 s
.venv/bin/python codigos/05_interpretacao/01_shap_xgboost.py          # 23 s
(cd codigos/06_modelagem_20 && ../../.venv/bin/python 02_robustez_mvp.py)   # 348 s
cd codigos/07_econometria
../../.venv/bin/python 00_base20.py                 # 7 s
../../.venv/bin/python 01_logit_ordinal.py          # 3.357 s (56 min: OrderedModel com gradiente numérico, n_treino = 53.639)
DML_FRAC_20=0.25 ../../.venv/bin/python 02_double_ml.py   # 1.682 s (28 min; MVP inteiro ≈ 20 min)
../../.venv/bin/python 03_did_congonhas.py          # 535 s
../../.venv/bin/python 04_shap_ausentes.py          # 3 s
../../.venv/bin/python 05_did_permutacao_mensal.py  # 1 s
```

Logs:
- MVP: `codigos/10_serie_historica/logs_econ/*.log`
- econometria: `codigos/07_econometria/resultados/log_0X.txt`
- tempos: `logs_econ/tempos.txt`

Teste rápido do DiD: `DID_FRAC=0.05` (amostra estratificada; usar só com as saídas redirecionadas para outra pasta).

---

## 3. NÚMEROS-CHAVE

### 3.1 MVP (GRU/CGH/VCP)

Base com satisfação válida: **76.629** respondentes, contra 14.219 em 2023. Split: treino 53.639, validação 11.495, teste 11.495.

| Período | SBGR | SBKP | SBSP |
|---|---|---|---|
| 2021T1 | 546 | 415 | 397 |
| 2021T3 (só ago–set) | 797 | 653 | 659 |
| 2021T4 | 1.273 | 1.105 | 1.171 |
| 2022T1–T3 | ~1.460–1.800 | ~1.200–1.260 | ~1.200–1.240 |
| 2022T4 (só out.) | 600 | 400 | 400 |
| 2023T1 (só mar.) | 617 | 443 | 415 |
| 2023T2–2026T1 | ~1.720–1.830 | ~1.200–1.250 | ~1.200–1.240 |
| 2026T2 | 1.497 | 993 | 1.043 |
| **Total** | **31.880** | **22.489** | **22.321** |

2021T2 não tem coleta. Tabela completa: `base_unificada_mvp.parquet` agrupado por período × aeroporto.

Modelos (teste):

| Modelo | MAE | R² | QWK | 2023 |
|---|---|---|---|---|
| XGBoost | 0,365 | 0,482 | 0,626 | MAE 0,377 |
| LightGBM (validação) | 0,364 | | | |
| Ridge (validação) | 0,410 | | | |

O baseline logit_ordinal_top30 falha, como já falhava em 2023.

Top-4 SHAP, em ordem: Localização e deslocamento, Limpeza geral, **Sinalização**, Restituição de bagagens. Em 2023 Restituição vinha antes de Sinalização.

Robustez dos ausentes (`02_robustez_mvp.py`), teste:
- Todas as variantes ficam em MAE 0,363–0,370; o ΔMAE em relação a (a) vai de −0,002 a +0,004.
- (b) NaN + PROCESSO: −0,0014 [−0,0026; −0,0003] (XGB).
- (d) só itens-pai piora: +0,0037 [0,0018; 0,0056].
- Rankings: Spearman com (a) entre 0,96 e 0,99 nas variantes (b), (c) e (c'), e 4/4 do top-4 em (b) com XGB.
- (d) cai para 0,46 no nível de atributo, mas fica em 0,99 no de dimensão.
- Fração do |SHAP| top-10 que vem de ausentes: 0,42 em (a), 0,25 em (b) e 0,22 em (c).
- **Conclusão igual à de 2023**: o tratamento dos ausentes não muda o desempenho e quase não muda a ordem por dimensão.

### 3.2 Logit ordinal (base MVP, EF de aeroporto + período)

Treino n = 53.639. Atributos: 61 detectados, 30 com cobertura ≥ 3%. Matriz econométrica: 119 colunas.

| Especificação | k | pseudo-R² | MAE teste | R² | QWK |
|---|---|---|---|---|---|
| [C] completo | 119 | 0,296 | 0,387 | 0,456 | 0,623 |
| [L] post-LASSO | 75 | 0,295 | 0,387 | 0,456 | 0,621 |
| [D] post-double-selection (30 atributos + 56 controles) | 109 | 0,296 | 0,386 | 0,457 | 0,622 |

Em 2023 o MAE de teste era 0,392.

Maiores coeficientes da especificação (D):

| Atributo | β (EP) | OR | AME em P(5) |
|---|---|---|---|
| Restituição de bagagens | 1,190 (0,036) | 3,29 | +0,203 |
| Localização e deslocamento | 1,077 (0,022) | 2,94 | +0,174 |
| Limpeza geral | 0,874 (0,025) | 2,40 | +0,127 |
| Controle aduaneiro | 0,699 (0,088) | | |
| Conforto da sala de embarque | 0,551 (0,031) | | |
| Sinalização | 0,468 (0,032) | | |

- Brant aproximado rejeita a hipótese de odds proporcionais: W = 468, 30 gl.
- Concordância com SHAP:

| Medida | Série | 2023 |
|---|---|---|
| Spearman (índice) | **0,568** (p = 0,001; n = 30) | 0,476 (p = 0,012) |
| Spearman com \|AME\| | 0,408 | |
| Top-10 em comum | 7 | 6 |
| Spearman só com itens observados (`04`) | 0,684 | |
| Spearman só com itens principais (`04`) | **0,909** (n = 16) | |

- Top-4: SHAP = Localização, Limpeza, Sinalização, Restituição; logit = Localização, Limpeza, Restituição, Conforto da sala de embarque.
- Parcela média do |SHAP| que vem de ausentes: 0,51 nos sub-itens e 0,35 nos principais. **A mensagem de 2023 se mantém e fica mais forte**: a discordância vem dos sub-itens da lógica de salto.

### 3.3 DML (PLR com LightGBM, 5 dobras × 5 repetições). Efeito de +1 ponto no atributo sobre a satisfação

| Base | Atributo | n | MQO simples | MQO + X | **DML [IC 95%]** | Inclinação SHAP | RV | 2023 (DML) |
|---|---|---|---|---|---|---|---|---|
| MVP | Localização | 76.468 | 0,377 | 0,242 | **0,238 [0,232; 0,244]** | 0,205 | 0,30 | 0,239 |
| MVP | Limpeza | 42.403 | 0,540 | 0,223 | **0,216 [0,205; 0,226]** | 0,187 | 0,21 | 0,215 |
| MVP | Sinalização | 13.463 | 0,326 | 0,168 | **0,158 [0,143; 0,173]** | 0,143 | 0,19 | 0,166 |
| MVP | Restituição | 24.796 | 0,383 | 0,285 | **0,280 [0,269; 0,291]** | 0,102 | 0,37 | 0,266 |
| 20 aer. (25%) | Localização | 115.084 | 0,426 | 0,258 | **0,253 [0,248; 0,259]** | — | 0,28 | 0,271 |
| 20 aer. (25%) | Limpeza | 64.212 | 0,553 | 0,210 | **0,207 [0,197; 0,216]** | — | 0,19 | 0,219 |
| 20 aer. (25%) | Sinalização | 11.587 | 0,361 | 0,187 | **0,183 [0,167; 0,200]** | — | 0,21 | 0,184 |
| 20 aer. (25%) | Restituição | 35.261 | 0,395 | 0,270 | **0,265 [0,255; 0,275]** | — | 0,33 | 0,261 |

Na base de 20 aeroportos, o DML usa a **subamostra estratificada de 25%** por aeroporto × período, **semente 42**: 115.405 respondentes, contra 85 mil na base inteira de 2023. As estimativas ficam praticamente iguais às de 2023 e os ICs encolhem.

Por aeroporto (MVP): estimativas entre 0,20 e 0,35. O destaque é Restituição em SBSP, com 0,350.

### 3.4 DiD Congonhas (SBSP: Infraero → Aena em 17/10/2023)

**Desenho**
- Pré: 2021T1 a 16/10/2023. Pós: 17/10/2023 a 2026T2, definido pela data da entrevista; nas linhas sem data, pelo mês.
- Controles: os 14 aeroportos sem troca de operador na série (SBBR, SBCF, SBCY, SBFL, SBFZ, SBGL, SBGR, SBKP, SBMO, SBPA, SBRF, SBRJ, SBSV, SBVT).
- Efeitos fixos de aeroporto e de PERIODO; covariáveis do respondente (30 dummies + PROCESSO).
- Inferência de referência: **permutação** (placebo no espaço, J = 14, menor p possível = 1/15 = 0,067). Também reportados HC1, CR1 e o wild cluster bootstrap restrito (Webb, B = 9.999).
- n = 348.170. SBSP: 22.285 entrevistas (9.204 pré, 13.081 pós). Controles: 325.885.

| Especificação (y = média 1–5) | δ | IC 95% HC1 | EP cluster | p HC1 | p perm. | MDE |
|---|---|---|---|---|---|---|
| (1) Principal | **+0,003** | [−0,015; 0,022] | 0,025 | 0,72 | **1,00** (WCR 0,90) | 0,30 |
| (2) EF mensais | +0,004 | [−0,015; 0,022] | 0,026 | 0,69 | 1,00 | 0,30 |
| (3) Sem covariáveis | +0,004 | [−0,015; 0,023] | 0,026 | 0,67 | 0,93 | 0,31 |
| (4) Sem SBRJ | +0,006 | [−0,013; 0,024] | 0,027 | 0,56 | 1,00 | 0,31 |
| (5) Sem Rio | +0,000 | [−0,018; 0,019] | 0,029 | 0,98 | 1,00 | 0,32 |
| (6) Só SP (GRU, VCP) | +0,041 | [0,019; 0,062] | — | <0,001 | — | — |
| (7) Só SBRJ | −0,035 | [−0,061; −0,010] | — | 0,006 | — | — |
| (8) Painel agregado (MQP) | +0,003 | [−0,056; 0,062] | — | 0,91 | 0,93 | 0,31 |

| Especificação (P(nota 5)) | δ | IC 95% HC1 | p cluster | p perm. |
|---|---|---|---|---|
| (1) Principal | **+0,038** | [0,025; 0,051] | 0,037 | **0,67** (WCR 0,44) |
| (3) Sem covariáveis | +0,037 | | | |
| (4) Sem SBRJ | +0,037 | | | |
| (5) Sem Rio | +0,032 | | | |
| (6) Só SP | +0,054 | | | |
| (7) Só SBRJ | +0,041 | | | |

P(nota ≤ 3), especificação (1): **+0,017** [0,009; 0,025], p perm. 0,40. O pós tem ao mesmo tempo mais notas 5 e mais notas ≤ 3, ou seja, a distribuição ficou mais dispersa.

Horizonte, especificação (1) com dois termos:

| Desfecho | Curto prazo (17/10/2023–2024T2; n pós = 3.525) | Médio prazo (2024T3–2026T2; n pós = 9.556) |
|---|---|---|
| Média | −0,002 (p perm. 1,00) | +0,006 (p perm. 1,00) |
| P(5) | +0,023 (p perm. 0,60) | +0,043 (p perm. 0,67) |

Event study mensal, médias relativas a set./2023:

| | Pós | Curto | Médio |
|---|---|---|---|
| Média | +0,029 (p perm. 0,87) | +0,022 | +0,032 |
| P(5) | +0,084 (p perm. 0,40) | | |

Robustez adicional:

| Verificação | δ (média) | p perm. | δ (P(5)) |
|---|---|---|---|
| Sem 2023T4 | +0,005 | 0,93 | +0,042 |
| Sem SBPA (enchentes) | +0,002 | 1,00 | |
| Placebo temporal 17/10/2022 (só pré) | −0,077 | 0,47 | −0,085 |

O placebo temporal é significativo pelo HC1, mas não pela permutação; isso reflete a tendência pré de SBSP, discutida a seguir.

**Tendências paralelas** (event study trimestral, referência 2023T3; leads de 2021T1 a 2023T2; lags até 2026T2):
- **SBSP vinha caindo em relação aos controles**. Leads em pontos: +0,22 (2021T1), +0,17, +0,13, +0,08, +0,11, −0,03, −0,10, +0,03, 0,00.
- Tendência linear pré de −0,022 ponto por trimestre (EP 0,002). Wald dos 9 leads: χ² = 120, p < 0,001 pelo HC1.
- **Pela permutação, porém, o padrão não é atípico**: p dos leads 0,47; p da tendência linear 0,27. Vários controles têm trajetórias tão ou mais erráticas, e a figura mostra uma faixa de placebos de ±0,4.
- Lags: entre −0,03 e +0,12. Média mensal pós relativa a set./2023 = +0,03.
- P(nota 5): mesmo padrão. Tendência pré −0,018 por trimestre; o P(5) bruto de SBSP sobe de 0,27 (2023T3) para 0,43–0,47 em 2025T2–2026T2, enquanto os controles ficam estáveis em torno de 0,55.

**Leitura**
- **Não há efeito detectável da troca de operador sobre a satisfação média.** A estimativa pontual fica perto de zero em todas as especificações com o grupo de controle completo, e o p de permutação fica entre 0,93 e 1,00.
- O MDE é de cerca de 0,30 ponto: o teste só detectaria efeitos grandes. Com 5 anos de série o MDE **aumentou** em relação a 2023 (0,21), porque a heterogeneidade entre os controles cresce no horizonte longo.
- Em P(nota 5) há um aumento de cerca de 4 p.p. que é significativo pelo HC1 e pelo cluster, mas **não pela permutação** (0,67).
- A tendência pré negativa de SBSP indica que a média do pré inteiro é uma referência fraca. Contra 2023T3, o efeito pós é positivo (+0,03 a +0,08 em P(5)), mas não se distingue dos placebos.
- Comparação com 2023: antes δ(y) = +0,029 com p perm. 0,63 e δ(y5) = +0,050 com p 0,26. A conclusão é a mesma (nulo), com MDE maior.

### 3.5 Secundário: Belém (SBBE, NOA em 04/09/2023)

Mesmos 14 controles, SBSP fora da amostra. SBBE tem 22.795 entrevistas, 14.232 no pós.

| | δ média | p perm. | WCR | δ P(5) | p perm. | δ P(≤3) | p perm. |
|---|---|---|---|---|---|---|---|
| Principal | **−0,170** (EP HC1 0,011) | 0,20 | 0,43 | −0,100 | 0,20 | +0,044 | 0,13 |
| Sem covariáveis | −0,143 | 0,20 | | | | | |
| Sem SBRJ | −0,167 | 0,21 | | | | | |

Horizonte (média): curto −0,090; médio −0,203, ambos com p perm. 0,13.

Leads trimestrais (referência 2023T2): HC1 p = 0,023, permutação p = 0,67; tendência pré −0,010 por trimestre (p perm. 0,53).

Leitura: a estimativa é negativa e grande (é o 3º maior |δ| entre 15 aeroportos), mas **não é significativa pela permutação**. A série bruta mostra uma queda forte em 2025T2–T3 (média de cerca de 3,7) com recuperação em 2025T4–2026T2. Isso é compatível com as **obras no aeroporto antes da COP30** (nov./2025), o que **confunde** o efeito do operador no médio prazo. Esse é um juízo meu e precisa ser confirmado antes de entrar no texto.

---

## 4. Problemas de dados encontrados (fora do meu escopo, mas afetam outros)

1. **`mpor_2021_2026_20.parquet`**, gerado por `10_serie_historica/01`:
   - Há **33.600 DATAs gravadas como serial do Excel** (aparecem como 1970 + n ns), em 2024T1–T2 e parte de 2025T3–T4.
   - Há **7.000 NaT** em 2022T4.
   - Há **610 linhas vazias** em 2022T2.
   - O `00_base20.py` corrige isso só na base de econometria. Quem usar DATA direto da série (DiD, app, NLP por data) precisa da mesma correção, ou o `01_carregar_serie.py` deve corrigi-la na origem.
2. **libomp**: o symlink `/opt/homebrew/opt/libomp` sumiu em 06/11. É preciso usar `DYLD_LIBRARY_PATH` ou rodar `brew reinstall libomp`.
3. SBPA não tem coleta em 2024T3–T4 (enchentes de 2024). O DiD sem SBPA está na robustez.
4. `texto/_paralelo/econometria.md` ainda cita `regime_operador_2023.csv`, que agora é `regime_operador_serie.csv`. Não editei.
