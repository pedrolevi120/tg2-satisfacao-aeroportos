# Extensão para a série 2021T1–2026T2 — especificação comum

Decisão do autor (08/11/2026): refazer todas as análises do TG com todas as bases trimestrais do portal
com a metodologia atual (2021T1 a 2026T2), em vez de só 2023. Resultados antigos (só 2023) estão salvos em
`versao_enviada_prof/codigos_e_resultados_2023.tar.gz`; não é preciso preservar arquivos de saída antigos.

## Dados
- **Base**: `dados/processados/mpor_2021_2026_20.parquet` (462.616 linhas, 20 aeroportos, gerada por
  `codigos/10_serie_historica/01_carregar_serie.py`).
  - Tem as mesmas 98 colunas e os mesmos tipos de `mpor_2023_20.parquet`, mais:
    - `ANO` (int);
    - `TRIMESTRE` (int, 1–4);
    - `PERIODO` (str, "2021T1" … "2026T2"; 21 valores, porque 2021T2 não existe);
    - `CHAVE_SERIE` (int, identificador único da série).
  - `CHAVE` só existe de 2023 em diante e é única dentro de 2023.
  - As linhas de 2023 são idênticas às da base antiga (os mesmos arquivos), exceto pela padronização de
    rótulos de RENDA, viagens, nacionalidade, idade e escolaridade.
- **Lógica de salto**: verificada idêntica nos 21 períodos (subitem só quando o item-pai ≤ 3).
- **Contexto por aeroporto × período**: `dados/processados/contexto_aeroportos_20_serie.csv`.
  - Chaves: `icao, ano, trimestre, periodo`.
  - Regime e operador: `regime_gestao` (concessionada / infraero / misto), `frac_dias_concessao`, `operador`,
    `data_assuncao`.
  - Movimento no trimestre (ANAC Dados Estatísticos): `pax_trim_mi`, `decolagens_trim_k`.
  - Fixos por aeroporto: `pistas`, `populacao_2022_mi`, `pib_per_capita_k`, `idh_2010`,
    `renda_per_capita_2010`, `populacao_2022_polo_mi`, `pib_per_capita_polo_k`, `idh_2010_polo`.
- **Datas de troca de operador** dentro da série:
  - SBEG 11/01/2022, SBGO 24/03/2022, SBCT 31/03/2022 (Infraero → concessão);
  - SBBE 04/09/2023, SBSP 17/10/2023 (Infraero → concessão);
  - SBSG 19/02/2024 (Inframerica → Zurich, ambas privadas).
  - SBRJ é Infraero o tempo todo.
- **Lacunas de coleta** (descrições do portal):
  - sem pesquisa de abr/2020 a jul/2021, exceto 2021T1;
  - 2021T3 só ago–set;
  - 2022T4 só outubro e 2023T1 só março (troca da empresa executora);
  - SBPA sem dados de mai–dez/2024 (enchentes);
  - pesquisa suspensa em 16/06/2026 (corte de orçamento).

## Regras de adaptação
1. **Período**: onde o código usava `TRIMESTRE` como efeito fixo, categoria, chave de junção, cluster ou
   eixo de gráfico, use `PERIODO`. Comparações numéricas como `TRIMESTRE == 4` ou `<= 3` precisam ser
   reescritas com datas (`DATA`) ou com `PERIODO`.
2. **Sem números fixos de 2023**: asserts de tamanho, `n=84.662`, `TOP4` fixo, "MAE 0,352" etc. passam a ser
   calculados ou removidos. Textos de figuras e tabelas: "2021–2026" em vez de "2023".
3. **Split**: o mesmo de antes (70/15/15 aleatório, estratificado por aeroporto × nota, seed 42), agora sobre
   a série inteira. A validação temporal (treino 2021–2024, teste 2025–2026) é um experimento novo, separado.
4. **Junções por passageiro**: use `CHAVE_SERIE`. Para casar com artefatos de 2023 (amostra manual de
   comentários), use `ANO == 2023` + `CHAVE` + `AEROPORTO`.
5. **Caminhos de saída**: os mesmos de antes (sobrescrever). Arquivos com "2023" no nome podem ganhar nome
   novo, desde que todos os consumidores sejam atualizados.
6. **Memória (Mac com 24 GB, várias sessões)**: teste em amostra (≤ 10% das linhas) e nunca rode o pipeline
   completo de um pacote em paralelo com outro. As rodadas completas são feitas pela sessão principal, uma
   por vez. Bootstraps grandes devem ser processados em blocos.
7. **Análises novas que a série permite** (cada pacote, quando couber):
   - ranking de prioridades por ano (estabilidade no tempo);
   - evolução da satisfação por período;
   - DiD de Congonhas com pós-tratamento até 2026T2;
   - temas dos comentários por ano.
