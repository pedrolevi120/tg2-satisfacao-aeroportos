#!/usr/bin/env bash
# Pipeline completo do pacote modelos20 (T5), série 2021T1–2026T2.
# Uso: bash codigos/06_modelagem_20/rodar_tudo.sh   (~1 h; ver codigos/10_serie_historica/notas_modelagem.md)
# Teste rápido sem sobrescrever artefatos: AMOSTRA_FRAC=0.1 SAIDA_TESTE=/caminho/tmp bash rodar_tudo.sh
# 02_robustez_mvp.py (Tarefa 1, MVP) é mantido por outro pacote e não roda aqui.
set -euo pipefail
cd "$(dirname "$0")"
source ../../.venv/bin/activate
export PYTHONUNBUFFERED=1
# libomp do Homebrew sem o link em /opt/homebrew/opt (xgboost/lightgbm não carregam): usa o Cellar
if [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && ls -d /opt/homebrew/Cellar/libomp/*/lib >/dev/null 2>&1; then
  export DYLD_LIBRARY_PATH="$(ls -d /opt/homebrew/Cellar/libomp/*/lib | tail -1)"
fi
python 01_carregar_mpor_20.py    > resultados/log_01.txt 2>&1   # confere/gera mpor_2021_2026_20.parquet
python 03_base_features_20.py    > resultados/log_03.txt 2>&1   # Tarefa 2 (+ contexto por aeroporto × período)
python 04_comparar_modelos_20.py > resultados/log_04.txt 2>&1   # Tarefa 3 (9 modelos, IC bootstrap)
python 05_loao_20.py             > resultados/log_05.txt 2>&1   # Tarefa 3b (LOAO)
python 06_shap_20.py             > resultados/log_06.txt 2>&1   # Tarefa 4 (SHAP, rankings, ranking por ano)
python ../10_serie_historica/03_validacao_temporal.py > ../10_serie_historica/resultados/log_validacao_temporal.txt 2>&1
