#!/usr/bin/env bash
# Experimentos 09 (E1–E7), série 2021T1–2026T2. Uso: bash codigos/09_experimentos/rodar_tudo.sh
# (~4 h com 4 threads; ver codigos/10_serie_historica/notas_modelagem.md).
# Só lê/importa de ../06_modelagem_20 — não altera o pipeline nem o texto.
# E1 com N_TRIALS=25 por padrão aqui (no script o padrão é 50) para caber no tempo.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export N_JOBS="${N_JOBS:-4}" N_TRIALS="${N_TRIALS:-25}" PYTHONUNBUFFERED=1
# libomp do Homebrew sem o link em /opt/homebrew/opt (xgboost/lightgbm não carregam): usa o Cellar
if [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && ls -d /opt/homebrew/Cellar/libomp/*/lib >/dev/null 2>&1; then
  export DYLD_LIBRARY_PATH="$(ls -d /opt/homebrew/Cellar/libomp/*/lib | tail -1)"
fi
rm -f resultados/_cache_e5_retreinos.npz
$PY e1_ajuste_hiperparametros.py > resultados/log_e1.txt 2>&1   # Optuna, 3 modelos
$PY e2_ebm.py                    > resultados/log_e2.txt 2>&1
$PY e3_ordinal.py                > resultados/log_e3.txt 2>&1
$PY e4_perfil.py                 > resultados/log_e4.txt 2>&1
$PY e5_estabilidade_ranking.py   > resultados/log_e5.txt 2>&1   # usa E1 e E2
$PY e6_kano_penalty_reward.py    > resultados/log_e6.txt 2>&1
$PY e7_sintese.py                > resultados/log_e7.txt 2>&1   # lê todos
rm -f resultados/_cache_e5_retreinos.npz
