#!/usr/bin/env bash
# =============================================================
#  Compila o TG localmente em PDF.
#  Uso: ./compile.sh           (compila tudo, com biblio)
#       ./compile.sh quick     (1 passada, sem biblio — útil pra preview rápido)
#       ./compile.sh clean     (apaga arquivos temporários)
# =============================================================
set -e

cd "$(dirname "$0")"   # garante que estamos em texto/
JOB="main"

case "${1:-full}" in
  clean)
    echo "Limpando arquivos auxiliares..."
    rm -f $JOB.aux $JOB.log $JOB.toc $JOB.lof $JOB.lot \
          $JOB.bbl $JOB.bcf $JOB.blg $JOB.run.xml \
          $JOB.out $JOB.fdb_latexmk $JOB.fls $JOB.synctex.gz
    echo "Pronto."
    ;;

  quick)
    echo "Compilação rápida (sem biblio)..."
    pdflatex -interaction=nonstopmode -halt-on-error $JOB.tex
    echo ""
    echo "PDF gerado em $JOB.pdf (sem citações finais)."
    ;;

  full|*)
    echo "Compilação completa (4 passadas com biber)..."
    pdflatex -interaction=nonstopmode $JOB.tex
    # O biber "universal" do TeX Live chama o lipo para extrair a arquitetura; com o lipo do
    # Xcode isso falha ("extracting arm64 binary with lipo failed"). Fallback: binário fino.
    if ! biber $JOB; then
      BIBER_FINO="${TMPDIR:-/tmp}/biber-$(uname -m)"
      [[ -x "$BIBER_FINO" ]] || lipo "$(command -v biber)" -thin "$(uname -m)" -output "$BIBER_FINO"
      "$BIBER_FINO" $JOB
    fi
    pdflatex -interaction=nonstopmode $JOB.tex
    pdflatex -interaction=nonstopmode $JOB.tex
    echo ""
    echo "✅ PDF gerado em $JOB.pdf"
    [[ "$(uname)" == "Darwin" ]] && open $JOB.pdf
    ;;
esac
