#!/usr/bin/env bash
# Gera texto/overleaf_TG.zip pronto para "New Project > Upload Project" no Overleaf.
# Estrutura do zip: main.tex, referencias.bib, midias/figuras/*.pdf, midias/imagens/*.png
set -e
cd "$(dirname "$0")"
TMP=$(mktemp -d)
cp main.tex referencias.bib "$TMP/"
mkdir -p "$TMP/midias/figuras" "$TMP/midias/imagens"
cp ../midias/figuras/*.pdf "$TMP/midias/figuras/"
cp ../midias/imagens/*.png "$TMP/midias/imagens/" 2>/dev/null || true
rm -f overleaf_TG.zip
(cd "$TMP" && zip -qr "$OLDPWD/overleaf_TG.zip" .)
rm -rf "$TMP"
echo "OK: texto/overleaf_TG.zip ($(du -h overleaf_TG.zip | cut -f1))"
echo "No Overleaf: New Project > Upload Project > overleaf_TG.zip; Menu > Compiler: pdfLaTeX; bibliografia: Biber."
