"""Inspeciona as bases trimestrais do portal (2020 T1 a 2026 T2): linhas, colunas, aeroportos e
diferenças de colunas em relação a 2023 T2 (base de referência do TG)."""
from pathlib import Path
import re, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PORTAL = ROOT / "dados" / "brutos" / "mpor" / "portal"


def ler(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        for enc in ("utf-8", "latin1"):
            try:
                return pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
    if path.suffix == ".ods":
        return pd.read_excel(path, sheet_name=0, engine="odf")
    # xlsx: cabeçalho na linha 2 (como em 2023); detecta
    raw = pd.read_excel(path, sheet_name=0, header=None, nrows=5)
    h = next(i for i in range(5) if raw.iloc[i].astype(str).str.contains("AEROPORTO").any())
    return pd.read_excel(path, sheet_name=0, header=h)


def periodo(p: Path):
    m = re.search(r"(\d)-tri-(\d{4})", p.name)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"1-trim-2020", p.name)
    return (2020, 1) if m else None


if __name__ == "__main__":
    ref = ler(PORTAL / "pesquisa-de-satisfacao-do-passageiro-2-tri-2023.xlsx")
    refc = [str(c).strip() for c in ref.columns]
    arqs = sorted([p for p in PORTAL.iterdir() if periodo(p)], key=periodo)
    for p in arqs:
        if len(sys.argv) > 1 and str(periodo(p)[0]) not in sys.argv[1:]:
            continue
        df = ler(p)
        cols = [str(c).strip() for c in df.columns]
        falta = [c for c in refc if c not in cols]
        sobra = [c for c in cols if c not in refc]
        sg = "SATISFAÇÃO GERAL" in cols
        print(f"\n### {periodo(p)} {p.name}: {len(df)} linhas, {len(cols)} col, "
              f"{df['AEROPORTO'].nunique() if 'AEROPORTO' in df else '?'} aerop., SAT GERAL={sg}")
        print("  faltam vs 2023T2:", falta[:40])
        print("  a mais:", sobra[:40])
