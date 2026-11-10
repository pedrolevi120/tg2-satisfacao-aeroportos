"""
Série histórica — carrega as bases trimestrais do portal de 2021 T1 a 2026 T2 (mesma metodologia e
mesmo leiaute de colunas de 2023) e grava uma base única no mesmo formato de mpor_2023_20.parquet.

Entrada:  dados/brutos/mpor/portal/pesquisa-de-satisfacao-do-passageiro-{t}-tri-{ano}.{csv,xlsx}
Saída:    dados/processados/mpor_2021_2026_20.parquet  (+ resultados/cobertura_periodos.csv)

Fora da série: 2013–2019 (metodologia antiga, arquivo único) e 2020 T1 (nova metodologia, mas leiaute
diferente e isolado pela pandemia: não há pesquisa de abr/2020 a jul/2021, exceto 2021 T1).
"""
from pathlib import Path
import re
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PORTAL = ROOT / "dados" / "brutos" / "mpor" / "portal"
PROC = ROOT / "dados" / "processados"
OUT = Path(__file__).resolve().parent / "resultados"
OUT.mkdir(exist_ok=True)

_spec = importlib.util.spec_from_file_location("insp", Path(__file__).with_name("00_inspecionar.py"))
insp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(insp)

RENOMEAR = {"UTILIZOU O ESTACIONAMENTO?": "UTILIZOU O ESTACIONAMENTO"}
# variantes de "não sabe / não respondeu" -> forma usada em 2023
NSNR = {"NS": "NS/NR", "NR": "NS/NR", "NS/NR ": "NS/NR", "Ns/Nr": "NS/NR"}


def _renda(v):
    if v is None or pd.isna(v):
        return v
    t = str(v).lower()
    for chave, rot in [("até 1", "Até 1 salário mínimo"), ("1 e 2", "Entre 1 e 2 SM"), ("1 and 2", "Entre 1 e 2 SM"),
                       ("2 e 4", "Entre 2 e 4 SM"), ("2 and 4", "Entre 2 e 4 SM"), ("4 e 10", "Entre 4 e 10 SM"),
                       ("4 and 10", "Entre 4 e 10 SM"), ("10 e 20", "Entre 10 e 20 SM"), ("10 and 20", "Entre 10 e 20 SM"),
                       ("mais de 20", "Mais de 20 SM"), ("more than 20", "Mais de 20 SM"),
                       ("desempregado", "Desempregado/Sem Renda"), ("unemployed", "Desempregado/Sem Renda")]:
        if chave in t:
            return rot
    return v


def _viagens(v):
    if v is None or pd.isna(v):
        return v
    t = str(v).lower()
    if t.startswith("1"):
        return "1 viagem"
    if "6 a 10" in t:
        return "6 a 10"
    if "11" in t:
        return "Mais de 11"
    return v


# rótulos que mudaram de grafia entre anos -> forma única
PADRONIZAR = {
    "NACIONALIDADE": {"Brasileiro": "Brasileira", "Estrangeiro": "Estrangeira"},
    "IDADE": {"56 a 64 anos": "56 a 65 anos"},
    "ESCOLARIDADE": {"Master's degree": "Mestrado"},
}


def padronizar(df):
    for c, m in PADRONIZAR.items():
        df[c] = df[c].replace(m)
    df["RENDA FAMILIAR"] = df["RENDA FAMILIAR"].map(_renda)
    df["QUANTIDADE DE VIAGENS NOS ÚLTIMOS 12 MESES"] = df["QUANTIDADE DE VIAGENS NOS ÚLTIMOS 12 MESES"].map(_viagens)
    return df


def carregar():
    ref = pd.read_parquet(PROC / "mpor_2023_20.parquet")
    colunas = [c for c in ref.columns if c not in ("TRIMESTRE", "ANO")]
    dfs, cob = [], []
    arqs = sorted([p for p in PORTAL.glob("pesquisa-de-satisfacao-do-passageiro-*-tri-*")],
                  key=insp.periodo)
    for p in arqs:
        ano, tri = insp.periodo(p)
        if ano < 2021:
            continue
        df = insp.ler(p)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=RENOMEAR)
        for c in colunas:
            if c not in df:
                df[c] = None
        df = df[colunas].copy()
        df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
        for c in colunas:
            if c in ("DATA",):
                continue
            if str(ref[c].dtype) in ("str", "object", "string"):
                s = df[c].astype(str).str.strip()
                s = s.where(df[c].notna(), None)
                # notas inteiras lidas como float ("5.0") voltam a "5"
                s = s.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
                df[c] = s.replace(NSNR)
        df = padronizar(df)
        df["TRIMESTRE"], df["ANO"] = tri, ano
        df["PERIODO"] = f"{ano}T{tri}"
        meses = sorted(df["DATA"].dropna().dt.month.unique().tolist())
        cob.append({"ano": ano, "tri": tri, "linhas": len(df), "aeroportos": df["AEROPORTO"].nunique(),
                    "meses": ",".join(map(str, meses)), "arquivo": p.name})
        dfs.append(df)
        print(f"  {ano}T{tri}: {len(df):>6} linhas, {df['AEROPORTO'].nunique()} aeroportos, meses {meses}")
    base = pd.concat(dfs, ignore_index=True)
    # CHAVE não existe em 2021–2022: cria identificador único da série
    base["CHAVE_SERIE"] = range(1, len(base) + 1)
    return base, pd.DataFrame(cob)


if __name__ == "__main__":
    base, cob = carregar()
    print(f"Total: {len(base):,} linhas, {base['AEROPORTO'].nunique()} aeroportos")
    print(base.groupby("ANO").size())
    base["CHAVE"] = pd.to_numeric(base["CHAVE"], errors="coerce")
    for c in base.columns:
        if base[c].dtype == object:
            base[c] = base[c].astype("string")
    base.to_parquet(PROC / "mpor_2021_2026_20.parquet", index=False)
    cob.to_csv(OUT / "cobertura_periodos.csv", index=False)
    print("  -> dados/processados/mpor_2021_2026_20.parquet")
