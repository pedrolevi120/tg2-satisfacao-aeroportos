"""
SO1.5 — Base MVP (GRU/CGH/VCP) a partir da série histórica MPor 2021T1–2026T2.

A leitura dos 21 trimestres do portal (com padronização de rótulos) é feita em
codigos/10_serie_historica/01_carregar_serie.py; aqui só se filtra o MVP.
O nome do arquivo de saída foi mantido (mpor_2023_mvp.parquet) porque os consumidores
o leem por esse caminho; o conteúdo agora é a série inteira (colunas ANO, TRIMESTRE,
PERIODO "2021T1"…"2026T2", CHAVE_SERIE).

Entrada:  dados/processados/mpor_2021_2026_20.parquet
          dados/brutos/mpor/2023/pesquisa_satisfacao_2023_T2.xlsx (aba LEGENDA, só p/ o dicionário)
Saída:    dados/processados/mpor_2023_mvp.parquet
          codigos/02_preprocessamento/dicionario_colunas.{csv,md}
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRUTOS = ROOT / "dados" / "brutos" / "mpor" / "2023"
PROCESSADOS = ROOT / "dados" / "processados"
PROCESSADOS.mkdir(parents=True, exist_ok=True)
SERIE = PROCESSADOS / "mpor_2021_2026_20.parquet"

AEROPORTOS_MVP = ["SBGR", "SBSP", "SBKP"]


def gerar_dicionario_colunas() -> pd.DataFrame:
    """Lê LEGENDA do T2 e gera dicionário de colunas em formato tabular."""
    path = BRUTOS / "pesquisa_satisfacao_2023_T2.xlsx"
    legenda = pd.read_excel(path, sheet_name="LEGENDA", header=None)
    # Linha 5 do raw é o header (FORMULÁRIO | CÓDIGO | QUESTÃO | RESPOSTAS)
    legenda.columns = ["FORMULARIO", "CODIGO", "QUESTAO", "RESPOSTAS"]
    legenda = legenda.iloc[6:].reset_index(drop=True)
    return legenda


def main():
    print("=== Carregando série 2021T1–2026T2 ===")
    df_full = pd.read_parquet(SERIE)
    print(f"Total bruto: {len(df_full):>6} respondentes, {df_full['PERIODO'].nunique()} períodos")

    print(f"\n=== Filtrando MVP ({', '.join(AEROPORTOS_MVP)}) ===")
    df_mvp = df_full[df_full["AEROPORTO"].isin(AEROPORTOS_MVP)].copy().reset_index(drop=True)
    print(f"Após filtro:  {len(df_mvp):>6} respondentes")
    print("\nQuebra por aeroporto x período:")
    print(df_mvp.groupby(["PERIODO", "AEROPORTO"]).size().unstack(fill_value=0))

    print("\n=== Salvando Parquet ===")
    out_path = PROCESSADOS / "mpor_2023_mvp.parquet"
    # Normalizar colunas object para string (evita erro de tipo misto no pyarrow)
    for col in df_mvp.select_dtypes(include="object").columns:
        df_mvp[col] = df_mvp[col].astype(str).where(df_mvp[col].notna(), None)
    df_mvp.to_parquet(out_path, index=False)
    print(f"  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    print("\n=== Gerando dicionário de colunas ===")
    dic = gerar_dicionario_colunas()
    dic_out = ROOT / "codigos" / "02_preprocessamento" / "dicionario_colunas.csv"
    dic.to_csv(dic_out, index=False)
    print(f"  → {dic_out} ({len(dic)} linhas)")

    # Markdown legível
    md_lines = ["# Dicionário de colunas — Pesquisa MPor 2023\n",
                "Extraído da aba LEGENDA do arquivo `pesquisa_satisfacao_2023_T2.xlsx`.\n\n",
                "| Formulário | Código | Questão | Respostas |\n",
                "|---|---|---|---|\n"]
    for _, row in dic.iterrows():
        cells = [str(row[c]).replace("|", "/").replace("\n", " ")[:100] for c in dic.columns]
        md_lines.append("| " + " | ".join(cells) + " |\n")
    md_out = ROOT / "codigos" / "02_preprocessamento" / "dicionario_colunas.md"
    md_out.write_text("".join(md_lines))
    print(f"  → {md_out}")

    print("\n=== Resumo final do dataset MVP ===")
    print(f"  Respondentes: {len(df_mvp)}")
    print(f"  Aeroportos:   {df_mvp['AEROPORTO'].nunique()}")
    print(f"  Períodos:     {df_mvp['PERIODO'].nunique()} ({df_mvp['PERIODO'].min()}–{df_mvp['PERIODO'].max()})")
    print(f"  Colunas:      {df_mvp.shape[1]}")
    # Quick check: SATISFAÇÃO GERAL
    if "SATISFAÇÃO GERAL" in df_mvp.columns:
        print(f"\n  Distribuição da SATISFAÇÃO GERAL:")
        print(df_mvp["SATISFAÇÃO GERAL"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
