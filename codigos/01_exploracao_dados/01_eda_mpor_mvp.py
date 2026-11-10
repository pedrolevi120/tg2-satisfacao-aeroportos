"""
SO1.6 — Análise Exploratória dos Dados (EDA) da MPor 2023 MVP.

Entrada: dados/processados/mpor_2023_mvp.parquet
Saída:   midias/figuras/eda_*.pdf
         codigos/01_exploracao_dados/resumo_eda.md
"""
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_modelagem_20"))
from comum import curto  # nomes legíveis dos atributos (ex.: FILAS2 -> filas inspeção)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "dados" / "processados" / "mpor_2023_mvp.parquet"
FIG_DIR = ROOT / "midias" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = ROOT / "codigos" / "01_exploracao_dados" / "resumo_eda.md"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 10

NOME_AERO = {"SBGR": "Guarulhos (GRU)", "SBSP": "Congonhas (CGH)", "SBKP": "Viracopos (VCP)"}
COR_AERO = {"SBGR": "#1f77b4", "SBSP": "#ff7f0e", "SBKP": "#2ca02c"}
ORDEM_AERO = ["SBGR", "SBSP", "SBKP"]


def to_numeric_likert(s: pd.Series) -> pd.Series:
    """Converte série Likert string para numérico (1-5); NaN para 'NS/NR' e não-numéricos."""
    return pd.to_numeric(s, errors="coerce")


def identificar_likert(df: pd.DataFrame) -> list[str]:
    """Identifica colunas com valores predominantemente em {1,2,3,4,5}."""
    likert_cols = []
    for col in df.columns:
        if col in {"AEROPORTO", "TRIMESTRE", "ANO", "CHAVE", "DATA", "MÊS",
                   "INÍCIO COLETA", "FIM COLETA", "PROCESSO"}:
            continue
        valores = df[col].dropna().astype(str).str.strip().unique()
        # se >80% dos valores são '1'..'5' ou 'NS/NR', consideramos Likert
        likert_set = {"1", "2", "3", "4", "5"}
        likert_extras = likert_set | {"NS/NR", "Não Existe", "nan"}
        if len(valores) == 0:
            continue
        share = sum(v in likert_extras for v in valores) / len(valores)
        if share >= 0.8 and len(set(valores) & likert_set) >= 3:
            likert_cols.append(col)
    return likert_cols


def main():
    print("=== Carregando ===")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df)} respondentes, {df.shape[1]} colunas")

    likert_cols = identificar_likert(df)
    # Remover o próprio target da lista de features
    likert_cols = [c for c in likert_cols if c != "SATISFAÇÃO GERAL"]
    print(f"  {len(likert_cols)} colunas Likert detectadas (excluindo target)")

    # Numerificar Likert
    df_lik = df[likert_cols].apply(to_numeric_likert)
    df_lik["AEROPORTO"] = df["AEROPORTO"]
    df_lik["TRIMESTRE"] = df["TRIMESTRE"]

    # === Figura 1: Distribuição da SATISFAÇÃO GERAL ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    target = pd.to_numeric(df["SATISFAÇÃO GERAL"], errors="coerce")
    target.value_counts().sort_index().plot.bar(
        ax=axes[0], color="#1f77b4", edgecolor="black", alpha=0.85
    )
    axes[0].set_title("Distribuição geral da Satisfação Geral")
    axes[0].set_xlabel("Nota Likert")
    axes[0].set_ylabel("Respondentes")

    # Faceted por aeroporto
    counts = (df.assign(SG=pd.to_numeric(df["SATISFAÇÃO GERAL"], errors="coerce"))
              .groupby(["AEROPORTO", "SG"]).size().unstack(fill_value=0))
    counts = counts.div(counts.sum(axis=1), axis=0) * 100  # percentuais
    counts.reindex(ORDEM_AERO).plot.bar(
        ax=axes[1], stacked=False, edgecolor="black", alpha=0.85,
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, counts.shape[1]))
    )
    axes[1].set_title("Distribuição percentual por aeroporto")
    axes[1].set_xticklabels([NOME_AERO[a] for a in ORDEM_AERO], rotation=15, ha="right")
    axes[1].set_ylabel("% dos respondentes")
    axes[1].legend(title="Nota", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = FIG_DIR / "eda_satisfacao_geral.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # === Figura 2: Satisfação média por atributo (top 20) ===
    medias = df_lik[likert_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    medias.tail(20).plot.barh(ax=ax, color="#2ca02c", edgecolor="black", alpha=0.85)
    medias.head(10).plot.barh(ax=ax, color="#d62728", edgecolor="black", alpha=0.85)
    ax.set_title("Atributos com maior e menor satisfação média (MVP)")
    ax.set_xlabel("Nota Likert média (1-5)")
    ax.axvline(3, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = FIG_DIR / "eda_atributos_top_bottom.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # === Figura 3: Completude por atributo (heatmap aeroporto x atributo) ===
    completude = (df_lik[likert_cols + ["AEROPORTO"]]
                  .groupby("AEROPORTO").apply(lambda g: g.notna().mean()))
    completude = completude.drop(columns=["AEROPORTO"], errors="ignore")
    fig, ax = plt.subplots(figsize=(min(0.3 * len(likert_cols), 20), 4))
    sns.heatmap(completude.reindex(ORDEM_AERO), cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Completude"})
    ax.set_yticklabels([NOME_AERO[a] for a in ORDEM_AERO], rotation=0)
    ax.set_title("Completude (% não-nulos) por atributo Likert e aeroporto")
    ax.set_xticklabels([curto(c) for c in completude.columns], rotation=90, ha="center", fontsize=6)
    plt.tight_layout()
    out = FIG_DIR / "eda_completude_atributos.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # === Figura 4: Correlação entre top atributos e SATISFAÇÃO GERAL ===
    df_corr = df_lik[likert_cols].copy()
    df_corr["TARGET"] = pd.to_numeric(df["SATISFAÇÃO GERAL"], errors="coerce")
    corrs = df_corr.corr()["TARGET"].drop("TARGET").sort_values(ascending=False)
    top = pd.concat([corrs.head(15), corrs.tail(5)])
    fig, ax = plt.subplots(figsize=(8, 7))
    top.plot.barh(ax=ax, color=["#2ca02c" if x > 0 else "#d62728" for x in top],
                  edgecolor="black", alpha=0.85)
    ax.set_title("Correlação de Pearson com Satisfação Geral (top 15 + bottom 5)")
    ax.set_xlabel("Pearson r")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out = FIG_DIR / "eda_correlacao_target.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # === Resumo .md ===
    lines = [
        "# Resumo da EDA inicial — Pesquisa MPor 2023 MVP",
        "",
        f"- **Dataset**: `{PARQUET.relative_to(ROOT)}`",
        f"- **Respondentes**: {len(df):,}",
        f"- **Aeroportos**: {df['AEROPORTO'].nunique()} ({', '.join(sorted(df['AEROPORTO'].unique()))})",
        f"- **Trimestres**: {df['TRIMESTRE'].nunique()}",
        f"- **Colunas Likert detectadas**: {len(likert_cols)}",
        "",
        "## Distribuição da variável-alvo (`SATISFAÇÃO GERAL`)",
        "",
        "| Nota | Respondentes | % |",
        "|---:|---:|---:|",
    ]
    target_counts = pd.to_numeric(df["SATISFAÇÃO GERAL"], errors="coerce").value_counts(dropna=False).sort_index()
    total = target_counts.sum()
    for nota, n in target_counts.items():
        rotulo = "NaN/NS" if pd.isna(nota) else int(nota)
        lines.append(f"| {rotulo} | {n:,} | {100*n/total:.2f}% |")

    lines += [
        "",
        "## Top 10 atributos mais correlacionados com Satisfação Geral",
        "",
        "| Atributo | Pearson r |",
        "|---|---:|",
    ]
    for atr, r in corrs.head(10).items():
        lines.append(f"| {atr} | {r:+.3f} |")

    lines += [
        "",
        "## Bottom 5 (mais fracos)",
        "",
        "| Atributo | Pearson r |",
        "|---|---:|",
    ]
    for atr, r in corrs.tail(5).items():
        lines.append(f"| {atr} | {r:+.3f} |")

    lines += [
        "",
        "## Satisfação média por aeroporto",
        "",
        "| Aeroporto | Média | Desvio | Mediana | N |",
        "|---|---:|---:|---:|---:|",
    ]
    target_num = pd.to_numeric(df["SATISFAÇÃO GERAL"], errors="coerce")
    for aero in ORDEM_AERO:
        s = target_num[df["AEROPORTO"] == aero].dropna()
        lines.append(f"| {NOME_AERO[aero]} | {s.mean():.3f} | {s.std():.3f} | {s.median():.1f} | {len(s):,} |")

    OUT_MD.write_text("\n".join(lines))
    print(f"  → {OUT_MD.relative_to(ROOT)}")

    print("\n=== Top 10 correlações com target ===")
    for atr, r in corrs.head(10).items():
        print(f"  {r:+.3f}  {atr[:70]}")


if __name__ == "__main__":
    main()
