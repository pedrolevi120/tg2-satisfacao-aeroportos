"""
SO5 — Estudo agregado do panorama de satisfação aeroportuária no Brasil, série 2021T1–2026T2.

Trabalha com os 20 aeroportos da Pesquisa MPor em 21 trimestres (2021T2 não existe).
Lacunas de coleta (descrições do portal): 2021T3 só ago–set; 2022T4 só outubro; 2023T1 só março;
Porto Alegre (SBPA) sem coleta de maio a dezembro de 2024 (enchentes); pesquisa suspensa em 16/06/2026.
Regime de gestão por período: dados/processados/contexto_aeroportos_20_serie.csv.

Entradas: dados/processados/mpor_2021_2026_20.parquet (codigos/10_serie_historica/01_carregar_serie.py)
          dados/processados/contexto_aeroportos_20_serie.csv (codigos/10_serie_historica/02_contexto_periodos.py)
Saídas:   dados/processados/panorama_aeroportos_2021_2026.parquet  (uma linha por respondente; lida pelo app)
          codigos/06_visualizacoes/panorama_resumo.csv             (por aeroporto: série, 2025, por ano, tendência)
          codigos/06_visualizacoes/panorama_por_periodo.csv        (aeroporto x período)
          codigos/06_visualizacoes/panorama_nacional_ano.csv       (média nacional por ano)
          codigos/06_visualizacoes/panorama_regiao_ano.csv         (região x ano)
          codigos/06_visualizacoes/panorama_regime_periodo.csv     (regime de gestão x período)
          midias/figuras/panorama_*.pdf  (+ cópias PNG em apresentacoes/img_png/)

Uso: python codigos/06_visualizacoes/01_panorama_agregado.py
"""
from pathlib import Path
import subprocess
import sys as _sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_modelagem_20"))
from comum import curto  # nomes legíveis dos atributos (ex.: FILAS2 -> filas inspeção)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROCESSADOS = ROOT / "dados" / "processados"
BASE = PROCESSADOS / "mpor_2021_2026_20.parquet"
CTX_SERIE = PROCESSADOS / "contexto_aeroportos_20_serie.csv"
FIG_DIR = ROOT / "midias" / "figuras"
PNG_DIR = ROOT / "apresentacoes" / "img_png"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROTULO = "2021–2026"
ANO_RECENTE = 2025            # último ano completo da série
MVP = ["SBGR", "SBSP", "SBKP"]
# todos os trimestres do intervalo, inclusive o que não teve coleta (eixo do tempo contínuo)
PERIODOS = [f"{a}T{t}" for a in range(2021, 2027) for t in range(1, 5) if (a, t) <= (2026, 2)]
LACUNAS = {"2021T2": "sem coleta", "2021T3": "só ago–set", "2022T4": "só out.",
           "2023T1": "só mar.", "2026T2": "até 16/06"}

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

# Coordenadas e metadados dos 20 aeroportos (o regime vem do contexto por período)
AEROPORTOS_INFO = {
    "SBBE": {"nome": "Belém",         "uf": "PA", "regiao": "Norte",        "lat": -1.379,  "lon": -48.476},
    "SBBR": {"nome": "Brasília",      "uf": "DF", "regiao": "Centro-Oeste", "lat": -15.871, "lon": -47.918},
    "SBCF": {"nome": "Confins (BH)",  "uf": "MG", "regiao": "Sudeste",      "lat": -19.624, "lon": -43.971},
    "SBCT": {"nome": "Curitiba",      "uf": "PR", "regiao": "Sul",          "lat": -25.532, "lon": -49.176},
    "SBCY": {"nome": "Cuiabá",        "uf": "MT", "regiao": "Centro-Oeste", "lat": -15.652, "lon": -56.117},
    "SBEG": {"nome": "Manaus",        "uf": "AM", "regiao": "Norte",        "lat": -3.039,  "lon": -60.050},
    "SBFL": {"nome": "Florianópolis", "uf": "SC", "regiao": "Sul",          "lat": -27.673, "lon": -48.547},
    "SBFZ": {"nome": "Fortaleza",     "uf": "CE", "regiao": "Nordeste",     "lat": -3.776,  "lon": -38.532},
    "SBGL": {"nome": "Galeão (RJ)",   "uf": "RJ", "regiao": "Sudeste",      "lat": -22.808, "lon": -43.243},
    "SBGO": {"nome": "Goiânia",       "uf": "GO", "regiao": "Centro-Oeste", "lat": -16.632, "lon": -49.221},
    "SBGR": {"nome": "Guarulhos",     "uf": "SP", "regiao": "Sudeste",      "lat": -23.436, "lon": -46.473},
    "SBKP": {"nome": "Viracopos",     "uf": "SP", "regiao": "Sudeste",      "lat": -23.007, "lon": -47.135},
    "SBMO": {"nome": "Maceió",        "uf": "AL", "regiao": "Nordeste",     "lat": -9.510,  "lon": -35.792},
    "SBPA": {"nome": "Porto Alegre",  "uf": "RS", "regiao": "Sul",          "lat": -29.994, "lon": -51.171},
    "SBRF": {"nome": "Recife",        "uf": "PE", "regiao": "Nordeste",     "lat": -8.126,  "lon": -34.923},
    "SBRJ": {"nome": "Santos Dumont", "uf": "RJ", "regiao": "Sudeste",      "lat": -22.910, "lon": -43.163},
    "SBSG": {"nome": "Natal",         "uf": "RN", "regiao": "Nordeste",     "lat": -5.911,  "lon": -35.249},
    "SBSP": {"nome": "Congonhas",     "uf": "SP", "regiao": "Sudeste",      "lat": -23.626, "lon": -46.655},
    "SBSV": {"nome": "Salvador",      "uf": "BA", "regiao": "Nordeste",     "lat": -12.910, "lon": -38.331},
    "SBVT": {"nome": "Vitória",       "uf": "ES", "regiao": "Sudeste",      "lat": -20.258, "lon": -40.286},
}

CORES_REGIAO = {
    "Norte": "#1f77b4",
    "Nordeste": "#ff7f0e",
    "Centro-Oeste": "#2ca02c",
    "Sudeste": "#d62728",
    "Sul": "#9467bd",
}
CORES_REGIME = {"concessionada": "#1f4e79", "infraero": "#c0392b", "misto": "#e08e0b"}


def carregar():
    """Série 2021T1–2026T2 dos 20 aeroportos, com regime de gestão do período."""
    df = pd.read_parquet(BASE)
    # 610 linhas de 2022T2 vêm sem aeroporto nem respostas (só 349 com texto em COMENTÁRIOS ADICIONAIS)
    df = df[df["AEROPORTO"].notna()].copy()
    df["SATISFACAO_GERAL_NUM"] = pd.to_numeric(df["SATISFAÇÃO GERAL"], errors="coerce")
    ctx = pd.read_csv(CTX_SERIE, usecols=["icao", "periodo", "regime_gestao", "operador"])
    df = df.merge(ctx.rename(columns={"icao": "AEROPORTO", "periodo": "PERIODO"}),
                  on=["AEROPORTO", "PERIODO"], how="left")
    assert df["regime_gestao"].notna().all(), "período sem regime no contexto da série"
    for k, col in (("nome", "nome_aero"), ("regiao", "regiao"), ("uf", "uf"), ("lat", "lat"), ("lon", "lon")):
        df[col] = df["AEROPORTO"].map({a: i[k] for a, i in AEROPORTOS_INFO.items()})
    return df


def resumo_aeroporto(v):
    """n, média, desvio, mediana e % de notas 4–5 por aeroporto (sobre respostas válidas)."""
    g = v.groupby("AEROPORTO")["SATISFACAO_GERAL_NUM"]
    return pd.DataFrame({"n_valid": g.size(), "media": g.mean(), "std": g.std(), "mediana": g.median(),
                         "pct_4_5": g.apply(lambda s: 100 * (s >= 4).mean())})


def tendencia(por_periodo):
    """Inclinação (pontos de satisfação por ano) da reta ajustada às médias trimestrais."""
    out = {}
    for a, g in por_periodo.groupby("AEROPORTO"):
        x = g["ANO"] + (g["TRIMESTRE"] - 1) / 4
        out[a] = np.polyfit(x, g["media"], 1)[0] if len(g) >= 4 else np.nan
    return pd.Series(out)


def png(nome, dpi=150):
    """Cópia PNG da figura para os slides (mesmo padrão de nome: <nome>-1.png)."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                    str(FIG_DIR / f"{nome}.pdf"), str(PNG_DIR / f"{nome}-1")], check=True)


def marcar_lacunas(ax, pos, y_texto):
    """Sombreia os trimestres sem coleta ou com coleta parcial."""
    for per, txt in LACUNAS.items():
        x = pos[per]
        ax.axvspan(x - 0.5, x + 0.5, color="0.55" if txt == "sem coleta" else "0.85",
                   alpha=0.35, lw=0, zorder=0)
        ax.text(x, y_texto, txt, rotation=90, ha="center", va="top", fontsize=7.5, color="0.3")


def main():
    print(f"=== Série {ROTULO} (20 aeroportos) ===")
    df = carregar()
    print(f"  {len(df):,} respondentes, {df['AEROPORTO'].nunique()} aeroportos, "
          f"{df['PERIODO'].nunique()} períodos")
    v = df.dropna(subset=["SATISFACAO_GERAL_NUM"])

    # === Estatísticas por aeroporto: série inteira, ano recente, por ano e tendência ===
    meta = df.groupby("AEROPORTO")[["nome_aero", "regiao", "uf"]].first()
    stats = meta.join(df.groupby("AEROPORTO").size().rename("n")).join(resumo_aeroporto(v))
    stats = stats.sort_values("media", ascending=False)
    stats["posicao"] = range(1, len(stats) + 1)
    rec = resumo_aeroporto(v[v["ANO"] == ANO_RECENTE])
    rec = rec.sort_values("media", ascending=False).assign(posicao=range(1, len(rec) + 1))
    stats = stats.join(rec[["n_valid", "media", "pct_4_5", "posicao"]].add_suffix(f"_{ANO_RECENTE}"))
    por_ano = v.groupby(["AEROPORTO", "ANO"])["SATISFACAO_GERAL_NUM"].mean().unstack()
    stats = stats.join(por_ano.drop(columns=ANO_RECENTE).add_prefix("media_"))
    por_periodo = (v.groupby(["AEROPORTO", "ANO", "TRIMESTRE", "PERIODO"])["SATISFACAO_GERAL_NUM"]
                   .agg(n_valid="size", media="mean", pct_4_5=lambda s: 100 * (s >= 4).mean())
                   .reset_index())
    stats["tendencia_pontos_ano"] = tendencia(por_periodo)
    stats["dif_2025_2022"] = stats["media_2025"] - stats["media_2022"]
    regime_atual = df[df["PERIODO"] == "2026T2"].groupby("AEROPORTO")["regime_gestao"].first()
    stats["regime_2026T2"] = regime_atual
    stats = stats.reset_index()
    stats.to_csv(HERE / "panorama_resumo.csv", index=False)

    reg_per = df.groupby(["AEROPORTO", "PERIODO"])["regime_gestao"].first()
    por_periodo["regime_gestao"] = [reg_per[(a, p)] for a, p in zip(por_periodo["AEROPORTO"], por_periodo["PERIODO"])]
    por_periodo.to_csv(HERE / "panorama_por_periodo.csv", index=False)

    # === Nacional, regional e por regime ===
    nac = v.groupby("ANO").agg(n_valid=("SATISFACAO_GERAL_NUM", "size"),
                               media=("SATISFACAO_GERAL_NUM", "mean"),
                               pct_4_5=("SATISFACAO_GERAL_NUM", lambda s: 100 * (s >= 4).mean()),
                               periodos=("PERIODO", lambda s: ", ".join(sorted(s.unique()))))
    nac["media_dos_aeroportos"] = por_ano.mean()
    tot = pd.DataFrame({"n_valid": [len(v)], "media": [v["SATISFACAO_GERAL_NUM"].mean()],
                        "pct_4_5": [100 * (v["SATISFACAO_GERAL_NUM"] >= 4).mean()],
                        "periodos": [f"{v['PERIODO'].nunique()} períodos"],
                        "media_dos_aeroportos": [stats["media"].mean()]}, index=pd.Index([ROTULO], name="ANO"))
    nac = pd.concat([nac.reset_index().astype({"ANO": str}), tot.reset_index()], ignore_index=True)
    nac.to_csv(HERE / "panorama_nacional_ano.csv", index=False)
    nac_per = v.groupby("PERIODO")["SATISFACAO_GERAL_NUM"].mean()

    reg_ano = (v.groupby(["regiao", "ANO"])["SATISFACAO_GERAL_NUM"]
               .agg(n_valid="size", media="mean").reset_index())
    reg_ano.to_csv(HERE / "panorama_regiao_ano.csv", index=False)
    regime = (v.groupby(["PERIODO", "regime_gestao"])
              .agg(n_aeroportos=("AEROPORTO", "nunique"), aeroportos=("AEROPORTO", lambda s: " ".join(sorted(s.unique()))),
                   n_valid=("SATISFACAO_GERAL_NUM", "size"), media=("SATISFACAO_GERAL_NUM", "mean"))
              .reset_index())
    regime.to_csv(HERE / "panorama_regime_periodo.csv", index=False)

    print("\n=== Ranking da série (top 5 / bottom 5) ===")
    cols = ["nome_aero", "regiao", "media", "pct_4_5", "n_valid", f"media_{ANO_RECENTE}",
            f"posicao_{ANO_RECENTE}", "tendencia_pontos_ano"]
    print(stats.head(5)[cols].round(3).to_string(index=False))
    print(stats.tail(5)[cols].round(3).to_string(index=False))
    print("\n=== Média nacional por ano ===")
    print(nac.round(3).to_string(index=False))

    # Parquet para o app (uma linha por respondente)
    df_min = df[["AEROPORTO", "nome_aero", "regiao", "uf", "ANO", "TRIMESTRE", "PERIODO",
                 "regime_gestao", "PROCESSO", "SATISFACAO_GERAL_NUM"]].copy()
    df_min.to_parquet(PROCESSADOS / "panorama_aeroportos_2021_2026.parquet", index=False)

    info = pd.DataFrame.from_dict(AEROPORTOS_INFO, orient="index")
    media_nac = v["SATISFACAO_GERAL_NUM"].mean()

    # === FIGURA 1: Mapa do Brasil com pontos coloridos por satisfação (série) ===
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.add_patch(patches.Rectangle((-74, -34), 40, 39, fill=False,
                                   edgecolor="gray", linewidth=0.3, alpha=0.4))
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--", alpha=0.5)
    ax.text(-73.5, 0.3, "Equador", fontsize=8, color="gray")
    ax.axhline(-23.5, color="gray", linewidth=0.3, linestyle=":", alpha=0.5)
    ax.text(-73.5, -23.2, "Trópico de Capricórnio", fontsize=8, color="gray")
    pos_map = info.reindex(stats["AEROPORTO"])
    sc = ax.scatter(pos_map["lon"], pos_map["lat"], s=stats["n_valid"].values / 150,
                    c=stats["media"].values, cmap="RdYlGn",
                    vmin=np.floor(stats["media"].min() * 10) / 10, vmax=np.ceil(stats["media"].max() * 10) / 10,
                    edgecolor="black", linewidth=0.8, alpha=0.9, zorder=5)
    for _, row in stats.iterrows():
        i = AEROPORTOS_INFO[row["AEROPORTO"]]
        ax.annotate(row["AEROPORTO"], (i["lon"], i["lat"]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Satisfação geral média por aeroporto — Brasil, {ROTULO}\n"
                 "(tamanho do círculo = n. de respondentes)")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Satisfação geral média (1–5)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_mapa_brasil.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\n  → panorama_mapa_brasil.pdf")

    # === FIGURA 2: Ranking da série (IC 95%) com a média de 2025 ===
    fig, ax = plt.subplots(figsize=(10, 9))
    s_ord = stats.sort_values("media")
    ax.barh(s_ord["nome_aero"], s_ord["media"], color=[CORES_REGIAO[r] for r in s_ord["regiao"]],
            edgecolor="black", alpha=0.85)
    se = s_ord["std"] / np.sqrt(s_ord["n_valid"])
    ax.errorbar(s_ord["media"], s_ord["nome_aero"], xerr=1.96 * se, fmt="none",
                ecolor="black", alpha=0.4, capsize=2)
    ax.scatter(s_ord[f"media_{ANO_RECENTE}"], s_ord["nome_aero"], marker="D", s=28, color="black",
               zorder=5)
    ax.axvline(media_nac, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Satisfação geral média (1–5); barra = IC 95%")
    ax.set_title(f"Ranking de aeroportos brasileiros por satisfação geral, {ROTULO}")
    ax.set_xlim(3.5, 4.85)
    handles = [Patch(facecolor=c, label=r, edgecolor="black", alpha=0.85) for r, c in CORES_REGIAO.items()]
    handles += [plt.Line2D([0], [0], color="black", linestyle="--", label=f"Média nacional ({ROTULO})"),
                plt.Line2D([0], [0], marker="D", color="black", linestyle="none", label=f"Média em {ANO_RECENTE}")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_ranking.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → panorama_ranking.pdf")

    # === FIGURA 2b: Ranking de 2025 ===
    fig, ax = plt.subplots(figsize=(10, 9))
    r_ord = stats.sort_values(f"media_{ANO_RECENTE}")
    ax.barh(r_ord["nome_aero"], r_ord[f"media_{ANO_RECENTE}"],
            color=[CORES_REGIAO[r] for r in r_ord["regiao"]], edgecolor="black", alpha=0.85)
    media_rec = v.loc[v["ANO"] == ANO_RECENTE, "SATISFACAO_GERAL_NUM"].mean()
    ax.axvline(media_rec, color="black", linestyle="--", alpha=0.6)
    for y, (m, p) in enumerate(zip(r_ord[f"media_{ANO_RECENTE}"], r_ord["posicao"])):
        ax.text(m + 0.01, y, f"{m:.2f}  (série: {p}º)", va="center", fontsize=8)
    ax.set_xlabel(f"Satisfação geral média (1–5), {ANO_RECENTE}")
    ax.set_title(f"Ranking de aeroportos brasileiros por satisfação geral, {ANO_RECENTE}")
    ax.set_xlim(3.5, 4.95)
    handles = [Patch(facecolor=c, label=r, edgecolor="black", alpha=0.85) for r, c in CORES_REGIAO.items()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label=f"Média nacional ({ANO_RECENTE})"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_ranking_2025.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → panorama_ranking_2025.pdf")

    # === FIGURA 2c: Ranking da série para os slides (MVP em negrito) ===
    with plt.rc_context({"font.size": 13}):
        fig, ax = plt.subplots(figsize=(6.1, 6.25))
        ax.barh(s_ord["nome_aero"], s_ord["media"], color=[CORES_REGIAO[r] for r in s_ord["regiao"]],
                edgecolor="black", height=0.72)
        for y, m in enumerate(s_ord["media"]):
            ax.text(m + 0.01, y, f"{m:.2f}", va="center", fontsize=11, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
        ax.axvline(media_nac, color="black", linestyle="--", lw=1.4)
        for t, a in zip(ax.get_yticklabels(), s_ord["AEROPORTO"]):
            if a in MVP:
                t.set_fontweight("bold")
        ax.set_xlim(3.5, 4.95)
        ax.set_xlabel("Satisfação geral média (1 a 5)")
        ax.set_title(f"Ranking nacional de satisfação, {ROTULO} (MVP em negrito)", fontweight="bold", fontsize=13)
        ax.grid(False)
        sns.despine(ax=ax)
        handles = [Patch(facecolor=c, label=r, edgecolor="black") for r, c in CORES_REGIAO.items()]
        handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="Média nacional"))
        ax.legend(handles=handles, loc="lower right", fontsize=10)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "panorama_ranking_apres.pdf", bbox_inches="tight")
        plt.close(fig)
    print("  → panorama_ranking_apres.pdf")

    # === FIGURA 3: Regiões por ano ===
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, c in CORES_REGIAO.items():
        g = reg_ano[reg_ano["regiao"] == r]
        ax.plot(g["ANO"], g["media"], marker="o", color=c, lw=2, label=r)
    na = nac[nac["ANO"] != ROTULO]
    ax.plot(na["ANO"].astype(int), na["media"], color="black", ls="--", marker="s", lw=1.5,
            label="Média nacional")
    ax.set_xticks(range(2021, 2027))
    ax.set_xticklabels(["2021\n(T1, T3–T4)", "2022\n(T4: só out.)", "2023\n(T1: só mar.)", "2024",
                        "2025", "2026\n(até 16/06)"], fontsize=9)
    ax.set_ylabel("Satisfação geral média (1–5)")
    ax.set_title(f"Satisfação geral média por região e ano, {ROTULO}")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_por_regiao.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → panorama_por_regiao.pdf")

    # === FIGURA 4: Evolução trimestral (21 períodos + lacunas) ===
    pos = {p: i for i, p in enumerate(PERIODOS)}
    fig, ax = plt.subplots(figsize=(13, 6.5))
    tab = por_periodo.pivot(index="AEROPORTO", columns="PERIODO", values="media").reindex(columns=PERIODOS)
    for aero, row in tab.iterrows():
        i = AEROPORTOS_INFO[aero]
        destaque = aero in MVP
        ax.plot(range(len(PERIODOS)), row.values, marker="o", ms=3 if not destaque else 4,
                lw=1.0 if not destaque else 1.8, color=CORES_REGIAO[i["regiao"]],
                alpha=0.35 if not destaque else 0.9)
        if destaque:
            ult = row.last_valid_index()
            ax.annotate(aero, (pos[ult], row[ult]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, fontweight="bold", color=CORES_REGIAO[i["regiao"]], va="center")
    nac_line = nac_per.reindex(PERIODOS)
    ax.plot(range(len(PERIODOS)), nac_line.values, color="black", lw=2.6, marker="s", ms=4,
            label="Média nacional (respondentes)", zorder=6)
    y_top = float(np.nanmax(tab.values)) + 0.12
    ax.set_ylim(float(np.nanmin(tab.values)) - 0.08, y_top + 0.02)
    marcar_lacunas(ax, pos, y_top)
    x_pa = [pos["2024T2"], pos["2024T4"]]
    ax.annotate("SBPA sem coleta\nmai–dez/2024 (enchentes)", xy=(np.mean(x_pa), float(np.nanmin(tab.values)) - 0.02),
                ha="center", fontsize=7.5, color=CORES_REGIAO["Sul"])
    ax.set_xticks(range(len(PERIODOS)))
    ax.set_xticklabels(PERIODOS, rotation=60, fontsize=8.5)
    ax.set_xlim(-0.6, len(PERIODOS) - 0.2)
    ax.set_ylabel("Satisfação geral média (1–5)")
    ax.set_title(f"Evolução trimestral da satisfação geral — 20 aeroportos, {ROTULO}")
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=r) for r, c in CORES_REGIAO.items()]
    handles.append(plt.Line2D([0], [0], color="black", lw=2.6, marker="s", label="Média nacional"))
    handles.append(Patch(facecolor="0.85", label="Coleta parcial / sem coleta"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5,
              title="Região\n(MVP em destaque)", title_fontsize=8.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_evolucao_trimestral.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → panorama_evolucao_trimestral.pdf")

    # === FIGURA 5: Regime de gestão por período ===
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for rg, c in CORES_REGIME.items():
        g = regime[regime["regime_gestao"] == rg].set_index("PERIODO").reindex(PERIODOS)
        ok = g["media"].notna()
        ax.plot(np.arange(len(PERIODOS))[ok.values], g.loc[ok, "media"], marker="o", lw=2 if rg != "misto" else 0,
                color=c, label={"concessionada": "Concessionados", "infraero": "Infraero",
                                "misto": "Trimestre da troca (misto)"}[rg])
        for x, (m, k) in zip(np.arange(len(PERIODOS))[ok.values], zip(g.loc[ok, "media"], g.loc[ok, "n_aeroportos"])):
            ax.annotate(f"{int(k)}", (x, m), xytext=(0, 6 if rg == "concessionada" else -11),
                        textcoords="offset points", ha="center", fontsize=7, color=c)
    lo = regime["media"].min() - 0.08
    marcar_lacunas(ax, pos, regime["media"].max() + 0.1)
    ax.set_ylim(lo, regime["media"].max() + 0.12)
    ax.set_xticks(range(len(PERIODOS)))
    ax.set_xticklabels(PERIODOS, rotation=60, fontsize=8.5)
    ax.set_ylabel("Satisfação geral média (1–5)")
    ax.set_title(f"Satisfação geral por regime de gestão no trimestre, {ROTULO} (número = aeroportos no grupo)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_regime_periodo.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → panorama_regime_periodo.pdf")

    # === FIGURA 6: Heatmap (atributos × aeroportos) com média Likert ===
    likert_set = {"1", "2", "3", "4", "5"}

    def is_likert(s):
        if s.dtype.kind in "biufc":
            u = s.dropna().unique()
            return len(u) > 0 and set(u).issubset({1, 2, 3, 4, 5})
        valores = s.dropna().astype(str).str.strip().unique()
        if len(valores) == 0:
            return False
        share = sum(x in likert_set for x in valores) / len(valores)
        return share >= 0.8 and len(set(valores) & likert_set) >= 3

    excluir = {"SATISFAÇÃO GERAL", "SATISFACAO_GERAL_NUM", "TRIMESTRE", "ANO", "CHAVE", "CHAVE_SERIE", "MÊS"}
    likert_cols = [c for c in BASE_COLS(df) if c not in excluir and is_likert(df[c])]
    df_lik = df[likert_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    df_lik["AEROPORTO"] = df["AEROPORTO"]
    medias_aero = df_lik.groupby("AEROPORTO").mean().reindex(stats["AEROPORTO"].tolist())
    top_var = medias_aero.std().sort_values(ascending=False).head(20).index
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(medias_aero[top_var].T, cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "Nota Likert média"},
                xticklabels=[AEROPORTOS_INFO[a]["nome"] for a in medias_aero.index], annot=False)
    ax.set_yticklabels([curto(c) for c in top_var], rotation=0)
    ax.set_title(f"Mapa de calor: atributos com maior variação entre aeroportos ({ROTULO})")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "panorama_heatmap_atributos.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → panorama_heatmap_atributos.pdf")
    print("  Atributos do mapa de calor:", ", ".join(curto(c) for c in top_var))

    # Cópias PNG para os slides
    for nome in ["panorama_mapa_brasil", "panorama_ranking", "panorama_ranking_2025", "panorama_por_regiao",
                 "panorama_evolucao_trimestral", "panorama_regime_periodo", "panorama_heatmap_atributos"]:
        png(nome)
    png("panorama_ranking_apres", dpi=200)
    print(f"  → PNGs em {PNG_DIR.relative_to(ROOT)}")

    print("\n=== Resumo final ===")
    print(f"  Média nacional ({ROTULO}, respondentes): {media_nac:.3f}; média dos aeroportos: {stats['media'].mean():.3f}")
    print(f"  Média nacional {ANO_RECENTE}: {media_rec:.3f}")
    print(f"  Melhor aeroporto (série): {stats.iloc[0]['nome_aero']} ({stats.iloc[0]['media']:.3f})")
    print(f"  Pior aeroporto (série):   {stats.iloc[-1]['nome_aero']} ({stats.iloc[-1]['media']:.3f})")
    print(f"  Melhor em {ANO_RECENTE}: {rec.index[0]} ({rec['media'].iloc[0]:.3f}); "
          f"pior: {rec.index[-1]} ({rec['media'].iloc[-1]:.3f})")
    print("\n=== Regime por período ===")
    print(regime.pivot(index="PERIODO", columns="regime_gestao", values="media").round(3).to_string())


def BASE_COLS(df):
    """Colunas do questionário (exclui as derivadas por este script)."""
    derivadas = {"regime_gestao", "operador", "nome_aero", "regiao", "uf", "lat", "lon", "PERIODO"}
    return [c for c in df.columns if c not in derivadas]


if __name__ == "__main__":
    main()
