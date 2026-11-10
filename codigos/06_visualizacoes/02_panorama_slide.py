"""Versão enxuta da evolução trimestral para o slide do panorama (deck da defesa).
Média nacional, faixa entre o pior e o melhor aeroporto e três aeroportos em destaque.
Lê panorama_por_periodo.csv (gerado por 01_panorama_agregado.py)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
pp = pd.read_csv(AQUI / "panorama_por_periodo.csv")
base = pd.read_parquet(RAIZ / "dados/processados/mpor_2021_2026_20.parquet", columns=["PERIODO", "SATISFAÇÃO GERAL"])
base["s"] = pd.to_numeric(base["SATISFAÇÃO GERAL"], errors="coerce")
nac = base.dropna(subset=["s"]).groupby("PERIODO")["s"].mean()

per = sorted(pp["PERIODO"].unique(), key=lambda p: (int(p[:4]), int(p[-1])))
x = {p: i for i, p in enumerate(per)}
tab = pp.pivot(index="PERIODO", columns="AEROPORTO", values="media").reindex(per)

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
xs = [x[p] for p in per]
ax.fill_between(xs, tab.min(axis=1), tab.max(axis=1), color="#cfd8e3", alpha=0.6, lw=0, label="pior a melhor aeroporto")
cores = {"SBFL": ("#2a6fdb", "Florianópolis"), "SBGR": ("#e08a00", "Guarulhos"), "SBBE": ("#c0392b", "Belém")}
for a, (c, nome) in cores.items():
    s = tab[a]
    ax.plot(xs, s.values, color=c, lw=1.6, marker="o", ms=3, label=nome)
ax.plot(xs, nac.reindex(per).values, color="black", lw=2.6, marker="s", ms=4, label="média nacional")
anos = [p for p in per if p.endswith("T1") or p == "2021T3"]
ax.set_xticks([x[p] for p in per if p.endswith("T1")] )
ax.set_xticklabels([p[:4] for p in per if p.endswith("T1")])
ax.set_ylabel("Satisfação geral (1–5)")
ax.set_ylim(3.6, 4.9)
ax.grid(axis="y", alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.38))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(RAIZ / f"midias/figuras/panorama_evolucao_slide.{ext}", dpi=200, bbox_inches="tight")
print("ok")
