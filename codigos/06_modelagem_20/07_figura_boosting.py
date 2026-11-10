"""Figura do deck: como a previsão do XGBoost (ramo Embarque) se forma árvore a árvore,
para três passageiros reais do teste. Escolha determinística: o primeiro passageiro de cada
perfil (nota real 5, 4 e 2) cuja previsão final fica a menos de 0,35 da nota real."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, xgboost as xgb
import matplotlib.pyplot as plt
from comum import SPLITS, FIG_DIR

m = joblib.load(Path(__file__).with_name("models") / "xgb_20.joblib")
X = pd.read_parquet(SPLITS / "X.parquet"); meta = pd.read_parquet(SPLITS / "meta.parquet")
te = ((meta.split == "test") & (meta.PROCESSO == "Embarque")).values
Xe = X[te][m.colunas["Embarque"]]; y = meta.y[te].values
mod = m.modelos["Embarque"]; bst = mod.get_booster()
nfim = mod.best_iteration + 1
pf = mod.predict(Xe)
LOC, LIM, CON = "LOCALIZAÇÃO E DESLOCAMENTO", "LIMPEZA GERAL DO AEROPORTO", "CONFORTO DA SALA DE EMBARQUE"
perfis = [(5, 5, 5, 5, "#2a6fdb"), (4, 4, 4, 4, "#7a7a7a"), (2, 1, 3, 1, "#c0392b")]
ks = np.unique(np.r_[0, np.round(np.geomspace(1, nfim, 40)).astype(int)])
base = float(json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for nota, loc, lim, con, cor in perfis:
    ok = np.where((y == nota) & (Xe[LOC] == loc) & (Xe[LIM] == lim) & (Xe[CON] == con) & (np.abs(pf - nota) < 0.35))[0]
    i = ok[0]
    d = xgb.DMatrix(Xe.iloc[[i]])
    vals = [float(bst.predict(d, iteration_range=(0, k))[0]) if k > 0 else np.nan for k in ks]
    vals[0] = base
    rot = f"nota real {nota} (Localização {loc}, Limpeza {lim}, Conforto {con})"
    ax.plot(ks + 1, vals, color=cor, lw=2.2, label=rot)
    ax.axhline(nota, color=cor, ls=":", lw=1)
    print(nota, i, [round(v, 2) for v in vals][::8], round(vals[-1], 2))
ax.set_xscale("log")
ax.set_xticks([1, 2, 11, 51, 101, nfim + 1]); ax.set_xticklabels(["0", "1", "10", "50", "100", str(nfim)])
ax.set_xlabel("número de árvores somadas")
ax.set_ylabel("satisfação prevista (1–5)")
ax.set_ylim(1.5, 5.1)
ax.text(1.05, vals[0] + 0.12, "começa na média", fontsize=9, color="#444")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8.5, frameon=False, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"xgb_previsao_arvores.{ext}", dpi=200, bbox_inches="tight")
print("ok", FIG_DIR)
