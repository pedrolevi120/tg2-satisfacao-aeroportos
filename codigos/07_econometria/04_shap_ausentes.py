"""
T6/econometria — Decomposição do |SHAP| por atributo: parcela vinda de linhas em que o atributo
NÃO foi respondido (codificação -1 do pipeline de ML) e ranking SHAP restrito às linhas observadas.
Testa se a discordância logit x SHAP concentra-se nos sub-itens da lógica de salto.
Saída: resultados/shap_decomposicao_ausentes.csv, resultados/shap_ausentes_spearman.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from comum import *

shap = pd.read_parquet(PROCESSADOS / "shap_values_test.parquet").drop(columns="AEROPORTO").astype(float)
Xte = pd.read_parquet(SPLITS / "X_test.parquet")
rk = pd.read_csv(OUT_DIR / "logit_vs_shap_ranking.csv")
hier = json.loads((OUT_DIR / "logit_resultados.json").read_text())["hierarquia"]
linhas = []
for a in rk.atributo:
    obs = Xte[a] > 0
    phi = shap[a].abs()
    linhas.append({"atributo": a, "sub_item": a in hier, "pai": hier.get(a, ""),
                   "cobertura_teste": float(obs.mean()),
                   "shap_mean_abs": float(phi.mean()),
                   "share_absshap_ausentes": float(phi[~obs].sum() / phi.sum()),
                   "shap_mean_abs_so_observados": float(phi.where(obs, 0).mean())})
d = pd.DataFrame(linhas).merge(rk[["atributo", "logit_idx_mean_abs", "rank_shap_mean_abs",
                                   "rank_logit_idx_mean_abs"]], on="atributo")
d["rank_shap_obs"] = d.shap_mean_abs_so_observados.rank(ascending=False)
d["desloc_rank"] = d.rank_logit_idx_mean_abs - d.rank_shap_mean_abs
d.to_csv(OUT_DIR / "shap_decomposicao_ausentes.csv", index=False)
out = {}
rho, p = stats.spearmanr(d.shap_mean_abs_so_observados, d.logit_idx_mean_abs)
out["spearman_logit_vs_shap_observados"] = {"rho": float(rho), "p": float(p), "n": len(d)}
nd = d[~d.sub_item]
rho2, p2 = stats.spearmanr(nd.shap_mean_abs, nd.logit_idx_mean_abs)
out["spearman_so_itens_principais"] = {"rho": float(rho2), "p": float(p2), "n": len(nd)}
out["share_ausentes_medio_subitens"] = float(d[d.sub_item].share_absshap_ausentes.mean())
out["share_ausentes_medio_principais"] = float(d[~d.sub_item].share_absshap_ausentes.mean())
out["desloc_medio_subitens"] = float(d[d.sub_item].desloc_rank.mean())
out["desloc_medio_principais"] = float(d[~d.sub_item].desloc_rank.mean())
(OUT_DIR / "shap_ausentes_spearman.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
pd.set_option("display.width", 200)
print(d[["atributo", "sub_item", "cobertura_teste", "share_absshap_ausentes", "rank_shap_mean_abs",
         "rank_shap_obs", "rank_logit_idx_mean_abs"]].round(3).to_string())
print(json.dumps(out, indent=2))

# ---- tabela LaTeX: rankings logit x SHAP
def p_txt(p):
    return r"$p<0{,}001$" if p < 0.001 else f"$p$ = {fmt(p, 3)}"


sp = json.loads((OUT_DIR / "logit_spearman.json").read_text())
L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
     r"\caption{Rankings de atributos: SHAP (XGBoost) $\times$ logit ordinal e parcela do $|\text{SHAP}|$ "
     r"vinda de respostas ausentes}", r"\label{tab:econometria-logit-shap}",
     r"\begin{tabular}{lrrrr}", r"\toprule",
     r"Atributo & Posição SHAP & Posição logit & Cobertura (\%) & $|\text{SHAP}|$ ausentes (\%) \\",
     r"\midrule"]
for _, r in d.sort_values("rank_shap_mean_abs").iterrows():
    nome = tex_escape(curto(r.atributo)) + (r"$^{\dagger}$" if r.sub_item else "")
    L.append(f"{nome} & {int(r.rank_shap_mean_abs)} & {r.rank_logit_idx_mean_abs:.0f} & "
             f"{fmt(100 * r.cobertura_teste, 1)} & {fmt(100 * r.share_absshap_ausentes, 1)} \\\\")
L += [r"\midrule",
      f"Spearman, todos os {len(d)} atributos & \\multicolumn{{4}}{{r}}{{{fmt(sp['logit_idx_mean_abs']['spearman'], 2)} "
      f"({p_txt(sp['logit_idx_mean_abs']['p'])})}} \\\\",
      f"Spearman, só itens principais ({out['spearman_so_itens_principais']['n']}) & \\multicolumn{{4}}{{r}}"
      f"{{{fmt(out['spearman_so_itens_principais']['rho'], 2)} ({p_txt(out['spearman_so_itens_principais']['p'])})}} \\\\",
      r"\bottomrule", r"\end{tabular}",
      r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: posição SHAP = ordem do $|\phi|$ médio no teste "
      r"(XGBoost, codificação $-1$ para ausentes); posição logit = ordem de $\overline{|\hat\beta_j(x_{ij}-\bar x_j)|}$ "
      r"no teste (análogo linear do $|\text{SHAP}|$ médio; especificação \emph{post-double-selection}). "
      r"Cobertura = fração do teste com nota válida. $^{\dagger}$Sub-item da lógica de salto (perguntado só se o "
      r"item-pai $\leq 3$): sua ausência revela que o pai recebeu nota $\geq 4$.}", r"\end{table}"]
(TEX_DIR / "econometria_logit_shap.tex").write_text("\n".join(L) + "\n")
print("tabela ok")

# ---- figura (substitui a de 01): rankings com sub-itens destacados
import matplotlib.pyplot as plt
estilo()
fig, ax = plt.subplots(figsize=(6.3, 5.4))
mx = len(d) + 1
ax.plot([0, mx], [0, mx], ls="--", color="grey", lw=0.8)
for sub, cor, mk, lab in [(False, "#2b6cb0", "o", "Item principal"),
                          (True, "#c05621", "^", "Sub-item (lógica de salto)")]:
    q = d[d.sub_item == sub]
    ax.scatter(q.rank_shap_mean_abs, q.rank_logit_idx_mean_abs, color=cor, marker=mk, s=32, label=lab, zorder=3)
for _, r in d.iterrows():
    if r.rank_shap_mean_abs <= 10 or r.rank_logit_idx_mean_abs <= 8:
        ax.annotate(curto(r.atributo), (r.rank_shap_mean_abs, r.rank_logit_idx_mean_abs),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.invert_xaxis(); ax.invert_yaxis()
ax.set_xlabel("Posição no ranking SHAP (XGBoost, teste)")
ax.set_ylabel("Posição no ranking do logit ordinal")
t1 = f"{sp['logit_idx_mean_abs']['spearman']:.2f}".replace(".", ",")
t2 = f"{out['spearman_so_itens_principais']['rho']:.2f}".replace(".", ",")
ax.set_title(f"Rankings de atributos: SHAP × logit ordinal\n"
             f"Spearman: todos = {t1}; só itens principais = {t2}", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "econometria_logit_vs_shap.pdf", bbox_inches="tight")
plt.close(fig)
print("figura ok")
