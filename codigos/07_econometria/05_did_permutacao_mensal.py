"""
T6/econometria — permutação no event study MENSAL do DiD de Congonhas (série 2021T1–2026T2),
a partir de resultados/did_event_study.json (gerado por 03_did_congonhas.py).

Mês relativo inteiro à troca (17/10/2023): -1 = set./2023 (referência); -0,5 = 01–16/10/2023 (Infraero);
0 = 17–31/10/2023 (Aena); 1 = nov./2023; ... 32 = jun./2026.
  * leads: soma dos quadrados dos coeficientes pré (todos os meses < 0, exceto a referência) de SBSP
    vs. a dos aeroportos-controle tratados como placebo;
  * pós: média dos coeficientes mensais no pós total (0..32), no curto prazo (0..8: out./2023–jun./2024)
    e no médio prazo (9..32: jul./2024–jun./2026), comparada com a dos placebos (|.|).
Atualiza (de forma idempotente) as linhas correspondentes em tex/econometria_did_eventstudy.tex e
redesenha a figura de placebos da especificação principal.
"""
import json
import numpy as np
import pandas as pd
from comum import *

FIM_CURTO = 8      # jun./2024 em meses relativos a out./2023


def p_abs(obs, pl):
    pl = np.asarray(pl)
    return float((1 + (np.abs(pl) >= abs(obs)).sum()) / (1 + len(pl)))


es = json.loads((OUT_DIR / "did_event_study.json").read_text())
out = {}
for o in ["y", "y5"]:
    e = es["mensal"][o]
    coef = {float(k): v[0] for k, v in e["coef"].items()}
    plac = {c: {float(k): v for k, v in p.items()} for c, p in e["placebos"].items()}
    leads = [k for k in coef if k < 0 and k != -1.0]
    s_obs = sum(coef[k] ** 2 for k in leads)
    s_pl = np.array([sum(p[k] ** 2 for k in leads if k in p) for p in plac.values()])
    out[o] = {"n_leads": len(leads), "stat_obs": s_obs,
              "p_perm": float((1 + (s_pl >= s_obs).sum()) / (1 + len(s_pl))),
              "rank": int(1 + (s_pl > s_obs).sum()), "J": len(s_pl)}
    for nome, sel in [("pos", lambda k: k >= 0), ("curto", lambda k: 0 <= k <= FIM_CURTO),
                      ("medio", lambda k: k > FIM_CURTO)]:
        ks = [k for k in coef if sel(k)]
        m_obs = float(np.mean([coef[k] for k in ks]))
        m_pl = [np.mean([p[k] for k in ks if k in p]) for p in plac.values()]
        out[o][f"media_{nome}"] = m_obs
        out[o][f"n_meses_{nome}"] = len(ks)
        out[o][f"p_perm_media_{nome}"] = p_abs(m_obs, m_pl)
(OUT_DIR / "did_permutacao_mensal.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

# ---- linhas na tabela do event study (substitui se já existirem)
p = TEX_DIR / "econometria_did_eventstudy.tex"
linhas = [
    (r"Leads mensais, $p$ permutação", " & ".join(fmt(out[o]["p_perm"], 3) for o in ["y", "y5"])),
    (r"Média mensal pós vs. set./2023 (out./2023--jun./2026), [$p$ perm.]",
     " & ".join(f"{fmt(out[o]['media_pos'])} [{fmt(out[o]['p_perm_media_pos'], 3)}]" for o in ["y", "y5"])),
    (r"\quad curto prazo (até jun./2024)",
     " & ".join(f"{fmt(out[o]['media_curto'])} [{fmt(out[o]['p_perm_media_curto'], 3)}]" for o in ["y", "y5"])),
    (r"\quad médio prazo (jul./2024--jun./2026)",
     " & ".join(f"{fmt(out[o]['media_medio'])} [{fmt(out[o]['p_perm_media_medio'], 3)}]" for o in ["y", "y5"])),
]
t = p.read_text().split("\n")
rotulos = [r for r, _ in linhas]
t = [l for l in t if not any(l.startswith(r + " & ") for r in rotulos)]
i = t.index(r"\bottomrule")
t[i:i] = [f"{r} & & {v} \\\\" for r, v in linhas]
p.write_text("\n".join(t))

# ---- figura (substitui a de 03): placebos ordenados, espec. principal
import matplotlib.pyplot as plt
estilo()
pl = pd.read_csv(OUT_DIR / "did_placebos.csv")
res = pd.read_csv(OUT_DIR / "did_resultados.csv")
res = res[(res.tratado == "SBSP") & (res.termo == "DiD")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), sharey=False)
for ax, o, tit in [(axes[0], "y", "Satisfação média (pontos)"), (axes[1], "y5", "P(nota 5)")]:
    q = pl[(pl.espec == "(1) Principal") & (pl.outcome == o) & (pl.termo == "DiD")][["placebo", "delta"]]
    r = res[(res.espec == "(1) Principal") & (res.outcome == o)].iloc[0]
    q = pd.concat([q, pd.DataFrame({"placebo": ["SBSP"], "delta": [r.delta]})]).sort_values("delta")
    cores = ["#c53030" if a == "SBSP" else "#718096" for a in q.placebo]
    ax.barh(range(len(q)), q.delta, color=cores, height=0.7)
    ax.set_yticks(range(len(q)))
    ax.set_yticklabels(q.placebo, fontsize=7)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(f"{tit}\nSBSP: $\\hat\\delta$ = {r.delta:+.3f}; p perm. = {r.p_perm:.2f}".replace(".", ","),
                 fontsize=9)
    ax.set_xlabel("$\\hat\\delta$ (pós 17/10/2023 vs. pré, 2021–2026)")
fig.suptitle("DiD de cada aeroporto-controle como pseudo-tratado (vermelho: Congonhas)", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "econometria_did_placebos.pdf", bbox_inches="tight")
plt.close(fig)
print("figura ok")
