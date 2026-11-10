"""
T5/modelos20 — Tarefa 4: SHAP no modelo selecionado (XGBoost por ramo do questionário).

* SHAP (TreeExplainer) nas linhas de teste; cada linha é explicada pelo modelo do
  seu ramo; atributos do outro ramo recebem SHAP = 0 por construção.
* Rankings de PRIORIDADE no nível de DIMENSÃO: item-pai + subitens da lógica de
  salto (subitem só perguntado quando o pai recebe nota <= 3), somando os SHAP
  linha a linha (aditividade). Importância = média de |SHAP da dimensão|.
* Recortes: global, por aeroporto (20), por região e por regime de gestão
  (regime vem do pacote T2 — coluna CTX_*regime* — quando disponível).
* Estabilidade: bootstrap (500) das linhas de teste de cada aeroporto — prob. de
  cada item do top-5 permanecer no top-5.

Série 2021T1–2026T2 (08/11/2026): SHAP nas linhas de teste da série inteira;
shap_values_20.parquet ganha ANO, PERIODO e CHAVE_SERIE. NOVO: ranking de
prioridades por ano (global e por aeroporto × ano quando n_teste >= N_MIN_ANO),
com estabilidade bootstrap do top-4 em cada ano.

Saída: dados/processados/shap_values_20.parquet
       codigos/05_interpretacao/ranking_prioridades_20.csv  (aeroporto, rank, atributo, mean_abs_shap, ...)
       codigos/06_modelagem_20/resultados/ranking_{atributos,regiao,regime,global}_20.csv
       texto/_paralelo/tex/modelos20_top5_aeroportos.tex, modelos20_shap_regiao.tex
       midias/figuras/modelos20_shap_{global,heatmap,regiao,regime}.pdf
       codigos/06_modelagem_20/resultados/ranking_por_ano_20.csv
       midias/figuras/shap_ranking_por_ano.pdf (+ apresentacoes/img_png/shap_ranking_por_ano-1.png)
"""
import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from comum import (ROOT, PROC_OUT, SPLITS, FIG_DIR, TEX_DIR, OUT_DIR, PNG_DIR, INTERP_DIR, SEED, fmt,
                   curto, estilo, tex_escape, hierarquia, grupos_dimensao, shap_por_dimensao)
from modelos import NOMES

warnings.filterwarnings("ignore")
RANKING_OUT = INTERP_DIR / "ranking_prioridades_20.csv"
N_MIN_ANO = int(os.environ.get("N_MIN_ANO", "200"))  # mínimo de linhas de teste p/ ranking aeroporto × ano
NOME_AERO = {"SBBE": "Belém", "SBBR": "Brasília", "SBCF": "Confins", "SBCT": "Curitiba",
             "SBCY": "Cuiabá", "SBEG": "Manaus", "SBFL": "Florianópolis", "SBFZ": "Fortaleza",
             "SBGL": "Galeão", "SBGO": "Goiânia", "SBGR": "Guarulhos", "SBKP": "Viracopos",
             "SBMO": "Maceió", "SBPA": "Porto Alegre", "SBRF": "Recife", "SBRJ": "Santos Dumont",
             "SBSG": "Natal", "SBSP": "Congonhas", "SBSV": "Salvador", "SBVT": "Vitória"}
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def calcular_shap(model, X, ramo):
    sv = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    base = pd.Series(np.nan, index=X.index)
    for r, m in model.modelos.items():
        mask = ramo == r
        cols = model.colunas[r]
        ex = shap.TreeExplainer(m)
        sv.loc[mask, cols] = ex.shap_values(X.loc[mask, cols])
        base.loc[mask] = float(np.atleast_1d(ex.expected_value)[0])
    return sv, base


def top_k_estabilidade(D, k=5, n_boot=500, seed=SEED):
    """D: |SHAP| por dimensão (linhas do aeroporto). Retorna ranking e P(item do top-k ficar no top-k)."""
    imp = D.mean().sort_values(ascending=False)
    top = list(imp.index[:k])
    rng = np.random.default_rng(seed)
    V = D.values
    cont = np.zeros(k)
    for _ in range(n_boot):
        mb = V[rng.integers(0, len(V), len(V))].mean(axis=0)
        topb = set(D.columns[np.argsort(-mb)[:k]])
        cont += [t in topb for t in top]
    return imp, dict(zip(top, cont / n_boot))


def main():
    estilo()
    X = pd.read_parquet(SPLITS / "X.parquet")
    meta = pd.read_parquet(SPLITS / "meta.parquet")
    fm = json.loads((SPLITS / "feature_metadata.json").read_text())
    likert = fm["likert_cols"]
    melhor = json.loads((OUT_DIR / "modelo_escolhido_20.json").read_text())["modelo"]
    model = joblib.load(OUT_DIR.parent / "models" / f"{melhor}_20.joblib")
    tr, te = (meta.split == "train").values, (meta.split == "test").values

    pais = hierarquia(X.loc[tr], likert)
    grupos = grupos_dimensao(likert, pais)
    (OUT_DIR / "hierarquia_20.json").write_text(json.dumps({"subitem->pai": pais, "dimensoes": grupos},
                                                           indent=2, ensure_ascii=False))
    print(f"  {len(pais)} subitens; {len(grupos)} dimensões")

    Xte, mte = X.loc[te], meta.loc[te].reset_index(drop=True)
    sv, base = calcular_shap(model, Xte, mte.PROCESSO.values)
    pred = model.predict(Xte, mte.PROCESSO.values)
    assert np.allclose(sv.sum(axis=1).values + base.values, pred, atol=1e-3), "SHAP não soma à predição"
    sv.index = mte.index

    # --- persistência dos valores SHAP ---
    out = sv.copy()
    for c in ["AEROPORTO", "PROCESSO", "REGIAO", "ANO", "TRIMESTRE", "PERIODO", "CHAVE_SERIE", "y"]:
        out[c] = mte[c].values
    reg_col = next((c for c in mte.columns if c.startswith("CTX_") and "regime" in c.lower()), None)
    if reg_col:
        out["REGIME"] = mte[reg_col].astype(str).values
    out["pred"], out["base_value"] = pred, base.values
    out.index = X.index[te]
    out.to_parquet(PROC_OUT / "shap_values_20.parquet")
    print(f"  → shap_values_20.parquet {out.shape}")

    absv = sv.abs()
    frac_likert = absv[likert].sum().sum() / absv.sum().sum()
    print(f"  fração do |SHAP| total atribuída a itens Likert: {frac_likert:.3f}")
    D = shap_por_dimensao(sv, grupos).abs()
    Dsigned = shap_por_dimensao(sv, grupos)
    Xpai = Xte.reset_index(drop=True)
    aero = mte.AEROPORTO.values

    # --- global ---
    g = D.mean().sort_values(ascending=False)
    pai_share = pd.Series({d: absv[d].mean() / max(absv[grupos[d]].sum(axis=1).mean(), 1e-12) for d in g.index})
    pd.DataFrame({"dimensao": g.index, "mean_abs_shap": g.values,
                  "n_subitens": [len(grupos[d]) - 1 for d in g.index]}).to_csv(OUT_DIR / "ranking_global_20.csv", index=False)
    print("  Global:", [(curto(k), round(v, 4)) for k, v in g.head(8).items()])

    # --- por aeroporto (CSV exigido + estabilidade) ---
    rows, top5, estab = [], {}, {}
    for A in sorted(np.unique(aero)):
        m = aero == A
        imp, st = top_k_estabilidade(D[m])
        top5[A], estab[A] = list(imp.index[:5]), st
        for r, (d, v) in enumerate(imp.items(), 1):
            pai_obs = Xpai.loc[m, d]
            rows.append({"aeroporto": A, "rank": r, "atributo": d, "mean_abs_shap": float(v),
                         "shap_medio": float(Dsigned.loc[m, d].mean()),
                         "nota_media": float(pai_obs.mean()) if pai_obs.notna().any() else np.nan,
                         "pct_nota_ate_3": float((pai_obs <= 3).sum() / max(pai_obs.notna().sum(), 1)),
                         "n_respostas": int(pai_obs.notna().sum()), "n_teste": int(m.sum()),
                         "subitens": " | ".join(grupos[d][1:])})
    rk = pd.DataFrame(rows)
    rk.to_csv(RANKING_OUT, index=False)
    print(f"  → {RANKING_OUT} ({len(rk)} linhas)")

    # ranking também no nível de atributo (sem agregação), para transparência
    ra = []
    for A in sorted(np.unique(aero)):
        imp = absv.loc[aero == A, likert].mean().sort_values(ascending=False)
        ra += [{"aeroporto": A, "rank": r, "atributo": a, "mean_abs_shap": float(v)} for r, (a, v) in enumerate(imp.items(), 1)]
    pd.DataFrame(ra).to_csv(OUT_DIR / "ranking_atributos_20.csv", index=False)

    # --- por região e por regime ---
    reg = mte.REGIAO.values
    R = pd.DataFrame({r: D[reg == r].mean() for r in REGIOES})
    R.to_csv(OUT_DIR / "ranking_regiao_20.csv")
    if reg_col:
        regv = mte[reg_col].astype(str).values
        RG = pd.DataFrame({r: D[regv == r].mean() for r in sorted(set(regv))})
        RG.to_csv(OUT_DIR / "ranking_regime_20.csv")
    else:
        RG = None
        print("  [regime] coluna de regime do T2 ausente — recorte por regime não calculado")

    S = pd.DataFrame({A: Dsigned[aero == A].mean() for A in sorted(np.unique(aero))}).T

    def perda(A):
        d = S.loc[A].idxmin()
        return "--" if S.loc[A, d] > -0.005 else f"{tex_escape(curto(d))} ({fmt(S.loc[A, d])})"

    # ================= Tabelas LaTeX =================
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{Cinco dimensões prioritárias por aeroporto (média de $|\\text{{SHAP}}|$ da dimensão no teste, série 2021--2026, {NOMES[melhor]} por ramo do questionário). "
         "Dimensão = item-pai somado aos subitens perguntados apenas quando o pai recebe nota $\\leq 3$. "
         "Estab.: probabilidade média, em 500 reamostragens \\textit{bootstrap} das linhas do aeroporto, de cada item do top-5 permanecer no top-5. "
         "Maior perda: dimensão com a contribuição média (SHAP com sinal) mais negativa no aeroporto, em pontos da escala 1--5 (``--'': nenhuma abaixo de $-0{,}005$).}",
         "\\label{tab:modelos20-top5}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrlllllcl}", "\\toprule",
         "Aeroporto & $n$ & 1\\textordmasculine & 2\\textordmasculine & 3\\textordmasculine & 4\\textordmasculine & 5\\textordmasculine & Estab. & Maior perda \\\\", "\\midrule"]
    for A in sorted(top5):
        n = int((aero == A).sum())
        L.append(f"{NOME_AERO[A]} & {n} & " + " & ".join(tex_escape(curto(d)) for d in top5[A])
                 + f" & {fmt(np.mean(list(estab[A].values())), 2)} & {perda(A)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_top5_aeroportos.tex").write_text("\n".join(L), encoding="utf-8")

    topdims = list(g.index[:10])
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Importância (média de $|\\text{SHAP}|$ no teste, série 2021--2026) das dez principais dimensões, global e por região"
         + (" e regime de gestão" if RG is not None else "") + ".}",
         "\\label{tab:modelos20-shap-regiao}"]
    colsR = REGIOES + (list(RG.columns) if RG is not None else [])
    L += ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "c" * (1 + len(colsR)) + "}", "\\toprule",
          "Dimensão & Global & " + " & ".join(tex_escape(c) for c in colsR) + " \\\\", "\\midrule"]
    for d in topdims:
        vals = [fmt(g[d])] + [fmt(R.loc[d, r]) for r in REGIOES] + ([fmt(RG.loc[d, c]) for c in RG.columns] if RG is not None else [])
        L.append(f"{tex_escape(curto(d))} & " + " & ".join(vals) + " \\\\")
    nrow = [str(len(D))] + [str(int((reg == r).sum())) for r in REGIOES] + ([str(int((mte[reg_col].astype(str).values == c).sum())) for c in RG.columns] if RG is not None else [])
    L += ["\\midrule", "$n$ (teste) & " + " & ".join(nrow) + " \\\\", "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_shap_regiao.tex").write_text("\n".join(L), encoding="utf-8")

    # ================= Figuras =================
    # Global: pai vs subitens
    gg = g.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    parte_pai = pd.Series({d: absv[d].mean() for d in gg.index})
    sub = (gg - parte_pai).clip(lower=0)
    ax.barh([curto(d) for d in gg.index], gg.values, color="#1f77b4")
    ax.set_xlabel("média de |SHAP| da dimensão (teste, 20 aeroportos, 2021–2026)")
    ax.set_title(f"Importância global das dimensões de serviço — {NOMES[melhor]} por ramo ($n$={len(D):,})".replace(",", "."))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_shap_global.pdf", bbox_inches="tight")
    plt.close(fig)

    # Heatmap aeroporto × dimensão (participação % na importância das dimensões do aeroporto; números = posição)
    dims = list(g.index[:14])
    H = pd.DataFrame({A: D[aero == A].mean() for A in sorted(np.unique(aero))}).T
    Hs = H.div(H.sum(axis=1), axis=0)[dims] * 100
    rank = H.rank(axis=1, ascending=False)[dims].astype(int)
    ordem = sorted(Hs.index, key=lambda a: (REGIOES.index(meta.loc[meta.AEROPORTO == a, "REGIAO"].iloc[0]), NOME_AERO[a]))
    Hs, rank = Hs.loc[ordem], rank.loc[ordem]
    fig, ax = plt.subplots(figsize=(13, 8.5))
    sns.heatmap(Hs, cmap="YlOrRd", annot=rank, fmt="d", annot_kws={"fontsize": 7},
                cbar_kws={"label": "% da importância das dimensões no aeroporto"}, ax=ax, linewidths=0.3)
    ax.set_yticklabels([f"{NOME_AERO[a]} ({meta.loc[meta.AEROPORTO == a, 'REGIAO'].iloc[0]})" for a in ordem], rotation=0)
    ax.set_xticklabels([curto(d) for d in dims], rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Importância relativa das dimensões por aeroporto (cor: % do |SHAP|; número: posição no ranking)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_shap_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Heatmap com sinal: contribuição média de cada dimensão à satisfação prevista do aeroporto
    Sg = S.loc[ordem, dims]
    lim = np.abs(Sg.values).max()
    fig, ax = plt.subplots(figsize=(13, 8.5))
    sns.heatmap(Sg, cmap="RdBu", center=0, vmin=-lim, vmax=lim, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6.5}, cbar_kws={"label": "SHAP médio (pontos da escala 1–5)"},
                ax=ax, linewidths=0.3)
    ax.set_yticklabels([f"{NOME_AERO[a]} ({meta.loc[meta.AEROPORTO == a, 'REGIAO'].iloc[0]})" for a in ordem], rotation=0)
    ax.set_xticklabels([curto(d) for d in dims], rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Contribuição média das dimensões para a satisfação prevista, por aeroporto (azul eleva, vermelho reduz)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_shap_heatmap_sinal.pdf", bbox_inches="tight")
    plt.close(fig)
    S.to_csv(OUT_DIR / "shap_medio_sinal_aeroporto_20.csv")

    # Região
    fig, ax = plt.subplots(figsize=(10, 5.5))
    Rn = R.loc[dims[:10], REGIOES]
    Rn.index = [curto(d) for d in Rn.index]
    Rn.plot.barh(ax=ax, width=0.8, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax.invert_yaxis()
    ax.set_xlabel("média de |SHAP| da dimensão (teste)")
    ax.set_title("Importância das dez principais dimensões por região")
    ax.legend(title="Região", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_shap_regiao.pdf", bbox_inches="tight")
    plt.close(fig)

    if RG is not None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        Rg = RG.loc[dims[:10]]
        Rg.index = [curto(d) for d in Rg.index]
        Rg.plot.barh(ax=ax, width=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("média de |SHAP| da dimensão (teste)")
        ax.set_title("Importância das dez principais dimensões por regime de gestão")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "modelos20_shap_regime.pdf", bbox_inches="tight")
        plt.close(fig)

    # ================= Ranking por ano (estabilidade no tempo) =================
    anos_v = mte.ANO.values.astype(int)
    anos = sorted(np.unique(anos_v))
    top4_g = list(g.index[:4])
    rng = np.random.default_rng(SEED)
    linhas_ano, resumo_ano, aero_ano = [], {}, []
    for a in anos:
        ma = anos_v == a
        imp_a = D[ma].mean().sort_values(ascending=False)
        for r, (dm, v) in enumerate(imp_a.items(), 1):
            linhas_ano.append({"escopo": "global", "aeroporto": "TODOS", "ANO": a, "rank": r, "dimensao": dm,
                               "nome": curto(dm), "mean_abs_shap": float(v), "n_teste": int(ma.sum())})
        # estabilidade do conjunto top-4 do ano: 500 reamostragens das linhas de teste do ano
        V = D[ma].values
        mesmo = 0
        for _ in range(500):
            mb = V[rng.integers(0, len(V), len(V))].mean(axis=0)
            mesmo += set(D.columns[np.argsort(-mb)[:4]]) == set(top4_g)
        rk_a = imp_a.rank(ascending=False).astype(int)
        resumo_ano[a] = {"n_teste": int(ma.sum()), "top4": list(imp_a.index[:4]),
                         "top4_mesmo_conjunto_global": set(imp_a.index[:4]) == set(top4_g),
                         "top4_mesma_ordem_global": list(imp_a.index[:4]) == top4_g,
                         "pct_boot_top4_igual_global": 100 * mesmo / 500,
                         "posicoes_top6_global": {dm: int(rk_a[dm]) for dm in g.index[:6]},
                         "mean_abs_shap_top6_global": {dm: float(imp_a[dm]) for dm in g.index[:6]},
                         "spearman_vs_global": float(pd.Series(imp_a).corr(g, method="spearman"))}
        for A in sorted(np.unique(aero)):
            mA = ma & (aero == A)
            if mA.sum() < N_MIN_ANO:
                continue
            imp_A = D[mA].mean().sort_values(ascending=False)
            for r, (dm, v) in enumerate(imp_A.items(), 1):
                linhas_ano.append({"escopo": "aeroporto", "aeroporto": A, "ANO": a, "rank": r, "dimensao": dm,
                                   "nome": curto(dm), "mean_abs_shap": float(v), "n_teste": int(mA.sum())})
            aero_ano.append({"aeroporto": A, "ANO": a, "top1_global": imp_A.index[0] == g.index[0],
                             "top2_global": imp_A.index[1] == g.index[1],
                             "top4_conjunto_global": set(imp_A.index[:4]) == set(top4_g)})
    # Ranking por ano DENTRO de cada ramo: separa mudança de prioridade de mudança de composição
    # (em 2021–22 a amostra tinha ~70% de embarque; de 2023 em diante, ~50/50). A média global de
    # |SHAP| inclui zeros estruturais (atributo do outro ramo), então mais embarque infla Limpeza e
    # Conforto e reduz Restituição sem que a ordem dentro de cada ramo mude.
    ramo_te = mte.PROCESSO.values
    resumo_ramo = {}
    for rr in sorted(np.unique(ramo_te)):
        mr = ramo_te == rr
        imp_r = D[mr].mean().sort_values(ascending=False)
        top4_r = list(imp_r.index[:4])
        por_ano_r = {}
        for a in anos:
            mra = mr & (anos_v == a)
            imp_ra = D[mra].mean().sort_values(ascending=False)
            for r, (dm, v) in enumerate(imp_ra.items(), 1):
                linhas_ano.append({"escopo": f"ramo_{rr}", "aeroporto": "TODOS", "ANO": a, "rank": r, "dimensao": dm,
                                   "nome": curto(dm), "mean_abs_shap": float(v), "n_teste": int(mra.sum())})
            por_ano_r[int(a)] = {"n_teste": int(mra.sum()), "top4": list(imp_ra.index[:4]),
                                 "top4_mesma_ordem_ramo": list(imp_ra.index[:4]) == top4_r,
                                 "top3_mesma_ordem_ramo": list(imp_ra.index[:3]) == top4_r[:3]}
        resumo_ramo[rr] = {"top4_ramo": top4_r,
                           "pct_teste_por_ano": {int(a): float(np.mean(mr[anos_v == a])) for a in anos},
                           "por_ano": por_ano_r}
        print(f"  ramo {rr}: top-4 " + " > ".join(curto(d) for d in top4_r) + " | mesma ordem top-3 em "
              f"{sum(v['top3_mesma_ordem_ramo'] for v in por_ano_r.values())}/{len(anos)} anos")
    RA = pd.DataFrame(linhas_ano)
    RA.to_csv(OUT_DIR / "ranking_por_ano_20.csv", index=False)
    AA = pd.DataFrame(aero_ano, columns=["aeroporto", "ANO", "top1_global", "top2_global", "top4_conjunto_global"])
    resumo_aero_ano = {"n_min_teste": N_MIN_ANO, "n_aeroporto_ano": int(len(AA)),
                       "n_top1_global": int(AA.top1_global.sum()), "n_top2_global": int(AA.top2_global.sum()),
                       "n_top4_conjunto_global": int(AA.top4_conjunto_global.sum()),
                       "por_ano": {int(a): {"n": int(len(gA)), "top1_global": int(gA.top1_global.sum()),
                                            "top2_global": int(gA.top2_global.sum()),
                                            "top4_conjunto_global": int(gA.top4_conjunto_global.sum())}
                                   for a, gA in AA.groupby("ANO")}}
    print("  Ranking por ano (global):")
    for a in anos:
        ra = resumo_ano[a]
        print(f"    {a} (n={ra['n_teste']}): " + " > ".join(curto(dm) for dm in ra["top4"])
              + f" | mesmo conjunto={ra['top4_mesmo_conjunto_global']} ({ra['pct_boot_top4_igual_global']:.0f}% boot)"
              + f" | ρ={ra['spearman_vs_global']:.3f}")
    print(f"  aeroporto × ano (n ≥ {N_MIN_ANO}): {json.dumps(resumo_aero_ano)}")

    # Figura: posição (esq.) e |SHAP| médio (dir.) por ano das dimensões que já estiveram no top-6
    dims_ano = [dm for dm in g.index if (RA[(RA.escopo == "global") & (RA.dimensao == dm)]["rank"] <= 6).any()][:8]
    piv_r = RA[RA.escopo == "global"].pivot(index="ANO", columns="dimensao", values="rank")[dims_ano]
    piv_v = RA[RA.escopo == "global"].pivot(index="ANO", columns="dimensao", values="mean_abs_shap")[dims_ano]
    cores = sns.color_palette("tab10", len(dims_ano))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for c, dm in zip(cores, dims_ano):
        ax1.plot(piv_r.index, piv_r[dm], marker="o", color=c, lw=2, label=curto(dm))
        ax1.annotate(curto(dm), (piv_r.index[-1], piv_r[dm].iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=c)
        ax2.plot(piv_v.index, piv_v[dm], marker="o", color=c, lw=2, label=curto(dm))
    ax1.invert_yaxis()
    ax1.set_yticks(range(1, int(np.nanmax(piv_r.values)) + 1))
    ax1.set_ylabel("posição no ranking (1 = mais importante)")
    ax1.set_xlim(anos[0] - 0.3, anos[-1] + 1.6)
    rot = [f"{a}\n$n$={resumo_ano[a]['n_teste']:,}".replace(",", ".") for a in anos]
    for ax in (ax1, ax2):
        ax.set_xticks(anos)
        ax.set_xticklabels(rot, fontsize=8)
    ax1.set_title("Posição das dimensões por ano")
    ax2.set_ylabel("média de |SHAP| da dimensão")
    ax2.set_title("Importância das dimensões por ano")
    ax2.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"Ranking de prioridades por ano — teste, {NOMES[melhor]} por ramo, 20 aeroportos "
                 f"(2026: só 1º semestre)")
    if "Embarque" in resumo_ramo:
        pe = resumo_ramo["Embarque"]["pct_teste_por_ano"]
        n_est = {rr: sum(v["top3_mesma_ordem_ramo"] for v in resumo_ramo[rr]["por_ano"].values()) for rr in resumo_ramo}
        fig.text(0.01, -0.02, "Parcela de embarque no teste: " + ", ".join(f"{a}: {100 * pe[int(a)]:.0f}%" for a in anos)
                 + ". A média de |SHAP| inclui zeros do outro ramo; dentro de cada ramo, o top-3 tem a mesma ordem em "
                 + " e ".join(f"{n_est[rr]}/{len(anos)} anos ({rr.lower()})" for rr in n_est) + ".",
                 fontsize=7.5, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_ranking_por_ano.pdf", bbox_inches="tight")
    fig.savefig(PNG_DIR / "shap_ranking_por_ano-1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # resumo p/ o relatório
    contagem_top2 = pd.Series([t[1] for t in top5.values()]).value_counts().to_dict()
    resumo = {"modelo": melhor, "n_teste": int(len(D)), "frac_shap_likert": float(frac_likert),
              "global_top4": top4_g,
              "n_aeroportos_top1_global": int(sum(t[0] == g.index[0] for t in top5.values())),
              "n_aeroportos_top2_global": int(sum(t[1] == g.index[1] for t in top5.values())),
              "n_aeroportos_top4_conjunto_global": int(sum(set(t[:4]) == set(top4_g) for t in top5.values())),
              "top2_contagem": contagem_top2,
              "ranking_por_ano": {int(a): v for a, v in resumo_ano.items()},
              "ranking_aeroporto_ano": resumo_aero_ano,
              "ranking_por_ano_ramo": resumo_ramo,
              "global_top10": [(d, float(v)) for d, v in g.head(10).items()],
              "top5": top5, "estabilidade_top5": {A: float(np.mean(list(v.values()))) for A, v in estab.items()},
              "estabilidade_item": estab,
              "top1_contagem": pd.Series([t[0] for t in top5.values()]).value_counts().to_dict(),
              "regime_disponivel": reg_col is not None}
    (OUT_DIR / "shap_resumo_20.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps({k: resumo[k] for k in ["top1_contagem", "estabilidade_top5"]}, indent=1, ensure_ascii=False))
    for A in sorted(top5):
        print(f"  {A}: " + " > ".join(curto(d) for d in top5[A]))


if __name__ == "__main__":
    main()
