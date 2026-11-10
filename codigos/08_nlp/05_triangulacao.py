"""
T7/NLP — Etapa 5: triangulação entre (i) temas mais criticados nos comentários, (ii) ranking SHAP de
importância dos atributos e (iii) atributos com pior nota Likert média, por aeroporto.

Série 2021–2026: críticas por mil respondentes e notas Likert agregam a série inteira (agregado_aeroporto_tema.csv);
o ranking SHAP deve ser o regenerado sobre a série (mesmas colunas) — conferir a data de ranking_prioridades_20.csv.

Ranking SHAP: usa codigos/05_interpretacao/ranking_prioridades_20.csv (T5, 20 aeroportos) se existir;
senão, o ranking do MVP (ranking_prioridades_por_aeroporto.csv: SBGR, SBSP, SBKP; top 15).

Para cada aeroporto e cada atributo Likert a:
  crit_nlp(a) = Σ críticos-por-mil dos temas específicos (coberto/parcial) que mapeiam para a
  shap(a)     = |SHAP| médio do atributo (features não Likert — dummies de aeroporto, perfil — são descartadas)
  nota(a)     = média Likert (1–5) do atributo entre respondentes do aeroporto que o avaliaram
Métricas: sobreposição top-5 NLP × top-5 SHAP (nº de atributos em comum), ρ de Spearman crit_nlp × shap e
crit_nlp × (−nota) entre os atributos presentes no ranking SHAP.

Saídas: saida/triangulacao_por_aeroporto.csv, saida/triangulacao_atributos.csv,
        texto/_paralelo/tex/nlp_triangulacao.tex, texto/_paralelo/tex/nlp_temas_nao_cobertos.tex
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from comum import ROOT, SAIDA, TEX, NOMES, BASE_COM, ANOS_TEX, fmt
from temas import TEMAS

INTERP = ROOT / "codigos" / "05_interpretacao"
K = 5


def carregar_shap():
    """Retorna (ranking, fonte, mapa subitem→atributo-pai, notas por aeroporto×atributo ou None)."""
    f20 = INTERP / "ranking_prioridades_20.csv"
    f = f20 if f20.exists() else INTERP / "ranking_prioridades_por_aeroporto.csv"
    r = pd.read_csv(f)
    col_f = "atributo" if "atributo" in r.columns else "feature"
    col_v = "mean_abs_shap" if "mean_abs_shap" in r.columns else "shap_abs_medio"
    r = r.rename(columns={col_f: "atributo", col_v: "shap"})
    pai = {}
    if "subitens" in r.columns:  # T5 agrega subitens no atributo-pai do questionário
        for _, x in r.dropna(subset=["subitens"]).drop_duplicates("atributo").iterrows():
            for sub in x.subitens.split(" | "):
                pai[sub.strip()] = x.atributo
    notas = r.set_index(["aeroporto", "atributo"])["nota_media"] if "nota_media" in r.columns else None
    return r[["aeroporto", "atributo", "shap"]], f.name, pai, notas


def pais_do_tema(tema, pai, validos):
    return {pai.get(a, a) for a in TEMAS[tema][2]} & validos


def main():
    shap, fonte, pai, notas_t5 = carregar_shap()
    base = pd.read_parquet(BASE_COM)
    likert = [c for c in base.columns if c not in ("CHAVE_SERIE", "CHAVE", "ANO", "TRIMESTRE", "PERIODO", "PROCESSO",
                                                   "AEROPORTO", "TIPO DE VOO", "SATISFAÇÃO GERAL", "COMENTÁRIOS ADICIONAIS")]
    shap = shap[shap.atributo.isin(likert)]
    validos = set(shap.atributo)
    notas = base.groupby("AEROPORTO")[likert].agg(lambda s: pd.to_numeric(s, errors="coerce").mean())
    ag = pd.read_csv(SAIDA / "agregado_aeroporto_tema.csv")
    esp = ag[ag.cobertura.isin(["coberto", "parcial"])]

    linhas, attr_rows = [], []
    for ap in sorted(shap.aeroporto.unique()):
        s = shap[shap.aeroporto == ap].sort_values("shap", ascending=False)
        e = esp[esp.aeroporto == ap].sort_values("criticos_por_mil", ascending=False)
        crit = {}
        for _, r in e.iterrows():
            for a in pais_do_tema(r.tema, pai, validos):
                crit[a] = crit.get(a, 0) + r.criticos_por_mil
        top_shap = s.atributo.head(K).tolist()
        top_temas = e.tema.head(K).tolist()
        attrs_nlp = {a for t in top_temas for a in pais_do_tema(t, pai, validos)}
        comuns = [a for a in top_shap if a in attrs_nlp]
        temas_hit = [t for t in top_temas if pais_do_tema(t, pai, validos) & set(top_shap)]
        nota = s.set_index(["aeroporto", "atributo"]).index.map(notas_t5) if notas_t5 is not None else s.atributo.map(notas.loc[ap])
        df = s.assign(crit_nlp=s.atributo.map(crit).fillna(0), nota=np.asarray(nota, dtype=float))
        rho_s, p_s = spearmanr(df.crit_nlp, df.shap)
        dn = df.dropna(subset=["nota"])
        rho_n, p_n = spearmanr(dn.crit_nlp, -dn.nota)
        rho_sn, _ = spearmanr(dn.shap, -dn.nota)
        nc = ag[(ag.aeroporto == ap) & (ag.cobertura == "nao_coberto")].sort_values("criticos_por_mil", ascending=False)
        linhas.append({
            "aeroporto": ap, "top5_shap": " | ".join(top_shap), "top5_temas_criticos": " | ".join(TEMAS[t][0] for t in top_temas),
            "n_atrib_top5_shap_cobertos_por_top5_nlp": len(comuns), "n_temas_top5_nlp_em_top5_shap": len(temas_hit),
            "tema1_nlp_em_top5_shap": bool(pais_do_tema(top_temas[0], pai, validos) & set(top_shap)),
            "rho_nlp_shap": rho_s, "p_nlp_shap": p_s, "rho_nlp_nota": rho_n, "p_nlp_nota": p_n, "rho_shap_nota": rho_sn,
            "n_atributos": len(df),
            "top_nao_coberto": nc.tema_rotulo.iloc[0] if len(nc) else "", "top_nao_coberto_por_mil": nc.criticos_por_mil.iloc[0] if len(nc) else np.nan,
            "rank_top_nao_coberto_entre_todos": int((ag[(ag.aeroporto == ap) & ~ag.cobertura.isin(["geral", "sem_tema"])]
                                                     .criticos_por_mil > nc.criticos_por_mil.iloc[0]).sum() + 1) if len(nc) else np.nan,
        })
        attr_rows.append(df.assign(aeroporto=ap))
    tri = pd.DataFrame(linhas)
    # Benchmark: se a concordância fosse só estrutura nacional, o NLP do aeroporto A concordaria tanto com o SHAP
    # de outro aeroporto B quanto com o seu. Compara ρ(A,A) com a média de ρ(A,B), B≠A.
    todos = pd.concat(attr_rows)
    fora = []
    for a in tri.aeroporto:
        ca = todos[todos.aeroporto == a].set_index("atributo").crit_nlp
        for b in tri.aeroporto:
            if a != b:
                sb = todos[todos.aeroporto == b].set_index("atributo").shap
                j = ca.index.intersection(sb.index)
                fora.append(spearmanr(ca[j], sb[j])[0])
    tri["rho_nlp_shap_fora_diagonal_media"] = np.nanmean(fora)
    # mesmo benchmark para a nota Likert: NLP de A × nota de B
    fora_n = []
    for a in tri.aeroporto:
        ca = todos[todos.aeroporto == a].set_index("atributo").crit_nlp
        for b in tri.aeroporto:
            if a != b:
                nb = todos[todos.aeroporto == b].set_index("atributo").nota.dropna()
                j = ca.index.intersection(nb.index)
                fora_n.append(spearmanr(ca[j], -nb[j])[0])
    tri["rho_nlp_nota_fora_diagonal_media"] = np.nanmean(fora_n)
    # Especificidade no nível do tema: para cada tema, correlação entre aeroportos (n=20) da taxa de críticas
    # por mil com a nota Likert média do atributo-pai correspondente (quanto mais crítica, pior a nota?)
    esp_rows = []
    for t in esp.tema.unique():
        ps = pais_do_tema(t, pai, validos)
        if len(ps) != 1:
            continue
        p_ = next(iter(ps))
        x = esp[esp.tema == t].set_index("aeroporto").criticos_por_mil.reindex(tri.aeroporto).fillna(0)
        y = todos[todos.atributo == p_].set_index("aeroporto").nota.reindex(tri.aeroporto)
        z = todos[todos.atributo == p_].set_index("aeroporto").shap.reindex(tri.aeroporto)
        esp_rows.append({"tema": t, "tema_rotulo": TEMAS[t][0], "atributo_pai": p_, "n_comentarios": int(esp[(esp.tema == t) & (esp.aeroporto != "BRASIL")].n_comentarios.sum()),
                         "rho_entre_aeroportos_crit_x_nota": spearmanr(x, -y, nan_policy="omit")[0],
                         "p_nota": spearmanr(x, -y, nan_policy="omit")[1],
                         "rho_entre_aeroportos_crit_x_shap": spearmanr(x, z, nan_policy="omit")[0]})
    et = pd.DataFrame(esp_rows).sort_values("n_comentarios", ascending=False)
    et.to_csv(SAIDA / "triangulacao_entre_aeroportos_por_tema.csv", index=False)
    print(et.round(3).to_string())
    ln = []
    for _, r in et[et.tema.map(lambda k: TEMAS[k][1]) != "geral"].iterrows():
        if r.tema in ("facilidade", "limpeza_org", "atendimento"):  # temas majoritariamente de elogio
            continue
        sig = "$^{*}$" if r.p_nota < 0.05 else ""
        ln.append(f"{r.tema_rotulo} & {r.atributo_pai.capitalize()} & {r.n_comentarios} & {fmt(r.rho_entre_aeroportos_crit_x_nota)}{sig} & {fmt(r.rho_entre_aeroportos_crit_x_shap)} \\\\")
    (TEX / "nlp_triangulacao_entre_aeroportos.tex").write_text(r"""\begin{table}[htbp]
\centering
\caption{Especificidade por aeroporto: correlação de Spearman, entre os 20 aeroportos, da taxa de comentários críticos por mil respondentes de cada tema com a nota Likert média invertida e com o $|$SHAP$|$ médio do atributo-pai correspondente. $^{*}$: $p<0{,}05$. Temas majoritariamente de elogio omitidos.}
\label{tab:nlp-triangulacao-entre}
\footnotesize
\begin{tabular}{p{4.6cm}p{4.2cm}rcc}
\toprule
Tema (NLP) & Atributo-pai & $N$ & $\rho$ com nota & $\rho$ com $|$SHAP$|$ \\
\midrule
""" + "\n".join(ln) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    tri.to_csv(SAIDA / "triangulacao_por_aeroporto.csv", index=False)
    pd.concat(attr_rows).to_csv(SAIDA / "triangulacao_atributos.csv", index=False)
    pd.set_option("display.width", 250)
    print("Fonte SHAP:", fonte)
    print(tri.drop(columns=["top5_shap", "top5_temas_criticos"]).round(3).to_string())
    for _, r in tri.iterrows():
        print(f"\n{r.aeroporto}\n  SHAP: {r.top5_shap}\n  NLP : {r.top5_temas_criticos}")
    print("\nMédias: comuns top5 =", tri.n_atrib_top5_shap_cobertos_por_top5_nlp.mean(),
          "| rho NLP×SHAP =", tri.rho_nlp_shap.mean().round(3), "| rho NLP×nota =", tri.rho_nlp_nota.mean().round(3),
          "| rho SHAP×nota =", tri.rho_shap_nota.mean().round(3),
          "| rho NLP(A)×SHAP(B≠A) médio =", round(float(np.nanmean(fora)), 3),
          "| rho NLP(A)×nota(B≠A) médio =", round(float(np.nanmean(fora_n)), 3),
          "| tema1 no top5 SHAP:", int(tri.tema1_nlp_em_top5_shap.sum()), "/", len(tri))

    # ---- LaTeX triangulação
    abrev = lambda a: a.capitalize().replace("Facilidade de identificação da esteira de restituição", "Ident. da esteira") \
        .replace("Processo de restituição de bagagens", "Restituição de bagagens").replace("Localização e deslocamento", "Localização/deslocamento") \
        .replace("Limpeza geral do aeroporto", "Limpeza geral").replace("Conforto da sala de embarque", "Conforto sala embarque") \
        .replace("Disponibilidade de ", "Disp. de ").replace("Facilidade de desembarque no meio-fio", "Meio-fio")
    ln = []
    for _, r in tri.iterrows():
        sh = "; ".join(abrev(a) for a in r.top5_shap.split(" | ")[:3])
        nl = "; ".join(r.top5_temas_criticos.split(" | ")[:3])
        ln.append(f"{r.aeroporto} & {sh} & {nl} & {r.n_atrib_top5_shap_cobertos_por_top5_nlp}/5 & {fmt(r.rho_nlp_shap)} & {fmt(r.rho_nlp_nota)} \\\\")
    (TEX / "nlp_triangulacao.tex").write_text(r"""{\footnotesize
\begin{longtable}{lp{4.2cm}p{4.8cm}ccc}
\caption{Triangulação por aeroporto: três atributos de maior $|$SHAP$|$ médio e três temas mais criticados nos comentários (críticos por mil respondentes). ``Top-5'': nº de atributos do top-5 SHAP cobertos pelos temas do top-5 NLP. $\rho$: Spearman, entre os atributos do ranking SHAP, da taxa de críticas do NLP com o $|$SHAP$|$ e com a nota Likert invertida. Fonte SHAP: \texttt{""" + fonte.replace("_", r"\_") + r"""}.}
\label{tab:nlp-triangulacao}\\
\toprule
Aerop. & Top-3 SHAP & Top-3 temas críticos (NLP) & Top-5 & $\rho_{\text{SHAP}}$ & $\rho_{\text{nota}}$ \\
\midrule
\endfirsthead
\toprule
Aerop. & Top-3 SHAP & Top-3 temas críticos (NLP) & Top-5 & $\rho_{\text{SHAP}}$ & $\rho_{\text{nota}}$ \\
\midrule
\endhead
""" + "\n".join(ln) + r"""
\bottomrule
\end{longtable}
}
""")

    # ---- LaTeX temas não cobertos (nacional + onde é mais citado)
    nc = ag[ag.cobertura == "nao_coberto"]
    br = nc[nc.aeroporto == "BRASIL"].sort_values("n_comentarios", ascending=False)
    ln = []
    for _, r in br.iterrows():
        loc = nc[(nc.tema == r.tema) & (nc.aeroporto != "BRASIL")].sort_values("criticos_por_mil", ascending=False).head(3)
        onde = ", ".join(f"{a} ({fmt(v,1)})" for a, v in zip(loc.aeroporto, loc.criticos_por_mil))
        ln.append(f"{r.tema_rotulo} & {int(r.n_comentarios)} & {fmt(r.pct_dos_comentarios,1)} & {fmt(r.sentimento_medio)} & {fmt(r.pct_critico,0)} & {onde} \\\\")
    (TEX / "nlp_temas_nao_cobertos.tex").write_text(r"""\begin{table}[htbp]
\centering
\caption{Temas presentes nos comentários abertos sem atributo correspondente no questionário MPor (20 aeroportos, """ + ANOS_TEX + r"""). \% com.: participação no total de comentários válidos; aeroportos com maior taxa de comentários críticos por mil respondentes entre parênteses.}
\label{tab:nlp-temas-nao-cobertos}
\footnotesize
\begin{tabular}{p{4.4cm}rrrrp{4.6cm}}
\toprule
Tema & $N$ & \% com. & Sent. & \% crít. & Aeroportos mais críticos (por mil) \\
\midrule
""" + "\n".join(ln).replace('("melhorou")', "(``melhorou'')") + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    main()
