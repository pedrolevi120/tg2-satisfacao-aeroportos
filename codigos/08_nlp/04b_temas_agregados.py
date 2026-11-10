"""
T7/NLP — Etapa 4b: aplica a rotulagem manual (temas.py) às atribuições do BERTopic, junta sentimento
e produz os artefatos consumidos pelo MVP e pelo texto.

Definições
  crítico = comentário cujo sentimento previsto (XLM-R) NÃO é positivo. Justificativa: na validação, a maioria
            das sugestões de melhoria (rotuladas negativas pela rubrica) é prevista como neutra; a versão binária
            positivo × não positivo é a mais confiável (acurácia 0,875).
  críticos por mil = nº de comentários críticos do tema no aeroporto / nº de respondentes do aeroporto × 1000.

Série 2021T1–2026T2: tópicos vêm de 04a2_atribuir_temas_serie.py (2023 = atribuição original do BERTopic;
demais anos = k vizinhos mais próximos entre os comentários de 2023). agregado_aeroporto_tema.csv agrega a série
inteira (mesmo formato lido pelo MVP); agregado_tema_ano.csv abre por ano.
Ressalvas para 2021–2022: (i) o campo de comentário vem em branco com mais frequência (taxa de comentários
válidos de 10–14% contra 18–20% em 2023–2026), o que reduz as taxas por mil respondentes daqueles anos — a
participação do tema nos comentários (pct_dos_comentarios) não sofre esse efeito; (ii) a amostra tem ~71% de
embarque (50% a partir de 2023), o que reduz temas de desembarque (bagagem) independentemente da qualidade.

Saídas: saida/comentarios_topicos.parquet (1 linha por comentário válido)
        saida/agregado_aeroporto_tema.csv   (aeroporto × tema, inclui linha agregada 'BRASIL'; série inteira)
        saida/agregado_tema_ano.csv         (aeroporto × tema × ano, inclui 'BRASIL')
        saida/temas_por_ano_brasil.csv      (Brasil: críticos por mil e posição de cada tema por ano)
        saida/temas_2023_robustez_metodo.csv (2023: atribuição original × kNN fora da dobra)
        saida/temas_resumo.csv
        texto/_paralelo/tex/nlp_topicos_atributos.tex
        midias/figuras/nlp_temas_sentimento.pdf, midias/figuras/nlp_heatmap_criticas.pdf,
        midias/figuras/nlp_temas_evolucao_ano.pdf (+ apresentacoes/img_png/nlp_temas_evolucao_ano-1.png)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comum import ROOT, SAIDA, FIG, TEX, NOMES, BASE_COM, CHAVE, ANOS_ROT, ANOS_TEX, estilo, fmt
from temas import TEMAS, TOPICO_PARA_TEMA

COB_LAB = {"coberto": "Coberto", "parcial": "Parcial", "nao_coberto": "Não coberto", "geral": "Global", "sem_tema": "---"}


def main():
    s = pd.read_parquet(SAIDA / "sentimento.parquet")
    at = pd.read_parquet(SAIDA / "bertopic_atribuicoes.parquet")
    d = s.merge(at.drop(columns=["ANO"]), on=[CHAVE, "AEROPORTO"], how="left")
    assert len(d) == len(s)
    # texto exportado: anonimização por NER (02b) — nomes de pessoas → [NOME]
    an = pd.read_parquet(SAIDA / "anonimizacao_ner.parquet")
    d = d.merge(an, on=[CHAVE, "AEROPORTO"], how="left")
    assert d.texto_anon.notna().all()
    d["texto"] = d["texto_anon"]
    # Configuração principal: atribuição direta do HDBSCAN (outliers −1 → 'sem_tema'). A realocação de outliers
    # por similaridade (coluna topico_realocado) foi testada e rejeitada: precisão de 0,38 nos realocados contra
    # 0,83 nos atribuídos diretamente (validação manual, 04c_validacao_temas.py).
    d["tema"] = np.where(d["classe"] == "generico", "elogio_generico", d["topico_hdbscan"].map(TOPICO_PARA_TEMA))
    d["tema_realocado"] = np.where(d["classe"] == "generico", "elogio_generico", d["topico"].map(TOPICO_PARA_TEMA))
    d["tema_knn_oof"] = np.where(d["classe"] == "generico", "elogio_generico", d["topico_knn_oof"].map(TOPICO_PARA_TEMA))
    d.loc[d.ANO != 2023, "tema_knn_oof"] = None  # só definido para 2023 (robustez do método)
    assert d["tema"].notna().all()
    d["tema_rotulo"] = d["tema"].map(lambda k: TEMAS[k][0])
    d["cobertura"] = d["tema"].map(lambda k: TEMAS[k][1])
    d["atributos_likert"] = d["tema"].map(lambda k: "; ".join(TEMAS[k][2]))
    d["critico"] = d["xlmr_classe"] != "positivo"
    cols = [CHAVE, "CHAVE", "ANO", "PERIODO", "AEROPORTO", "PROCESSO", "TIPO DE VOO", "TRIMESTRE", "sat", "classe", "texto", "n_palavras",
            "topico_hdbscan", "topico", "tema_realocado", "tema", "tema_rotulo", "cobertura", "atributos_likert",
            "xlmr_classe", "xlmr_escore", "xlmr_p_negativo", "xlmr_p_neutro", "xlmr_p_positivo",
            "bertweetpt_classe", "bertweetpt_escore", "critico", "origem", "tema_knn_oof"]
    oof = d[["tema_knn_oof", "ANO", "AEROPORTO", "critico", "classe"]].copy()
    d = d[cols].rename(columns={"TIPO DE VOO": "TIPO_VOO", "topico_hdbscan": "topico_bertopic", "topico": "topico_realocado", "sat": "SATISFACAO_GERAL",
                               "origem": "origem_tema"})
    d.to_parquet(SAIDA / "comentarios_topicos.parquet", index=False)

    base = pd.read_parquet(BASE_COM, columns=["AEROPORTO", "ANO"])
    n_resp = base.groupby("AEROPORTO").size()
    n_resp["BRASIL"] = len(base)
    n_val = d.groupby("AEROPORTO").size()
    n_val["BRASIL"] = len(d)

    def agrega(g, ap):
        return pd.Series({
            "n_comentarios": len(g), "n_criticos": int(g.critico.sum()),
            "sentimento_medio": g.xlmr_escore.mean(), "pct_negativo": 100 * (g.xlmr_classe == "negativo").mean(),
            "pct_critico": 100 * g.critico.mean(),
            "pct_dos_comentarios": 100 * len(g) / n_val[ap],
            "comentarios_por_mil": 1000 * len(g) / n_resp[ap], "criticos_por_mil": 1000 * g.critico.sum() / n_resp[ap],
            "satisfacao_media_autores": g.SATISFACAO_GERAL.mean(),
        })
    rows = [agrega(g, ap).rename((ap, t)) for (ap, t), g in d.groupby(["AEROPORTO", "tema"])]
    rows += [agrega(g, "BRASIL").rename(("BRASIL", t)) for t, g in d.groupby("tema")]
    ag = pd.DataFrame(rows)
    ag.index = pd.MultiIndex.from_tuples(ag.index, names=["aeroporto", "tema"])
    ag = ag.reset_index()
    ag["tema_rotulo"] = ag.tema.map(lambda k: TEMAS[k][0])
    ag["cobertura"] = ag.tema.map(lambda k: TEMAS[k][1])
    ag["atributos_likert"] = ag.tema.map(lambda k: "; ".join(TEMAS[k][2]))
    # ranking de criticidade dentro do aeroporto (apenas temas específicos)
    esp = ~ag.cobertura.isin(["geral", "sem_tema"])
    ag["rank_critico"] = np.nan
    ag.loc[esp, "rank_critico"] = ag[esp].groupby("aeroporto")["criticos_por_mil"].rank(ascending=False, method="first")
    ag = ag.sort_values(["aeroporto", "rank_critico"])
    ag.to_csv(SAIDA / "agregado_aeroporto_tema.csv", index=False)

    br = ag[ag.aeroporto == "BRASIL"].set_index("tema")
    res = br[["tema_rotulo", "cobertura", "atributos_likert", "n_comentarios", "pct_dos_comentarios", "sentimento_medio",
              "pct_critico", "satisfacao_media_autores"]].sort_values("n_comentarios", ascending=False)
    res["topicos_bertopic"] = [", ".join(map(str, TEMAS[k][3])) for k in res.index]
    res.to_csv(SAIDA / "temas_resumo.csv")
    print(res.round(2).to_string())
    subst = d[d.classe == "substantivo"]
    print("\nCobertura (% dos comentários substantivos):", (100 * subst.cobertura.value_counts(normalize=True)).round(1).to_dict())
    crit = subst[subst.critico]
    print("Cobertura (% dos substantivos CRÍTICOS):", (100 * crit.cobertura.value_counts(normalize=True)).round(1).to_dict())

    # ---- LaTeX tópico → atributo (longtable)
    ln = []
    for cob in ["coberto", "parcial", "nao_coberto", "geral", "sem_tema"]:
        for k, r in res[res.cobertura == cob].iterrows():
            atr = r.atributos_likert.replace("; ", "; ").lower().capitalize() if r.atributos_likert else "---"
            atr = atr.replace("x qualidade", "$\\times$ qualidade").replace("x benefício", "$\\times$ benefício")
            top = r.topicos_bertopic if r.topicos_bertopic else "---"
            n_fmt = f"{int(r.n_comentarios):,}".replace(",", ".")
            ln.append(f"{r.tema_rotulo} & {top} & {n_fmt}" +
                      f" & {fmt(r.sentimento_medio)} & {fmt(r.pct_critico,0)} & {COB_LAB[cob]} & {atr} \\\\")
        ln.append("\\midrule")
    ln = ln[:-1]
    (TEX / "nlp_topicos_atributos.tex").write_text(r"""{\footnotesize
\begin{longtable}{p{3.6cm}p{1.5cm}rrrp{1.3cm}p{4.6cm}}
\caption{Temas dos comentários abertos (""" + ANOS_TEX + r"""; BERTopic ajustado em 2023, rotulagem manual, demais anos atribuídos por vizinhos mais próximos) e mapeamento para os atributos Likert do questionário MPor. Sent.: escore médio $P(\text{pos})-P(\text{neg})$ (XLM-R); \% crít.: comentários não positivos. ``Não coberto'': tema sem atributo correspondente no questionário.}
\label{tab:nlp-topicos-atributos}\\
\toprule
Tema & Tópicos & $N$ & Sent. & \% crít. & Cobertura & Atributo(s) Likert \\
\midrule
\endfirsthead
\toprule
Tema & Tópicos & $N$ & Sent. & \% crít. & Cobertura & Atributo(s) Likert \\
\midrule
\endhead
""" + "\n".join(ln).replace('("melhorou")', "(``melhorou'')") + r"""
\bottomrule
\end{longtable}
}
""")

    # ---- Figuras
    estilo()
    r2 = res[~res.cobertura.isin(["sem_tema"])].sort_values("n_comentarios")
    fig, ax = plt.subplots(figsize=(7, 7.5))
    cmap = plt.get_cmap("RdBu")
    cores = [cmap((v + 1) / 2) for v in r2.sentimento_medio]
    bars = ax.barh(range(len(r2)), r2.n_comentarios, color=cores, edgecolor="black", linewidth=0.3)
    for b, cob in zip(bars, r2.cobertura):
        if cob == "nao_coberto":
            b.set_hatch("///")
    ax.set_yticks(range(len(r2)), [t if len(t) < 48 else t[:46] + "…" for t in r2.tema_rotulo], fontsize=7.5)
    ax.set_xlabel(f"Nº de comentários válidos (20 aeroportos, {ANOS_ROT})")
    ax.set_title("Temas dos comentários abertos: volume e sentimento médio")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Sentimento médio (XLM-R)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="///", label="Tema sem atributo no questionário")],
              loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "nlp_temas_sentimento.pdf")

    esp = ~ag.cobertura.isin(["geral", "sem_tema"])
    h = ag[(ag.aeroporto != "BRASIL") & esp].pivot(index="aeroporto", columns="tema", values="criticos_por_mil").fillna(0)
    ordem = br[~br.cobertura.isin(["geral", "sem_tema"])].sort_values("criticos_por_mil", ascending=False).index[:18]
    h = h[ordem]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(h.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(h.shape[1]), [TEMAS[k][0][:32] + ("*" if TEMAS[k][1] == "nao_coberto" else "") for k in h.columns],
                  rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(h.shape[0]), [f"{a} ({NOMES[a]})" for a in h.index], fontsize=7.5)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > h.values.max() * 0.6 else "black")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Comentários críticos por mil respondentes")
    ax.set_title(f"Menções críticas por tema e aeroporto, {ANOS_ROT} (* = tema sem atributo no questionário)")
    fig.savefig(FIG / "nlp_heatmap_criticas.pdf")

    por_ano(d, base, oof)


def por_ano(d, base, oof):
    """Temas por ano: críticos por mil respondentes (Brasil e aeroporto × ano) + figura de evolução."""
    anos = sorted(base.ANO.unique())
    n_resp = base.groupby(["AEROPORTO", "ANO"]).size()
    for a in anos:
        n_resp[("BRASIL", a)] = (base.ANO == a).sum()
    dd = pd.concat([d, d.assign(AEROPORTO="BRASIL")])
    n_val = dd.groupby(["AEROPORTO", "ANO"]).size()
    n_sub = dd[dd.classe == "substantivo"].groupby(["AEROPORTO", "ANO"]).size()
    g = dd.groupby(["AEROPORTO", "tema", "ANO"])
    ag = pd.DataFrame({"n_comentarios": g.size(), "n_criticos": g.critico.sum(),
                       "sentimento_medio": g.xlmr_escore.mean(), "pct_critico": 100 * g.critico.mean()})
    grade = pd.MultiIndex.from_product([dd.AEROPORTO.dropna().unique(), list(TEMAS), anos], names=["AEROPORTO", "tema", "ANO"])
    ag = ag.reindex(grade)
    ag[["n_comentarios", "n_criticos"]] = ag[["n_comentarios", "n_criticos"]].fillna(0).astype(int)
    ag = ag.reset_index()
    idx = pd.MultiIndex.from_frame(ag[["AEROPORTO", "ANO"]])
    ag["n_respondentes"] = n_resp.reindex(idx).values
    ag["n_validos"] = n_val.reindex(idx).fillna(0).values
    ag["n_substantivos"] = n_sub.reindex(idx).fillna(0).values
    ag["criticos_por_mil"] = 1000 * ag.n_criticos / ag.n_respondentes
    ag["comentarios_por_mil"] = 1000 * ag.n_comentarios / ag.n_respondentes
    ag["pct_dos_comentarios"] = 100 * ag.n_comentarios / ag.n_validos
    ag["pct_dos_substantivos"] = np.where(ag.tema == "elogio_generico", np.nan, 100 * ag.n_comentarios / ag.n_substantivos)
    ag["tema_rotulo"] = ag.tema.map(lambda k: TEMAS[k][0])
    ag["cobertura"] = ag.tema.map(lambda k: TEMAS[k][1])
    esp = ~ag.cobertura.isin(["geral", "sem_tema"])
    ag["rank_critico"] = np.nan
    ag.loc[esp, "rank_critico"] = ag[esp].groupby(["AEROPORTO", "ANO"])["criticos_por_mil"].rank(ascending=False, method="first")
    ag = ag.rename(columns={"AEROPORTO": "aeroporto"}).sort_values(["aeroporto", "ANO", "rank_critico"])
    ag.to_csv(SAIDA / "agregado_tema_ano.csv", index=False)

    br = ag[ag.aeroporto == "BRASIL"]
    wide = br.pivot(index="tema", columns="ANO", values="criticos_por_mil")
    rk = br.pivot(index="tema", columns="ANO", values="rank_critico").add_prefix("pos_")
    pc = br.pivot(index="tema", columns="ANO", values="pct_dos_comentarios").add_prefix("pct_")
    tot = pd.read_csv(SAIDA / "agregado_aeroporto_tema.csv").query("aeroporto == 'BRASIL'").set_index("tema")
    wide = wide.join(rk).join(pc)
    wide.insert(0, "serie", tot.criticos_por_mil.reindex(wide.index))
    wide.insert(0, "cobertura", wide.index.map(lambda k: TEMAS[k][1]))
    wide.insert(0, "tema_rotulo", wide.index.map(lambda k: TEMAS[k][0]))
    wide = wide.sort_values("serie", ascending=False)
    # 2023 pelo mesmo método dos outros anos (kNN fora da dobra) — mostra o efeito do método na comparação entre anos
    o = oof[oof.ANO == 2023]
    n23 = (base.ANO == 2023).sum()
    wide["2023_knn_oof"] = (o[o.critico].groupby("tema_knn_oof").size() * 1000 / n23).reindex(wide.index).fillna(0)
    wide.to_csv(SAIDA / "temas_por_ano_brasil.csv")
    pd.set_option("display.width", 250)
    print("\nCríticos por mil respondentes, Brasil, por ano:")
    print(wide[["cobertura", "serie"] + anos + ["2023_knn_oof"]].round(2).to_string())
    print("\nPosição (temas específicos) por ano:")
    print(wide[wide.cobertura.isin(["coberto", "parcial", "nao_coberto"])][[f"pos_{a}" for a in anos]].to_string())

    # robustez do método em 2023: atribuição original × kNN fora da dobra (mesmo método dos outros anos)
    o = oof[oof.ANO == 2023]
    n23 = (base.ANO == 2023).sum()
    rob = pd.DataFrame({"criticos_por_mil_original": d[d.ANO == 2023].groupby("tema").critico.sum() * 1000 / n23,
                        "criticos_por_mil_knn_oof": o.groupby("tema_knn_oof").critico.sum() * 1000 / n23}).fillna(0)
    rob["razao"] = rob.criticos_por_mil_knn_oof / rob.criticos_por_mil_original
    rob = rob.sort_values("criticos_por_mil_original", ascending=False)
    rob.to_csv(SAIDA / "temas_2023_robustez_metodo.csv")
    from scipy.stats import spearmanr
    e = rob.drop(index=[t for t in rob.index if TEMAS[t][1] in ("geral", "sem_tema")])
    print("\nRobustez 2023 (original × kNN fora da dobra): rho de Spearman entre temas específicos =",
          round(spearmanr(e.criticos_por_mil_original, e.criticos_por_mil_knn_oof)[0], 3))
    print(rob.round(2).to_string())

    # ---- Figura: evolução anual dos 12 temas específicos mais criticados na série (Brasil)
    estilo()
    top = wide[wide.cobertura.isin(["coberto", "parcial", "nao_coberto"])].index[:12]
    fig, axs = plt.subplots(3, 4, figsize=(10, 6.6), sharex=True)
    cor = {"coberto": "#2166ac", "parcial": "#2166ac", "nao_coberto": "#d6604d"}
    for ax, t in zip(axs.flat, top):
        y = wide.loc[t, anos].values.astype(float)
        ax.axvspan(min(anos) - 0.4, 2022.5, color="#eeeeee", zorder=0, lw=0)
        ax.plot(anos, y, color=cor[TEMAS[t][1]], lw=1.8, marker="o", ms=4)
        ax.plot([2023], [wide.loc[t, "2023_knn_oof"]], ls="none", marker="o", ms=6, mfc="none", mec="#555555", mew=0.9)
        ax.set_ylim(0, max(y.max() * 1.25, 1))
        ax.set_xlim(min(anos) - 0.4, max(anos) + 0.4)
        rot = TEMAS[t][0]
        rot = rot if len(rot) <= 34 else rot[:32] + "…"
        ax.set_title(rot + (" *" if TEMAS[t][1] == "nao_coberto" else ""), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(anos, [f"{a % 100:02d}" for a in anos])
    for ax in axs[:, 0]:
        ax.set_ylabel("Críticos por mil resp.", fontsize=8)
    for ax in axs[-1, :]:
        ax.set_xlabel("Ano (20xx)", fontsize=8)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color="#2166ac", marker="o", label="Tema com atributo no questionário"),
                        Line2D([], [], color="#d6604d", marker="o", label="* Tema sem atributo no questionário"),
                        Line2D([], [], ls="none", marker="o", mfc="none", mec="#555555", label="2023 atribuído como os demais anos (kNN)"),
                        Patch(facecolor="#eeeeee", label="2021–2022: campo mais vezes em branco; 71% embarque")],
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Comentários críticos por tema e ano, 20 aeroportos ({ANOS_ROT}; 2026 até junho)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "nlp_temas_evolucao_ano.pdf")
    png = ROOT / "apresentacoes" / "img_png"
    if png.exists():
        fig.savefig(png / "nlp_temas_evolucao_ano-1.png", dpi=200)


if __name__ == "__main__":
    main()
