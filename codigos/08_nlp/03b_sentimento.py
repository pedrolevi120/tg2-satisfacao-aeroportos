"""
T7/NLP — Etapa 3b: sentimento dos comentários com modelos abertos locais, validação na amostra
rotulada (n=200) e correlação com a SATISFAÇÃO GERAL.

Modelos (sem seleção na amostra de validação — o primário foi fixado a priori):
  primário:   cardiffnlp/twitter-xlm-roberta-base-sentiment (XLM-R multilíngue, 3 classes)
  robustez:   pysentimiento/bertweet-pt-sentiment (BERTweet treinado em PT-BR, 3 classes)
Escore contínuo: s = P(positivo) − P(negativo) ∈ [−1, 1].

Série 2021T1–2026T2: inferência em todos os comentários válidos da série; a validação continua na amostra
FIXA de 200 comentários de 2023 (casada por ANO == 2023 + CHAVE + AEROPORTO).

Saídas: saida/sentimento.parquet, saida/sentimento_validacao.json, saida/sentimento_por_aeroporto.csv (série
        agregada), saida/sentimento_por_ano.csv (Brasil e aeroporto × ano),
        texto/_paralelo/tex/nlp_sentimento_validacao.tex, texto/_paralelo/tex/nlp_sentimento_satisfacao.tex,
        midias/figuras/nlp_sentimento_satisfacao.pdf
"""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from comum import SAIDA, AQUI, FIG, TEX, SEED, NOMES, BASE_COM, ANOS_TEX, estilo, fmt

MODELOS = {
    "xlmr": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "bertweetpt": "pysentimiento/bertweet-pt-sentiment",
}
CLASSES = ["negativo", "neutro", "positivo"]
torch.manual_seed(SEED)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def mapear_rotulos(id2label):
    m = {}
    for i, l in id2label.items():
        l = l.lower()
        m[int(i)] = "negativo" if l.startswith("neg") else "positivo" if l.startswith("pos") else "neutro"
    return m


def inferir(nome, textos, bs=64):
    tok = AutoTokenizer.from_pretrained(nome)
    mod = AutoModelForSequenceClassification.from_pretrained(nome).to(DEV).eval()
    mp = mapear_rotulos(mod.config.id2label)
    probs = []
    # ordena por comprimento (menos padding); a ordem original é restaurada no fim
    ordem = np.argsort([len(t) for t in textos], kind="stable")
    tx = [textos[i] for i in ordem]
    with torch.no_grad():
        for i in range(0, len(tx), bs):
            enc = tok(tx[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
            probs.append(torch.softmax(mod(**enc).logits, -1).float().cpu().numpy())
            if (i // bs) % 200 == 0:
                print(f"  {i}/{len(tx)}", flush=True)
    P = np.empty((len(tx), probs[0].shape[1]), dtype=np.float32)
    P[ordem] = np.vstack(probs)
    del mod
    if DEV == "mps":
        torch.mps.empty_cache()
    out = pd.DataFrame({f"p_{mp[j]}": P[:, j] for j in range(P.shape[1])})
    out["classe"] = out[[f"p_{c}" for c in CLASSES]].idxmax(axis=1).str[2:]
    out["escore"] = out["p_positivo"] - out["p_negativo"]
    return out


def metricas(y, p):
    yb, pb = (np.array(y) == "positivo"), (np.array(p) == "positivo")
    return {
        "acuracia": accuracy_score(y, p), "f1_macro": f1_score(y, p, average="macro", labels=CLASSES, zero_division=0),
        "f1_ponderado": f1_score(y, p, average="weighted", labels=CLASSES, zero_division=0),
        "kappa": cohen_kappa_score(y, p),
        "f1_por_classe": dict(zip(CLASSES, f1_score(y, p, average=None, labels=CLASSES, zero_division=0).round(4).tolist())),
        "bin_acuracia": accuracy_score(yb, pb), "bin_f1_macro": f1_score(yb, pb, average="macro"),
        "matriz_confusao": confusion_matrix(y, p, labels=CLASSES).tolist(),
    }


def boot_spearman(x, y, n=1000):
    rng = np.random.default_rng(SEED)
    x, y = np.asarray(x), np.asarray(y)
    rs = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.percentile(rs, [2.5, 97.5])


def main():
    d = pd.read_parquet(SAIDA / "comentarios_limpos.parquet").reset_index(drop=True)
    textos = d["texto"].tolist()
    for k, nome in MODELOS.items():
        print("Inferindo", nome, "em", DEV, "...")
        r = inferir(nome, textos)
        for c in r.columns:
            d[f"{k}_{c}"] = r[c].values
    d.to_parquet(SAIDA / "sentimento.parquet", index=False)

    # ---- validação
    rot = pd.read_csv(AQUI / "rotulos_manuais_200.csv")
    v = rot.merge(d[d.ANO == 2023], on=["CHAVE", "AEROPORTO"], how="left")
    assert v["xlmr_classe"].notna().all() and len(v) == 200
    res = {"n": 200, "dist_manual": rot.rotulo_manual.value_counts().to_dict()}
    for k in MODELOS:
        res[k] = metricas(v.rotulo_manual, v[f"{k}_classe"])
        print(f"\n== {k} ==\n", classification_report(v.rotulo_manual, v[f"{k}_classe"], labels=CLASSES, zero_division=0))
        print("Matriz (linhas=manual neg/neu/pos):", res[k]["matriz_confusao"])
    # erros do primário (para análise qualitativa)
    err = v[v.rotulo_manual != v.xlmr_classe][["texto", "rotulo_manual", "xlmr_classe", "bertweetpt_classe"]]
    err.to_csv(SAIDA / "sentimento_erros_validacao.csv", index=False)

    # ---- correlação com SATISFAÇÃO GERAL
    s = d.dropna(subset=["sat"])
    corr = {}
    for k in MODELOS:
        for sub, mask in {"validos": np.ones(len(s), bool), "substantivos": (s.classe == "substantivo").values}.items():
            x, y = s.loc[mask, f"{k}_escore"], s.loc[mask, "sat"]
            rho, pv = spearmanr(x, y)
            lo, hi = boot_spearman(x.values, y.values)
            corr[f"{k}_{sub}"] = {"n": int(mask.sum()), "rho": rho, "p": pv, "ic95": [lo, hi]}
    res["correlacao_sat"] = corr
    res["sat_media_por_classe"] = {k: s.groupby(f"{k}_classe")["sat"].agg(["mean", "count"]).round(4).to_dict("index") for k in MODELOS}
    # correlação ecológica: por aeroporto, % negativo vs satisfação média de TODOS os respondentes
    base = pd.read_parquet(BASE_COM, columns=["AEROPORTO", "ANO", "SATISFAÇÃO GERAL"])
    base["sat"] = pd.to_numeric(base["SATISFAÇÃO GERAL"], errors="coerce")

    def resumo(dd, bb, by):
        return pd.DataFrame({"sat_media": bb.groupby(by)["sat"].mean(),
                             "escore_medio": dd.groupby(by)["xlmr_escore"].mean(),
                             "pct_neg": dd.groupby(by)["xlmr_classe"].apply(lambda c: 100 * (c == "negativo").mean()),
                             "pct_pos": dd.groupby(by)["xlmr_classe"].apply(lambda c: 100 * (c == "positivo").mean()),
                             "n_comentarios": dd.groupby(by).size()})
    ag = resumo(d, base, "AEROPORTO")  # série agregada (formato lido pelo MVP)
    rho_e, p_e = spearmanr(ag.escore_medio, ag.sat_media)
    res["correlacao_ecologica_20"] = {"rho": rho_e, "p": p_e}
    ag.drop(columns=["pct_pos", "n_comentarios"]).to_csv(SAIDA / "sentimento_por_aeroporto.csv")
    # por ano: Brasil e aeroporto × ano
    pa = resumo(d, base, ["AEROPORTO", "ANO"]).reset_index()
    pb = resumo(d, base, "ANO").reset_index().assign(AEROPORTO="BRASIL")
    pa = pd.concat([pb, pa])[["AEROPORTO", "ANO", "n_comentarios", "escore_medio", "pct_neg", "pct_pos", "sat_media"]]
    pa["pct_critico"] = 100 - pa["pct_pos"]
    pa.to_csv(SAIDA / "sentimento_por_ano.csv", index=False)
    print(pb.round(3).to_string())
    ce = {}
    for ano, g in pa[pa.AEROPORTO != "BRASIL"].groupby("ANO"):
        r_, p_ = spearmanr(g.escore_medio, g.sat_media)
        ce[int(ano)] = {"rho": r_, "p": p_, "n": len(g)}
    cel = pa[pa.AEROPORTO != "BRASIL"]
    r_, p_ = spearmanr(cel.escore_medio, cel.sat_media)
    res["correlacao_ecologica_por_ano"] = ce
    res["correlacao_ecologica_aeroporto_ano"] = {"rho": r_, "p": p_, "n": len(cel)}
    print(json.dumps({k: res[k] for k in ["correlacao_sat", "sat_media_por_classe", "correlacao_ecologica_20",
                                          "correlacao_ecologica_por_ano", "correlacao_ecologica_aeroporto_ano"]}, indent=1, default=float))
    (SAIDA / "sentimento_validacao.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))

    # ---- LaTeX: validação
    lab = {"xlmr": "XLM-R (Cardiff, multilíngue)", "bertweetpt": "BERTweet-PT (pysentimiento)"}
    ln = []
    for k in MODELOS:
        m = res[k]
        ln.append(f"{lab[k]} & {fmt(m['acuracia'])} & {fmt(m['f1_macro'])} & {fmt(m['f1_por_classe']['negativo'])} & "
                  f"{fmt(m['f1_por_classe']['neutro'])} & {fmt(m['f1_por_classe']['positivo'])} & {fmt(m['kappa'])} & "
                  f"{fmt(m['bin_acuracia'])} & {fmt(m['bin_f1_macro'])} \\\\")
    dm = res["dist_manual"]
    (TEX / "nlp_sentimento_validacao.tex").write_text(r"""\begin{table}[htbp]
\centering
\caption{Validação dos classificadores de sentimento na amostra rotulada manualmente ($n=200$ comentários de 2023; """ +
        f"{dm.get('negativo',0)} negativos, {dm.get('neutro',0)} neutros, {dm.get('positivo',0)} positivos" +
        r"""). A rubrica trata sugestões de melhoria como negativas. ``Binário'': positivo vs.\ não positivo.}
\label{tab:nlp-sentimento-validacao}
\small
\begin{tabular}{lcccccccc}
\toprule
 & & & \multicolumn{3}{c}{F1 por classe} & & \multicolumn{2}{c}{Binário} \\
\cmidrule(lr){4-6}\cmidrule(lr){8-9}
Modelo & Acurácia & F1 macro & Neg. & Neu. & Pos. & $\kappa$ & Acurácia & F1 macro \\
\midrule
""" + "\n".join(ln) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # ---- LaTeX: satisfação média por classe predita (primário)
    sm = s.groupby("xlmr_classe")["sat"].agg(["mean", "std", "count"]).reindex(CLASSES)
    c1, c2 = corr["xlmr_validos"], corr["xlmr_substantivos"]
    ln = [f"{c.capitalize()} & {int(r['count']):,}".replace(",", ".") + f" & {fmt(r['mean'])} & {fmt(r['std'])} \\\\" for c, r in sm.iterrows()]
    (TEX / "nlp_sentimento_satisfacao.tex").write_text(r"""\begin{table}[htbp]
\centering
\caption{Satisfação geral (1--5) segundo o sentimento previsto (XLM-R) do comentário (""" + ANOS_TEX + r"""). Correlação de Spearman entre o escore contínuo $P(\text{pos})-P(\text{neg})$ e a satisfação geral: """ +
        f"$\\rho={fmt(c1['rho'])}$ (IC 95\\% bootstrap [{fmt(c1['ic95'][0])}; {fmt(c1['ic95'][1])}], $n={c1['n']:_}$) nos comentários válidos e $\\rho={fmt(c2['rho'])}$ ([{fmt(c2['ic95'][0])}; {fmt(c2['ic95'][1])}]) nos substantivos.".replace("_", ".") +
        r"""}
\label{tab:nlp-sentimento-satisfacao}
\begin{tabular}{lrcc}
\toprule
Sentimento previsto & $N$ & Satisfação média & Desvio-padrão \\
\midrule
""" + "\n".join(ln) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # ---- Figura
    estilo()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ct = pd.crosstab(s["sat"].astype(int), s["xlmr_classe"], normalize="index").reindex(columns=CLASSES) * 100
    cores = {"negativo": "#d6604d", "neutro": "#bababa", "positivo": "#4393c3"}
    left = np.zeros(len(ct))
    for c in CLASSES:
        axs[0].barh(ct.index.astype(str), ct[c], left=left, color=cores[c], label=c.capitalize())
        left += ct[c].values
    n_sat = s["sat"].astype(int).value_counts().sort_index()
    axs[0].set_yticks(range(len(ct)), [f"{i} (n={n_sat[i]:,})".replace(",", ".") for i in ct.index])
    axs[0].set_xlabel("% dos comentários válidos"); axs[0].set_ylabel("Satisfação geral")
    axs[0].set_title("(a) Sentimento previsto por nível de satisfação")
    axs[0].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    axs[0].grid(axis="y", visible=False)
    axs[1].scatter(ag.escore_medio, ag.sat_media, color="#2166ac", s=18)
    for ap, r in ag.iterrows():
        axs[1].annotate(ap, (r.escore_medio, r.sat_media), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
    axs[1].set_xlabel("Escore médio de sentimento dos comentários")
    axs[1].set_ylabel("Satisfação geral média (todos)")
    axs[1].set_title(f"(b) Por aeroporto ($\\rho$ de Spearman = {rho_e:.2f})".replace(".", ","))
    fig.tight_layout()
    fig.savefig(FIG / "nlp_sentimento_satisfacao.pdf")


if __name__ == "__main__":
    main()
