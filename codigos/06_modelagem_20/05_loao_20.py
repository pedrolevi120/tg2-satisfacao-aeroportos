"""
T5/modelos20 — Tarefa 3b: leave-one-airport-out (20 folds) para o modelo
selecionado (XGBoost) e o Ridge, na especificação por ramo do questionário.

Fold A: treino = linhas de treino dos outros 19 aeroportos; early stopping nas
linhas de validação dos outros 19; avaliação nas linhas de teste de A (mesmas
linhas do reporte "dentro da distribuição", comparação pareada) e em todas as
linhas de A. As dummies de AEROPORTO são removidas (o aeroporto de teste é
desconhecido); features de contexto (T2), se existirem, permanecem — são o único
canal pelo qual o modelo pode "conhecer" um aeroporto novo.
Série 2021T1–2026T2 (08/11/2026): mesmo protocolo sobre a série inteira; as dummies
de PERIODO permanecem (o período do aeroporto excluído é conhecido).

Saída: resultados/loao_20.csv, texto/_paralelo/tex/modelos20_loao.tex,
       midias/figuras/modelos20_loao.pdf
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comum import SPLITS, FIG_DIR, TEX_DIR, OUT_DIR, avaliar, fmt, estilo
from modelos import NOMES, PorRamo

warnings.filterwarnings("ignore")
NOME_AERO = {"SBBE": "Belém", "SBBR": "Brasília", "SBCF": "Confins", "SBCT": "Curitiba",
             "SBCY": "Cuiabá", "SBEG": "Manaus", "SBFL": "Florianópolis", "SBFZ": "Fortaleza",
             "SBGL": "Galeão", "SBGO": "Goiânia", "SBGR": "Guarulhos", "SBKP": "Viracopos",
             "SBMO": "Maceió", "SBPA": "Porto Alegre", "SBRF": "Recife", "SBRJ": "Santos Dumont",
             "SBSG": "Natal", "SBSP": "Congonhas", "SBSV": "Salvador", "SBVT": "Vitória"}


def main():
    estilo()
    X = pd.read_parquet(SPLITS / "X.parquet")
    meta = pd.read_parquet(SPLITS / "meta.parquet")
    P = pd.read_parquet(OUT_DIR / "predicoes_20.parquet")
    melhor = json.loads((OUT_DIR / "modelo_escolhido_20.json").read_text())["modelo"]
    Xl = X.drop(columns=[c for c in X.columns if c.startswith("AEROPORTO_")])
    y, s, ramo, aero = meta.y.values, meta.split.values, meta.PROCESSO.values, meta.AEROPORTO.values
    tem_ctx = any(c.startswith("CTX_") for c in X.columns)
    linhas = []
    for A in sorted(np.unique(aero)):
        outros = aero != A
        tr, va = outros & (s == "train"), outros & (s == "val")
        te_A, all_A = (aero == A) & (s == "test"), aero == A
        for nome in [melhor, "ridge"]:
            m = PorRamo(nome).fit(Xl[tr], y[tr], ramo[tr], Xl[va], y[va], ramo[va])
            p_te = m.predict(Xl[te_A], ramo[te_A])
            p_all = m.predict(Xl[all_A], ramo[all_A])
            loao = avaliar(y[te_A], p_te)
            dentro = avaliar(y[te_A], P.loc[te_A, f"pred_{nome}"].values)
            base = avaliar(y[te_A], np.full(te_A.sum(), 4.0))
            linhas.append({"aeroporto": A, "modelo": nome, "n_teste": int(te_A.sum()),
                           "n_total": int(all_A.sum()),
                           **{f"{k}_loao": v for k, v in loao.items()},
                           **{f"{k}_dentro": v for k, v in dentro.items()},
                           "MAE_loao_todas": avaliar(y[all_A], p_all)["MAE"],
                           "vies_loao": float((p_te - y[te_A]).mean()),
                           "vies_dentro": float((P.loc[te_A, f"pred_{nome}"].values - y[te_A]).mean()),
                           "MAE_baseline4": base["MAE"], "media_y": float(y[te_A].mean())})
            r = linhas[-1]
            print(f"  {A} {nome:6s} MAE loao={r['MAE_loao']:.4f} dentro={r['MAE_dentro']:.4f} "
                  f"R2 loao={r['R2_loao']:.3f} dentro={r['R2_dentro']:.3f} viés={r['vies_loao']:+.3f}", flush=True)
    df = pd.DataFrame(linhas)
    df["dMAE"] = df.MAE_loao - df.MAE_dentro
    df.to_csv(OUT_DIR / "loao_20.csv", index=False)
    resumo = df.groupby("modelo")[["MAE_loao", "MAE_dentro", "dMAE", "R2_loao", "R2_dentro",
                                   "QWK_loao", "QWK_dentro"]].agg(["mean", "median", "min", "max"])
    print(resumo.round(4).T.to_string())
    resumo.to_csv(OUT_DIR / "loao_20_resumo.csv")

    # ---------- Tabela LaTeX ----------
    w = df.pivot(index="aeroporto", columns="modelo")
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
         f"\\caption{{\\textit{{Leave-one-airport-out}} (20 \\textit{{folds}}): MAE e R\\textsuperscript{{2}} nas linhas de teste de cada aeroporto "
         f"quando ele é excluído do treino (LOAO), comparados ao modelo treinado com todos os aeroportos (dentro), série 2021--2026. "
         f"{NOMES[melhor]} e Ridge, especificação por ramo do questionário"
         + (", com variáveis de contexto ANAC/IBGE" if tem_ctx else ", sem variáveis de contexto") + ".}",
         "\\label{tab:modelos20-loao}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrcccccccc}", "\\toprule",
         f" & & \\multicolumn{{4}}{{c}}{{{NOMES[melhor]}}} & \\multicolumn{{4}}{{c}}{{Ridge}} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
         "Aeroporto & $n$ & MAE\\textsubscript{LOAO} & MAE\\textsubscript{dentro} & R\\textsuperscript{2}\\textsubscript{LOAO} & R\\textsuperscript{2}\\textsubscript{dentro} & "
         "MAE\\textsubscript{LOAO} & MAE\\textsubscript{dentro} & R\\textsuperscript{2}\\textsubscript{LOAO} & R\\textsuperscript{2}\\textsubscript{dentro} \\\\", "\\midrule"]
    for A in w.index:
        c = lambda k, mo: fmt(w.loc[A, (k, mo)])
        L.append(f"{NOME_AERO[A]} ({A}) & {int(w.loc[A, ('n_teste', melhor)])} & {c('MAE_loao', melhor)} & {c('MAE_dentro', melhor)} & "
                 f"{c('R2_loao', melhor)} & {c('R2_dentro', melhor)} & {c('MAE_loao', 'ridge')} & {c('MAE_dentro', 'ridge')} & "
                 f"{c('R2_loao', 'ridge')} & {c('R2_dentro', 'ridge')} \\\\")
    L.append("\\midrule")
    mm = df.groupby("modelo").mean(numeric_only=True)
    L.append(f"Média & -- & {fmt(mm.loc[melhor,'MAE_loao'])} & {fmt(mm.loc[melhor,'MAE_dentro'])} & {fmt(mm.loc[melhor,'R2_loao'])} & "
             f"{fmt(mm.loc[melhor,'R2_dentro'])} & {fmt(mm.loc['ridge','MAE_loao'])} & {fmt(mm.loc['ridge','MAE_dentro'])} & "
             f"{fmt(mm.loc['ridge','R2_loao'])} & {fmt(mm.loc['ridge','R2_dentro'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_loao.tex").write_text("\n".join(L), encoding="utf-8")

    # ---------- Figura ----------
    d = w.sort_values(("MAE_dentro", melhor))
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = np.arange(len(d))
    ax.scatter(d[("MAE_dentro", melhor)], yy, marker="o", c="#1f77b4", label=f"{NOMES[melhor]} — dentro", zorder=3)
    ax.scatter(d[("MAE_loao", melhor)], yy, marker="o", facecolors="none", edgecolors="#1f77b4", label=f"{NOMES[melhor]} — LOAO", zorder=3)
    ax.scatter(d[("MAE_dentro", "ridge")], yy, marker="s", c="#ff7f0e", label="Ridge — dentro", zorder=3)
    ax.scatter(d[("MAE_loao", "ridge")], yy, marker="s", facecolors="none", edgecolors="#ff7f0e", label="Ridge — LOAO", zorder=3)
    for i, A in enumerate(d.index):
        ax.plot([d.loc[A, ("MAE_dentro", melhor)], d.loc[A, ("MAE_loao", melhor)]], [i, i], c="#1f77b4", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{NOME_AERO[a]} ({a})" for a in d.index])
    ax.set_xlabel("MAE nas linhas de teste do aeroporto")
    ax.set_title("Generalização para aeroporto não visto (leave-one-airport-out), 2021–2026")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_loao.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
