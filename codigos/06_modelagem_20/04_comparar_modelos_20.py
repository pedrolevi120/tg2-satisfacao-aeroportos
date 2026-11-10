"""
T5/modelos20 — Tarefa 3: comparação dos 8 modelos do MVP + CatBoost nos 20 aeroportos.

Especificação principal (Tarefa 1): um modelo por ramo do questionário (PROCESSO),
ausentes como NaN (nativo em árvores/boosting; média + indicador nos demais).
Split 70/15/15 estratificado por aeroporto × target (03_base_features_20.py).
Seleção pelo MAE de validação; reporte no teste com IC 95% bootstrap (1.000
reamostragens) e diferença pareada em relação ao selecionado.
Série 2021T1–2026T2 (08/11/2026): mesma especificação sobre a série inteira; a
diferença pareada é reamostrada em blocos de 100 (com ~69 mil linhas de teste a
matriz 1.000 × n de uma vez ocuparia ~0,5 GB), com a mesma sequência de índices.

Saída: codigos/06_modelagem_20/resultados/comparacao_modelos_20.csv, predicoes_20.parquet
       codigos/06_modelagem_20/models/<modelo>_20.joblib
       texto/_paralelo/tex/modelos20_comparacao.tex
       midias/figuras/modelos20_comparacao.pdf
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comum import (SPLITS, FIG_DIR, TEX_DIR, OUT_DIR, avaliar, bootstrap_ic, bootstrap_medias, fmt, estilo)
from modelos import NOMES, ORDEM, PorRamo

warnings.filterwarnings("ignore")
MODELS = OUT_DIR.parent / "models"
MODELS.mkdir(exist_ok=True)


def carregar():
    X = pd.read_parquet(SPLITS / "X.parquet")
    meta = pd.read_parquet(SPLITS / "meta.parquet")
    return X, meta


def main():
    estilo()
    X, meta = carregar()
    s = meta["split"].values
    tr, va, te = s == "train", s == "val", s == "test"
    y = meta["y"].values
    ramo = meta["PROCESSO"].values
    print(f"  X={X.shape}; treino={tr.sum()}, val={va.sum()}, teste={te.sum()}")

    linhas, preds, boots = [], {}, {}
    for nome in ORDEM:
        t0 = time.time()
        m = PorRamo(nome).fit(X[tr], y[tr], ramo[tr], X[va], y[va], ramo[va])
        tfit = time.time() - t0
        pv, pt = m.predict(X[va], ramo[va]), m.predict(X[te], ramo[te])
        preds[nome] = (pv, pt)
        mv = avaliar(y[va], pv)
        ic, amostras = bootstrap_ic(y[te], pt)
        boots[nome] = pt
        linhas.append({"modelo": nome, "nome": NOMES[nome], "tempo_s": tfit,
                       **{f"{k}_val": v for k, v in mv.items()},
                       **{f"{k}_test": v[0] for k, v in ic.items()},
                       **{f"{k}_test_lo": v[1] for k, v in ic.items()},
                       **{f"{k}_test_hi": v[2] for k, v in ic.items()}})
        joblib.dump(m, MODELS / f"{nome}_20.joblib")
        print(f"  {nome:8s} [{tfit:6.1f}s] val MAE={mv['MAE']:.4f} R2={mv['R2']:.4f} QWK={mv['QWK']:.4f} | "
              f"test MAE={ic['MAE'][0]:.4f} [{ic['MAE'][1]:.4f};{ic['MAE'][2]:.4f}] "
              f"R2={ic['R2'][0]:.4f} [{ic['R2'][1]:.4f};{ic['R2'][2]:.4f}]", flush=True)

    df = pd.DataFrame(linhas)
    melhor = df.loc[df["MAE_val"].idxmin(), "modelo"]
    print(f"\n  Selecionado em validação (MAE): {melhor}")
    # Diferença pareada no teste: MAE(modelo) - MAE(selecionado), em blocos (memória)
    yt = y[te]
    eb = np.abs(yt - preds[melhor][1])
    for i, r in df.iterrows():
        d = np.abs(yt - preds[r.modelo][1]) - eb
        dd = bootstrap_medias(d, n_boot=1000, seed=42)
        df.loc[i, "dMAE_test"] = d.mean()
        df.loc[i, "dMAE_test_lo"] = np.quantile(dd, 0.025)
        df.loc[i, "dMAE_test_hi"] = np.quantile(dd, 0.975)
    df["selecionado"] = df["modelo"] == melhor
    df.to_csv(OUT_DIR / "comparacao_modelos_20.csv", index=False)
    (OUT_DIR / "modelo_escolhido_20.json").write_text(json.dumps({"modelo": melhor, "criterio": "MAE_val"}))

    P = meta[["AEROPORTO", "PROCESSO", "REGIAO", "ANO", "PERIODO", "split", "y"]].copy()
    for nome, (pv, pt) in preds.items():
        P[f"pred_{nome}"] = np.nan
        P.loc[va, f"pred_{nome}"] = pv
        P.loc[te, f"pred_{nome}"] = pt
    P.to_parquet(OUT_DIR / "predicoes_20.parquet")
    # métricas do selecionado no teste, por ano (série 2021–2026)
    Pt = P.loc[te]
    por_ano = pd.DataFrame([{"ANO": int(a), "n_teste": int(len(g)), "media_y": float(g.y.mean()),
                             **avaliar(g.y, g[f"pred_{melhor}"])} for a, g in Pt.groupby("ANO")])
    por_ano.to_csv(OUT_DIR / "metricas_por_ano_20.csv", index=False)
    print(por_ano.round(4).to_string())
    print(df[["modelo", "MAE_val", "R2_val", "QWK_val", "MAE_test", "R2_test", "QWK_test",
              "dMAE_test", "dMAE_test_lo", "dMAE_test_hi"]].round(4).to_string())

    # ---------- Tabela LaTeX ----------
    best = {k: (df[f"{k}_val"].min() if k in ("MAE", "RMSE") else df[f"{k}_val"].max()) for k in ["MAE", "RMSE", "R2", "QWK"]}
    def b(v, k):
        s = fmt(v)
        return f"\\textbf{{{s}}}" if np.isclose(v, best[k]) else s
    n_te, n_va = int(te.sum()), int(va.sum())
    nva, nte = f"{n_va:,}".replace(",", "."), f"{n_te:,}".replace(",", ".")
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{Comparação dos modelos nos 20 aeroportos, série 2021--2026 (especificação por ramo do questionário). "
         f"Seleção em validação ($n={nva}$, melhor valor por métrica em negrito) e reporte em teste "
         f"($n={nte}$) com IC 95\\% \\textit{{bootstrap}} (1.000 reamostragens). $\\Delta$MAE: diferença pareada "
         f"em relação ao modelo selecionado ({NOMES[melhor]}).}}",
         "\\label{tab:modelos20-comparacao}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lccccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{Validação} & \\multicolumn{4}{c}{Teste [IC 95\\%]} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}",
         "Modelo & MAE & R\\textsuperscript{2} & QWK & MAE & R\\textsuperscript{2} & QWK & $\\Delta$MAE \\\\", "\\midrule"]
    for _, r in df.iterrows():
        ic = lambda k: f"{fmt(r[f'{k}_test'])} [{fmt(r[f'{k}_test_lo'])}; {fmt(r[f'{k}_test_hi'])}]"
        dm = "--" if r.selecionado else f"{fmt(r.dMAE_test)} [{fmt(r.dMAE_test_lo)}; {fmt(r.dMAE_test_hi)}]"
        L.append(f"{r.nome} & {b(r.MAE_val,'MAE')} & {b(r.R2_val,'R2')} & {b(r.QWK_val,'QWK')} & "
                 f"{ic('MAE')} & {ic('R2')} & {ic('QWK')} & {dm} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_comparacao.tex").write_text("\n".join(L), encoding="utf-8")

    # ---------- Figura: MAE e R² no teste com IC ----------
    d = df[df.modelo != "baseline"].sort_values("MAE_test", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, k, lab in [(axes[0], "MAE", "MAE no teste (menor é melhor)"),
                       (axes[1], "R2", "R² no teste (maior é melhor)")]:
        cores = ["#d62728" if m == melhor else "#1f77b4" for m in d.modelo]
        ax.errorbar(d[f"{k}_test"], [n.replace("$k{=}10$", "k=10") for n in d.nome],
                    xerr=[d[f"{k}_test"] - d[f"{k}_test_lo"], d[f"{k}_test_hi"] - d[f"{k}_test"]],
                    fmt="none", ecolor="gray", capsize=3)
        ax.scatter(d[f"{k}_test"], range(len(d)), c=cores, zorder=3)
        ax.scatter(d[f"{k}_val"], range(len(d)), marker="x", c="black", zorder=3, label="validação")
        ax.set_xlabel(lab)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Modelos nos 20 aeroportos, 2021–2026 — teste com IC 95% bootstrap (vermelho: selecionado em validação, {NOMES[melhor]})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_comparacao.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
