"""
SO2.4 — Consolida resultados dos 8 modelos, gera tabela + figura comparativa.

Entrada: codigos/03_modelos_baseline/resultados_baselines.json
         codigos/04_modelos_ml/resultados_ensembles.json
Saída:   midias/figuras/comparacao_modelos.pdf
         codigos/04_modelos_ml/tabela_comparativa.csv
         codigos/04_modelos_ml/tabela_comparativa_latex.tex
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "codigos" / "03_modelos_baseline" / "resultados_baselines.json"
ENSEMBLES = ROOT / "codigos" / "04_modelos_ml" / "resultados_ensembles.json"
FIG_DIR = ROOT / "midias" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "codigos" / "04_modelos_ml" / "tabela_comparativa.csv"
OUT_TEX = ROOT / "codigos" / "04_modelos_ml" / "tabela_comparativa_latex.tex"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

# Mapeamento para nomes legíveis
NOMES = {
    "Baseline-chute-4": "Baseline (chuta 4)",
    "ols_ridge": "Ridge (linear)",
    "knn_k10": "KNN ($k=10$)",
    "decision_tree_d10": "Decision Tree",
    "random_forest_n300_d15": "Random Forest",
    "xgboost_n500_d6": "XGBoost",
    "lightgbm_n500_l63": "LightGBM",
    "mlp_128_64": "MLP",
}
ORDEM = ["Baseline-chute-4", "knn_k10", "ols_ridge", "decision_tree_d10",
         "mlp_128_64", "random_forest_n300_d15", "lightgbm_n500_l63",
         "xgboost_n500_d6"]


def main():
    base = json.loads(BASELINES.read_text())
    ens = json.loads(ENSEMBLES.read_text())
    todos = base + ens
    # filtrar entradas que rodaram (têm 'val')
    todos = [r for r in todos if "val" in r]
    # ordenar por ORDEM
    todos.sort(key=lambda r: ORDEM.index(r["modelo"]) if r["modelo"] in ORDEM else 99)

    # Build DataFrame
    rows = []
    for r in todos:
        rows.append({
            "modelo": r["modelo"],
            "nome": NOMES.get(r["modelo"], r["modelo"]),
            "MAE_val": r["val"]["MAE"],
            "RMSE_val": r["val"]["RMSE"],
            "R2_val": r["val"]["R2"],
            "QWK_val": r["val"]["QWK"],
            "MAE_test": r["test"]["MAE"],
            "RMSE_test": r["test"]["RMSE"],
            "R2_test": r["test"]["R2"],
            "QWK_test": r["test"]["QWK"],
            "tempo_s": r["tempo_fit_s"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Tabela salva: {OUT_CSV.relative_to(ROOT)}\n")

    # === Figura: 4 painéis (MAE, RMSE, R², QWK) ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metricas = [("MAE_val", "MAE (menor=melhor)", "#1f77b4"),
                ("RMSE_val", "RMSE (menor=melhor)", "#ff7f0e"),
                ("R2_val", "R² (maior=melhor)", "#2ca02c"),
                ("QWK_val", "QWK (maior=melhor)", "#9467bd")]
    for ax, (col, titulo, cor) in zip(axes.flat, metricas):
        sorted_df = df.sort_values(col, ascending=(col.startswith("MAE") or col.startswith("RMSE")))
        bars = ax.barh(sorted_df["nome"], sorted_df[col],
                       color=cor, edgecolor="black", alpha=0.85)
        # destacar vencedor
        for bar, val in zip(bars, sorted_df[col]):
            if bar.get_y() == bars[0].get_y():
                bar.set_color("#d62728")
                bar.set_alpha(0.95)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel(col.split("_")[0])
        for bar, val in zip(bars, sorted_df[col]):
            ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
    plt.suptitle("Comparação dos 8 modelos — métricas no conjunto de validação",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "comparacao_modelos.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {out.relative_to(ROOT)}\n")

    # === Tabela LaTeX (formato pronto pra colar no main.tex) ===
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparação dos oito modelos na predição da satisfação geral. Métricas no conjunto de validação (2.133 respondentes). Vencedor por métrica em \\textbf{negrito}.}",
        "\\label{tab:resultados-modelos}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Modelo & MAE & RMSE & R\\textsuperscript{2} & QWK & Tempo (s) \\\\",
        "\\midrule",
    ]
    # Identificar vencedores
    best_mae = df["MAE_val"].min()
    best_rmse = df["RMSE_val"].min()
    best_r2 = df["R2_val"].max()
    best_qwk = df["QWK_val"].max()
    for _, r in df.iterrows():
        nome = r["nome"].replace("$k=10$", "$k{=}10$")
        mae = f"\\textbf{{{r['MAE_val']:.3f}}}" if r["MAE_val"] == best_mae else f"{r['MAE_val']:.3f}"
        rmse = f"\\textbf{{{r['RMSE_val']:.3f}}}" if r["RMSE_val"] == best_rmse else f"{r['RMSE_val']:.3f}"
        r2 = f"\\textbf{{{r['R2_val']:.3f}}}" if r["R2_val"] == best_r2 else f"{r['R2_val']:.3f}"
        qwk = f"\\textbf{{{r['QWK_val']:.3f}}}" if r["QWK_val"] == best_qwk else f"{r['QWK_val']:.3f}"
        lines.append(f"{nome} & {mae} & {rmse} & {r2} & {qwk} & {r['tempo_s']:.2f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines))
    print(f"Tabela LaTeX salva: {OUT_TEX.relative_to(ROOT)}\n")

    print("=== Resumo final ===")
    print(df[["nome", "MAE_val", "RMSE_val", "R2_val", "QWK_val", "tempo_s"]].to_string(index=False))
    print(f"\nVencedor por MAE:  {df.loc[df['MAE_val'].idxmin(), 'nome']}")
    print(f"Vencedor por RMSE: {df.loc[df['RMSE_val'].idxmin(), 'nome']}")
    print(f"Vencedor por R²:   {df.loc[df['R2_val'].idxmax(), 'nome']}")
    print(f"Vencedor por QWK:  {df.loc[df['QWK_val'].idxmax(), 'nome']}")


if __name__ == "__main__":
    main()
