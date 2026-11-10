"""
SO2.5 — Análise SHAP no modelo vencedor (XGBoost) — global + local por aeroporto.

Entrada: codigos/04_modelos_ml/models/xgboost_n500_d6.joblib
         dados/processados/splits/X_*.parquet
         dados/processados/base_unificada_mvp.parquet  (pra recuperar AEROPORTO)
Saída:   midias/figuras/shap_global.pdf
         midias/figuras/shap_local_{SBGR,SBSP,SBKP}.pdf
         dados/processados/shap_values_test.parquet (pra usar na camada LLM)
         codigos/05_interpretacao/ranking_prioridades_por_aeroporto.csv
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "dados" / "processados" / "splits"
MODEL_PATH = ROOT / "codigos" / "04_modelos_ml" / "models" / "xgboost_n500_d6.joblib"
FIG_DIR = ROOT / "midias" / "figuras"
PROCESSADOS = ROOT / "dados" / "processados"
RANKING_OUT = Path(__file__).resolve().parent / "ranking_prioridades_por_aeroporto.csv"

NOME_AERO = {"SBGR": "Guarulhos (GRU)", "SBSP": "Congonhas (CGH)", "SBKP": "Viracopos (VCP)"}


def main():
    print("=== Carregando modelo XGBoost e dados ===")
    model = joblib.load(MODEL_PATH)
    Xte = pd.read_parquet(SPLITS / "X_test.parquet")
    yte = pd.read_parquet(SPLITS / "y_test.parquet")["y"].values
    print(f"  teste: {Xte.shape}")

    # Recuperar AEROPORTO de cada linha do teste a partir dos dummies
    aero_cols = [c for c in Xte.columns if c.startswith("AEROPORTO_")]
    aero_array = np.array([
        c.replace("AEROPORTO_", "") for c in aero_cols
    ])
    aero_test = aero_array[Xte[aero_cols].values.argmax(axis=1)]
    print(f"\n  Distribuição aeroporto no teste:")
    for a, n in zip(*np.unique(aero_test, return_counts=True)):
        print(f"    {a}: {n}")

    print("\n=== Calculando valores SHAP no teste ===")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    print(f"  shape SHAP: {sv.shape}")

    # Salvar SHAP values pra usar depois na camada LLM
    sv_df = pd.DataFrame(sv, columns=Xte.columns, index=Xte.index)
    sv_df["AEROPORTO"] = aero_test
    for col in sv_df.select_dtypes(include="object").columns:
        sv_df[col] = sv_df[col].astype(str)
    sv_df.to_parquet(PROCESSADOS / "shap_values_test.parquet")
    print(f"  → shap_values_test.parquet ({(PROCESSADOS/'shap_values_test.parquet').stat().st_size/1024:.1f} KB)")

    # === Figura 1: SHAP global (summary plot bar) ===
    print("\n=== Figura: SHAP global ===")
    plt.figure(figsize=(9, 8))
    shap.summary_plot(sv, Xte, plot_type="bar", max_display=20, show=False)
    plt.title("Importância global das features (média do |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global.pdf", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  → shap_global.pdf")

    # === Figura 2: SHAP global beeswarm (densidade) ===
    plt.figure(figsize=(10, 9))
    shap.summary_plot(sv, Xte, max_display=20, show=False)
    plt.title("Distribuição dos valores SHAP por feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.pdf", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  → shap_beeswarm.pdf")

    # === Rankings por aeroporto ===
    print("\n=== Rankings por aeroporto ===")
    ranking_rows = []
    for aero in ["SBGR", "SBSP", "SBKP"]:
        mask = aero_test == aero
        if mask.sum() == 0:
            continue
        # importância local: |SHAP| médio das amostras desse aeroporto
        imp = np.abs(sv[mask]).mean(axis=0)
        top = pd.Series(imp, index=Xte.columns).sort_values(ascending=False).head(15)
        for rank, (feat, val) in enumerate(top.items(), 1):
            ranking_rows.append({
                "aeroporto": aero, "rank": rank,
                "feature": feat, "shap_abs_medio": float(val),
            })

        # Figura: ranking por aeroporto
        fig, ax = plt.subplots(figsize=(8, 7))
        top.sort_values().plot.barh(ax=ax, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.set_title(f"Top 15 prioridades — {NOME_AERO[aero]} (média $|$SHAP$|$, $n={mask.sum()}$)")
        ax.set_xlabel("Importância média ($|$SHAP$|$)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_local_{aero}.pdf", bbox_inches="tight", dpi=200)
        plt.close()
        print(f"  → shap_local_{aero}.pdf")

        print(f"\n  Top 5 — {NOME_AERO[aero]}:")
        for f, v in top.head(5).items():
            print(f"    {v:+.4f}  {f[:60]}")

    pd.DataFrame(ranking_rows).to_csv(RANKING_OUT, index=False)
    print(f"\n  → {RANKING_OUT.relative_to(ROOT)} ({len(ranking_rows)} linhas)")


if __name__ == "__main__":
    main()
