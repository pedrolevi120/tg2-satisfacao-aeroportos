"""
Síntese dos experimentos de PREVISÃO (E1–E4) numa tabela e numa figura únicas.

Todas as linhas são comparadas, de forma pareada (mesmas 1.000 reamostragens do
teste, semente 42), com o XGBoost atual do pipeline (MAE lido de predicoes_20). Inclui as médias
simples (ensembles) dos três boostings atuais e dos três ajustados no E1.
Faixa de irrelevância prática: |ΔMAE| ≤ 0,003 ponto (critério combinado com o
Pedro); a variação do MAE só por trocar a semente/reamostrar o treino (E5) é
mostrada como régua.

Saída: resultados/sintese_previsao.csv, figuras/sintese_ganhos_previsao.png
"""
import json

import numpy as np
import pandas as pd

from exp_comum import (RES, carregar, comparar_pareado, preds_atuais, estilo_slide, salvar, fmt_n,
                       AZUL, LARANJA, CINZA, TINTA2, CINZA_CLARO)


def br(x, nd=3, sinal=False):
    return (f"{x:+.{nd}f}" if sinal else f"{x:.{nd}f}").replace(".", ",")


def main():
    d = carregar()
    y, te = d["y"], d["te"]
    P = preds_atuais()
    yt = y[te]
    ref = P.loc[te, "pred_xgb"].values
    linhas = []

    def add(grupo, rotulo, p, **kw):
        r = comparar_pareado(yt, ref, p, **kw)
        linhas.append({"grupo": grupo, "variante": rotulo, **r})

    add("Atual (TG)", "XGBoost atual (referência)", ref)
    add("Atual (TG)", "LightGBM atual", P.loc[te, "pred_lgbm"].values)
    add("Atual (TG)", "CatBoost atual", P.loc[te, "pred_cat"].values)
    add("Atual (TG)", "Random Forest atual", P.loc[te, "pred_rf"].values)
    add("Atual (TG)", "Ridge atual", P.loc[te, "pred_ridge"].values)
    media_atual = P.loc[te, ["pred_xgb", "pred_lgbm", "pred_cat"]].mean(axis=1).values
    add("E1 ajuste", "Média XGB+LGBM+Cat atuais", media_atual)
    aj = {}
    for k, nome in [("xgb", "XGBoost ajustado"), ("lgbm", "LightGBM ajustado"), ("cat", "CatBoost ajustado")]:
        f = RES / f"e1_pred_test_{k}.parquet"
        if f.exists():
            aj[k] = pd.read_parquet(f)["pred"].values
            add("E1 ajuste", nome, aj[k])
    if len(aj) == 3:
        add("E1 ajuste", "Média dos 3 ajustados", np.mean(list(aj.values()), axis=0))
    for k, nome in [("ebm_gam", "EBM só efeitos principais"), ("ebm_ga2m", "EBM + 20 interações")]:
        f = RES / f"e2_pred_test_{k}.parquet"
        if f.exists():
            add("E2 EBM", nome, pd.read_parquet(f)["pred"].values)
    f = RES / "e3_pred_test.parquet"
    if f.exists():
        E3 = pd.read_parquet(f)
        cortes = json.loads((RES / "e3_cortes.json").read_text())
        c = np.sort(cortes["regressao"]["cortes"])
        add("E3 ordinal", "Regressão + cortes p/ QWK", ref, qwk_new=1 + np.searchsorted(c, ref, side="right"))
        add("E3 ordinal", "Multiclasse, valor esperado", E3["b_multiclasse_valor_esperado"].values)
        add("E3 ordinal", "Multiclasse, mediana (inteira)", E3["b_multiclasse_mediana"].values)
        add("E3 ordinal", "Ordinal cumulativo, valor esperado", E3["c_ordinal_cumulativo_EV"].values)
    f = RES / "e4_perfil_metricas.csv"
    if f.exists():
        M = pd.read_csv(f)
        for _, r in M[M.variante != "V0_atual_refit"].iterrows():
            linhas.append({"grupo": "E4 perfil", "variante": {"V1_perfil": "+ perfil/viagem faltante",
                                                               "V2_perfil_operacional": "+ perfil + operacionais"}[r.variante],
                           **{k: r[k] for k in r.index if k not in ("variante", "n_features")}})
    S = pd.DataFrame(linhas)
    S.to_csv(RES / "sintese_previsao.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["grupo", "variante", "MAE", "dMAE", "dMAE_lo", "dMAE_hi", "R2", "dR2", "dR2_lo", "dR2_hi",
             "QWK", "dQWK", "dQWK_lo", "dQWK_hi"]].round(4).to_string())

    # ---------- figura: ΔMAE vs XGBoost atual ----------
    plt = estilo_slide()
    F = S[S.variante != "XGBoost atual (referência)"].reset_index(drop=True)
    F = F[~F.variante.str.contains("mediana|cortes")]  # inteira / só muda o QWK (ver RELATORIO)
    F = F.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(F) + 1.8))
    ax.set_axisbelow(True)
    ax.axvspan(-0.003, 0.003, color="#eef4fc", zorder=0)
    ax.axvline(0, color=CINZA, lw=1.2)
    yy = np.arange(len(F))
    cores = [AZUL if g.startswith("E") else CINZA for g in F.grupo]
    ax.hlines(yy, F.dMAE_lo, F.dMAE_hi, color=cores, lw=2.2)
    ax.scatter(F.dMAE, yy, color=cores, s=60, zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{v}" for v in F.variante])
    ax.grid(axis="y", visible=False)
    # rótulos de grupo à direita
    for g in F.grupo.unique():
        ii = np.where(F.grupo == g)[0]
        ax.text(1.01, (ii.mean() + 0.5) / len(F), g, transform=ax.transAxes, fontsize=11, color=TINTA2,
                va="center", ha="left", fontweight="bold")
    ax.set_xlabel("ΔMAE no teste vs. XGBoost atual (pontos da escala 1–5; < 0 = melhor), IC 95%")
    best = F.loc[F.dMAE.idxmin()]
    ganho = -best.dMAE
    ax.set_title((f"Nenhuma alternativa muda o quadro: o maior ganho é de {br(ganho)} ponto ({best.variante})"
                  if ganho <= 0.003 else f"Maior ganho: {br(ganho)} ponto de MAE ({best.variante})")
                 if ganho > 0 else "Nenhuma alternativa reduz o MAE do XGBoost atual")
    ax.set_ylim(-1.3, len(F) - 0.5)
    ax.text(0, -0.95, "faixa ±0,003 (irrelevante)", ha="center", va="center", fontsize=10, color=AZUL)
    o = S[S.variante == "Ordinal cumulativo, valor esperado"]
    nota_ord = (f"Ordinal cumulativo (valor esperado): ΔMAE {br(o.dMAE.iloc[0], 4, True)}, ΔR² {br(o.dR2.iloc[0], 4, True)}, "
                f"ΔQWK {br(o.dQWK.iloc[0], 4, True)}." if len(o) else "")
    fig.text(0.01, -0.01, f"Diferença pareada no mesmo teste (série 2021–2026, n = {fmt_n(te.sum())}; 1.000 reamostragens bootstrap). "
             "Cinza: modelos já no TG; azul: experimentos novos.\n" + nota_ord,
             fontsize=9.5, color=TINTA2, ha="left", va="top")
    fig.tight_layout()
    salvar(fig, "sintese_ganhos_previsao.png")


if __name__ == "__main__":
    main()
