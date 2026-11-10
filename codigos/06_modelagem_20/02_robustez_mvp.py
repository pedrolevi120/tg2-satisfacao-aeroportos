"""
T5/modelos20 — Tarefa 1: robustez do tratamento de ausentes estruturais no MVP.

Os atributos Likert ausentes são, em grande parte, estruturais: o ramo do
questionário (PROCESSO = Embarque/Desembarque) define quais blocos são
perguntados, e a coluna PROCESSO havia sido descartada no MVP. Comparam-se:
  (a) atual     : ausente -> -1, sem PROCESSO                (especificação do MVP)
  (b) nan+proc  : ausente -> NaN nativo (XGB/LGBM) + PROCESSO como feature
  (c) separado  : um modelo por PROCESSO (NaN nativo dentro do ramo)
  (c') separado com -1 dentro do ramo (ablação)
  (d) separado por PROCESSO, apenas itens-pai (sem os subitens da lógica de salto)
Achado: além dos ramos, há uma 2ª camada estrutural — subitens (p.ex. SINALIZAÇÃO,
CONFORTO ACÚSTICO) só são perguntados quando o item-pai recebe nota <= 3. Por isso
os rankings também são reportados por DIMENSÃO (pai + subitens; SHAP somados).
Mesmos splits (índices salvos em dados/processados/splits), seed e
hiperparâmetros do MVP (XGBoost n500/d6/lr0.05; LightGBM n500/l63/lr0.05).
Base: série MPor 2021T1–2026T2 filtrada no MVP (02_preprocessamento/01–03); as categóricas
de tempo/regime são PERIODO e REGIME_NO_PERIODO. n_jobs=4 (máquina compartilhada).

Entrada: dados/processados/base_unificada_mvp.parquet, dados/processados/splits/*
Saída:   codigos/06_modelagem_20/resultados/robustez_*.csv|json
         texto/_paralelo/tex/modelos20_robustez*.tex
         midias/figuras/modelos20_robustez_*.pdf
"""
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr

from comum import (ROOT, PROCESSADOS, FIG_DIR, TEX_DIR, OUT_DIR, SEED, TARGET_COL,
                   avaliar, fmt, curto, estilo, tex_escape, hierarquia, grupos_dimensao,
                   shap_por_dimensao)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SPLITS = PROCESSADOS / "splits"

# Reusa a lógica do pré-processamento do MVP (is_likert, listas de colunas)
spec = importlib.util.spec_from_file_location(
    "fe", ROOT / "codigos" / "02_preprocessamento" / "03_features_e_split.py")
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)

META = json.loads((SPLITS / "feature_metadata.json").read_text())
LIKERT = META["likert_cols"]
ATRIBUTOS = [c for c in LIKERT if c != "TERMINAL"]  # TERMINAL é identificador, não atributo


def construir_X(df, na_mode="-1", com_processo=False):
    """Replica 03_features_e_split.py; muda só o tratamento de ausentes/PROCESSO."""
    Xl = df[LIKERT].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    Xl = Xl.fillna(-1).astype(int) if na_mode == "-1" else Xl.astype(float)
    cat_cols = [c for c in META["categoricas_original"] if c in df.columns]
    Xc = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols, drop_first=False)
    Xc.columns = fe.limpar_nomes(Xc.columns)
    Xn = df[META["numericas"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    partes = [Xl, Xc, Xn]
    if com_processo:
        partes.append((df["PROCESSO"] == "Embarque").astype(int).rename("PROCESSO_EMBARQUE"))
    X = pd.concat(partes, axis=1)
    X.columns = X.columns.astype(str)
    return X


def carregar():
    df = pd.read_parquet(PROCESSADOS / "base_unificada_mvp.parquet")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    ok = y.notna()
    df = df.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True).astype(float)
    idx = {s: pd.read_parquet(SPLITS / f"X_{s}.parquet", columns=[]).index for s in ["train", "val", "test"]}
    # Verificação: a variante (a) reproduz exatamente o X salvo do MVP
    Xa = construir_X(df, "-1", False)
    Xsaved = pd.read_parquet(SPLITS / "X_test.parquet")
    assert list(Xa.columns) == list(Xsaved.columns), "colunas divergem do MVP"
    # Só o contexto numérico poderia divergir (se o arquivo de contexto for atualizado depois
    # do split); com a base atual a reprodução é exata.
    difs = [c for c in Xa.columns if not np.allclose(Xa.loc[idx["test"], c].values, Xsaved[c].values)]
    assert all(c in META["numericas"] for c in difs), difs
    print(f"  Colunas de contexto divergentes do X salvo: {difs}")
    ysaved = pd.read_parquet(SPLITS / "y_test.parquet")["y"].values
    assert np.allclose(y.loc[idx["test"]].values, ysaved)
    print("  Variante (a) reproduz X/y do MVP: OK")
    return df, y, idx


def xgb_fit(Xtr, ytr, Xv, yv):
    m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=4, random_state=SEED, early_stopping_rounds=30,
                         verbosity=0)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return m


def lgb_fit(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                          random_state=SEED, verbosity=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return m


FITTERS = {"XGBoost": xgb_fit, "LightGBM": lgb_fit}


def rodar_unico(X, y, idx, modelo):
    Xtr, Xv, Xte = X.loc[idx["train"]], X.loc[idx["val"]], X.loc[idx["test"]]
    m = FITTERS[modelo](Xtr, y.loc[idx["train"]], Xv, y.loc[idx["val"]])
    pv, pt = m.predict(Xv), m.predict(Xte)
    sv = shap.TreeExplainer(m).shap_values(Xte)
    return pv, pt, pd.DataFrame(sv, index=Xte.index, columns=X.columns)


def rodar_separado(X, y, idx, modelo, processo):
    """Um modelo por ramo; predições e SHAP recombinados nas linhas originais."""
    pv = pd.Series(np.nan, index=idx["val"])
    pt = pd.Series(np.nan, index=idx["test"])
    sv_all = pd.DataFrame(0.0, index=idx["test"], columns=X.columns)
    for ramo in ["Embarque", "Desembarque"]:
        r = processo == ramo
        tr = idx["train"][r.loc[idx["train"]].values]
        va = idx["val"][r.loc[idx["val"]].values]
        te = idx["test"][r.loc[idx["test"]].values]
        m = FITTERS[modelo](X.loc[tr], y.loc[tr], X.loc[va], y.loc[va])
        pv.loc[va] = m.predict(X.loc[va])
        pt.loc[te] = m.predict(X.loc[te])
        sv_all.loc[te] = shap.TreeExplainer(m).shap_values(X.loc[te])
    return pv.values, pt.values, sv_all


def ranking_info(sv, idx_te, aero, miss_te, grupos, miss_pai):
    absv = sv.abs()
    att = absv[ATRIBUTOS]
    dim = shap_por_dimensao(sv, grupos).abs()
    a_te = aero.loc[idx_te].values
    return {"feat": absv.mean().sort_values(ascending=False),
            "global": att.mean(),
            "aero": {a: att[a_te == a].mean() for a in AEROS},
            "frac_miss": (att.where(miss_te.values, 0).sum() / att.sum()).fillna(0),
            "dim_global": dim.mean(),
            "dim_aero": {a: dim[a_te == a].mean() for a in AEROS},
            "dim_frac_miss": (dim.where(miss_pai.values, 0).sum() / dim.sum()).fillna(0)}


AEROS = ["SBGR", "SBSP", "SBKP"]


def comparar(info, ref, chave_g, chave_a):
    g = info[chave_g].sort_values(ascending=False)
    gr = ref[chave_g].sort_values(ascending=False)
    row = {"spearman_global": spearmanr(info[chave_g], ref[chave_g]).statistic,
           "overlap_top10": len(set(g.index[:10]) & set(gr.index[:10])),
           "top4_global": sum(x in set(g.index[:4]) for x in gr.index[:4]),
           "top4_ordem_igual": list(g.index[:4]) == list(gr.index[:4])}
    for a in AEROS:
        ga, gra = info[chave_a][a], ref[chave_a][a]
        row[f"spearman_{a}"] = spearmanr(ga, gra).statistic
        row[f"top4_{a}"] = sum(x in set(ga.sort_values(ascending=False).index[:4])
                               for x in gra.sort_values(ascending=False).index[:4])
    return row


def main():
    estilo()
    df, y, idx = carregar()
    aero, processo = df["AEROPORTO"], df["PROCESSO"]
    X_m1 = construir_X(df, "-1", False)
    X_nan_p = construir_X(df, "nan", True)
    X_nan = construir_X(df, "nan", False)
    X_m1_p = construir_X(df, "-1", True)
    # hierarquia detectada no treino
    pais = hierarquia(X_nan.loc[idx["train"]], ATRIBUTOS)
    grupos = grupos_dimensao(ATRIBUTOS, pais)
    print(f"  {len(pais)} subitens com lógica de salto; {len(grupos)} dimensões")
    (OUT_DIR / "hierarquia_mvp.json").write_text(json.dumps(pais, indent=2, ensure_ascii=False))
    X_pais = X_nan.drop(columns=list(pais))
    miss_te = X_nan.loc[idx["test"], ATRIBUTOS].isna()
    miss_pai = pd.DataFrame({d: miss_te[d] for d in grupos})

    variantes = {
        "(a) -1, sem PROCESSO": ("unico", X_m1),
        "(b) NaN + PROCESSO": ("unico", X_nan_p),
        "(c) separado por PROCESSO (NaN)": ("sep", X_nan),
        "(c') separado por PROCESSO (-1)": ("sep", X_m1),
        "(d) separado, só itens-pai (NaN)": ("sep", X_pais),
        "(abl.) -1 + PROCESSO": ("unico", X_m1_p),
    }
    linhas, info, preds = [], {}, {}
    for nome, (tipo, X) in variantes.items():
        for modelo in ["XGBoost", "LightGBM"]:
            f = rodar_unico if tipo == "unico" else (lambda *a: rodar_separado(*a, processo))
            pv, pt, sv = f(X, y, idx, modelo)
            sv = sv.reindex(columns=X_nan_p.columns.union(X.columns), fill_value=0.0)
            mv, mt = avaliar(y.loc[idx["val"]], pv), avaliar(y.loc[idx["test"]], pt)
            preds[(nome, modelo)] = (np.asarray(pv), np.asarray(pt))
            linhas.append({"variante": nome, "modelo": modelo,
                           **{f"{k}_val": v for k, v in mv.items()},
                           **{f"{k}_test": v for k, v in mt.items()}})
            info[(nome, modelo)] = ranking_info(sv, idx["test"], aero, miss_te, grupos, miss_pai)
            print(f"  {nome:34s} {modelo:8s} val MAE={mv['MAE']:.4f} R2={mv['R2']:.4f} | "
                  f"test MAE={mt['MAE']:.4f} R2={mt['R2']:.4f} QWK={mt['QWK']:.4f}")
    met = pd.DataFrame(linhas)
    # Bootstrap pareado (1.000 reamostragens) da diferença de MAE vs. (a), mesmo modelo
    rng = np.random.default_rng(SEED)
    yv, yt = y.loc[idx["val"]].values, y.loc[idx["test"]].values
    bv = rng.integers(0, len(yv), (1000, len(yv)))
    bt = rng.integers(0, len(yt), (1000, len(yt)))
    for i, r in met.iterrows():
        ref = preds[("(a) -1, sem PROCESSO", r.modelo)]
        cur = preds[(r.variante, r.modelo)]
        for nome_s, yy, B, k in [("val", yv, bv, 0), ("test", yt, bt, 1)]:
            d = np.abs(yy - cur[k]) - np.abs(yy - ref[k])
            dd = d[B].mean(axis=1)
            met.loc[i, f"dMAE_{nome_s}"] = d.mean()
            met.loc[i, f"dMAE_{nome_s}_lo"] = np.quantile(dd, 0.025)
            met.loc[i, f"dMAE_{nome_s}_hi"] = np.quantile(dd, 0.975)
    print(met[["variante", "modelo", "dMAE_val", "dMAE_val_lo", "dMAE_val_hi", "dMAE_test", "dMAE_test_lo", "dMAE_test_hi"]].round(4).to_string())
    met.to_csv(OUT_DIR / "robustez_metricas.csv", index=False)

    ref_nome = "(a) -1, sem PROCESSO"
    comp, tops = [], {}
    for (nome, modelo), d in info.items():
        ref = info[(ref_nome, modelo)]
        ra = comparar(d, ref, "global", "aero")
        rd = comparar(d, ref, "dim_global", "dim_aero")
        g = d["global"].sort_values(ascending=False)
        gd = d["dim_global"].sort_values(ascending=False)
        comp.append({"variante": nome, "modelo": modelo,
                     **{f"atr_{k}": v for k, v in ra.items()},
                     **{f"dim_{k}": v for k, v in rd.items()},
                     "frac_shap_ausente_top10_atr": float(d["frac_miss"][g.index[:10]].mean()),
                     "frac_shap_ausente_top10_dim": float(d["dim_frac_miss"][gd.index[:10]].mean())})
        tops[f"{nome} | {modelo}"] = {
            "top10_atributos_global": [(k, float(v), float(d["frac_miss"][k])) for k, v in g.head(10).items()],
            "top10_dimensoes_global": [(k, float(v), float(d["dim_frac_miss"][k])) for k, v in gd.head(10).items()],
            "top10_features_global": [(k, float(v)) for k, v in d["feat"].head(10).items()],
            "top10_atributos_por_aeroporto": {a: [(k, float(v)) for k, v in d["aero"][a].sort_values(ascending=False).head(10).items()] for a in AEROS},
            "top10_dimensoes_por_aeroporto": {a: [(k, float(v)) for k, v in d["dim_aero"][a].sort_values(ascending=False).head(10).items()] for a in AEROS},
        }
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT_DIR / "robustez_rankings.csv", index=False)
    (OUT_DIR / "robustez_tops.json").write_text(json.dumps(tops, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 250)
    print(comp.round(3).to_string())

    # ---------- Tabela LaTeX 1: métricas ----------
    nv, nt = (f"{len(idx[k]):,}".replace(",", ".") for k in ["val", "test"])
    L = ["\\begin{table}[htbp]", "\\centering", "\\small",
         "\\caption{Robustez do tratamento de ausentes estruturais no MVP (GRU, CGH, VCP; 2021--2026). "
         "Mesmos \\textit{splits}, semente e hiperparâmetros; métricas em validação (seleção) e teste (reporte), "
         + f"$n_{{\\text{{val}}}}={nv}$, $n_{{\\text{{teste}}}}={nt}$. " +
         "$\\Delta$MAE: diferença pareada em relação a (a) com o mesmo algoritmo, IC 95\\% por \\textit{bootstrap} (1.000 reamostragens).}",
         "\\label{tab:modelos20-robustez-metricas}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llccccccc}", "\\toprule",
         " & & \\multicolumn{3}{c}{Validação} & \\multicolumn{4}{c}{Teste} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}",
         "Variante & Modelo & MAE & R\\textsuperscript{2} & QWK & MAE & R\\textsuperscript{2} & QWK & $\\Delta$MAE vs.\\ (a) [IC 95\\%] \\\\", "\\midrule"]
    for _, r in met.iterrows():
        L.append(f"{tex_escape(r.variante)} & {r.modelo} & {fmt(r.MAE_val)} & {fmt(r.R2_val)} & {fmt(r.QWK_val)} & "
                 f"{fmt(r.MAE_test)} & {fmt(r.R2_test)} & {fmt(r.QWK_test)} & "
                 + ("--" if r.variante.startswith("(a)") else
                    f"{fmt(r.dMAE_test,4)} [{fmt(r.dMAE_test_lo,4)}; {fmt(r.dMAE_test_hi,4)}]") + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_robustez_metricas.tex").write_text("\n".join(L), encoding="utf-8")

    # ---------- Tabela LaTeX 2: top-10 atributos (XGBoost) ----------
    vs = ["(a) -1, sem PROCESSO", "(b) NaN + PROCESSO", "(c) separado por PROCESSO (NaN)"]
    rb = comp.set_index(["variante", "modelo"])
    L = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
         "\\caption{Top-10 atributos por média de $|\\text{SHAP}|$ no teste do MVP (XGBoost; 2021--2026) em cada especificação. "
         "Entre parênteses, a fração do $|\\text{SHAP}|$ do atributo proveniente de linhas em que ele está ausente.}",
         "\\label{tab:modelos20-robustez-top10}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{rlll}", "\\toprule",
         "\\# & (a) $-1$, sem PROCESSO & (b) NaN + PROCESSO & (c) Separado por PROCESSO \\\\", "\\midrule"]
    cols = {v: info[(v, "XGBoost")]["global"].sort_values(ascending=False).head(10) for v in vs}
    for i in range(10):
        cel = [f"{tex_escape(curto(cols[v].index[i]))} ({fmt(100*info[(v,'XGBoost')]['frac_miss'][cols[v].index[i]],0)}\\%)" for v in vs]
        L.append(f"{i+1} & " + " & ".join(cel) + " \\\\")
    L += ["\\midrule", "Spearman vs.\\ (a) & 1{,}00 & " + fmt(rb.loc[(vs[1], "XGBoost"), "atr_spearman_global"], 2) +
          " & " + fmt(rb.loc[(vs[2], "XGBoost"), "atr_spearman_global"], 2) + " \\\\",
          "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_robustez_top10.tex").write_text("\n".join(L), encoding="utf-8")

    # ---------- Tabela LaTeX 3: estabilidade (atributo e dimensão) ----------
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Estabilidade dos rankings em relação à especificação atual (a), XGBoost: correlação de Spearman "
         "($\\rho$) entre os vetores de importância e quantos dos quatro primeiros itens de (a) permanecem no top-4. "
         "Nível ``atributo'': cada item Likert; nível ``dimensão'': item-pai somado aos subitens da lógica de salto.}",
         "\\label{tab:modelos20-robustez-spearman}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llcccccccc}", "\\toprule",
         " & & \\multicolumn{2}{c}{Global} & \\multicolumn{2}{c}{GRU} & \\multicolumn{2}{c}{CGH} & \\multicolumn{2}{c}{VCP} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
         "Variante & Nível & $\\rho$ & top-4 & $\\rho$ & top-4 & $\\rho$ & top-4 & $\\rho$ & top-4 \\\\", "\\midrule"]
    for _, r in comp[(comp.variante != ref_nome) & (comp.modelo == "XGBoost")].iterrows():
        for nv, pre in [("atributo", "atr_"), ("dimensão", "dim_")]:
            L.append(f"{tex_escape(r.variante)} & {nv} & {fmt(r[pre+'spearman_global'],2)} & {r[pre+'top4_global']}/4 & "
                     + " & ".join(f"{fmt(r[pre+'spearman_'+a],2)} & {r[pre+'top4_'+a]}/4" for a in AEROS) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (TEX_DIR / "modelos20_robustez_spearman.tex").write_text("\n".join(L), encoding="utf-8")

    # ---------- Figura ----------
    vsf = ["(a) -1, sem PROCESSO", "(b) NaN + PROCESSO", "(c) separado por PROCESSO (NaN)", "(d) separado, só itens-pai (NaN)"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5), sharex=True, sharey=True)
    ordem = info[(ref_nome, "XGBoost")]["global"].sort_values(ascending=False).index[:12][::-1]
    for ax, v in zip(axes, vsf):
        d = info[(v, "XGBoost")]
        g = d["global"][ordem]
        aus = g * d["frac_miss"][ordem]
        ax.barh([curto(a) for a in ordem], g - aus, color="#1f77b4", label="linhas com resposta")
        ax.barh([curto(a) for a in ordem], aus, left=g - aus, color="#bbbbbb", label="linhas com ausente")
        ax.set_title(v, fontsize=10)
        ax.set_xlabel("média de |SHAP| (teste)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Importância dos 12 principais atributos de (a) sob cada tratamento de ausentes (XGBoost, MVP 2021–2026)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelos20_robustez_shap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("OK")


if __name__ == "__main__":
    main()
