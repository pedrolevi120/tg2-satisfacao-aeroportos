"""
E4 — Perfil do passageiro e da viagem.

1) Inventário: quais variáveis de perfil/viagem do questionário o modelo atual
   já usa (as 11 categóricas do 03_base_features_20.py) e quais ficaram de fora.
2) Teste: acrescentar as que faltam (sem o comentário livre nem o campo livre
   MOTIVO) ao X atual, com o MESMO XGBoost (mesmos hiperparâmetros, split,
   semente, um modelo por ramo):
     V1 = atual + perfil/viagem faltante (frequência de viagens, já esteve no
          aeroporto, antecedência, tempo de espera, nº de acompanhantes, quem
          comprou a passagem, canal de compra, meio de transporte, deficiência,
          usou estacionamento, forma de check-in);
     V2 = V1 + operacionais (companhia aérea, hora da entrevista, mês da série, terminal).
   Série 2021–2026: o "mês" é o mês da série (ano-mês, 1 = jan/2021), para não
   misturar anos; o período (PERIODO) já está no X atual como dummies.
   Codificação: faixas ordenadas viram números (1, 2, 3…; NS/NR = ausente);
   categorias nominais viram dummies, com as < 0,5% agrupadas em "Outros" (como
   no pipeline). Ganho pareado no teste contra o XGBoost atual.
3) O top-4 por dimensão muda? SHAP por dimensão do modelo V2 (TreeSHAP), global
   e por aeroporto; e quanto pesa o bloco de perfil (soma do SHAP das variáveis
   de perfil, tratada como uma "dimensão").

Saída: resultados/e4_inventario_perfil.csv, e4_perfil_metricas.csv,
       e4_perfil_ranking.csv, e4_perfil_top4_aeroporto.csv
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from exp_comum import (RES, SEED, N_JOBS, TOP4, PorRamoF, Timer, carregar, comparar_pareado,
                       preds_atuais, grupos_dim, nome_dim, fmt_ic, base_unificada_alinhada,
                       shap_dim_tree)

warnings.filterwarnings("ignore")
JA_USADAS = ["AEROPORTO", "PERIODO", "GÊNERO", "ESCOLARIDADE", "RENDA FAMILIAR", "MOTIVO DA VIAGEM",
             "TIPO DE VOO", "NACIONALIDADE", "VIAJANDO SOZINHO", "IDADE", "CONEXÃO"]
FAIXA_TEMPO = {"30min a 1h": 1, "1h a 1h30min": 2, "1h30min a 2h": 3, "2h a 2h30min": 4,
               "2h30min a 3h": 5, "Mais de 3h": 6}


def freq_viagens(s):
    t = s.astype(str).str.lower().str.strip()
    out = pd.Series(np.nan, index=s.index)
    out[t.str.startswith("1")] = 1
    out[t == "2 a 3"] = 2
    out[t == "4 a 5"] = 3
    out[t == "6 a 10"] = 4
    out[t.str.startswith("mais de 11")] = 5
    return out


def dummies(s, prefixo, min_freq=0.005):
    t = s.astype(str).str.strip().str.replace(r"^Outros:\s*", "", regex=True).str.lower()
    t = t.where(s.notna(), None)
    freq = t.value_counts(normalize=True)
    t = t.where(~t.isin(freq[freq < min_freq].index), "outros")
    return pd.get_dummies(t, prefix=prefixo, dummy_na=False).astype(np.uint8)


def construir_extras(df):
    A = pd.DataFrame(index=df.index)
    A["P_freq_viagens_12m"] = freq_viagens(df["QUANTIDADE DE VIAGENS NOS ÚLTIMOS 12 MESES"])
    A["P_ja_esteve_aeroporto"] = df["JÁ EMBARCOU/DESEMBARCOU ANTES NO AEROPORTO"].map({"Sim": 1, "Não": 0})
    A["P_antecedencia"] = df["ANTECEDÊNCIA"].map(FAIXA_TEMPO)
    A["P_tempo_espera"] = df["TEMPO DE ESPERA"].map(FAIXA_TEMPO)
    ac = df["NÚMERO DE ACOMPANHANTES"].map({"1 pessoa": 1, "2 pessoas": 2, "3 pessoas": 3, "4 ou mais pessoas": 4})
    ac[df["VIAJANDO SOZINHO"] == "Sim"] = 0
    A["P_n_acompanhantes"] = ac
    A["P_passagem_por_terceiros"] = df["AQUISIÇÃO DA PASSAGEM"].map({"Por terceiros": 1, "Pelo passageiro": 0})
    A["P_deficiencia"] = (df["POSSUI DEFICIÊNCIA"].fillna("Nenhuma") != "Nenhuma").astype(float)
    A["P_usou_estacionamento"] = df["UTILIZOU O ESTACIONAMENTO"].map({"Sim": 1, "Não": 0})
    A = pd.concat([A,
                   dummies(df["MEIO DE AQUISIÇÃO DA PASSAGEM"], "P_canal_compra"),
                   dummies(df["MEIO DE  TRANSPORTE PARA O AEROPORTO"], "P_transporte"),
                   dummies(df["FORMA DE CHECK-IN"], "P_forma_checkin")], axis=1)
    B = pd.DataFrame(index=df.index)
    hora = pd.to_numeric(df["INÍCIO COLETA"].astype(str).str.extract(r"^(\d{1,2}):")[0], errors="coerce")
    B["O_hora_entrevista"] = hora.where(hora.between(0, 23))
    dt = pd.to_datetime(df["DATA"])
    B["O_mes_serie"] = ((dt.dt.year - 2021) * 12 + dt.dt.month).astype(float)  # ano-mês, sem misturar anos
    B["O_terminal"] = pd.to_numeric(df["TERMINAL"], errors="coerce")
    B = pd.concat([B, dummies(df["CIA AÉREA"], "O_cia")], axis=1)
    A.columns = [re.sub(r"[^\w]+", "_", c) for c in A.columns]
    B.columns = [re.sub(r"[^\w]+", "_", c) for c in B.columns]
    return A.astype(float), B.astype(float)


def inventario(df):
    linhas = []
    for c, desc, tipo in [
        ("GÊNERO", "gênero", "passageiro"), ("IDADE", "faixa etária", "passageiro"),
        ("ESCOLARIDADE", "escolaridade", "passageiro"), ("RENDA FAMILIAR", "renda familiar", "passageiro"),
        ("NACIONALIDADE", "nacionalidade", "passageiro"), ("POSSUI DEFICIÊNCIA", "deficiência", "passageiro"),
        ("MOTIVO DA VIAGEM", "motivo da viagem", "viagem"), ("TIPO DE VOO", "doméstico/internacional", "viagem"),
        ("CONEXÃO", "em conexão", "viagem"), ("VIAJANDO SOZINHO", "viajando sozinho", "viagem"),
        ("NÚMERO DE ACOMPANHANTES", "nº de acompanhantes", "viagem"),
        ("QUANTIDADE DE VIAGENS NOS ÚLTIMOS 12 MESES", "frequência de viagens (12 meses)", "passageiro"),
        ("JÁ EMBARCOU/DESEMBARCOU ANTES NO AEROPORTO", "já esteve no aeroporto", "passageiro"),
        ("ANTECEDÊNCIA", "antecedência de chegada", "viagem"), ("TEMPO DE ESPERA", "tempo de espera", "viagem"),
        ("AQUISIÇÃO DA PASSAGEM", "quem comprou a passagem", "viagem"),
        ("MEIO DE AQUISIÇÃO DA PASSAGEM", "canal de compra", "viagem"),
        ("MEIO DE  TRANSPORTE PARA O AEROPORTO", "meio de transporte", "viagem"),
        ("UTILIZOU O ESTACIONAMENTO", "usou estacionamento", "viagem"),
        ("FORMA DE CHECK-IN", "forma de check-in", "viagem"),
        ("CIA AÉREA", "companhia aérea", "operacional"), ("INÍCIO COLETA", "hora da entrevista", "operacional"),
        ("DATA", "mês da série (ano-mês)", "operacional"), ("TERMINAL", "terminal", "operacional"),
        ("MOTIVO", "campo livre (texto)", "excluído"), ("COMENTÁRIOS ADICIONAIS", "comentário livre", "excluído")]:
        cob = df.groupby("PROCESSO")[c].apply(lambda s: s.notna().mean())
        linhas.append({"coluna": c, "descricao": desc, "tipo": tipo, "ja_no_modelo": c in JA_USADAS,
                       "cobertura_embarque": round(float(cob.get("Embarque", 0)), 3),
                       "cobertura_desembarque": round(float(cob.get("Desembarque", 0)), 3)})
    return pd.DataFrame(linhas)


def main():
    d = carregar()
    X, y, ramo, tr, va, te, aero = d["X"], d["y"], d["ramo"], d["tr"], d["va"], d["te"], d["aero"]
    df = base_unificada_alinhada()
    assert (df["AEROPORTO"].values == aero).all() and (df["PROCESSO"].values == ramo).all()
    assert np.allclose(pd.to_numeric(df["SATISFAÇÃO GERAL"]).values, y)
    inv = inventario(df)
    inv.to_csv(RES / "e4_inventario_perfil.csv", index=False)
    print(inv.to_string())

    A, B = construir_extras(df)
    print(f"  bloco perfil/viagem: {A.shape[1]} colunas; bloco operacional: {B.shape[1]} colunas")
    P = preds_atuais()
    ref = P.loc[te, "pred_xgb"].values
    _, grupos = grupos_dim()

    def fab(r):
        return xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
                                early_stopping_rounds=30, verbosity=0)

    def aj(m, Xt, yt, Xv, yv):
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        return m

    linhas, ranks, top4 = [], {}, []
    perfil_cols_ja = [c for c in X.columns if any(c.startswith(p + "_") for p in JA_USADAS
                                                  if p not in ("AEROPORTO", "PERIODO"))]
    for nome, Xv in [("V0_atual_refit", X), ("V1_perfil", pd.concat([X, A], axis=1)),
                     ("V2_perfil_operacional", pd.concat([X, A, B], axis=1))]:
        with Timer(nome):
            m = PorRamoF(fab, aj).fit(Xv[tr], y[tr], ramo[tr], Xv[va], y[va], ramo[va])
        pt = m.predict(Xv[te], ramo[te])
        r = comparar_pareado(y[te], ref, pt)
        linhas.append({"variante": nome, "n_features": Xv.shape[1], **r})
        print(f"  {nome}: MAE {fmt_ic(r['MAE'], r['MAE_lo'], r['MAE_hi'], 4)} R2 {r['R2']:.4f} QWK {r['QWK']:.4f} | "
              f"ΔMAE {fmt_ic(r['dMAE'], r['dMAE_lo'], r['dMAE_hi'], 4, True)} ΔR2 {fmt_ic(r['dR2'], r['dR2_lo'], r['dR2_hi'], 4, True)}",
              flush=True)
        # SHAP por dimensão + blocos de perfil
        with Timer(f"SHAP {nome}"):
            D, sv = shap_dim_tree(m, Xv.loc[te], ramo[te], grupos)
        novos = [c for c in Xv.columns if c.startswith("P_")]
        oper = [c for c in Xv.columns if c.startswith("O_")]
        D["[perfil já usado]"] = sv[perfil_cols_ja].sum(axis=1)
        if novos:
            D["[perfil novo]"] = sv[novos].sum(axis=1)
        if oper:
            D["[operacional novo]"] = sv[oper].sum(axis=1)
        D["[aeroporto]"] = sv[[c for c in Xv.columns if c.startswith("AEROPORTO_")]].sum(axis=1)
        Da = D.abs()
        imp = Da.mean().sort_values(ascending=False)
        ranks[nome] = imp
        so_dims = [k for k in imp.index if not k.startswith("[")]
        print("   top-8: " + " > ".join(f"{nome_dim(k) if not k.startswith('[') else k} ({v:.3f})"
                                       for k, v in imp.head(8).items()))
        for Aa in sorted(np.unique(aero[te])):
            ma = aero[te] == Aa
            ia = Da.loc[ma, so_dims].mean().sort_values(ascending=False)
            top4.append({"variante": nome, "aeroporto": Aa, "top4": " > ".join(nome_dim(k) for k in ia.index[:4]),
                         "mesmo_conjunto_top4": set(ia.index[:4]) == set(TOP4)})
        if nome != "V0_atual_refit":
            fi = sv[novos + oper].abs().mean().sort_values(ascending=False)
            print("   novas variáveis mais usadas: " + ", ".join(f"{k} ({v:.4f})" for k, v in fi.head(8).items()))
            fi.to_csv(RES / f"e4_importancia_novas_{nome}.csv")

    met = pd.DataFrame(linhas)
    met.to_csv(RES / "e4_perfil_metricas.csv", index=False)
    R = pd.DataFrame(ranks)
    for c in list(R.columns):
        R[f"rank_{c}"] = R[c].rank(ascending=False).astype("Int64")
    R.insert(0, "nome", [nome_dim(k) if not k.startswith("[") else k for k in R.index])
    R = R.sort_values("V0_atual_refit", ascending=False)
    R.to_csv(RES / "e4_perfil_ranking.csv")
    T = pd.DataFrame(top4)
    T.to_csv(RES / "e4_perfil_top4_aeroporto.csv", index=False)
    print(R.head(14).round(4).to_string())
    print(T.groupby("variante").mesmo_conjunto_top4.sum())


if __name__ == "__main__":
    main()
