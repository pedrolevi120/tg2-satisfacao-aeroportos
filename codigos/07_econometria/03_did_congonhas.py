"""
T6/econometria — Quase-experimento Congonhas (SBSP): transição Infraero -> Aena em 17/10/2023,
com a série MPor 2021T1–2026T2 (pré = 2021T1–17/10/2023; pós = 17/10/2023–2026T2).

Diferenças em diferenças com microdados (respondente):
  y_iat = alpha_a + gamma_t + delta * (SBSP_a x Pós_it) + X_i'beta + e_iat
  * tratado: SBSP; Pós = entrevista a partir de 17/10/2023 (pela data, não pelo trimestre).
  * controles: aeroportos SEM troca de operador em 2021T1–2026T2 (regime_operador_serie.csv):
    excluídos SBEG, SBGO, SBCT (concessões de 2022), SBSG (troca de concessionária em 2024) e SBBE
    (concessão em 04/09/2023, analisado à parte como resultado secundário). SBRJ (Infraero a série
    toda) fica no grupo de controle; a coluna (4) o retira.
  * efeitos fixos de aeroporto e de PERIODO (trimestre); em (2), de mês com o mês da troca dividido
    na data. Covariáveis do respondente como antes.
  * inferência: com UM aeroporto tratado, EP clusterizados por aeroporto não são confiáveis
    (Conley & Taber, 2011; MacKinnon & Webb, 2018). Reportam-se EP robustos (HC1) e clusterizados
    como referência, o p de permutação (placebo no espaço: cada controle como pseudo-tratado; menor p
    possível = 1/(J+1)) e, na especificação principal, o p do wild cluster bootstrap restrito
    (pesos de Webb), que tende a ser conservador com um só cluster tratado (MacKinnon & Webb, 2017).
  * horizonte: curto prazo (17/10/2023–2024T2) x médio prazo (2024T3–2026T2), dois termos de DiD.
  * tendências paralelas: event study por trimestre relativo (ref. 2023T3; leads até 2021T1, lags até
    2026T2) e por mês relativo inteiro (ref. set./2023; o mês da troca dividido na data, parte pré
    com rótulo -0,5), teste conjunto dos leads (Wald HC1 e permutação) e tendência linear pré.
  * secundário: o mesmo desenho para Belém (SBBE), concessão à NOA em 04/09/2023 (ref. 2023T2).
  * poder: MDE ~ (z_0,975 + z_0,80) x DP(placebos).
Entrada: dados/processados/econometria_base20.parquet (00_base20.py)
Saídas: resultados/did_*.csv|json, tex/econometria_did*.tex, figuras econometria_did_*.pdf
"""
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from comum import *

warnings.filterwarnings("ignore")
TRATAMENTOS = {"SBSP": pd.Timestamp("2023-10-17"), "SBBE": pd.Timestamp("2023-09-04")}
TRATADO = "SBSP"
FIM_CURTO = pd.Timestamp("2024-07-01")   # curto prazo: até o fim de 2024T2
OUTCOMES = ["y", "y5", "y123"]
Z80 = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
RNG = np.random.default_rng(SEED)


# ------------------------------------------------------------------ dados
def carregar():
    cols = CONTROLES_CAT + ["AEROPORTO", "PERIODO", "DATA", "MES_NUM", "ANO", "TRIMESTRE", "PROCESSO",
                            "GRUPO_DID", "TIPO DE VOO"]
    df, nome = carregar_20(colunas=cols)
    df = df[df["GRUPO_DID"].isin(["tratado", "tratado_secundario", "controle"])].reset_index(drop=True)
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    df["y5"] = (df["y"] == 5).astype(float)
    df["y123"] = (df["y"] <= 3).astype(float)
    df["mes_abs"] = df["ANO"].astype(int) * 12 + df["MES_NUM"].astype(int) - 1
    df["q_abs"] = (df["ANO"].astype(int) - 2021) * 4 + df["TRIMESTRE"].astype(int) - 1
    frac = float(os.environ.get("DID_FRAC", "1.0"))          # só para teste rápido
    if frac < 1:
        df = df.groupby(["AEROPORTO", "PERIODO"], group_keys=False).sample(frac=frac, random_state=SEED)
        df = df.reset_index(drop=True)
    controles = sorted(df.loc[df.GRUPO_DID == "controle", "AEROPORTO"].unique())
    return df, nome, controles


def eventos(df, data_trat):
    """Pós (pela data; linhas sem DATA usam o mês), trimestre e mês relativos à troca."""
    m0 = data_trat.year * 12 + data_trat.month - 1
    q0 = (data_trat.year - 2021) * 4 + (data_trat.month - 1) // 3
    mesmo_mes = df["mes_abs"].values == m0
    pos = (df["mes_abs"].values > m0) | (mesmo_mes & (df["DATA"].values >= np.datetime64(data_trat)))
    q_rel = df["q_abs"].values - q0
    m_rel = (df["mes_abs"].values - m0).astype(float)
    m_rel[mesmo_mes & ~pos] = -0.5            # parte pré do mês da troca
    return pos, q_rel, m_rel


# ------------------------------------------------------------------ MQO rápido
class Desenho:
    """Matrizes da amostra-base reaproveitadas por todas as regressões (covariáveis calculadas uma vez)."""

    def __init__(self, df):
        self.df = df
        C, _ = dummies_controles(controles_limpos(df))
        C["PROCESSO=Embarque"] = (df["PROCESSO"].astype(str) == "Embarque").astype(float)
        self.C = C.values.astype(float)
        self.cnames = list(C.columns)
        self.aero = df["AEROPORTO"].astype(str).values
        self.Y = {o: df[o].values.astype(float) for o in OUTCOMES}

    @staticmethod
    def dummies(lab):
        u, inv = np.unique(lab, return_inverse=True)
        M = np.zeros((len(lab), max(len(u) - 1, 0)))
        r = np.where(inv > 0)[0]
        M[r, inv[r] - 1] = 1.0
        return M

    def ajustar(self, rows, outcome, regs, tempo, cov=True, se=True, w=None):
        """MQO (ou MQP com pesos w) de y em [const, regs, EF aeroporto, EF tempo, covariáveis].
        regs: dict nome -> vetor (amostra-base); tempo: rótulos de tempo (amostra-base).
        Retorna coeficientes dos regs e, se se=True, matrizes HC1 e CR1 (cluster aeroporto) deles."""
        y = self.Y[outcome][rows] if isinstance(outcome, str) else np.asarray(outcome, float)[rows]
        R = np.column_stack([np.asarray(v, float)[rows] for v in regs.values()])
        partes = [np.ones((len(y), 1)), R, self.dummies(self.aero[rows]), self.dummies(np.asarray(tempo)[rows])]
        if cov:
            Cs = self.C[rows]
            partes.append(Cs[:, Cs.std(0) > 0])
        X = np.hstack(partes)
        if w is not None:
            sw = np.sqrt(w[rows])
            X, y = X * sw[:, None], y * sw
        XtX = X.T @ X
        XtXi = np.linalg.pinv(XtX)
        b = XtXi @ (X.T @ y)
        J = list(range(1, 1 + R.shape[1]))
        out = {"b": b[J], "n": len(y), "k": X.shape[1]}
        if not se:
            return out
        e = y - X @ b
        n, k = X.shape
        A = X @ XtXi[J].T                       # n x |J| (linhas de (X'X)^-1 X')
        S = A * e[:, None]
        out["V_hc1"] = S.T @ S * n / (n - k)
        g = pd.factorize(self.aero[rows])[0]
        G = g.max() + 1
        Sg = np.zeros((G, len(J)))
        np.add.at(Sg, g, S)
        out["V_cl"] = Sg.T @ Sg * G / (G - 1) * (n - 1) / (n - k)
        out["G"] = G
        out["X"], out["y"], out["XtXi"], out["e"] = X, y, XtXi, e
        return out


def wcr_bootstrap(fit, j, grupos, B=9999):
    """Wild cluster bootstrap restrito (H0: coef j = 0), pesos de Webb, estatística t com EP CR1.
    Usa FWL e agregação por cluster (custo independente de n por réplica)."""
    X, y = fit["X"], fit["y"]
    jj = 1 + j
    Xr = np.delete(X, jj, axis=1)
    U, sv, _ = np.linalg.svd(Xr, full_matrices=False)     # base ortonormal de col(Xr), robusta a colinearidade
    Q = U[:, sv > sv[0] * 1e-10]
    del U
    D = X[:, jj]
    d = D - Q @ (Q.T @ D)
    u = y - Q @ (Q.T @ y)
    dd = d @ d
    g = pd.factorize(grupos)[0]
    G = g.max() + 1
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    c = np.array([d[g == h] @ u[g == h] for h in range(G)])
    s = np.array([d[g == h] @ d[g == h] for h in range(G)])
    P = np.column_stack([Q[g == h].T @ d[g == h] for h in range(G)])     # (k-1) x G
    Qm = np.column_stack([Q[g == h].T @ u[g == h] for h in range(G)])    # (k-1) x G
    Bm = np.diag(c) - P.T @ Qm - np.outer(s, c) / dd                     # escores por cluster = Bm @ w
    delta = c.sum() / dd
    se = np.sqrt(adj * ((Bm @ np.ones(G)) ** 2).sum()) / dd
    # conferência: w = 1 reproduz o estimador e o EP CR1 da regressão completa
    assert np.isclose(delta, fit["b"][j]) and np.isclose(se, np.sqrt(fit["V_cl"][j, j]))
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    W = RNG.choice(webb, size=(G, B))
    d_b = c @ W / dd
    se_b = np.sqrt(adj * ((Bm @ W) ** 2).sum(0)) / dd
    return float(np.mean(np.abs(d_b / se_b) >= abs(delta / se)))


def pval(b, se):
    return float(2 * stats.norm.sf(abs(b / se))) if se > 0 else np.nan


def p_perm(obs, plac):
    plac = np.asarray(plac)
    return (float((1 + (np.abs(plac) >= abs(obs)).sum()) / (1 + len(plac))),
            int(1 + (np.abs(plac) > abs(obs)).sum()))


# ------------------------------------------------------------------ DiD
def did_espec(Dz, tratado, data_trat, controles, outcome, tempo="PERIODO", cov=True, filtro=None,
              horizonte=False, data_post=None, wcr=False):
    """DiD de `tratado` (+ permutação nos controles). horizonte=True: termos curto e médio prazo.
    data_post: data alternativa de corte (placebo temporal). filtro: máscara extra de linhas."""
    df = Dz.df
    data_post = data_post or data_trat
    pos, _, m_rel = eventos(df, data_post)
    if tempo == "PERIODO":
        tlab = df["PERIODO"].astype(str).values
    else:                                       # mês calendário, com o mês da troca dividido
        tlab = np.where(m_rel == -0.5, "pre", "") + df["mes_abs"].astype(str).values
    base = np.ones(len(df), bool) if filtro is None else filtro

    def regs(unid):
        t = (Dz.aero == unid)
        if not horizonte:
            return {"DiD": (t & pos).astype(float)}
        curto = t & pos & (df["mes_abs"].values < FIM_CURTO.year * 12 + FIM_CURTO.month - 1)
        medio = t & pos & ~curto
        return {"DiD_curto": curto.astype(float), "DiD_medio": medio.astype(float)}

    rows = base & np.isin(Dz.aero, [tratado] + list(controles))
    f = Dz.ajustar(rows, outcome, regs(tratado), tlab, cov, se=True)
    nomes = list(regs(tratado).keys())
    plac = {c: Dz.ajustar(base & np.isin(Dz.aero, controles), outcome, regs(c), tlab, cov, se=False)["b"]
            for c in controles} if len(controles) >= 5 else {}
    linhas = []
    for j, nm in enumerate(nomes):
        b = float(f["b"][j])
        se1, sec = float(np.sqrt(f["V_hc1"][j, j])), float(np.sqrt(f["V_cl"][j, j]))
        if f["G"] < 5:                          # 2–3 clusters: EP clusterizado sem sentido
            sec = np.nan
        r = {"termo": nm, "delta": b, "ep_hc1": se1, "p_hc1": pval(b, se1), "ep_cluster": sec,
             "p_cluster": pval(b, sec), "n": int(f["n"]), "n_controles": len(controles),
             "n_tratado_pos": int(regs(tratado)[nm][rows].sum()),
             "n_tratado": int((rows & (Dz.aero == tratado)).sum())}
        if plac:
            v = np.array([p[j] for p in plac.values()])
            r["p_perm"], r["rank_perm"] = p_perm(b, v)
            r["sd_placebo"] = float(v.std(ddof=1))
            r["mde_80"] = Z80 * r["sd_placebo"]
        else:
            r.update(p_perm=np.nan, rank_perm=np.nan, sd_placebo=np.nan, mde_80=np.nan)
        r["p_wcr"] = wcr_bootstrap(f, j, Dz.aero[rows]) if wcr else np.nan
        linhas.append(r)
    return linhas, {c: dict(zip(nomes, map(float, p))) for c, p in plac.items()}


def painel_agregado(Dz, tratado, data_trat, controles, outcome):
    """Painel aeroporto x período (médias), DiD = fração das entrevistas do tratado no pós, MQP."""
    df = Dz.df
    pos, _, _ = eventos(df, data_trat)
    d = df.loc[np.isin(Dz.aero, [tratado] + list(controles)), ["AEROPORTO", "PERIODO", outcome]].copy()
    d["pos"] = pos[d.index]
    g = d.groupby(["AEROPORTO", "PERIODO"]).agg(m=(outcome, "mean"), n=(outcome, "size"),
                                               pos=("pos", "mean")).reset_index()

    def fit(gg, unid, se):
        Z = Desenho.__new__(Desenho)
        Z.aero = gg.AEROPORTO.values
        Z.Y = {"m": gg.m.values}
        Z.C = np.zeros((len(gg), 0))
        regs = {"DiD": ((gg.AEROPORTO == unid) * gg.pos).values}
        return Z.ajustar(np.ones(len(gg), bool), "m", regs, gg.PERIODO.values, cov=False, se=se,
                         w=gg.n.values.astype(float))

    f = fit(g, tratado, True)
    gc = g[g.AEROPORTO != tratado].reset_index(drop=True)
    plac = np.array([fit(gc, c, False)["b"][0] for c in controles])
    b = float(f["b"][0])
    se1 = float(np.sqrt(f["V_hc1"][0, 0]))
    pp, rk = p_perm(b, plac)
    return {"termo": "DiD", "delta": b, "ep_hc1": se1, "p_hc1": pval(b, se1), "ep_cluster": np.nan,
            "p_cluster": np.nan, "n": int(len(g)), "n_controles": len(controles), "n_tratado_pos": int((g.pos > 0)[g.AEROPORTO == tratado].sum()),
            "n_tratado": int((g.AEROPORTO == tratado).sum()), "p_perm": pp, "rank_perm": rk,
            "sd_placebo": float(plac.std(ddof=1)), "mde_80": Z80 * float(plac.std(ddof=1)), "p_wcr": np.nan}


# ------------------------------------------------------------------ event study
def event_study(Dz, tratado, data_trat, controles, outcome, escala="trimestre", cov=True):
    """Interações tratado x tempo relativo (ref. -1). Retorna coeficientes, teste dos leads e placebos."""
    df = Dz.df
    _, q_rel, m_rel = eventos(df, data_trat)
    rel = q_rel.astype(float) if escala == "trimestre" else m_rel
    if escala == "trimestre":
        tlab = df["PERIODO"].astype(str).values
    else:
        tlab = np.where(m_rel == -0.5, "pre", "") + df["mes_abs"].astype(str).values

    def ajusta(rows, unid, se):
        ks = sorted(set(rel[rows & (Dz.aero == unid)]) - {-1.0})
        regs = {k: ((Dz.aero == unid) & (rel == k)).astype(float) for k in ks}
        return ks, Dz.ajustar(rows, outcome, regs, tlab, cov, se=se)

    rows = np.isin(Dz.aero, [tratado] + list(controles))
    ks, f = ajusta(rows, tratado, True)
    coef = {-1.0: (0.0, 0.0, 0.0)}
    for j, k in enumerate(ks):
        coef[k] = (float(f["b"][j]), float(np.sqrt(f["V_hc1"][j, j])), float(np.sqrt(f["V_cl"][j, j])))
    leads = [j for j, k in enumerate(ks) if k < 0]
    bl = f["b"][leads]
    W = float(bl @ np.linalg.pinv(f["V_hc1"][np.ix_(leads, leads)]) @ bl)
    rc = np.isin(Dz.aero, controles)
    plac = {}
    for c in controles:
        kc, fc = ajusta(rc, c, False)
        plac[c] = {**{-1.0: 0.0}, **{k: float(b) for k, b in zip(kc, fc["b"])}}
    s_obs = float((bl ** 2).sum())
    s_pl = np.array([sum(v ** 2 for k, v in p.items() if k < 0 and k != -1.0) for p in plac.values()])
    # tendência linear pré: tratado x índice do trimestre, só no pré
    pos, _, _ = eventos(df, data_trat)
    pre = rows & ~pos
    t = df["q_abs"].values.astype(float)
    ft = Dz.ajustar(pre, outcome, {"tend": (Dz.aero == tratado) * t}, df["PERIODO"].astype(str).values, cov)
    tp = [Dz.ajustar(rc & ~pos, outcome, {"tend": (Dz.aero == c) * t}, df["PERIODO"].astype(str).values,
                     cov, se=False)["b"][0] for c in controles]
    b_t, se_t = float(ft["b"][0]), float(np.sqrt(ft["V_hc1"][0, 0]))
    return {"coef": coef, "wald_leads": W, "gl": len(leads), "p_leads": float(stats.chi2.sf(W, len(leads))),
            "p_leads_perm": float((1 + (s_pl >= s_obs).sum()) / (1 + len(s_pl))),
            "tend_pre": b_t, "tend_pre_ep": se_t, "tend_pre_p": pval(b_t, se_t),
            "tend_pre_p_perm": p_perm(b_t, tp)[0], "placebos": plac}


# ------------------------------------------------------------------ main
def main():
    estilo()
    t0 = time.time()
    df, nome, CTRL = carregar()
    Dz = Desenho(df)
    print(f"Base: {nome}; n={len(df):,}; controles ({len(CTRL)}): {', '.join(CTRL)}; "
          f"{len(Dz.cnames)} covariáveis", flush=True)
    d_trat = TRATAMENTOS[TRATADO]
    pos, _, _ = eventos(df, d_trat)
    print(f"SBSP: {int(((Dz.aero == 'SBSP') & ~pos).sum()):,} pré, {int(((Dz.aero == 'SBSP') & pos).sum()):,} pós; "
          f"controles: {int(np.isin(Dz.aero, CTRL).sum()):,}", flush=True)

    especs = {
        "(1) Principal": dict(controles=CTRL, wcr=True),
        "(2) EF mensais": dict(controles=CTRL, tempo="MES"),
        "(3) Sem covariáveis": dict(controles=CTRL, cov=False),
        "(4) Sem SBRJ (só concessões)": dict(controles=[c for c in CTRL if c != "SBRJ"]),
        "(5) Sem Rio (SBGL, SBRJ)": dict(controles=[c for c in CTRL if c not in ("SBGL", "SBRJ")]),
        "(6) Só SP (SBGR, SBKP)": dict(controles=["SBGR", "SBKP"]),
        "(7) Só Infraero (SBRJ)": dict(controles=["SBRJ"]),
        # robustez adicional (só no CSV e na tabela de horizonte)
        "(R1) Sem 2023T4 (transição)": dict(controles=CTRL, filtro=(df["PERIODO"] != "2023T4").values),
        "(R2) Sem SBPA (enchentes 2024)": dict(controles=[c for c in CTRL if c != "SBPA"]),
        "(R3) Placebo temporal 17/10/2022 (só pré)": dict(controles=CTRL, filtro=~pos,
                                                         data_post=pd.Timestamp("2022-10-17")),
        "(H) Curto x médio prazo": dict(controles=CTRL, horizonte=True),
    }
    linhas, placebos = [], {}
    for nome_e, e in especs.items():
        for out in OUTCOMES:
            ls, plac = did_espec(Dz, TRATADO, d_trat, outcome=out, **e)
            for r in ls:
                r.update({"tratado": TRATADO, "espec": nome_e, "outcome": out})
                linhas.append(r)
                print(f"  {nome_e:42s} {out:5s} {r['termo']:9s} delta={r['delta']:+.4f} HC1={r['ep_hc1']:.4f} "
                      f"cl={r['ep_cluster']:.4f} p_perm={r['p_perm']:.3f} (rank {r['rank_perm']}) "
                      f"MDE={r['mde_80']:.3f} p_wcr={r['p_wcr']:.3f}", flush=True)
            for c, v in plac.items():
                for termo, dl in v.items():
                    placebos.setdefault((nome_e, out, termo), {})[c] = dl
    for out in OUTCOMES:
        r = painel_agregado(Dz, TRATADO, d_trat, CTRL, out)
        r.update({"tratado": TRATADO, "espec": "(8) Painel agregado (MQP)", "outcome": out})
        linhas.append(r)
        print(f"  (8) Painel agregado {out:5s} delta={r['delta']:+.4f} HC1={r['ep_hc1']:.4f} p_perm={r['p_perm']:.3f}")

    # ---- SBBE (secundário): mesmo desenho, controles = os mesmos 14
    d_be = TRATAMENTOS["SBBE"]
    especs_be = {"(1) Principal": dict(controles=CTRL, wcr=True),
                 "(3) Sem covariáveis": dict(controles=CTRL, cov=False),
                 "(4) Sem SBRJ (só concessões)": dict(controles=[c for c in CTRL if c != "SBRJ"]),
                 "(H) Curto x médio prazo": dict(controles=CTRL, horizonte=True)}
    for nome_e, e in especs_be.items():
        for out in OUTCOMES:
            ls, _ = did_espec(Dz, "SBBE", d_be, outcome=out, **e)
            for r in ls:
                r.update({"tratado": "SBBE", "espec": nome_e, "outcome": out})
                linhas.append(r)
                print(f"  SBBE {nome_e:30s} {out:5s} {r['termo']:9s} delta={r['delta']:+.4f} "
                      f"HC1={r['ep_hc1']:.4f} p_perm={r['p_perm']:.3f} p_wcr={r['p_wcr']:.3f}", flush=True)
    res = pd.DataFrame(linhas)
    res.to_csv(OUT_DIR / "did_resultados.csv", index=False)
    pl = pd.DataFrame([{"espec": k[0], "outcome": k[1], "termo": k[2], "placebo": c, "delta": v}
                       for k, dct in placebos.items() for c, v in dct.items()])
    pl.to_csv(OUT_DIR / "did_placebos.csv", index=False)
    print(f"  [{time.time() - t0:.0f}s] DiD ok", flush=True)

    # ---- balanceamento: DiD em covariáveis (mudança de composição?)
    C = controles_limpos(df)
    bal_vars = {"Embarque": (df["PROCESSO"].astype(str) == "Embarque").astype(float),
                "Voo internacional": (df["TIPO DE VOO"].astype(str) == "Internacional").astype(float),
                "Feminino": (C["GÊNERO"] == "Feminino").astype(float),
                "Estrangeiro": (C["NACIONALIDADE"] == "Estrangeira").astype(float),
                "Motivo trabalho (embarque)": (C["MOTIVO DA VIAGEM"] == "Trabalho").astype(float)}
    bal = []
    for k, v in bal_vars.items():
        Dz.Y["bal"] = v.values
        filtro = (df["PROCESSO"].astype(str) == "Embarque").values if k.endswith("(embarque)") else None
        r = did_espec(Dz, TRATADO, d_trat, CTRL, "bal", cov=False, filtro=filtro)[0][0]
        bal.append({"variavel": k, "delta": r["delta"], "ep_hc1": r["ep_hc1"], "p_perm": r["p_perm"]})
    pd.DataFrame(bal).to_csv(OUT_DIR / "did_balanceamento.csv", index=False)
    print(pd.DataFrame(bal).round(4).to_string(), flush=True)

    # ---- event studies
    es, esm, es_be = {}, {}, {}
    for out in ["y", "y5"]:
        es[out] = event_study(Dz, TRATADO, d_trat, CTRL, out, "trimestre")
        esm[out] = event_study(Dz, TRATADO, d_trat, CTRL, out, "mes")
        es_be[out] = event_study(Dz, "SBBE", d_be, CTRL, out, "trimestre")
        for rot, e in [("trim", es[out]), ("mensal", esm[out]), ("SBBE trim", es_be[out])]:
            print(f"  ES {rot} {out}: leads W={e['wald_leads']:.2f} gl={e['gl']} p={e['p_leads']:.3f} "
                  f"p_perm={e['p_leads_perm']:.3f} | tend. pré={e['tend_pre']:+.4f} (p={e['tend_pre_p']:.3f}, "
                  f"p_perm={e['tend_pre_p_perm']:.3f})", flush=True)
        print("     " + ", ".join(f"{int(k):+d}={v[0]:+.3f}({v[1]:.3f})" for k, v in sorted(es[out]["coef"].items())))

    # ---- médias brutas por período (para figura)
    d = df[np.isin(Dz.aero, [TRATADO, "SBBE"] + CTRL)]
    grupo = np.where(d.AEROPORTO == TRATADO, "SBSP", np.where(d.AEROPORTO == "SBBE", "SBBE", "Controles"))
    med = d.assign(grupo=grupo).groupby(["grupo", "PERIODO"]).agg(
        mean=("y", "mean"), sem=("y", "sem"), size=("y", "size"), p5=("y5", "mean")).reset_index()
    med.to_csv(OUT_DIR / "did_medias_trimestrais.csv", index=False)
    d.groupby(["AEROPORTO", "PERIODO"])["y"].mean().unstack().to_csv(OUT_DIR / "did_medias_aeroporto_trimestre.csv")

    def ser(x):
        return {str(int(k)) if float(k).is_integer() else str(k): (list(v) if isinstance(v, tuple) else v)
                for k, v in sorted(x.items())}

    def ser_es(e):
        return {**{k: v for k, v in e.items() if k not in ("placebos", "coef")}, "coef": ser(e["coef"]),
                "placebos": {c: ser(p) for c, p in e["placebos"].items()}}
    (OUT_DIR / "did_event_study.json").write_text(json.dumps(
        {"trimestral": {o: ser_es(e) for o, e in es.items()},
         "mensal": {o: ser_es(e) for o, e in esm.items()},
         "SBBE": {"trimestral": {o: ser_es(e) for o, e in es_be.items()}},
         "controles": CTRL, "data_tratamento": str(d_trat.date()), "referencia_trimestre": "2023T3",
         "referencia_mes": "2023-09 (mês relativo -1); -0,5 = 01–16/10/2023",
         "fim_curto_prazo": str(FIM_CURTO.date())}, indent=2, ensure_ascii=False))

    figuras(med, es, esm, es_be, res, pl, CTRL)
    tabelas(res, es, esm, es_be, bal, CTRL)
    print(f"OK ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ figuras


def figuras(med, es, esm, es_be, res, pl, ctrl):
    import matplotlib.pyplot as plt
    pers = sorted(med.PERIODO.unique())
    x = {p: i for i, p in enumerate(pers)}
    # 1) tendências brutas (média e P(nota 5))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, col, lab in [(axes[0], "mean", "Satisfação geral média (1–5)"), (axes[1], "p5", "P(nota 5)")]:
        for g, cor, mk, rot in [("SBSP", "#c53030", "o", "Congonhas (SBSP)"),
                                ("Controles", "#2b6cb0", "s", f"Controles ({len(ctrl)} aeroportos)"),
                                ("SBBE", "#d69e2e", "^", "Belém (SBBE)")]:
            m = med[med.grupo == g]
            xx = [x[p] for p in m.PERIODO]
            if col == "mean":
                ax.errorbar(xx, m["mean"], yerr=1.96 * m["sem"], color=cor, marker=mk, ms=3.5, capsize=2,
                            lw=1, label=rot)
            else:
                ax.plot(xx, m["p5"], color=cor, marker=mk, ms=3.5, lw=1, label=rot)
        ax.axvline(x["2023T4"] - 0.5, color="grey", ls="--", lw=0.8)
        ax.axvline(x["2024T3"] - 0.5, color="grey", ls=":", lw=0.8)
        ax.set_ylabel(lab)
    axes[0].text(x["2023T4"] - 0.4, axes[0].get_ylim()[1] - 0.01, "Aena (17/10/2023)", fontsize=7, va="top")
    axes[0].legend(fontsize=7.5, frameon=False, ncol=3, loc="lower left")
    axes[1].set_xticks(range(len(pers)))
    axes[1].set_xticklabels(pers, rotation=60, fontsize=7.5)
    axes[0].set_title("Satisfação geral: Congonhas × controles, 2021–2026 (IC 95%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "econometria_did_tendencias.pdf", bbox_inches="tight"); plt.close(fig)

    # 2) event study trimestral e mensal com placebos
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
    for ax, e, titulo, xl in [(axes[0], es["y"], "Trimestral (ref. 2023T3 = −1)", "Trimestres desde a troca"),
                              (axes[1], esm["y"], "Mensal (ref. set./2023 = −1; −0,5 = 1–16/10/2023)",
                               "Meses desde a troca")]:
        ks = sorted(e["coef"].keys())
        for c, p in e["placebos"].items():
            kk = [k for k in ks if k in p]
            ax.plot(kk, [p[k] for k in kk], color="#cbd5e0", lw=0.6, zorder=1)
        b = np.array([e["coef"][k][0] for k in ks]); s = np.array([e["coef"][k][1] for k in ks])
        ax.errorbar(ks, b, yerr=1.96 * s, color="#c53030", marker="o", ms=3, capsize=1.5, lw=1.0, zorder=3,
                    label="SBSP (IC 95% HC1)")
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(-0.25 if "Mensal" in titulo else -0.5, color="grey", ls="--", lw=0.8)
        ax.set_title(titulo, fontsize=9); ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel("Coeficiente (pontos)")
    axes[0].plot([], [], color="#cbd5e0", label="Placebos (controles)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Event study: Congonhas vs. controles (satisfação média)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "econometria_did_eventstudy.pdf", bbox_inches="tight"); plt.close(fig)

    # 3) event study SBBE
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, o, tit in [(axes[0], "y", "Satisfação média"), (axes[1], "y5", "P(nota 5)")]:
        e = es_be[o]
        ks = sorted(e["coef"].keys())
        for c, p in e["placebos"].items():
            kk = [k for k in ks if k in p]
            ax.plot(kk, [p[k] for k in kk], color="#cbd5e0", lw=0.6, zorder=1)
        b = np.array([e["coef"][k][0] for k in ks]); s = np.array([e["coef"][k][1] for k in ks])
        ax.errorbar(ks, b, yerr=1.96 * s, color="#d69e2e", marker="o", ms=3, capsize=1.5, lw=1.0, zorder=3)
        ax.axhline(0, color="black", lw=0.6); ax.axvline(-0.5, color="grey", ls="--", lw=0.8)
        ax.set_title(tit, fontsize=9); ax.set_xlabel("Trimestres desde a troca (ref. 2023T2)", fontsize=8)
    fig.suptitle("Event study: Belém (SBBE, NOA desde 04/09/2023) vs. controles", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "econometria_did_belem.pdf", bbox_inches="tight"); plt.close(fig)

    # 4) placebos da especificação principal (substituída por 05_did_permutacao_mensal.py)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2))
    for ax, out, tit in [(axes[0], "y", "Satisfação média"), (axes[1], "y5", "P(nota 5)")]:
        p = pl[(pl.espec == "(1) Principal") & (pl.outcome == out)]
        r = res[(res.tratado == TRATADO) & (res.espec == "(1) Principal") & (res.outcome == out)].iloc[0]
        q = pd.concat([p[["placebo", "delta"]], pd.DataFrame({"placebo": ["SBSP"], "delta": [r.delta]})]).sort_values("delta")
        ax.barh(range(len(q)), q.delta, color=["#c53030" if a == "SBSP" else "#718096" for a in q.placebo], height=0.7)
        ax.set_yticks(range(len(q))); ax.set_yticklabels(q.placebo, fontsize=7)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{tit}\nSBSP: p perm. = {r.p_perm:.2f}".replace(".", ","), fontsize=9)
        ax.set_xlabel("$\\hat\\delta$ (pós 17/10/2023 vs. pré)")
    fig.suptitle("DiD de cada aeroporto-controle como pseudo-tratado (vermelho: Congonhas)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "econometria_did_placebos.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ tabelas
def _n(x):
    return f"{int(x):,}".replace(",", ".")


def tabelas(res, es, esm, es_be, bal, ctrl):
    r0 = res[res.tratado == TRATADO]
    esp = ["(1) Principal", "(2) EF mensais", "(3) Sem covariáveis", "(4) Sem SBRJ (só concessões)",
           "(5) Sem Rio (SBGL, SBRJ)", "(6) Só SP (SBGR, SBKP)", "(7) Só Infraero (SBRJ)", "(8) Painel agregado (MQP)"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Diferenças em diferenças: transição de Congonhas (SBSP) da Infraero para a Aena "
         r"(17/10/2023) --- efeito sobre a satisfação geral, 2021T1--2026T2}",
         r"\label{tab:econometria-did}",
         r"\begin{tabular}{l" + "r" * len(esp) + "}", r"\toprule",
         " & " + " & ".join(e.split(")")[0] + ")" for e in esp) + r" \\", r"\midrule"]
    for out, titulo in [("y", "A. Satisfação geral média (1--5)"), ("y5", r"B. $\Pr(\text{nota}=5)$"),
                        ("y123", r"C. $\Pr(\text{nota}\leq 3)$")]:
        L.append(f"\\multicolumn{{{len(esp) + 1}}}{{l}}{{\\textit{{{titulo}}}}} \\\\")
        rr = [r0[(r0.espec == e) & (r0.outcome == out)].iloc[0] for e in esp]
        L.append(r"$\hat\delta$ & " + " & ".join(fmt(r.delta) for r in rr) + r" \\")
        L.append(r"EP robusto (HC1) & " + " & ".join(f"({fmt(r.ep_hc1)})" for r in rr) + r" \\")
        L.append(r"EP cluster aeroporto & " + " & ".join(f"[{fmt(r.ep_cluster)}]" if r.ep_cluster == r.ep_cluster
                                                        else "--" for r in rr) + r" \\")
        L.append(r"$p$ permutação & " + " & ".join(fmt(r.p_perm, 3) if r.p_perm == r.p_perm else "--"
                                                  for r in rr) + r" \\")
        L.append(r"$p$ \emph{wild cluster bootstrap} & " + " & ".join(fmt(r.p_wcr, 3) if r.p_wcr == r.p_wcr else "--"
                                                                     for r in rr) + r" \\")
        if out == "y":
            L.append(r"MDE (80\%, 5\%) & " + " & ".join(fmt(r.mde_80) if r.mde_80 == r.mde_80 else "--"
                                                       for r in rr) + r" \\")
        L.append(r"\addlinespace")
    rr = [r0[(r0.espec == e) & (r0.outcome == "y")].iloc[0] for e in esp]
    L.append(r"\midrule")
    L.append(r"$N$ & " + " & ".join(_n(r.n) for r in rr) + r" \\")
    L.append(r"Controles & " + " & ".join(str(int(r.n_controles)) for r in rr) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: MQO com efeitos fixos de aeroporto e de período "
          r"(trimestre; em (2), de mês, com outubro/2023 dividido em 17/10) e covariáveis do respondente (PROCESSO, "
          r"tipo de voo, nacionalidade, gênero, frequência de viagem e perfil sociodemográfico do embarque), "
          r"exceto em (3). Pós = entrevistas a partir de 17/10/2023 (pré: 2021T1--16/10/2023; pós: até 2026T2). "
          f"(1) Controles: os {len(ctrl)} aeroportos sem troca de operador em 2021--2026 "
          r"(excluídos SBEG, SBGO e SBCT, concedidos em 2022, SBSG, troca de concessionária em 2024, e SBBE, "
          r"concedido em 09/2023); inclui o Santos Dumont (SBRJ), Infraero a série toda. "
          r"(8) Painel aeroporto $\times$ período ponderado pelo número de entrevistas (tratamento = fração pós "
          r"no período). Com um único aeroporto tratado, os EP clusterizados não são confiáveis "
          r"\parencite{conley2011inference}; a inferência de referência é o $p$ de permutação (placebo no espaço: "
          r"cada controle como pseudo-tratado; menor $p$ possível = $1/(J+1)$). O \emph{wild cluster bootstrap} "
          r"restrito (pesos de Webb, 9.999 réplicas) é conservador com um só cluster tratado. MDE = efeito mínimo "
          r"detectável com 80\% de poder a 5\%, $(z_{0{,}975}+z_{0{,}80})\times$DP dos placebos.}",
          r"\end{table}"]
    (TEX_DIR / "econometria_did.tex").write_text("\n".join(L) + "\n")

    # horizonte e robustez adicional
    def cel(e, out, termo="DiD", trat=TRATADO):
        q = res[(res.tratado == trat) & (res.espec == e) & (res.outcome == out) & (res.termo == termo)]
        if q.empty:
            return "--"
        r = q.iloc[0]
        return f"{fmt(r.delta)}{estrelas(r.p_perm) if r.p_perm == r.p_perm else ''} [{fmt(r.p_perm, 3)}]"
    linhas = [("Pós total (17/10/2023--2026T2)", "(1) Principal", "DiD"),
              ("Curto prazo (17/10/2023--2024T2)", "(H) Curto x médio prazo", "DiD_curto"),
              ("Médio prazo (2024T3--2026T2)", "(H) Curto x médio prazo", "DiD_medio"),
              ("Sem 2023T4 (trimestre da transição)", "(R1) Sem 2023T4 (transição)", "DiD"),
              ("Sem SBPA (enchentes de 2024)", "(R2) Sem SBPA (enchentes 2024)", "DiD"),
              ("Placebo temporal: troca fictícia em 17/10/2022", "(R3) Placebo temporal 17/10/2022 (só pré)", "DiD")]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{DiD de Congonhas por horizonte e verificações adicionais}",
         r"\label{tab:econometria-did-horizonte}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r" & Satisfação média & $\Pr(\text{nota}=5)$ & $\Pr(\text{nota}\leq 3)$ \\", r"\midrule"]
    for i, (rot, e, termo) in enumerate(linhas):
        if i == 3:
            L.append(r"\midrule")
        L.append(f"{rot} & " + " & ".join(cel(e, o, termo) for o in OUTCOMES) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{l}{\textit{Belém (SBBE), NOA desde 04/09/2023, mesmos controles}} \\")
    for rot, e, termo in [("Pós total", "(1) Principal", "DiD"), ("Curto prazo (até 2024T2)", "(H) Curto x médio prazo", "DiD_curto"),
                          ("Médio prazo (2024T3--2026T2)", "(H) Curto x médio prazo", "DiD_medio")]:
        L.append(f"{rot} & " + " & ".join(cel(e, o, termo, "SBBE") for o in OUTCOMES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: especificação da coluna (1) da "
          r"Tabela~\ref{tab:econometria-did}; $p$ de permutação entre colchetes (placebo no espaço, "
          f"$J={len(ctrl)}$ controles; menor $p$ possível = {fmt(1 / (len(ctrl) + 1), 3)}). Curto e médio prazo "
          r"entram na mesma regressão como dois termos de DiD. O placebo temporal usa só o período pré "
          r"(até 16/10/2023) com uma troca fictícia um ano antes. *** $p<0{,}01$; ** $p<0{,}05$; * $p<0{,}10$.}",
          r"\end{table}"]
    (TEX_DIR / "econometria_did_horizonte.tex").write_text("\n".join(L) + "\n")

    # event study trimestral (SBSP)
    ks = sorted(es["y"]["coef"].keys())
    rot_q = {k: p for k, p in zip(range(-11, 11), [f"{a}T{t}" for a in range(2021, 2027) for t in range(1, 5)][:22])}
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
         r"\caption{\emph{Event study} trimestral de Congonhas (referência: 2023T3) e testes de tendências paralelas}",
         r"\label{tab:econometria-did-eventstudy}",
         r"\begin{tabular}{llrr}", r"\toprule",
         r"Trimestre & Relativo & Satisfação média & $\Pr(\text{nota}=5)$ \\", r"\midrule"]
    for k in ks:
        ki = int(k)
        if ki == -1:
            L.append(f"{rot_q[ki]} & $-1$ (ref.) & 0 & 0 \\\\")
            continue
        c = [es[o]["coef"][k] for o in ["y", "y5"]]
        L.append(f"{rot_q[ki]} & {fmt(ki, 0, sinal=True)} & {fmt(c[0][0])} ({fmt(c[0][1])}) & "
                 f"{fmt(c[1][0])} ({fmt(c[1][1])}) \\\\")
        if ki == -2:
            L.append(r"\addlinespace")
    L.append(r"\midrule")
    L.append(f"Wald leads ({es['y']['gl']} trimestres pré = 0), $p$ (HC1) & & " +
             " & ".join(fmt(es[o]["p_leads"], 3) for o in ["y", "y5"]) + r" \\")
    L.append(r"Leads, $p$ permutação & & " + " & ".join(fmt(es[o]["p_leads_perm"], 3) for o in ["y", "y5"]) + r" \\")
    L.append(r"Tendência linear pré (por trimestre), SBSP & & " + " & ".join(
        f"{fmt(es[o]['tend_pre'], 4)} ({fmt(es[o]['tend_pre_ep'], 4)})" for o in ["y", "y5"]) + r" \\")
    L.append(r"Tendência linear pré, $p$ permutação & & " + " & ".join(
        fmt(es[o]["tend_pre_p_perm"], 3) for o in ["y", "y5"]) + r" \\")
    L.append(f"Wald leads mensais, $p$ (HC1) & & " + " & ".join(
        f"{fmt(esm[o]['p_leads'], 3)} ($\\chi^2_{{{esm[o]['gl']}}}$)" for o in ["y", "y5"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: especificação da coluna (1) da "
          r"Tabela~\ref{tab:econometria-did}, com interações SBSP $\times$ trimestre (relativo ao da troca, "
          r"2023T4 = 0; 2021T2 não tem coleta). EP robustos (HC1) entre parênteses. O $p$ de permutação compara a "
          r"soma dos quadrados dos \emph{leads} de SBSP com a dos aeroportos-controle tratados como placebo. "
          r"2023T4 mistura 01--16/10 (Infraero) e o restante (Aena). Coleta parcial em 2021T3 (ago--set), "
          r"2022T4 (out.) e 2023T1 (mar.).}",
          r"\end{table}"]
    (TEX_DIR / "econometria_did_eventstudy.tex").write_text("\n".join(L) + "\n")

    # SBBE
    rb = res[res.tratado == "SBBE"]
    espb = ["(1) Principal", "(3) Sem covariáveis", "(4) Sem SBRJ (só concessões)"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{DiD de Belém (SBBE): concessão à NOA em 04/09/2023 (resultado secundário)}",
         r"\label{tab:econometria-did-belem}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         " & " + " & ".join(e.split(")")[0] + ")" for e in espb) + r" \\", r"\midrule"]
    for out, titulo in [("y", "Satisfação média"), ("y5", r"$\Pr(\text{nota}=5)$"), ("y123", r"$\Pr(\text{nota}\leq 3)$")]:
        q = [rb[(rb.espec == e) & (rb.outcome == out)].iloc[0] for e in espb]
        L.append(f"{titulo}: $\\hat\\delta$ & " + " & ".join(fmt(r.delta) for r in q) + r" \\")
        L.append(r"\quad EP (HC1) & " + " & ".join(f"({fmt(r.ep_hc1)})" for r in q) + r" \\")
        L.append(r"\quad $p$ permutação & " + " & ".join(fmt(r.p_perm, 3) for r in q) + r" \\")
    L.append(r"\midrule")
    L.append(r"Wald leads trimestrais, $p$ (HC1) & " + " & ".join(
        [fmt(es_be["y"]["p_leads"], 3) + " / " + fmt(es_be["y5"]["p_leads"], 3), "", ""]) + r" \\")
    L.append(r"Leads, $p$ permutação & " + " & ".join(
        [fmt(es_be["y"]["p_leads_perm"], 3) + " / " + fmt(es_be["y5"]["p_leads_perm"], 3), "", ""]) + r" \\")
    q = rb[(rb.espec == "(1) Principal") & (rb.outcome == "y")].iloc[0]
    L.append(f"$N$ / entrevistas SBBE (pós) & {_n(q.n)} / {_n(q.n_tratado)} ({_n(q.n_tratado_pos)}) & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize Notas: mesmo desenho do DiD de Congonhas (efeitos fixos "
          r"de aeroporto e período, covariáveis do respondente, mesmos controles; SBSP fora da amostra), com "
          r"pós = entrevistas a partir de 04/09/2023 e \emph{event study} com referência em 2023T2. "
          r"Testes de \emph{leads}: satisfação média / $\Pr(\text{nota}=5)$.}",
          r"\end{table}"]
    (TEX_DIR / "econometria_did_belem.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
