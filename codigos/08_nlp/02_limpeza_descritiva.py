"""
T7/NLP — Etapa 2: limpeza, anonimização e estatística descritiva dos comentários.

Série 2021T1–2026T2 (20 aeroportos). Tabela e figura agregam a série inteira; descritiva_por_ano.csv
dá a cobertura por ano (Brasil e aeroporto). Atenção: em 2021–2022 o campo vem em branco com mais frequência
(e não "Não"), e a taxa de comentários válidos é menor em vários aeroportos — taxas "por mil respondentes"
desses anos ficam subestimadas em relação a 2023–2026.

Saídas: saida/comentarios_limpos.parquet (apenas linhas com comentário válido, i.e. genérico+substantivo)
        saida/descritiva_por_aeroporto.csv, saida/descritiva_por_ano.csv
        texto/_paralelo/tex/nlp_descritiva.tex
        midias/figuras/nlp_cobertura_comentarios.pdf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comum import SAIDA, FIG, TEX, BASE_COM, ANOS_ROT, ANOS_TEX, COL_COM, COL_SAT, NOMES, estilo, classificar, anonimizar, fmt


def main():
    df = pd.read_parquet(BASE_COM)
    df["classe"] = df[COL_COM].map(classificar)
    df["sat"] = pd.to_numeric(df[COL_SAT], errors="coerce")
    print("Total respondentes:", len(df))
    print(df["classe"].value_counts())
    print(pd.crosstab(df.PERIODO, df.classe, margins=True).to_string())
    print("Exemplos de ruído mais comuns:")
    print(df.loc[df.classe == "ruido", COL_COM].str.strip().str.lower().value_counts().head(15))

    valido = df[df.classe.isin(["generico", "substantivo"])].copy()
    valido["texto"] = valido[COL_COM].astype(str).str.strip().map(anonimizar)
    valido["n_chars"] = valido["texto"].str.len()
    valido["n_palavras"] = valido["texto"].str.split().str.len()
    n_anon = (valido["texto"] != valido[COL_COM].astype(str).str.strip()).sum()
    print("Comentários com trechos anonimizados:", n_anon)
    print("Comprimento (palavras) dos substantivos:", valido.loc[valido.classe == "substantivo", "n_palavras"].describe())
    # truncamento no instrumento: 200 caracteres
    print("Comentários com exatamente 200 caracteres (limite do campo):", (valido[COL_COM].str.len() == 200).sum())
    valido.drop(columns=[COL_COM]).to_parquet(SAIDA / "comentarios_limpos.parquet", index=False)

    g = df.groupby("AEROPORTO")
    tab = pd.DataFrame({
        "n": g.size(),
        "n_valido": g["classe"].apply(lambda s: s.isin(["generico", "substantivo"]).sum()),
        "n_subst": g["classe"].apply(lambda s: (s == "substantivo").sum()),
        "n_ruido": g["classe"].apply(lambda s: (s == "ruido").sum()),
    })
    tab["pct_valido"] = 100 * tab.n_valido / tab.n
    tab["pct_subst"] = 100 * tab.n_subst / tab.n
    tab["palavras_med"] = valido[valido.classe == "substantivo"].groupby("AEROPORTO")["n_palavras"].median()
    tab["sat_com"] = df[df.classe == "substantivo"].groupby("AEROPORTO")["sat"].mean()
    tab["sat_sem"] = df[~df.classe.isin(["generico", "substantivo"])].groupby("AEROPORTO")["sat"].mean()
    tab = tab.sort_values("pct_valido", ascending=False)
    tot = pd.Series({
        "n": len(df), "n_valido": len(valido), "n_subst": (df.classe == "substantivo").sum(),
        "n_ruido": (df.classe == "ruido").sum(),
        "pct_valido": 100 * len(valido) / len(df), "pct_subst": 100 * (df.classe == "substantivo").mean(),
        "palavras_med": valido.loc[valido.classe == "substantivo", "n_palavras"].median(),
        "sat_com": df.loc[df.classe == "substantivo", "sat"].mean(),
        "sat_sem": df.loc[~df.classe.isin(["generico", "substantivo"]), "sat"].mean(),
    }, name="Total")
    tab_out = pd.concat([tab, tot.to_frame().T])
    tab_out.to_csv(SAIDA / "descritiva_por_aeroporto.csv")
    print(tab_out.round(2).to_string())

    # Cobertura por ano (Brasil e aeroporto) — formato longo
    df["valido"] = df.classe.isin(["generico", "substantivo"])
    df["subst"] = df.classe == "substantivo"
    por_ano = []
    for chave, g in [("BRASIL", df)] + list(df.groupby("AEROPORTO")):
        a = g.groupby("ANO").agg(n=("valido", "size"), n_valido=("valido", "sum"), n_subst=("subst", "sum"),
                                 n_vazio=("classe", lambda c: (c == "vazio").sum()))
        a["pct_valido"] = 100 * a.n_valido / a.n
        a["pct_subst"] = 100 * a.n_subst / a.n
        a["pct_vazio"] = 100 * a.n_vazio / a.n
        por_ano.append(a.reset_index().assign(aeroporto=chave))
    por_ano = pd.concat(por_ano)[["aeroporto", "ANO", "n", "n_valido", "n_subst", "n_vazio", "pct_valido", "pct_subst", "pct_vazio"]]
    por_ano.to_csv(SAIDA / "descritiva_por_ano.csv", index=False)
    print(por_ano[por_ano.aeroporto == "BRASIL"].round(1).to_string(index=False))

    # LaTeX
    linhas = []
    for ap, r in tab_out.iterrows():
        nome = f"{ap} ({NOMES[ap]})" if ap in NOMES else r"\textbf{Total}"
        linhas.append(f"{nome} & {int(r.n):,} & {int(r.n_valido):,} & {fmt(r.pct_valido,1)} & {int(r.n_subst):,} & "
                      f"{fmt(r.pct_subst,1)} & {fmt(r.palavras_med,0)} & {fmt(r.sat_com)} & {fmt(r.sat_sem)} \\\\".replace(",", "."))
        linhas[-1] = linhas[-1].replace("{.}", "{,}")
    tex = r"""\begin{table}[htbp]
\centering
\caption{Cobertura dos comentários abertos por aeroporto (MPor, """ + ANOS_TEX + r""", 21 trimestres). ``Válidos'' excluem respostas vazias e ruído de preenchimento (``Não'', ``N'', ``Sim'' isolado, dígitos isolados); ``substantivos'' excluem, adicionalmente, elogios ou avaliações genéricas sem conteúdo temático (``tudo ótimo''). Satisfação geral média (1--5) entre respondentes com comentário substantivo e sem comentário válido.}
\label{tab:nlp-descritiva}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
Aeroporto & $N$ & Válidos & \% & Subst. & \% & Palavras (med.) & Sat. c/ com. & Sat. s/ com. \\
\midrule
""" + "\n".join(linhas[:-1]) + "\n\\midrule\n" + linhas[-1] + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TEX / "nlp_descritiva.tex").write_text(tex)

    # Figura
    estilo()
    t = tab.sort_values("pct_valido")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    y = np.arange(len(t))
    ax.barh(y, t.pct_valido, color="#9ecae1", label="Comentário válido (inclui genérico)")
    ax.barh(y, t.pct_subst, color="#3182bd", label="Comentário substantivo")
    ax.set_yticks(y, [f"{a} ({NOMES[a]})" for a in t.index], fontsize=8)
    ax.set_xlabel("% dos respondentes")
    ax.set_title(f"Cobertura dos comentários abertos por aeroporto, {ANOS_ROT}")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIG / "nlp_cobertura_comentarios.pdf")


if __name__ == "__main__":
    main()
