"""
Aba "Comentários" do MVP: resultados do pacote NLP (T7) sobre os comentários
abertos da MPor, série 2021T1–2026T2 (sentimento, temas críticos e triangulação
com o SHAP). Se as saídas do NLP não existirem, a aba é omitida.

As tabelas por aeroporto são as da série inteira. Se alguma vier com colunas ANO/PERIODO
(linhas por ano), só as linhas da série inteira (ANO/PERIODO vazio ou "total") são usadas;
os comentários individuais mostram o período em que foram escritos.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NLP = ROOT / "codigos" / "08_nlp" / "saida"
ARQS = {"agregado": NLP / "agregado_aeroporto_tema.csv",
        "sentimento": NLP / "sentimento_por_aeroporto.csv",
        "triang": NLP / "triangulacao_por_aeroporto.csv",
        "descr": NLP / "descritiva_por_aeroporto.csv",
        "coment": NLP / "comentarios_topicos.parquet"}
COBERTURA = {"coberto": "coberto pelo questionário", "parcial": "coberto parcialmente",
             "nao_coberto": "NÃO coberto pelo questionário", "geral": "geral"}


TOTAL = {"total", "todos", "serie", "série", "2021–2026", "2021-2026", "2021T1–2026T2"}


def sem_ano(df, chave):
    """Uma linha por `chave` (aeroporto, ou aeroporto + tema): descarta linhas por ano/período
    (ANO/PERIODO preenchidos) quando a tabela também traz a linha da série inteira, e a linha
    BRASIL. Tabelas sem ANO/PERIODO passam intactas."""
    chave = [chave] if isinstance(chave, str) else list(chave)
    df = df[df[chave[0]].astype(str).str.upper() != "BRASIL"]
    for col in ("ANO", "PERIODO"):
        if col in df.columns and df.duplicated(chave).any():
            tot = df[col].isna() | df[col].astype(str).str.strip().str.lower().isin(TOTAL)
            if not tot.any():
                raise ValueError(f"tabela do NLP só tem linhas por {col}; falta a linha da série inteira")
            df = df[tot]
    return df


def carregar():
    if not all(p.exists() for p in ARQS.values()):
        return None
    d = {k: (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
         for k, p in ARQS.items()}
    d["descr"] = sem_ano(d["descr"].rename(columns={"Unnamed: 0": "aeroporto"}), "aeroporto").set_index("aeroporto")
    d["sentimento"] = sem_ano(d["sentimento"], "AEROPORTO").set_index("AEROPORTO")
    d["triang"] = sem_ano(d["triang"], "aeroporto").set_index("aeroporto")
    d["agregado"] = sem_ano(d["agregado"], ["aeroporto", "tema"])
    return d


def _br(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def render(d, icao, nome):
    import altair as alt
    import streamlit as st

    st.subheader(f"O que os passageiros escrevem — {nome}")
    sen, desc = d["sentimento"].loc[icao], d["descr"].loc[icao]
    br = d["agregado"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Comentários válidos", f"{int(desc['n_valido']):,}".replace(",", "."),
              help=f"{_br(desc['pct_valido'])}% dos respondentes deixaram comentário substantivo")
    c2.metric("% com sentimento negativo", f"{_br(sen['pct_neg'])}%",
              f"{_br(sen['pct_neg'] - d['sentimento']['pct_neg'].mean())} p.p. vs. média dos 20",
              delta_color="inverse")
    c3.metric("Escore médio de sentimento", _br(sen["escore_medio"], 2),
              help="XLM-RoBERTa; −1 (negativo) a +1 (positivo)")

    ag = br[(br["aeroporto"] == icao) & (~br["cobertura"].isin(["sem_tema", "geral"]))]
    ag = ag.sort_values("criticos_por_mil", ascending=False).head(10).copy()
    ag["cobertura_txt"] = ag["cobertura"].map(COBERTURA)
    col_g, _ = st.columns([4, 1])
    with col_g:
        st.markdown("**Temas críticos mais citados** (comentários críticos por mil respondentes)")
        graf = alt.Chart(ag).mark_bar().encode(
            x=alt.X("criticos_por_mil:Q", title="Críticos por mil respondentes",
                    axis=alt.Axis(labelExpr="replace(format(datum.value, '.0f'), '.', ',')")),
            y=alt.Y("tema_rotulo:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=340, labelOverlap=False)),
            color=alt.Color("cobertura_txt:N", title=None,
                            legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=0),
                            scale=alt.Scale(domain=list(COBERTURA.values())[:3],
                                            range=["#1f4e79", "#7fa7cf", "#c0504d"])),
            tooltip=[alt.Tooltip("tema_rotulo:N", title="Tema"),
                     alt.Tooltip("n_comentarios:Q", title="Comentários"),
                     alt.Tooltip("pct_critico:Q", title="% críticos", format=".1f"),
                     alt.Tooltip("sentimento_medio:Q", title="Sentimento", format=".2f")],
        ).properties(height=34 * len(ag) + 40)
        st.altair_chart(graf, width="stretch")
    tr = d["triang"].loc[icao]
    with st.container():
        st.markdown("**Triangulação SHAP × comentários**")
        st.dataframe(pd.DataFrame({
            "Top 5 SHAP": [x.capitalize() for x in tr["top5_shap"].split(" | ")],
            "Top 5 temas críticos (NLP)": tr["top5_temas_criticos"].split(" | "),
        }), width="stretch", hide_index=True,
            column_config={"Top 5 SHAP": st.column_config.TextColumn(width="medium"),
                           "Top 5 temas críticos (NLP)": st.column_config.TextColumn(width="large")})
        st.caption(f"Correlação de Spearman entre as ordenações NLP e SHAP nos atributos "
                   f"comuns: ρ = {_br(tr['rho_nlp_shap'], 2)} (p = {_br(tr['p_nlp_shap'], 3)}). "
                   f"Tema crítico mais citado **não coberto** pelo questionário: "
                   f"*{tr['top_nao_coberto']}*.")

    st.markdown("**Exemplos de comentários críticos**")
    temas = ag["tema_rotulo"].tolist()
    if temas:
        tema = st.selectbox("Tema", temas, key="tema_coment")
        cm = d["coment"]
        ex = cm[(cm["AEROPORTO"] == icao) & (cm["tema_rotulo"] == tema) & (cm["critico"])]
        ex = ex.sample(min(5, len(ex)), random_state=42) if len(ex) else ex
        for _, r in ex.iterrows():
            quando = f" · {r['PERIODO']}" if "PERIODO" in r and pd.notna(r["PERIODO"]) else ""
            st.markdown(f"> {r['texto']}  \n<small>{r['PROCESSO']}{quando} · nota geral "
                        f"{int(r['SATISFACAO_GERAL']) if pd.notna(r['SATISFACAO_GERAL']) else 'n/d'}"
                        f"</small>", unsafe_allow_html=True)
