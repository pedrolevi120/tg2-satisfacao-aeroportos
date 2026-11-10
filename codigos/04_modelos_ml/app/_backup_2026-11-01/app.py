"""
MVP Streamlit — Framework de priorização de melhorias em aeroportos brasileiros.
Versão TG-2: 20 aeroportos da MPor 2023, seletor de LLM e aba de comentários.

Uso:
    cd tg2-satisfacao-aeroportos   # raiz do repositório
    source .venv/bin/activate
    streamlit run codigos/04_modelos_ml/app/app.py
"""
import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from contexto import (contexto_20, carregar_ranking, top_acionaveis,
                      PREFIXOS_NAO_ACIONAVEIS)
from db_init import init_db, salvar_relatorio, listar_relatorios
from llm_client import gerar_relatorio_stream, modelo_disponivel, MODELO_DEFAULT
from prompts import montar_prompt
import comentarios

ROOT = Path(__file__).resolve().parents[3]
MODELOS_LLM = ["llama3.1:8b", "mistral:7b", "qwen2.5:7b", "gemma2:9b"]
MODELO_PADRAO = "gemma2:9b"   # melhor avaliado na comparação (03_comparar_llms.py)
NOME_MODELO = {"llama3.1:8b": "Llama 3.1 8B", "mistral:7b": "Mistral 7B",
               "qwen2.5:7b": "Qwen 2.5 7B", "gemma2:9b": "Gemma 2 9B"}


QUADRANTES = ["Corrigir primeiro", "Manter", "Acompanhar", "Baixa prioridade"]
CORES_QUADRANTE = ["#c0392b", "#1f4e79", "#e08e0b", "#9aa5b1"]


def nome_legivel(atributo: str) -> str:
    """'ORGANIZAÇÃO DAS FILAS2' -> 'Organização das filas' (nomes da planilha MPor)."""
    return atributo.rstrip("0123456789").strip().capitalize()


def nota_nacional(ranking: pd.DataFrame) -> pd.Series:
    """Nota média nacional de cada dimensão, ponderada pelo nº de respostas."""
    r = ranking.dropna(subset=["nota_media", "n_respostas"])
    soma = (r["nota_media"] * r["n_respostas"]).groupby(r["feature"]).sum()
    return soma / r.groupby("feature")["n_respostas"].sum()


@st.cache_data
def dados_base():
    ctx, media_nac = contexto_20()
    ranking, fonte = carregar_ranking()
    return ctx, media_nac, ranking, fonte


def main():
    st.set_page_config(page_title="TG — Priorização Aeroportuária",
                       page_icon="✈️", layout="wide")
    init_db()
    CTX, media_nac, ranking, fonte_rank = dados_base()
    com = comentarios.carregar()

    st.markdown("""<style>
      .block-container {padding-top: 2rem;}
      h1 {font-size: 1.9rem !important; padding-bottom: .2rem !important;}
    </style>""", unsafe_allow_html=True)
    st.title("✈️ Priorização data-driven de melhorias em aeroportos brasileiros")
    st.caption("Ferramenta do TG — Pedro Levi Sales Cavalcante (ITA, Civil-Aeronáutica) · "
               f"MPor 2023 · {len(CTX)} aeroportos")

    # ---------- Sidebar ----------
    aeros_rank = sorted(ranking["aeroporto"].unique())
    with st.sidebar:
        st.header("Configuração")
        opcoes = [a for a in CTX if a in aeros_rank]   # ordem do ranking de satisfação
        icao = st.selectbox(
            "Aeroporto", options=opcoes,
            index=opcoes.index("SBGR") if "SBGR" in opcoes else 0,
            format_func=lambda k: f"{CTX[k]['posicao']:>2}º · {CTX[k]['nome_curto']} — {CTX[k]['uf']}")
        disp = [m for m in MODELOS_LLM if modelo_disponivel(m)] or [MODELO_DEFAULT]
        modelo = st.selectbox("Modelo LLM (Ollama)", options=disp,
                              index=disp.index(MODELO_PADRAO) if MODELO_PADRAO in disp else 0,
                              format_func=lambda m: f"{NOME_MODELO.get(m, m)}  ·  {m}")
        gerar = st.button("Gerar relatório", type="primary", width="stretch")
        st.divider()
        st.header("Status do LLM")
        if modelo_disponivel(modelo):
            st.success(f"✅ {modelo} pronto")
        else:
            st.warning(f"⚠️ {modelo} não disponível.\n\nRode: `ollama pull {modelo}`")
        st.divider()
        st.caption(f"Ranking SHAP: {'modelo dos 20 aeroportos' if fonte_rank == '20' else 'MVP (3 aeroportos)'}")

    ctx = CTX[icao]
    top5 = top_acionaveis(ranking, icao)

    # ---------- Indicadores ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satisfação média", f"{ctx['media']:.2f} / 5".replace(".", ","),
              f"{ctx['media'] - media_nac:+.2f}".replace(".", ",") + " vs. média nacional")
    c2.metric("Posição nacional", f"{ctx['posicao']}º / {len(CTX)}")
    c3.metric("Respondentes (2023)", f"{ctx['n']:,}".replace(",", "."))
    c4.metric("% notas 4–5", f"{ctx['pct_4_5']:.1f}%".replace(".", ","))
    st.divider()

    abas = ["Prioridades (SHAP)", "Relatório LLM", "Histórico"]
    if com is not None:
        abas.append("Comentários")
    tabs = st.tabs(abas)

    with tabs[0]:
        st.subheader(f"Prioridades de melhoria — {ctx['nome_curto']}")
        st.caption("Prioridade = importância do atributo para a satisfação geral (SHAP) "
                   "combinada com o desempenho do aeroporto nele, comparado à média nacional.")
        dims = ranking[ranking["aeroporto"] == icao].sort_values("rank")
        dims = dims[~dims["feature"].str.upper().str.startswith(PREFIXOS_NAO_ACIONAVEIS)].head(10)
        dims = dims.assign(nome=dims["feature"].map(nome_legivel),
                           nota_nac=dims["feature"].map(nota_nacional(ranking)))
        dims["gap"] = dims["nota_media"] - dims["nota_nac"]
        corte = dims["shap_abs_medio"].median()
        dims["quadrante"] = [
            ("Corrigir primeiro" if g < 0 else "Manter") if imp >= corte
            else ("Acompanhar" if g < 0 else "Baixa prioridade")
            for imp, g in zip(dims["shap_abs_medio"], dims["gap"])]
        escala = alt.Scale(domain=QUADRANTES, range=CORES_QUADRANTE)
        lim = max(0.3, float(dims["gap"].abs().max()) * 1.15)
        col_g, col_c = st.columns([3, 2])
        with col_g:
            base = alt.Chart(dims).encode(
                x=alt.X("gap:Q", scale=alt.Scale(domain=[-lim, lim]),
                        title="Desempenho: nota do aeroporto − média nacional",
                        axis=alt.Axis(labelExpr="replace(format(datum.value, '+.1f'), '.', ',')")),
                y=alt.Y("shap_abs_medio:Q", title="Importância (|SHAP| médio)",
                        axis=alt.Axis(labelExpr="replace(format(datum.value, '.2f'), '.', ',')")))
            pontos = base.mark_circle(size=170, opacity=0.9).encode(
                color=alt.Color("quadrante:N", scale=escala,
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=[alt.Tooltip("nome:N", title="Dimensão"),
                         alt.Tooltip("shap_abs_medio:Q", title="Importância", format=".3f"),
                         alt.Tooltip("nota_media:Q", title="Nota local", format=".2f"),
                         alt.Tooltip("nota_nac:Q", title="Nota nacional", format=".2f"),
                         alt.Tooltip("quadrante:N", title="Leitura")])
            # rótulo à direita do ponto; perto da borda direita, à esquerda dele
            importantes = base.transform_filter(alt.datum.shap_abs_medio >= corte)
            rotulos = (importantes.transform_filter(alt.datum.gap < lim * 0.25)
                       .mark_text(align="left", dx=10, fontSize=12).encode(text="nome:N")
                       + importantes.transform_filter(alt.datum.gap >= lim * 0.25)
                       .mark_text(align="right", dx=-10, fontSize=12).encode(text="nome:N"))
            linha_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="#666").encode(x="x:Q")
            linha_y = alt.Chart(pd.DataFrame({"y": [corte]})).mark_rule(strokeDash=[4, 4], color="#666").encode(y="y:Q")
            st.altair_chart((linha_x + linha_y + pontos + rotulos).properties(height=430),
                            width="stretch")
        with col_c:
            urgentes = dims[dims["quadrante"] == "Corrigir primeiro"]
            st.markdown("**Onde agir primeiro**")
            if urgentes.empty:
                st.success("Nas dimensões mais importantes, o aeroporto está na média "
                           "nacional ou acima dela.")
            for _, r in urgentes.iterrows():
                st.markdown(f"- **{r['nome']}**: nota {r['nota_media']:.2f} contra "
                            f"{r['nota_nac']:.2f} no país ({r['gap']:+.2f})".replace(".", ","))
            st.markdown("**Contexto operacional**")
            pax = f"{ctx['pax_mi']:.2f} milhões".replace(".", ",")
            voos = f"{ctx['voos_k']:.1f} mil".replace(".", ",")
            st.dataframe(pd.DataFrame([
                ("Cidade / UF", f"{ctx['cidade']} / {ctx['uf']}"),
                ("Região", ctx["regiao"]),
                ("Passageiros 2023", pax),
                ("Decolagens 2023", voos),
                ("Regime", ctx["regime"]),
                ("Operador", ctx["concessionaria"]),
            ], columns=["Campo", "Valor"]), width="stretch", hide_index=True)
        tabela = dims.assign(**{
            "Dimensão": dims["nome"], "Importância": dims["shap_abs_medio"],
            "Nota local": dims["nota_media"], "Nota nacional": dims["nota_nac"],
            "Diferença": dims["gap"], "% notas ≤ 3": 100 * dims["pct_nota_ate_3"],
            "Leitura": dims["quadrante"]})
        virgula = lambda f: (lambda v: f.format(v).replace(".", ","))
        st.dataframe(tabela[["Dimensão", "Importância", "Nota local", "Nota nacional",
                             "Diferença", "% notas ≤ 3", "Leitura"]].style.format({
            "Importância": virgula("{:.3f}"), "Nota local": virgula("{:.2f}"),
            "Nota nacional": virgula("{:.2f}"), "Diferença": virgula("{:+.2f}"),
            "% notas ≤ 3": virgula("{:.0f}%")}), width="stretch", hide_index=True)
        with st.expander("Top 5 enviado ao LLM"):
            st.dataframe(pd.DataFrame(top5).rename(columns={"feature": "Atributo",
                         "shap_abs": "|SHAP|"}).style.format(
                             {"|SHAP|": lambda v: f"{v:.3f}".replace(".", ",")}),
                         width="stretch", hide_index=True)

    with tabs[1]:
        prompt = montar_prompt(
            nome_completo=ctx["nome"], icao=icao, cidade=ctx["cidade"], uf=ctx["uf"],
            regiao=ctx["regiao"], pax_mi=ctx["pax_mi"], voos_k=ctx["voos_k"], regime=ctx["regime"],
            concessionaria=ctx["concessionaria"], n=ctx["n"], media=ctx["media"],
            posicao=ctx["posicao"], pct_4_5=ctx["pct_4_5"], top_atributos=top5)
        gerado_agora = False
        if gerar:
            if not modelo_disponivel(modelo):
                st.error(f"Modelo {modelo} não disponível. Rode `ollama pull {modelo}`.")
            else:
                st.caption(f"Gerando com **{NOME_MODELO.get(modelo, modelo)}** no computador "
                           "local; o texto aparece enquanto o modelo escreve.")
                t0 = time.time()
                with st.container(border=True):
                    texto = st.write_stream(gerar_relatorio_stream(prompt, modelo=modelo))
                tempo = time.time() - t0
                salvar_relatorio(aeroporto=icao, aeroporto_nome=ctx["nome_curto"],
                                 modelo_llm=modelo, top_atributos=top5,
                                 contexto_operacional=ctx, relatorio_texto=texto,
                                 tempo_geracao_s=tempo)
                st.success(f"Relatório gerado em {tempo:.0f} s e salvo no histórico.")
                gerado_agora = True
        ultimos = listar_relatorios(aeroporto=icao, limit=50)
        ultimo = next((r for r in ultimos if r["modelo_llm"] == modelo), None)
        if gerado_agora:
            pass
        elif ultimo:
            st.caption(f"Relatório mais recente com **{NOME_MODELO.get(modelo, modelo)}** · "
                       f"{ultimo['data_geracao'].replace('T', ' ')} · "
                       f"{ultimo['tempo_geracao_s']:.0f} s de geração")
            with st.container(border=True):
                st.markdown(ultimo["relatorio_texto"])
        else:
            st.info("Nenhum relatório deste aeroporto com este modelo. Clique em "
                    "**Gerar relatório** na barra lateral.")
        with st.expander("Ver prompt enviado ao LLM"):
            st.code(prompt, language="markdown")

    with tabs[2]:
        st.subheader("Histórico de relatórios gerados")
        cf1, cf2 = st.columns(2)
        filtro_a = cf1.checkbox(f"Apenas {icao}", value=False)
        filtro_m = cf2.multiselect("Modelos", MODELOS_LLM, default=[], placeholder="Todos os modelos",
                                   format_func=lambda m: NOME_MODELO.get(m, m))
        rel = listar_relatorios(aeroporto=icao if filtro_a else None, limit=500)
        if filtro_m:
            rel = [r for r in rel if r["modelo_llm"] in filtro_m]
        if not rel:
            st.info("Nenhum relatório gerado ainda.")
        else:
            tab = pd.DataFrame(rel)
            resumo = tab.assign(
                modelo=tab["modelo_llm"].map(lambda m: NOME_MODELO.get(m, m)),
                palavras=tab["relatorio_texto"].str.split().str.len(),
                data=tab["data_geracao"].str.replace("T", " "))
            st.dataframe(resumo[["data", "aeroporto", "aeroporto_nome", "modelo",
                                 "tempo_geracao_s", "palavras"]].rename(columns={
                "data": "Data", "aeroporto": "ICAO", "aeroporto_nome": "Aeroporto",
                "modelo": "Modelo", "tempo_geracao_s": "Tempo (s)", "palavras": "Palavras"}
            ).style.format({"Tempo (s)": "{:.0f}", "Palavras": lambda v: f"{v:,}".replace(",", ".")}),
                width="stretch",
                hide_index=True, height=260)
            st.caption(f"{len(rel)} relatórios")
            for r in rel[:15]:
                with st.expander(f"{r['aeroporto']} · {NOME_MODELO.get(r['modelo_llm'], r['modelo_llm'])}"
                                 f" · {r['data_geracao'].replace('T', ' ')}"):
                    st.markdown(r["relatorio_texto"])
                    st.json({"modelo": r["modelo_llm"],
                             "top_atributos": json.loads(r["top_atributos"])}, expanded=False)

    if com is not None:
        with tabs[3]:
            comentarios.render(com, icao, ctx["nome_curto"])


if __name__ == "__main__":
    main()
