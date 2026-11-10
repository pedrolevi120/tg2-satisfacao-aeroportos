"""
SO3 (TG-2) — Comparação sistemática de LLMs locais (Ollama) na geração dos
relatórios executivos dos 3 aeroportos do MVP (GRU/CGH/VCP).

Desenho: 4 modelos × 3 aeroportos × 2 repetições = 24 relatórios, todos com o
MESMO prompt (codigos/04_modelos_ml/app/prompts.py), temperatura 0,4,
num_predict 2200 e semente 42 + repetição.

Métricas por relatório (ver relatorios_avaliacao.py):
  (a) fidelidade numérica automática; (b) cobertura das 5 prioridades;
  (c) tempo de geração e tokens/s; (d) erros de PT-BR (LanguageTool);
  (e) nota qualitativa 1–5 por rubrica (coerência, acionabilidade, tom formal),
      lida de relatorios_comparacao/avaliacao_qualitativa.csv (preenchida à mão
      após leitura dos 24 textos cegos quanto ao modelo — ver RUBRICA abaixo).

Uso:
  python codigos/05_interpretacao/03_comparar_llms.py            # gera + avalia
  python codigos/05_interpretacao/03_comparar_llms.py --agregar  # só reavalia/agrega

Saídas em codigos/05_interpretacao/relatorios_comparacao/:
  textos/<modelo>__<ICAO>__r<k>.md / .json   relatório + metadados
  metricas_por_relatorio.csv                  uma linha por relatório
  resumo_por_modelo.csv                       médias por modelo
  numeros_inventados.csv                      números sem lastro no prompt
  cegos/                                      textos anonimizados p/ avaliação (e)
e a tabela LaTeX texto/_paralelo/tex/llm_mvp_comparacao.tex.
"""
import argparse
from datetime import datetime
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codigos" / "04_modelos_ml" / "app"))
sys.path.insert(0, str(HERE))

from contexto import contexto_20, top_acionaveis  # noqa: E402
from llm_client import gerar_relatorio_meta, modelo_disponivel  # noqa: E402
from prompts import montar_prompt  # noqa: E402
import relatorios_avaliacao as av  # noqa: E402

MODELOS = ["llama3.1:8b", "mistral:7b", "qwen2.5:7b", "gemma2:9b"]
NOME_MODELO = {"llama3.1:8b": "Llama 3.1 8B", "mistral:7b": "Mistral 7B",
               "qwen2.5:7b": "Qwen 2.5 7B", "gemma2:9b": "Gemma 2 9B"}
AEROPORTOS = ["SBGR", "SBSP", "SBKP"]
CTX, _ = contexto_20()
REPETICOES = 2
SEED = 42
RANKING = HERE / "ranking_prioridades_por_aeroporto.csv"
OUT = HERE / "relatorios_comparacao"
TXT = OUT / "textos"
CEGOS = OUT / "cegos"
QUALI = OUT / "avaliacao_qualitativa.csv"
TEX = ROOT / "texto" / "_paralelo" / "tex" / "llm_mvp_comparacao.tex"

RUBRICA = """Rubrica (e) — nota inteira de 1 a 5 em cada critério:
COERÊNCIA   5: argumentação lógica, sem contradições, cada prioridade ligada ao contexto do aeroporto
            3: coerente no geral, com trechos genéricos ou repetitivos
            1: contradições, afirmações falsas sobre o aeroporto ou texto desconexo
ACIONABILIDADE 5: ações específicas, com prazo (curto/médio) e KPI mensurável por prioridade
               3: ações plausíveis porém genéricas ("investir em", "melhorar")
               1: sem ações concretas
TOM FORMAL  5: registro técnico-formal PT-BR, sem adjetivos vagos, sem mistura de idiomas
            3: formal com deslizes (coloquialismos, anglicismos, adjetivos vagos)
            1: informal ou em outro idioma
"""


def slug(modelo):
    return modelo.replace(":", "-").replace(".", "")


def prompt_de(icao, ranking):
    c = CTX[icao]
    top = top_acionaveis(ranking, icao)
    p = montar_prompt(
        nome_completo=c["nome"], icao=icao, cidade=c["cidade"], uf=c["uf"],
        regiao=c["regiao"], pax_mi=c["pax_mi"], voos_k=c["voos_k"],
        regime=c["regime"], concessionaria=c["concessionaria"],
        n=c["n"], media=c["media"], posicao=c["posicao"],
        pct_4_5=c["pct_4_5"], top_atributos=top)
    return p, [t["feature"] for t in top]


def gerar(ranking):
    TXT.mkdir(parents=True, exist_ok=True)
    for modelo in MODELOS:
        if not modelo_disponivel(modelo):
            print(f"\n### {modelo}: indisponível no Ollama (ollama pull {modelo}); pulando")
            continue
        print(f"\n### {modelo}: aquecimento (carga do modelo fora da medição)")
        gerar_relatorio_meta("Responda apenas: ok.", modelo=modelo, max_tokens=5)
        for icao in AEROPORTOS:
            prompt, _ = prompt_de(icao, ranking)
            for rep in range(1, REPETICOES + 1):
                base = TXT / f"{slug(modelo)}__{icao}__r{rep}"
                if base.with_suffix(".json").exists():
                    print(f"  {base.name}: já existe, pulando")
                    continue
                meta = gerar_relatorio_meta(prompt, modelo=modelo, seed=SEED + rep)
                meta.update({"modelo": modelo, "aeroporto": icao, "rep": rep,
                             "seed": SEED + rep, "temperatura": 0.4, "prompt": prompt})
                base.with_suffix(".md").write_text(meta["texto"])
                base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
                print(f"  {base.name}: {meta['tokens_saida']} tok em {meta['tempo_s']:.1f}s "
                      f"({meta['tokens_por_s']:.1f} tok/s)")


def benchmark(ranking, n=3, tokens=400):
    """Velocidade isolada: n gerações de `tokens` tokens por modelo (prompt do SBGR),
    registrando a carga média do sistema (os tempos da geração principal sofreram
    concorrência de outros processos da máquina)."""
    import os
    prompt, _ = prompt_de("SBGR", ranking)
    linhas = []
    for modelo in MODELOS:
        gerar_relatorio_meta("Responda apenas: ok.", modelo=modelo, max_tokens=5)
        for i in range(n):
            m = gerar_relatorio_meta(prompt, modelo=modelo, max_tokens=tokens, seed=SEED + i)
            linhas.append({"modelo": modelo, "i": i, "tokens_saida": m["tokens_saida"],
                           "tokens_por_s": m["tokens_por_s"], "tempo_s": m["tempo_s"],
                           "carga_1min": os.getloadavg()[0]})
            print(linhas[-1])
    pd.DataFrame(linhas).to_csv(OUT / "bench_velocidade.csv", index=False)


def popular_db(ranking):
    """Insere os relatórios da comparação no SQLite do MVP (aba Histórico), sem duplicar."""
    from db_init import init_db, salvar_relatorio, listar_relatorios
    init_db()
    ja = {r["relatorio_texto"] for r in listar_relatorios(limit=100000)}
    n = 0
    for f in sorted(TXT.glob("*.json")):
        m = json.loads(f.read_text())
        if m["texto"] in ja:
            continue
        salvar_relatorio(aeroporto=m["aeroporto"], aeroporto_nome=CTX[m["aeroporto"]]["nome_curto"],
                         modelo_llm=m["modelo"], top_atributos=top_acionaveis(ranking, m["aeroporto"]),
                         contexto_operacional=CTX[m["aeroporto"]], relatorio_texto=m["texto"],
                         tempo_geracao_s=m["tempo_s"],
                         data_geracao=datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"))
        n += 1
    print(f"SQLite: {n} relatórios inseridos")


def exportar_cegos(df):
    """Cópias anonimizadas (ordem embaralhada, seed fixa) para a avaliação (e)."""
    CEGOS.mkdir(parents=True, exist_ok=True)
    ids = list(df["id"])
    random.Random(SEED).shuffle(ids)
    chave = []
    for i, rid in enumerate(ids, 1):
        cod = f"R{i:02d}"
        (CEGOS / f"{cod}.md").write_text((TXT / f"{rid}.md").read_text())
        chave.append({"codigo": cod, "id": rid})
    pd.DataFrame(chave).to_csv(CEGOS / "_chave.csv", index=False)
    (CEGOS / "_RUBRICA.txt").write_text(RUBRICA)


def avaliar(ranking):
    linhas, invent = [], []
    for f in sorted(TXT.glob("*.json")):
        m = json.loads(f.read_text())
        _, feats = prompt_de(m["aeroporto"], ranking)
        texto, prompt = m["texto"], m["prompt"]
        fid = av.fidelidade_numerica(texto, prompt)
        cob = av.cobertura_prioridades(texto, feats)
        est = av.estrutura(texto)
        ass = av.associacao_shap(texto, top_acionaveis(ranking, m["aeroporto"]))
        err = av.erros_ptbr(texto, prompt)
        for x in fid.pop("inventados"):
            invent.append({"id": f.stem, "modelo": m["modelo"], "aeroporto": m["aeroporto"], **x})
        linhas.append({
            "id": f.stem, "modelo": m["modelo"], "aeroporto": m["aeroporto"], "rep": m["rep"],
            "tempo_s": m["tempo_s"], "tokens_saida": m["tokens_saida"],
            "tokens_por_s": m["tokens_por_s"],
            "tempo_decod_s": m["tokens_saida"] / m["tokens_por_s"],
            "truncado": m["tokens_saida"] >= 2200, **fid,
            "cobertura": cob["cobertura"], "nao_cobertos": "; ".join(cob["nao_cobertos"]),
            **est, **ass, **{k: v for k, v in err.items() if k != "erros_exemplos"},
            "erros_exemplos": json.dumps(err["erros_exemplos"], ensure_ascii=False),
            "sha1_texto": hashlib.sha1(texto.encode()).hexdigest()[:10],
        })
    df = pd.DataFrame(linhas)
    pd.DataFrame(invent).to_csv(OUT / "numeros_inventados.csv", index=False)
    return df


def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", "{,}")


def tabela_latex(res, soma, tem_quali):
    """Tabela transposta: linhas = métricas (agrupadas), colunas = modelos."""
    vel = "tokens_por_s_bench" if "tokens_por_s_bench" in res else "tokens_por_s"
    pct = lambda v: fmt(100 * v, 0)
    grupos = [
        ("Fidelidade e cobertura", [
            ("Números citados (média)", "num_total", lambda v: fmt(v, 1), res),
            ("Números sem lastro (total)", "num_inventado", lambda v: fmt(v, 0), soma),
            ("Fidelidade numérica (\\%)", "fidelidade", lambda v: fmt(100 * v, 1), res),
            ("SHAP no atributo correto (\\%)", "shap_assoc", pct, res),
            ("Prioridades mencionadas (\\%)", "cobertura", pct, res)]),
        ("Aderência às instruções", [
            ("Seções presentes (de 6)", "n_secoes", lambda v: fmt(v, 1), res),
            ("Frase de encerramento (\\%)", "frase_final", pct, res),
            ("Entre 700 e 1000 palavras (\\%)", "extensao_ok", pct, res),
            ("Palavras (média)", "palavras", lambda v: fmt(v, 0), res),
            ("Truncados (total)", "truncado", lambda v: fmt(v, 0), soma)]),
        ("Língua portuguesa", [
            ("Erros por mil palavras", "erros_por_mil", lambda v: fmt(v, 1), res)]),
        ("Custo computacional", [
            ("Tokens gerados (média)", "tokens_saida", lambda v: fmt(v, 0), res),
            ("Tokens/s" + (" (\\textit{benchmark})" if vel.endswith("bench") else ""),
             vel, lambda v: fmt(v, 1), res),
            ("Tempo de decodificação (s)", "tempo_decod_s", lambda v: fmt(v, 0), res)]),
    ]
    if tem_quali:
        grupos.append(("Avaliação qualitativa cega (1 a 5)", [
            ("Coerência", "coerencia", lambda v: fmt(v, 1), res),
            ("Acionabilidade", "acionabilidade", lambda v: fmt(v, 1), res),
            ("Tom formal", "tom", lambda v: fmt(v, 1), res)]))
    modelos = list(res.index)
    linhas = []
    for g, itens in grupos:
        linhas.append(f"\\multicolumn{{{len(modelos) + 1}}}{{l}}{{\\textit{{{g}}}}} \\\\")
        for rot, col, f, tab in itens:
            linhas.append("\\quad " + rot + " & " + " & ".join(f(tab.loc[m, col]) for m in modelos) + " \\\\")
        linhas.append("\\addlinespace")
    corpo = "\n".join(linhas[:-1])
    head = " & ".join(["Métrica"] + [NOME_MODELO[m] for m in modelos])
    return f"""% Gerado por codigos/05_interpretacao/03_comparar_llms.py — não editar à mão
\\begin{{table}}[htbp]
\\centering
\\caption{{Comparação de LLMs locais (Ollama) na geração dos relatórios executivos dos três aeroportos do MVP (GRU, CGH e VCP). Médias de 6 relatórios por modelo (3 aeroportos $\\times$ 2 repetições), com o mesmo prompt e temperatura 0{{,}}4.}}
\\label{{tab:llm_mvp-comparacao}}
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{'r' * len(modelos)}}}
\\toprule
{head} \\\\
\\midrule
{corpo}
\\bottomrule
\\end{{tabular}}
\\par\\smallskip
\\parbox{{0.95\\linewidth}}{{\\footnotesize Fonte: elaboração própria. Números sem lastro: números do texto (excluída a numeração de seções) que não coincidem, nem por arredondamento nem como diferença entre valores fornecidos, com números do prompt. Erros de língua: LanguageTool pt-BR, sem regras de estilo e tipografia. Velocidade medida em Apple M4 (24 GB){', em execução isolada (3 gerações de 400 tokens por modelo)' if vel.endswith('bench') else ''}; o tempo de decodificação é tokens gerados divididos pela velocidade observada durante a geração. Notas qualitativas atribuídas às cegas, por rubrica explícita (ver texto).}}
\\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agregar", action="store_true", help="não gera, só avalia")
    ap.add_argument("--bench", action="store_true", help="só benchmark de velocidade")
    args = ap.parse_args()
    ranking = pd.read_csv(RANKING)
    if args.bench:
        return benchmark(ranking)
    if not args.agregar:
        gerar(ranking)
    df = avaliar(ranking)
    popular_db(ranking)
    if not (CEGOS / "_chave.csv").exists():
        exportar_cegos(df)
    tem_quali = QUALI.exists()
    if tem_quali:
        q = pd.read_csv(QUALI).merge(pd.read_csv(CEGOS / "_chave.csv"), on="codigo")
        df = df.merge(q[["id", "coerencia", "acionabilidade", "tom", "justificativa"]],
                      on="id", how="left")
    df.to_csv(OUT / "metricas_por_relatorio.csv", index=False)
    num = [c for c in ["fidelidade", "num_total", "num_inventado", "shap_citados", "shap_assoc", "cobertura", "n_secoes",
                       "frase_final", "truncado", "palavras", "extensao_ok", "tempo_s", "tempo_decod_s", "tokens_saida",
                       "tokens_por_s", "erros_orto", "erros_gram", "erros_por_mil",
                       "coerencia", "acionabilidade", "tom"] if c in df]
    res = df.groupby("modelo")[num].mean().reindex([m for m in MODELOS if m in set(df.modelo)])
    dp = df.groupby("modelo")[num].std().add_suffix("_dp")
    bench = OUT / "bench_velocidade.csv"
    if bench.exists():
        b = pd.read_csv(bench).groupby("modelo")
        res["tokens_por_s_bench"] = b["tokens_por_s"].mean()
        res["carga_bench"] = b["carga_1min"].mean()
    res.join(dp).to_csv(OUT / "resumo_por_modelo.csv")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    soma = df.groupby("modelo")[["num_inventado", "truncado"]].sum()
    TEX.write_text(tabela_latex(res, soma, tem_quali))
    pd.set_option("display.width", 200)
    print(res.round(3).T)
    print(f"\n→ {OUT.relative_to(ROOT)} e {TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
