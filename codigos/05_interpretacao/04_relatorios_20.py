"""
SO3 (TG-2) — Relatórios executivos dos 20 aeroportos da MPor, série 2021T1–2026T2.

Entradas:
  codigos/05_interpretacao/ranking_prioridades_20.csv   (pacote modelos20/T5)
  dados/processados/panorama_aeroportos_2021_2026.parquet (n, média, % 4–5, posição, por ano/período)
  codigos/02_preprocessamento/aeroportos_20_caracteristicas.csv (movimento ANAC 2025)
  dados/processados/contexto_externo_20.csv              (ANAC VRA/estatístico 2025, IBGE, Atlas)
  dados/processados/contexto_aeroportos_20_serie.csv     (regime por período, operador atual)
Saídas:
  codigos/05_interpretacao/relatorios_20/<ICAO>.md      (+ .json com métricas)
  codigos/05_interpretacao/relatorios_20/metricas_20.csv
  SQLite do MVP (histórico do app)

Uso: python codigos/05_interpretacao/04_relatorios_20.py [--modelo gemma2:9b] [--aeroportos SBGR ...]
         [--refazer]   gera de novo mesmo os aeroportos que já têm .json (sobrescreve .md/.json)
         [--dry-run]   só monta e imprime os prompts (sem LLM; não grava nada)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codigos" / "04_modelos_ml" / "app"))
sys.path.insert(0, str(HERE))

from contexto import (contexto_20, carregar_ranking, top_acionaveis, RANKING_20,  # noqa: E402
                      prioridades_detalhadas, satisfacao_detalhada, comentarios_resumo,
                      contexto_externo)
from db_init import init_db, salvar_relatorio  # noqa: E402
from llm_client import gerar_relatorio_meta, modelo_disponivel  # noqa: E402
from prompts import prompt_do_contexto  # noqa: E402
import relatorios_avaliacao as av  # noqa: E402

OUT = HERE / "relatorios_20"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="gemma2:9b")
    ap.add_argument("--aeroportos", nargs="*", help="ICAOs (padrão: os 20)")
    ap.add_argument("--refazer", action="store_true",
                    help="regenera também os aeroportos cujo .json já existe")
    ap.add_argument("--dry-run", action="store_true", help="só imprime os prompts, sem LLM")
    args = ap.parse_args()
    assert RANKING_20.exists(), "ranking_prioridades_20.csv ausente (pacote modelos20)"
    assert args.dry_run or modelo_disponivel(args.modelo), f"ollama pull {args.modelo}"
    ctx, media_nac = contexto_20()
    ranking, _ = carregar_ranking()
    OUT.mkdir(exist_ok=True)
    init_db()
    linhas = []
    for icao, c in ctx.items():
        if args.aeroportos and icao not in args.aeroportos:
            continue
        top = top_acionaveis(ranking, icao)
        prompt = prompt_do_contexto(
            c, icao, media_nac, prioridades=prioridades_detalhadas(ranking, icao),
            satisfacao=satisfacao_detalhada(icao), comentarios=comentarios_resumo(icao),
            externo=contexto_externo(icao))
        if args.dry_run:
            print(f"\n{'=' * 30} {icao} {'=' * 30}\n{prompt}")
            continue
        base = OUT / icao
        if base.with_suffix(".json").exists() and not args.refazer:
            meta = json.loads(base.with_suffix(".json").read_text())
            print(f"{icao}: já existe")
        else:
            meta = gerar_relatorio_meta(prompt, modelo=args.modelo, seed=SEED)
            meta.update({"modelo": args.modelo, "aeroporto": icao, "prompt": prompt,
                         "top_atributos": top})
            base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            base.with_suffix(".md").write_text(
                f"# Relatório automatizado — {c['nome_curto']}\n\n"
                f"**Gerado por**: {args.modelo} via Ollama · {meta['tempo_s']:.0f} s\n\n---\n\n"
                + meta["texto"])
            salvar_relatorio(aeroporto=icao, aeroporto_nome=c["nome_curto"],
                             modelo_llm=args.modelo, top_atributos=top,
                             contexto_operacional=c, relatorio_texto=meta["texto"],
                             tempo_geracao_s=meta["tempo_s"])
            print(f"{icao}: {meta['tokens_saida']} tok, {meta['tempo_s']:.0f}s")
        fid = av.fidelidade_numerica(meta["texto"], prompt)
        inv = fid.pop("inventados")
        cob = av.cobertura_prioridades(meta["texto"], [t["feature"] for t in top])
        linhas.append({"aeroporto": icao, "tempo_s": meta["tempo_s"],
                       "tokens_por_s": meta["tokens_por_s"], **fid,
                       "cobertura": cob["cobertura"], **av.estrutura(meta["texto"]),
                       "inventados": "; ".join(x["token"] for x in inv)})
    if args.dry_run:
        return
    df = pd.DataFrame(linhas)
    df.to_csv(OUT / "metricas_20.csv", index=False)
    print(df[["fidelidade", "num_inventado", "cobertura", "n_secoes", "palavras", "tempo_s"]]
          .describe().round(3))


if __name__ == "__main__":
    main()
