"""
SO3 (TG-2) — Relatórios executivos dos 20 aeroportos da MPor 2023.

Entradas:
  codigos/05_interpretacao/ranking_prioridades_20.csv   (pacote modelos20/T5)
  dados/processados/panorama_aeroportos_2023.parquet    (n, média, % 4–5, posição)
  codigos/02_preprocessamento/aeroportos_20_caracteristicas.csv (pacote dados/T2)
Saídas:
  codigos/05_interpretacao/relatorios_20/<ICAO>.md      (+ .json com métricas)
  codigos/05_interpretacao/relatorios_20/metricas_20.csv
  SQLite do MVP (histórico do app)

Uso: python codigos/05_interpretacao/04_relatorios_20.py [--modelo llama3.1:8b]
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

from contexto import contexto_20, top_acionaveis, RANKING_20  # noqa: E402
from db_init import init_db, salvar_relatorio  # noqa: E402
from llm_client import gerar_relatorio_meta, modelo_disponivel  # noqa: E402
from prompts import montar_prompt  # noqa: E402
import relatorios_avaliacao as av  # noqa: E402

OUT = HERE / "relatorios_20"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="llama3.1:8b")
    args = ap.parse_args()
    assert RANKING_20.exists(), "ranking_prioridades_20.csv ausente (pacote modelos20)"
    assert modelo_disponivel(args.modelo), f"ollama pull {args.modelo}"
    ctx, _ = contexto_20()
    ranking = pd.read_csv(RANKING_20)
    OUT.mkdir(exist_ok=True)
    init_db()
    linhas = []
    for icao, c in ctx.items():
        top = top_acionaveis(ranking, icao)
        prompt = montar_prompt(
            nome_completo=c["nome"], icao=icao, cidade=c["cidade"], uf=c["uf"],
            regiao=c["regiao"], pax_mi=c["pax_mi"], voos_k=c["voos_k"],
            regime=c["regime"], concessionaria=c["concessionaria"], n=c["n"],
            media=c["media"], posicao=c["posicao"], pct_4_5=c["pct_4_5"],
            top_atributos=top)
        base = OUT / icao
        if base.with_suffix(".json").exists():
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
    df = pd.DataFrame(linhas)
    df.to_csv(OUT / "metricas_20.csv", index=False)
    print(df[["fidelidade", "num_inventado", "cobertura", "n_secoes", "palavras", "tempo_s"]]
          .describe().round(3))


if __name__ == "__main__":
    main()
