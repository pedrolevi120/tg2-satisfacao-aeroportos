"""
SO3 — Gera 3 relatórios (GRU/CGH/VCP) via Llama 3.1 8B (Ollama).
Pipeline: ranking SHAP + contexto → prompt → LLM → texto.

Saídas:
  codigos/05_interpretacao/relatorios/{SBGR,SBSP,SBKP}.md
  Persiste também no SQLite do MVP (codigos/04_modelos_ml/app/tg_aeroportos.db).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "codigos" / "04_modelos_ml" / "app"
sys.path.insert(0, str(APP_DIR))

import pandas as pd
from db_init import init_db, salvar_relatorio
from llm_client import gerar_relatorio, modelo_disponivel, MODELO_DEFAULT
from prompts import montar_prompt

RANKING_PATH = ROOT / "codigos" / "05_interpretacao" / "ranking_prioridades_por_aeroporto.csv"
OUT_DIR = ROOT / "codigos" / "05_interpretacao" / "relatorios"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXTO = {
    "SBGR": {"nome": "Aeroporto Internacional de Guarulhos",
              "nome_curto": "Guarulhos (GRU)",
              "cidade": "Guarulhos", "uf": "SP", "regiao": "Sudeste",
              "pax_mi": 41.3, "voos_k": 277, "regime": "Concessão privada",
              "concessionaria": "GRU Airport (Invepar/ACSA)",
              "n": 6054, "media": 4.21, "posicao": 15, "pct_4_5": 88.7},
    "SBSP": {"nome": "Aeroporto de Congonhas",
              "nome_curto": "Congonhas (CGH)",
              "cidade": "São Paulo", "uf": "SP", "regiao": "Sudeste",
              "pax_mi": 22.0, "voos_k": 212, "regime": "Concessão privada (a partir de 17/out/2023)",
              "concessionaria": "Aena Brasil (anteriormente Infraero)",
              "n": 4033, "media": 4.16, "posicao": 18, "pct_4_5": 89.5},
    "SBKP": {"nome": "Aeroporto Internacional de Viracopos",
              "nome_curto": "Viracopos (VCP)",
              "cidade": "Campinas", "uf": "SP", "regiao": "Sudeste",
              "pax_mi": 12.5, "voos_k": 68, "regime": "Concessão privada",
              "concessionaria": "Aeroportos Brasil Viracopos (ABV)",
              "n": 4132, "media": 4.47, "posicao": 7, "pct_4_5": 94.1},
}


def main():
    if not modelo_disponivel(MODELO_DEFAULT):
        print(f"❌ Modelo {MODELO_DEFAULT} não disponível. Rode: ollama pull {MODELO_DEFAULT}")
        return

    init_db()
    ranking_df = pd.read_csv(RANKING_PATH)

    for icao, ctx in CONTEXTO.items():
        print(f"\n=== Gerando relatório: {ctx['nome_curto']} ===")
        top5 = ranking_df[ranking_df["aeroporto"] == icao].head(5)
        top_atributos = [
            {"feature": r["feature"], "shap_abs": r["shap_abs_medio"]}
            for _, r in top5.iterrows()
        ]

        prompt = montar_prompt(
            nome_completo=ctx["nome"], icao=icao,
            cidade=ctx["cidade"], uf=ctx["uf"], regiao=ctx["regiao"],
            pax_mi=ctx["pax_mi"], voos_k=ctx["voos_k"],
            regime=ctx["regime"], concessionaria=ctx["concessionaria"],
            n=ctx["n"], media=ctx["media"], posicao=ctx["posicao"],
            pct_4_5=ctx["pct_4_5"], top_atributos=top_atributos,
        )

        texto, tempo = gerar_relatorio(prompt, modelo=MODELO_DEFAULT)
        print(f"  ✓ {len(texto)} chars em {tempo:.1f}s")
        print(f"  ─" * 30)
        print(texto[:400] + "..." if len(texto) > 400 else texto)
        print(f"  ─" * 30)

        # Salvar arquivo .md
        out_md = OUT_DIR / f"{icao}.md"
        header = f"""# Relatório automatizado — {ctx['nome_curto']}

**Gerado por**: {MODELO_DEFAULT} via Ollama
**Tempo de geração**: {tempo:.1f}s
**Caracteres**: {len(texto)}

---

"""
        out_md.write_text(header + texto)
        print(f"  → {out_md.relative_to(ROOT)}")

        # Persistir no SQLite
        salvar_relatorio(
            aeroporto=icao,
            aeroporto_nome=ctx["nome_curto"],
            modelo_llm=MODELO_DEFAULT,
            top_atributos=top_atributos,
            contexto_operacional=ctx,
            relatorio_texto=texto,
            tempo_geracao_s=tempo,
        )

    print(f"\n✅ Todos os 3 relatórios gerados em {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
