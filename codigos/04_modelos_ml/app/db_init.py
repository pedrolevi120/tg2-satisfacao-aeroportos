"""
Inicializa o banco SQLite usado pelo MVP Streamlit.
Schema: tabela `relatorios` armazena cada análise gerada.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tg_aeroportos.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS relatorios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aeroporto TEXT NOT NULL,
    aeroporto_nome TEXT NOT NULL,
    data_geracao TEXT NOT NULL,
    modelo_llm TEXT NOT NULL,
    top_atributos TEXT NOT NULL,           -- JSON: top 5 atributos SHAP
    contexto_operacional TEXT NOT NULL,     -- JSON: pax, voos, etc.
    relatorio_texto TEXT NOT NULL,          -- output do LLM
    tempo_geracao_s REAL
);

CREATE INDEX IF NOT EXISTS idx_relatorios_aeroporto ON relatorios(aeroporto);
CREATE INDEX IF NOT EXISTS idx_relatorios_data ON relatorios(data_geracao DESC);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"Database inicializado em {DB_PATH}")


def salvar_relatorio(aeroporto, aeroporto_nome, modelo_llm, top_atributos,
                     contexto_operacional, relatorio_texto, tempo_geracao_s,
                     data_geracao=None):
    import json
    from datetime import datetime
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO relatorios
            (aeroporto, aeroporto_nome, data_geracao, modelo_llm, top_atributos,
             contexto_operacional, relatorio_texto, tempo_geracao_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (aeroporto, aeroporto_nome,
             data_geracao or datetime.now().isoformat(timespec="seconds"),
             modelo_llm, json.dumps(top_atributos, ensure_ascii=False),
             json.dumps(contexto_operacional, ensure_ascii=False),
             relatorio_texto, tempo_geracao_s)
        )


def listar_relatorios(aeroporto=None, limit=50):
    with get_conn() as conn:
        if aeroporto:
            rows = conn.execute(
                "SELECT * FROM relatorios WHERE aeroporto = ? ORDER BY data_geracao DESC LIMIT ?",
                (aeroporto, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM relatorios ORDER BY data_geracao DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
