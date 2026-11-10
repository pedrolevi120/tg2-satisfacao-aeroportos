"""
Cliente para o LLM local via Ollama.
"""
import time

import ollama


MODELO_DEFAULT = "gemma2:9b"   # melhor avaliado na comparação (03_comparar_llms.py)
# Janela de contexto explícita: o prompt com contexto externo passa de 2.500 tokens
# e, somado aos 2.200 de saída, estouraria a janela padrão do Ollama (4.096).
NUM_CTX = 8192


def gerar_relatorio(prompt: str, modelo: str = MODELO_DEFAULT,
                     temperatura: float = 0.4, max_tokens: int = 3000):
    """Chama o LLM via Ollama e retorna texto + tempo decorrido.

    Args:
        prompt: prompt completo (já formatado).
        modelo: tag do modelo no Ollama (ex.: 'llama3.1:8b').
        temperatura: 0.0 = determinístico, 1.0 = criativo. 0.3 = bom para
                     relatório técnico (factual com leve variação).
        max_tokens: limite de saída.
    Returns:
        (texto, tempo_s)
    """
    t0 = time.time()
    response = ollama.chat(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperatura, "num_predict": max_tokens, "num_ctx": NUM_CTX},
    )
    texto = response["message"]["content"].strip()
    tempo = time.time() - t0
    return texto, tempo


def gerar_relatorio_stream(prompt: str, modelo: str = MODELO_DEFAULT,
                           temperatura: float = 0.4, max_tokens: int = 3000):
    """Como `gerar_relatorio`, mas em streaming: devolve os trechos de texto à
    medida que o modelo escreve (para exibir com `st.write_stream`)."""
    for parte in ollama.chat(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperatura, "num_predict": max_tokens, "num_ctx": NUM_CTX},
        stream=True,
    ):
        yield parte["message"]["content"]


def modelo_disponivel(modelo: str = MODELO_DEFAULT) -> bool:
    """Verifica se o modelo está disponível localmente."""
    try:
        modelos = ollama.list()
        # Ollama 0.5+ retorna objeto com .models
        if hasattr(modelos, "models"):
            return any(getattr(m, "model", "") == modelo for m in modelos.models)
        # versões antigas retornam dict
        return any(m.get("name") == modelo for m in modelos.get("models", []))
    except Exception:
        return False


def gerar_relatorio_meta(prompt: str, modelo: str = MODELO_DEFAULT,
                          temperatura: float = 0.4, max_tokens: int = 3000,
                          seed: int = 42) -> dict:
    """Como `gerar_relatorio`, mas devolve também as métricas do Ollama.

    Returns:
        dict com texto, tempo_s, tokens_saida, tokens_prompt, tokens_por_s
        (tokens de saída / tempo de decodificação) e tempo_carga_s.
    """
    t0 = time.time()
    r = ollama.chat(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperatura, "num_predict": max_tokens,
                 "num_ctx": NUM_CTX, "seed": seed},
    )
    tempo = time.time() - t0
    eval_count = r.get("eval_count") or 0
    eval_dur = (r.get("eval_duration") or 0) / 1e9
    return {
        "texto": r["message"]["content"].strip(),
        "tempo_s": tempo,
        "tokens_saida": eval_count,
        "tokens_prompt": r.get("prompt_eval_count") or 0,
        "tokens_por_s": eval_count / eval_dur if eval_dur else float("nan"),
        "tempo_carga_s": (r.get("load_duration") or 0) / 1e9,
    }
