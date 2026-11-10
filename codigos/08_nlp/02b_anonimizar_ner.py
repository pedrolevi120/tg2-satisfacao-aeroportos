"""
T7/NLP — Etapa 2b: anonimização de nomes próprios de pessoas por NER (modelo local
Davlan/xlm-roberta-base-ner-hrl, multilíngue, inclui PT) + regras de 02 (e-mail, URL, números longos).
Entidades PER → "[NOME]". Filtro contra falsos positivos: o trecho é expandido para palavras inteiras e só é
mascarado se cada palavra começa com maiúscula e sua forma minúscula aparece < 3 vezes em minúsculas no
corpus (palavras comuns como "Gosto", "Maravilhoso" não são nomes). Aplicada ao texto EXPORTADO (comentarios_topicos.parquet / MVP / texto do TG).
Os modelos de sentimento e tópicos rodaram localmente sobre o texto com anonimização por regras; nenhum dado
sai da máquina.

Saída: saida/anonimizacao_ner.parquet (CHAVE_SERIE, AEROPORTO, texto_anon, n_nomes) — série 2021T1–2026T2.
O filtro de frequência em minúsculas (< 3) é calculado sobre o corpus da série inteira.
"""
import re
from collections import Counter
import pandas as pd
import torch
from transformers import pipeline
from comum import SAIDA, CHAVE

MODELO = "Davlan/xlm-roberta-base-ner-hrl"


def main():
    d = pd.read_parquet(SAIDA / "comentarios_limpos.parquet")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ner = pipeline("token-classification", model=MODELO, aggregation_strategy="simple", device=dev)
    textos = d["texto"].tolist()
    # ordena por comprimento para reduzir padding (resultado não depende da ordem)
    ordem = sorted(range(len(textos)), key=lambda i: len(textos[i]))
    ents_ord = ner([textos[i] for i in ordem], batch_size=64)
    ents = [None] * len(textos)
    for i, e in zip(ordem, ents_ord):
        ents[i] = e
    minus = Counter(w for t in textos for w in re.findall(r"\b[a-záéíóúâêôãõçà]+\b", t))
    anon, n = [], []
    for t, es in zip(textos, ents):
        spans = []
        for e in es:
            if e["entity_group"] != "PER" or e["score"] < 0.6:
                continue
            a, b = e["start"], e["end"]
            while a > 0 and t[a - 1].isalpha():
                a -= 1
            while b < len(t) and t[b].isalpha():
                b += 1
            palavras = re.findall(r"\w+", t[a:b])
            if palavras and all(w[0].isupper() and minus[w.lower()] < 3 for w in palavras):
                spans.append((a, b))
        # funde spans sobrepostos/adjacentes e substitui do fim para o início
        spans.sort()
        fund = []
        for a, b in spans:
            if fund and a <= fund[-1][1] + 1:
                fund[-1] = (fund[-1][0], max(b, fund[-1][1]))
            else:
                fund.append((a, b))
        for a, b in reversed(fund):
            t = t[:a] + "[NOME]" + t[b:]
        anon.append(t); n.append(len(fund))
    d["texto_anon"] = anon
    d["n_nomes"] = n
    d[[CHAVE, "AEROPORTO", "texto_anon", "n_nomes"]].to_parquet(SAIDA / "anonimizacao_ner.parquet", index=False)
    print("Comentários com nome mascarado:", (d.n_nomes > 0).sum(), "de", len(d))
    print(d.loc[d.n_nomes > 0, "texto_anon"].sample(min(25, (d.n_nomes > 0).sum()), random_state=42).to_string())


if __name__ == "__main__":
    main()
