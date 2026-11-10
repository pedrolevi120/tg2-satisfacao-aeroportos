"""
T7/NLP — Etapa 3a: amostra de validação do sentimento (200 comentários válidos de 2023).

A amostra foi sorteada UMA vez, em 2023 (d.sample(200, random_state=42) sobre os comentários válidos de 2023,
antes de rodar qualquer modelo) e rotulada manualmente às cegas (rotulos_manuais_200.csv, rubrica em
RUBRICA_SENTIMENTO.md). Na série 2021T1–2026T2 ela fica FIXA: este script não sorteia nem sobrescreve nada;
só confere que os 200 comentários existem na base da série (ANO == 2023 + CHAVE + AEROPORTO), com o mesmo
texto e a mesma classe, e imprime o CHAVE_SERIE correspondente.

Para reproduzir o sorteio original (só com a base de 2023): ver saida/_2023/ e o histórico do pacote.
"""
import pandas as pd
from comum import SAIDA, AQUI

a = pd.read_csv(AQUI / "amostra_validacao_200.csv")
rot = pd.read_csv(AQUI / "rotulos_manuais_200.csv")
d = pd.read_parquet(SAIDA / "comentarios_limpos.parquet")
d23 = d[d.ANO == 2023]
v = a.merge(d23[["CHAVE", "AEROPORTO", "CHAVE_SERIE", "classe", "texto"]], on=["CHAVE", "AEROPORTO"], how="left",
            suffixes=("", "_serie"))
assert len(v) == 200 and v.CHAVE_SERIE.notna().all(), "amostra de validação não encontrada na série"
assert (v.classe == v.classe_serie).all() and (v.texto == v.texto_serie).all()
assert set(zip(rot.CHAVE, rot.AEROPORTO)) == set(zip(a.CHAVE, a.AEROPORTO))
print("Amostra de validação (2023) localizada na série: 200/200, texto e classe idênticos.")
print(a.classe.value_counts().to_string())
