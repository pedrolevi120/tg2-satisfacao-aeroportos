"""
T7/NLP — Etapa 4c: validação manual da atribuição de temas nos comentários SUBSTANTIVOS da amostra de
validação (os 178 dos 200 sorteados com random_state=42). Julgamento do autor, feito lendo cada comentário
e o tema atribuído (saída de 04b):
  correto   — o tema atribuído corresponde a um dos assuntos do comentário;
  errado    — o tema atribuído não corresponde a nenhum assunto do comentário;
  residual  — atribuído a 'sem_tema' (abstenção do modelo).
Os índices abaixo referem-se à ordem da amostra filtrada (substantivos) — impressa por este script com --listar.
Série 2021–2026: a amostra e o julgamento continuam os de 2023 (casados por ANO == 2023 + CHAVE + AEROPORTO).
Como os comentários de 2023 mantêm a atribuição original do BERTopic (04a2), o julgamento vale sem alteração;
para os demais anos, a validade da transferência é medida pela concordância do kNN na reprodução de 2023
(saida/atribuicao_temas_serie.json) — aqui também se reporta a precisão do julgamento manual restrita aos
comentários em que o kNN fora da dobra reproduz o tema original.
"""
import sys
import numpy as np
import pandas as pd
from comum import SAIDA, AQUI

ERRADOS = {0, 2, 4, 6, 8, 12, 13, 18, 20, 22, 28, 31, 39, 49, 51, 52, 55, 59, 60, 62, 69, 76, 78, 81, 95, 99, 102,
           106, 109, 111, 114, 122, 127, 136, 149, 164, 165, 166, 168, 170, 174, 175, 176}


def main():
    d = pd.read_parquet(SAIDA / "comentarios_topicos.parquet")
    a = pd.read_csv(AQUI / "amostra_validacao_200.csv")[["CHAVE", "AEROPORTO"]]
    v = a.merge(d[d.ANO == 2023], on=["CHAVE", "AEROPORTO"])
    assert len(v) == 200 and (v.origem_tema[v.classe == "substantivo"] == "bertopic_2023").all()
    v = v[v.classe == "substantivo"].reset_index(drop=True)
    if "--listar" in sys.argv:
        for i, r in v.iterrows():
            print(i, r.tema, "|", r.texto[:120])
        return
    # julgamento feito sobre a configuração com realocação de outliers (tema_realocado)
    v["julgamento_realocado"] = ["residual" if t == "sem_tema" else "errado" if i in ERRADOS else "correto"
                                 for i, t in enumerate(v.tema_realocado)]
    assert not (v.loc[list(ERRADOS), "tema_realocado"] == "sem_tema").any()
    v["outlier_hdbscan"] = v.topico_bertopic == -1
    # configuração principal (sem realocação): temas idênticos fora dos outliers; outliers viram residual
    assert (v.loc[~v.outlier_hdbscan, "tema"] == v.loc[~v.outlier_hdbscan, "tema_realocado"]).all()
    v["julgamento"] = np.where(v.tema == "sem_tema", "residual", v.julgamento_realocado)
    res = {}
    for nome, col in [("principal_hdbscan", "julgamento"), ("com_realocacao", "julgamento_realocado")]:
        j = v[col]; n = len(j); c = (j == "correto").sum(); r = (j == "residual").sum()
        res[nome] = {"n": n, "corretos": int(c), "errados": int((j == "errado").sum()), "residual": int(r),
                     "acuracia": c / n, "precisao": c / (n - r), "cobertura": (n - r) / n}
        print(nome, {k: round(float(x), 3) for k, x in res[nome].items()})
    v["julgamento"] = v["julgamento_realocado"]
    v[["CHAVE", "AEROPORTO", "tema", "tema_realocado", "outlier_hdbscan", "julgamento"]].to_csv(AQUI / "validacao_temas_178.csv", index=False)
    n = len(v); c = (v.julgamento == "correto").sum(); r = (v.julgamento == "residual").sum()
    print(f"n={n} corretos={c} errados={(v.julgamento=='errado').sum()} residual={r}")
    print(f"acurácia global={c/n:.3f} | precisão entre atribuídos={c/(n-r):.3f} | cobertura={(n-r)/n:.3f}")
    atr = v[v.julgamento != "residual"]
    print("Precisão por origem (HDBSCAN direto vs. outlier realocado):")
    print(atr.groupby("outlier_hdbscan")["julgamento"].apply(lambda s: f"{(s=='correto').mean():.3f} (n={len(s)})"))
    print("Precisão por tema (n≥5):")
    g = atr.groupby("tema_realocado")["julgamento"].agg(lambda s: (s == "correto").mean()).rename("prec").to_frame()
    g["n"] = atr.groupby("tema_realocado").size()
    print(g[g.n >= 5].sort_values("prec").round(2).to_string())
    res["precisao_por_origem"] = {"hdbscan_direto": (atr[~atr.outlier_hdbscan].julgamento == "correto").mean(),
                                  "outlier_realocado": (atr[atr.outlier_hdbscan].julgamento == "correto").mean(),
                                  "n_direto": int((~atr.outlier_hdbscan).sum()), "n_realocado": int(atr.outlier_hdbscan.sum())}
    # transferência para os outros anos: o kNN fora da dobra (mesmo método aplicado a 2021–2022 e 2024–2026)
    # reproduz o tema original em parte dos 178; nos demais, o tema kNN não foi julgado → limites inferior/superior
    v["julg_principal"] = np.where(v.tema == "sem_tema", "residual", v.julgamento_realocado)
    k = v.tema_knn_oof
    igual = k == v.tema
    atrib = k != "sem_tema"
    c_ok = ((igual & atrib) & (v.julg_principal == "correto")).sum()
    nao_julg = (~igual & atrib).sum()
    res["knn_oof_178"] = {"concordancia_tema": float(igual.mean()), "n_atribuidos": int(atrib.sum()),
                          "n_atribuidos_tema_diferente_nao_julgado": int(nao_julg),
                          "precisao_limite_inferior": float(c_ok / atrib.sum()),
                          "precisao_limite_superior": float((c_ok + nao_julg) / atrib.sum()),
                          "precisao_entre_concordantes": float(c_ok / (igual & atrib).sum())}
    print("kNN fora da dobra nos 178:", {a: round(float(b), 3) for a, b in res["knn_oof_178"].items()})
    import json
    (SAIDA / "validacao_temas.json").write_text(json.dumps(res, indent=1, default=float))
    return
    pd.Series({"n": n, "corretos": int(c), "residual": int(r), "acuracia": c / n, "precisao": c / (n - r),
               "precisao_hdbscan": (atr[~atr.outlier_hdbscan].julgamento == "correto").mean(),
               "precisao_realocados": (atr[atr.outlier_hdbscan].julgamento == "correto").mean(),
               "n_realocados": int(atr.outlier_hdbscan.sum())}).to_json(SAIDA / "validacao_temas.json")


if __name__ == "__main__":
    main()
