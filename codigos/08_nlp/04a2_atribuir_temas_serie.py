"""
T7/NLP — Etapa 4a2: atribui os tópicos do modelo de REFERÊNCIA (BERTopic ajustado nos 13.868 comentários
substantivos de 2023, 04a_topicos_bertopic.py, saídas em saida/_2023/) a todos os comentários substantivos da
série 2021T1–2026T2, sem reajustar o BERTopic (os números de tópico de temas.py só valem para aquele modelo).

Método
  1. Embeddings: o mesmo modelo de sentenças (paraphrase-multilingual-MiniLM-L12-v2, normalizado). Para 2023,
     reaproveita o cache da referência (conferido contra uma reamostra re-codificada).
  2. Comentários de 2023: mantêm a atribuição original do BERTopic (são os dados de ajuste da referência; é o que
     o próprio BERTopic faz: documentos de treino ficam com o rótulo do ajuste). Assim a validação manual de
     temas (04c, 178 comentários de 2023) continua valendo sem alteração.
  3. Demais anos: classificador de k vizinhos mais próximos (cosseno, voto ponderado por 1/distância) sobre os
     comentários de 2023 rotulados pelo HDBSCAN, INCLUINDO a classe outlier (−1). O k é escolhido pela maior
     concordância em validação cruzada estratificada de 5 dobras dentro de 2023 (k ∈ {3, 5, 7, 10}).
     Alternativa pedida inicialmente — centróide mais próximo de cada tópico com limiar de outlier calibrado —
     é calculada e reportada para comparação: concorda bem menos com o HDBSCAN (os grupos do HDBSCAN no espaço
     UMAP não são convexos no espaço original e o limiar único não reproduz os ~20% de outliers).
  4. Coluna 'topico' (realocação de outliers, configuração secundária, como em 2023): para outliers, tópico do
     centróide mais próximo se cosseno ≥ 0,5. Isso reproduz exatamente o reduce_outliers(strategy="embeddings",
     threshold=0.5) do BERTopic usado em 2023 (conferido: 100% nos outliers de 2023).

Saídas: saida/embeddings_substantivos.npy + saida/embeddings_substantivos_chaves.parquet (cache da série),
        saida/bertopic_atribuicoes.parquet (CHAVE_SERIE, AEROPORTO, ANO, topico_hdbscan, topico, prob, origem,
            topico_centroide, sim_centroide, topico_knn_oof),
        saida/atribuicao_temas_serie.json (concordâncias na reprodução de 2023),
        saida/atribuicao_2023_reproducao_por_tema.csv (participação de cada tema em 2023: original × reproduzida).
"""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from comum import SAIDA, REF2023, SEED, CHAVE
from temas import TOPICO_PARA_TEMA

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KS = [3, 5, 7, 10]
LIM_REALOC = 0.5  # o mesmo limiar do reduce_outliers de 2023


def centroides(E, t):
    ids = np.array(sorted(set(t) - {-1}))
    C = np.vstack([E[t == k].mean(0) for k in ids])
    return ids, C / np.linalg.norm(C, axis=1, keepdims=True)


def mais_proximo(E, ids, C, bloco=20000):
    j, s = [], []
    for i in range(0, len(E), bloco):
        S = E[i:i + bloco] @ C.T
        j.append(S.argmax(1)); s.append(S.max(1))
    return ids[np.concatenate(j)], np.concatenate(s)


def knn(k):
    return KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance", algorithm="brute", n_jobs=1)


def tema(t):
    return pd.Series(t).map(TOPICO_PARA_TEMA).values


def main():
    d = pd.read_parquet(SAIDA / "comentarios_limpos.parquet")
    d = d[d.classe == "substantivo"].reset_index(drop=True)
    ref = pd.read_parquet(REF2023 / "bertopic_atribuicoes.parquet")
    Eref = np.load(REF2023 / "embeddings_substantivos.npy")
    ref_txt = pd.read_parquet(REF2023 / "comentarios_limpos.parquet", columns=["CHAVE", "AEROPORTO", "classe", "texto"])
    ref_txt = ref_txt[ref_txt.classe == "substantivo"].reset_index(drop=True)
    assert len(ref) == len(Eref) == len(ref_txt) and (ref.CHAVE.values == ref_txt.CHAVE.values).all()
    ref["i_ref"] = np.arange(len(ref))
    ref["texto_ref"] = ref_txt.texto.values

    # --- casa 2023 da série com a referência (ANO == 2023 + CHAVE + AEROPORTO)
    d23 = d[d.ANO == 2023].reset_index()
    m = d23.merge(ref, on=["CHAVE", "AEROPORTO"], how="outer", indicator=True)
    assert (m._merge == "both").all() and len(m) == len(ref), m._merge.value_counts()
    assert (m.texto == m.texto_ref).all()
    i_ref_de = pd.Series(m.i_ref.values, index=m["index"].values)  # índice em d -> índice na referência

    # --- embeddings da série (cache por CHAVE_SERIE)
    f_e, f_k = SAIDA / "embeddings_substantivos.npy", SAIDA / "embeddings_substantivos_chaves.parquet"
    if f_e.exists() and f_k.exists() and (pd.read_parquet(f_k)[CHAVE].values == d[CHAVE].values).all():
        E = np.load(f_e)
        print("Embeddings da série: cache", E.shape)
    else:
        from sentence_transformers import SentenceTransformer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        st = SentenceTransformer(EMB, device=dev)
        E = np.zeros((len(d), Eref.shape[1]), dtype=np.float32)
        e23 = i_ref_de.index.values
        E[e23] = Eref[i_ref_de.values]
        # conferência: re-codifica 300 comentários de 2023 e compara com o cache da referência
        chk = np.random.default_rng(SEED).choice(e23, 300, replace=False)
        Echk = st.encode(d.texto.iloc[chk].tolist(), batch_size=64, normalize_embeddings=True)
        cos = (Echk * E[chk]).sum(1)
        print(f"Conferência do cache 2023: cosseno mínimo {cos.min():.6f}, médio {cos.mean():.6f}")
        assert cos.min() > 0.999
        outros = np.setdiff1d(np.arange(len(d)), e23)
        E[outros] = st.encode(d.texto.iloc[outros].tolist(), batch_size=128, show_progress_bar=True,
                              normalize_embeddings=True)
        np.save(f_e, E)
        d[[CHAVE]].to_parquet(f_k, index=False)
        del st
    assert len(E) == len(d)

    t = ref.topico_hdbscan.values
    r = ref.topico.values
    T = tema(t)
    res = {"n_referencia_2023": int(len(ref)), "outliers_hdbscan_2023": float((t == -1).mean()),
           "sem_tema_2023": float((T == "sem_tema").mean())}

    # --- conferência: realocação de outliers = centróide mais próximo com cosseno ≥ 0,5
    ids, C = centroides(Eref, t)
    p_c, s_c = mais_proximo(Eref, ids, C)
    o = t == -1
    res["realocacao_reproduzida_outliers_2023"] = float((np.where(s_c[o] >= LIM_REALOC, p_c[o], -1) == r[o]).mean())

    # --- reprodução de 2023 pelo método do centróide (limiar calibrado na própria 2023)
    grade = np.round(np.arange(0.30, 0.801, 0.025), 3)
    conc = {float(l): float((np.where(s_c >= l, p_c, -1) == t).mean()) for l in grade}
    l_best = max(conc, key=conc.get)
    p_cb = np.where(s_c >= l_best, p_c, -1)
    res["centroide"] = {"limiar_calibrado": l_best, "concordancia_topico": conc[l_best],
                        "concordancia_tema": float((tema(p_cb) == T).mean()),
                        "pct_outlier": float((p_cb == -1).mean()), "curva": conc}
    print("Centróide: limiar", l_best, {k: round(v, 4) for k, v in res["centroide"].items() if k != "curva"})

    # --- reprodução de 2023 pelo kNN (validação cruzada estratificada, 5 dobras)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = {}
    res["knn_cv"] = {}
    for k in KS:
        p = np.empty_like(t)
        for tr, te in cv.split(Eref, t):
            p[te] = knn(k).fit(Eref[tr], t[tr]).predict(Eref[te])
        oof[k] = p
        res["knn_cv"][k] = {"concordancia_topico": float((p == t).mean()), "concordancia_tema": float((tema(p) == T).mean()),
                            "pct_outlier": float((p == -1).mean()), "pct_sem_tema": float((tema(p) == "sem_tema").mean())}
        print("kNN k =", k, {a: round(b, 4) for a, b in res["knn_cv"][k].items()})
    k_best = max(KS, key=lambda k: res["knn_cv"][k]["concordancia_topico"])
    res["k_escolhido"] = k_best
    p_oof = oof[k_best]
    # concordância só entre os não-outliers da referência e precisão entre os previstos não-outliers
    nn = t != -1
    res["knn_escolhido_detalhe"] = {
        "concordancia_entre_nao_outliers": float((p_oof[nn] == t[nn]).mean()),
        "concordancia_entre_outliers": float((p_oof[~nn] == -1).mean()),
        "concordancia_tema_entre_previstos_com_tema": float((tema(p_oof)[tema(p_oof) != "sem_tema"] == T[tema(p_oof) != "sem_tema"]).mean()),
    }
    # participação de cada tema em 2023: original × reproduzida (out-of-fold)
    rep = pd.DataFrame({"original": pd.Series(T).value_counts(normalize=True),
                        "knn_oof": pd.Series(tema(p_oof)).value_counts(normalize=True),
                        "centroide": pd.Series(tema(p_cb)).value_counts(normalize=True)}).fillna(0) * 100
    rep["razao_knn_original"] = rep.knn_oof / rep.original
    rep.sort_values("original", ascending=False).to_csv(SAIDA / "atribuicao_2023_reproducao_por_tema.csv")
    print(rep.sort_values("original", ascending=False).round(2).to_string())

    # --- atribuição final
    clf = knn(k_best).fit(Eref, t)
    out = d[[CHAVE, "AEROPORTO", "ANO"]].copy()
    out["topico_hdbscan"] = -1
    out["prob"] = np.nan
    out["origem"] = "knn_ref2023"
    novos = np.setdiff1d(np.arange(len(d)), i_ref_de.index.values)
    En = E[novos]
    pr = np.vstack([clf.predict_proba(En[i:i + 10000]) for i in range(0, len(En), 10000)])
    out.loc[novos, "topico_hdbscan"] = clf.classes_[pr.argmax(1)]
    out.loc[novos, "prob"] = pr.max(1)
    i23 = i_ref_de.index.values
    out.loc[i23, "topico_hdbscan"] = t[i_ref_de.values]
    out.loc[i23, "prob"] = ref.prob.values[i_ref_de.values]
    out.loc[i23, "origem"] = "bertopic_2023"
    pc_all, sc_all = mais_proximo(E, ids, C)
    out["topico_centroide"] = np.where(sc_all >= l_best, pc_all, -1)
    out["sim_centroide"] = sc_all
    out["topico"] = np.where(out.topico_hdbscan == -1, np.where(sc_all >= LIM_REALOC, pc_all, -1), out.topico_hdbscan)
    out.loc[i23, "topico"] = r[i_ref_de.values]
    out["topico_knn_oof"] = np.nan
    out.loc[i23, "topico_knn_oof"] = p_oof[i_ref_de.values]
    assert (out.loc[i23, "topico_hdbscan"].values == t[i_ref_de.values]).all()
    out.to_parquet(SAIDA / "bertopic_atribuicoes.parquet", index=False)

    out["tema"] = tema(out.topico_hdbscan.values)
    res["pct_outlier_por_ano"] = out.groupby("ANO")["topico_hdbscan"].apply(lambda s: float((s == -1).mean())).to_dict()
    res["pct_sem_tema_por_ano"] = out.groupby("ANO")["tema"].apply(lambda s: float((s == "sem_tema").mean())).to_dict()
    res["n_substantivos_por_ano"] = out.groupby("ANO").size().astype(int).to_dict()
    res = json.loads(json.dumps(res, default=float), parse_int=int)
    (SAIDA / "atribuicao_temas_serie.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("centroide",)}, indent=1))


if __name__ == "__main__":
    main()
