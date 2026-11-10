"""
T7/NLP — Etapa 4a: modelagem de tópicos (BERTopic) nos comentários SUBSTANTIVOS.

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilíngue, 384 dim.)
Pipeline:  UMAP(15 viz., 5 dim., cosseno, random_state=42) → HDBSCAN(min_cluster_size=40) → c-TF-IDF (1–2-gramas)
Comentários genéricos ("tudo ótimo") não entram na modelagem: recebem o tópico fixo "Elogio genérico" na etapa 4b.

SÉRIE 2021–2026: este script NÃO é rodado de novo. O modelo ajustado em 2023 é a REFERÊNCIA (os números de
tópico de temas.py só valem para ele). Entradas e saídas ficam em saida/_2023/ (cópia das saídas de 2023);
os comentários dos demais anos recebem tópicos por 04a2_atribuir_temas_serie.py (vizinhos mais próximos entre os
comentários de 2023 no mesmo espaço de embeddings).

Saídas (em saida/_2023/): embeddings_substantivos.npy, saida/bertopic_topicos_brutos.csv (id, n, palavras, exemplos),
        saida/bertopic_atribuicoes.parquet (CHAVE, AEROPORTO, topico_hdbscan, topico, prob)
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from comum import REF2023 as SAIDA, SEED, STOP_PT, sem_acento  # referência 2023 (ver docstring)

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    d = pd.read_parquet(SAIDA / "comentarios_limpos.parquet")
    d = d[d.classe == "substantivo"].reset_index(drop=True)
    docs = d["texto"].tolist()
    f = SAIDA / "embeddings_substantivos.npy"
    if f.exists():
        E = np.load(f)
    else:
        E = SentenceTransformer(EMB).encode(docs, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        np.save(f, E)
    assert len(E) == len(docs)

    stop = sorted(set(STOP_PT) | {sem_acento(w) for w in STOP_PT})
    vec = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, strip_accents=None, lowercase=True)
    tm = BERTopic(
        embedding_model=SentenceTransformer(EMB),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=40, min_samples=10, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=vec, top_n_words=10, calculate_probabilities=False, verbose=True,
    )
    topics, probs = tm.fit_transform(docs, E)
    d["topico_hdbscan"] = topics
    # realoca outliers (−1) ao tópico mais próximo no espaço de embeddings (limiar de similaridade)
    novos = tm.reduce_outliers(docs, topics, strategy="embeddings", embeddings=E, threshold=0.5)
    d["topico"] = novos
    d["prob"] = probs
    print("Outliers HDBSCAN:", (np.array(topics) == -1).mean().round(3), "| após realocação:", (np.array(novos) == -1).mean().round(3))

    info = tm.get_topic_info()
    rows = []
    for t in sorted(set(topics)):
        palavras = ", ".join(w for w, _ in tm.get_topic(t)[:10]) if t != -1 else ""
        ex = d.loc[d.topico_hdbscan == t, "texto"].sample(min(8, (d.topico_hdbscan == t).sum()), random_state=SEED).tolist()
        rows.append({"topico": t, "n_hdbscan": int((d.topico_hdbscan == t).sum()), "n_final": int((d.topico == t).sum()),
                     "palavras": palavras, "exemplos": " || ".join(ex)})
    pd.DataFrame(rows).to_csv(SAIDA / "bertopic_topicos_brutos.csv", index=False)
    d[["CHAVE", "AEROPORTO", "topico_hdbscan", "topico", "prob"]].to_parquet(SAIDA / "bertopic_atribuicoes.parquet", index=False)
    print("Nº tópicos (exceto −1):", len(set(topics)) - (1 if -1 in topics else 0))


if __name__ == "__main__":
    main()
