"""
T7/NLP — Etapa 1: carrega a série MPor 2021T1–2026T2 (20 aeroportos; base única gerada por
codigos/10_serie_historica/01_carregar_serie.py) mantendo identificadores, período, SATISFAÇÃO GERAL,
atributos Likert e COMENTÁRIOS ADICIONAIS.

Entrada: dados/processados/mpor_2021_2026_20.parquet
Saída:   codigos/08_nlp/saida/mpor_serie_20aero_comentarios.parquet  (base completa, 1 linha/respondente)

Chave de junção na série: CHAVE_SERIE. CHAVE (do portal) só existe de 2023 em diante e é usada apenas para
casar com os artefatos manuais de 2023 (amostra de validação de 200 comentários): ANO == 2023 + CHAVE + AEROPORTO.
"""
import pandas as pd
from comum import BASE, BASE_COM

# Colunas de perfil/identificação/período que não são atributos Likert
NAO_LIKERT = {
    "CHAVE", "PROCESSO", "AEROPORTO", "DATA", "MÊS", "INÍCIO COLETA", "FIM COLETA", "TERMINAL",
    "PORTÃO", "TIPO DE VOO", "CIA AÉREA", "VOO", "CONEXÃO", "AQUISIÇÃO DA PASSAGEM",
    "MEIO DE AQUISIÇÃO DA PASSAGEM", "MEIO DE  TRANSPORTE PARA O AEROPORTO", "POSSUI DEFICIÊNCIA",
    "UTILIZA RECURSO ASSISTIVO", "SOLICITOU ASSISTÊNCIA ESPECIAL", "FORMA DE DESEMBARQUE UTILIZADA",
    "UTILIZOU O ESTACIONAMENTO", "FORMA DE CHECK-IN", "MOTIVO", "NACIONALIDADE", "GÊNERO", "IDADE",
    "ESCOLARIDADE", "RENDA FAMILIAR", "VIAJANDO SOZINHO", "NÚMERO DE ACOMPANHANTES", "MOTIVO DA VIAGEM",
    "QUANTIDADE DE VIAGENS NOS ÚLTIMOS 12 MESES", "JÁ EMBARCOU/DESEMBARCOU ANTES NO AEROPORTO",
    "ANTECEDÊNCIA", "TEMPO DE ESPERA", "COMENTÁRIOS ADICIONAIS", "SATISFAÇÃO GERAL",
    "TRIMESTRE", "ANO", "PERIODO", "CHAVE_SERIE",
}
MANTER = ["CHAVE_SERIE", "CHAVE", "ANO", "TRIMESTRE", "PERIODO", "PROCESSO", "AEROPORTO", "TIPO DE VOO",
          "SATISFAÇÃO GERAL", "COMENTÁRIOS ADICIONAIS"]


def main():
    full = pd.read_parquet(BASE)
    assert full["CHAVE_SERIE"].is_unique
    # 610 linhas de 2022T2 vêm sem AEROPORTO (e sem nenhum atributo Likert) na base da série: não dá para
    # atribuí-las a um aeroporto nem a uma taxa por mil respondentes — ficam fora do pacote de comentários.
    sem_ap = full["AEROPORTO"].isna()
    print("Linhas sem AEROPORTO (descartadas):", int(sem_ap.sum()), full.loc[sem_ap, "PERIODO"].value_counts().to_dict())
    full = full[~sem_ap]
    likert = [c for c in full.columns if c not in NAO_LIKERT]
    out = full[MANTER + likert].copy()
    out["CHAVE"] = out["CHAVE"].astype("Int64")
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).where(out[col].notna(), None)
    out.to_parquet(BASE_COM, index=False)
    print("Total:", len(out), "| aeroportos:", out["AEROPORTO"].nunique(), "| períodos:", out["PERIODO"].nunique(),
          "| atributos Likert:", len(likert))
    print(out.groupby("ANO").size().to_string())
    print(out["SATISFAÇÃO GERAL"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()
