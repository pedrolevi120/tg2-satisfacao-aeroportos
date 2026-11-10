"""
T7/NLP — Dicionário de rotulagem manual: tópico BERTopic → tema (PT-BR) → atributos Likert do questionário.

Rotulagem feita pelo autor a partir das 10 palavras c-TF-IDF e de 8 exemplos aleatórios de cada tópico
(saida/bertopic_topicos_brutos.csv). Cobertura:
  'coberto'      — o tema corresponde diretamente a atributo(s) Likert;
  'parcial'      — relaciona-se a atributo(s), mas contém aspecto não perguntado;
  'nao_coberto'  — não existe atributo no questionário (candidato a contribuição dos comentários);
  'geral'        — avaliação global do aeroporto (≈ SATISFAÇÃO GERAL), sem atributo específico;
  'sem_tema'     — tópico heterogêneo/residual.
"""

TEMAS = {
    # código: (rótulo, cobertura, [atributos Likert], [tópicos BERTopic])
    "banheiros": ("Limpeza, manutenção e quantidade de banheiros", "coberto",
                  ["SANITÁRIOS", "LIMPEZA DOS BANHEIROS", "MANUTENÇÃO GERAL DOS SANITÁRIOS", "QUANTIDADE DE BANHEIROS"], [0, 52]),
    "bagagem": ("Bagagem: despacho, restituição e danos", "coberto",
                ["PROCESSO DE RESTITUIÇÃO DE BAGAGENS", "TEMPO DE RESTITUIÇÃO", "INTEGRIDADE DA BAGAGEM",
                 "FACILIDADE DE IDENTIFICAÇÃO DA ESTEIRA DE RESTITUIÇÃO"], [1]),
    # no questionário, cordialidade/atendimento dos funcionários só é perguntado dentro de check-in e inspeção → parcial
    "atendimento": ("Atendimento e cordialidade dos funcionários", "parcial",
                    ["CORDIALIDADE DOS FUNCIONÁRIOS", "ATENDIMENTO DOS FUNCIONÁRIOS"], [2, 20, 28, 38]),
    "climatizacao": ("Climatização e calor", "coberto", ["CONFORTO TÉRMICO"], [4]),
    "transporte": ("Acesso terrestre e transporte público", "coberto",
                   ["OPÇÕES DE TRANSPORTE ATÉ O AEROPORTO"], [5, 44]),
    "taxi_app": ("Táxi, aplicativos e meio-fio", "parcial",
                 ["FACILIDADE DE DESEMBARQUE NO MEIO-FIO", "OPÇÕES DE TRANSPORTE ATÉ O AEROPORTO"], [63]),
    "sinalizacao": ("Sinalização (geral e no desembarque)", "coberto", ["SINALIZAÇÃO"], [7, 17, 24]),
    "informacao": ("Informação e orientação por pessoas/painéis", "parcial",
                   ["DISPONIBILIDADE DE PAINÉIS DE INFORMAÇÕES DE VOO", "SINALIZAÇÃO"], [29, 53, 60]),
    "limpeza_org": ("Limpeza e organização do terminal", "coberto", ["LIMPEZA GERAL DO AEROPORTO"], [8, 30, 31, 74]),
    "precos": ("Preços de alimentação, comércio e serviços", "coberto",
               ["RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS DE ALIMENTAÇÃO",
                "RELAÇÃO PREÇO x QUALIDADE DOS ESTABELECIMENTOS COMERCIAIS"], [9, 10, 41]),
    "alimentacao": ("Oferta e variedade de alimentação", "coberto",
                    ["ESTABELECIMENTOS DE ALIMENTAÇÃO", "QUANTIDADE DE ESTABELECIMENTOS DE ALIMENTAÇÃO",
                     "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS DE ALIMENTAÇÃO"], [18, 19, 26, 42, 57]),
    "lojas": ("Oferta de lojas", "coberto",
              ["ESTABELECIMENTOS COMERCIAIS", "QUANTIDADE DE ESTABELECIMENTOS COMERCIAIS",
               "QUALIDADE E VARIEDADE DE OPÇÕES DE ESTABELECIMENTOS COMERCIAIS"], [58]),
    "assentos": ("Conforto dos assentos e da sala de embarque", "coberto",
                 ["CONFORTO DA SALA DE EMBARQUE", "DISPONIBILIDADE DE ASSENTOS"], [11, 32]),
    "tomadas": ("Tomadas: quantidade e funcionamento", "coberto", ["DISPONIBILIDADE DE TOMADAS"], [36, 48, 55, 62, 73]),
    "internet": ("Internet e Wi-Fi", "coberto",
                 ["INTERNET DISPONIBILIZADA PELO AEROPORTO", "VELOCIDADE DE CONEXÃO", "FACILIDADE DE ACESSO À REDE"], [21, 65]),
    "acustica": ("Acústica e anúncios sonoros", "coberto", ["CONFORTO ACÚSTICO"], [35]),
    "estacionamento": ("Estacionamento: preço, vagas e permanência", "coberto",
                       ["ESTACIONAMENTO", "RELAÇÃO CUSTO X BENEFÍCIO", "FACILIDADE PARA ENCONTRAR VAGAS",
                        "QUALIDADE DAS INSTALAÇÕES DE ESTACIONAMENTO"], [16]),
    "distancias": ("Distâncias internas e deslocamento até portões", "coberto", ["LOCALIZAÇÃO E DESLOCAMENTO"], [43, 49, 51]),
    "facilidade": ("Facilidade de acesso e localização (elogio)", "coberto",
                   ["LOCALIZAÇÃO E DESLOCAMENTO"], [40]),
    "checkin": ("Check-in e filas no balcão", "coberto",
                ["PROCESSO DE CHECK IN", "TEMPO DE ESPERA NA FILA", "ORGANIZAÇÃO DAS FILAS"], [50]),
    "inspecao": ("Inspeção de segurança", "coberto",
                 ["PROCESSO DE INSPEÇÃO DE SEGURANÇA", "TEMPO DE ESPERA EM FILA"], [75]),
    "acessibilidade": ("Acessibilidade, idosos e mobilidade reduzida (escadas, elevadores)", "coberto",
                       ["ACESSIBILIDADE DO TERMINAL"], [70, 72]),
    "cia_aerea": ("Companhia aérea e atendimento a estrangeiros", "parcial", ["ATENDIMENTO DA CIA. AÉREA"], [45]),
    # --- temas SEM atributo no questionário (contribuição dos comentários)
    "atrasos": ("Atrasos de voo e tempo de conexão", "nao_coberto", [], [12, 25]),
    "oferta_voos": ("Oferta de voos e rotas", "nao_coberto", [], [14]),
    "sala_vip": ("Sala VIP (ausência ou restrição)", "nao_coberto", [], [34, 37, 71]),
    "bebedouros": ("Bebedouros e água potável", "nao_coberto", [], [46]),
    "obras": ("Obras, reformas e capacidade da infraestrutura", "nao_coberto", [], [22, 66, 68]),
    "evolucao": ("Evolução percebida do aeroporto (\"melhorou\")", "nao_coberto", [], [6, 59]),
    # --- avaliação global e residual
    "elogio_geral": ("Elogio geral ao aeroporto", "geral", ["SATISFAÇÃO GERAL"], [3, 13, 23, 27, 39, 54, 56, 61, 64, 67, 69]),
    "elogio_generico": ("Elogio/avaliação genérica curta (não modelado)", "geral", ["SATISFAÇÃO GERAL"], []),
    "sem_tema": ("Residual/heterogêneo", "sem_tema", [], [-1, 15, 33, 47]),
}

TOPICO_PARA_TEMA = {t: k for k, v in TEMAS.items() for t in v[3]}
