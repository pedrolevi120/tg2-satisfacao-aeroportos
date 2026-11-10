"""T7/NLP — utilidades comuns (caminhos, estilo de figura, limpeza e anonimização)."""
from pathlib import Path
import re
import unicodedata
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "saida"
FIG = ROOT / "midias" / "figuras"
TEX = ROOT / "texto" / "_paralelo" / "tex"
for p in (SAIDA, FIG, TEX):
    p.mkdir(parents=True, exist_ok=True)

BASE = ROOT / "dados" / "processados" / "mpor_2021_2026_20.parquet"  # série 2021T1–2026T2 (ESPEC.md)
BASE_COM = SAIDA / "mpor_serie_20aero_comentarios.parquet"              # saída de 01 (1 linha/respondente)
REF2023 = SAIDA / "_2023"   # modelo de tópicos de referência (BERTopic ajustado em 2023) e saídas antigas
CHAVE = "CHAVE_SERIE"       # identificador único do respondente na série
ANOS_ROT = "2021–2026"      # rótulo do período em títulos de figuras
ANOS_TEX = "2021--2026"     # idem, em legendas LaTeX

SEED = 42
COL_COM = "COMENTÁRIOS ADICIONAIS"
COL_SAT = "SATISFAÇÃO GERAL"

NOMES = {
    "SBBE": "Belém", "SBBR": "Brasília", "SBCF": "Confins", "SBCT": "Curitiba", "SBCY": "Cuiabá",
    "SBEG": "Manaus", "SBFL": "Florianópolis", "SBFZ": "Fortaleza", "SBGL": "Galeão", "SBGO": "Goiânia",
    "SBGR": "Guarulhos", "SBKP": "Viracopos", "SBMO": "Maceió", "SBPA": "Porto Alegre", "SBRF": "Recife",
    "SBRJ": "Santos Dumont", "SBSG": "Natal", "SBSP": "Congonhas", "SBSV": "Salvador", "SBVT": "Vitória",
}


def estilo():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    })


def sem_acento(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def normalizar(s: str) -> str:
    """minúsculas, sem acento, só letras/dígitos/espaço — usado apenas para regras de ruído."""
    s = sem_acento(str(s).lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Respostas que significam "sem comentário" (ruído de preenchimento)
RUIDO_EXATO = {
    "nao", "n", "nh", "na", "nd", "nda", "nada", "nenhum", "nenhuma", "sem comentarios", "sem comentario",
    "nao tenho", "nada a declarar", "nada a acrescentar", "nao ha", "nao obrigado", "nao obrigada",
    "ns", "nr", "ns nr", "nsnr", "x", "xx", "sem", "nao tem", "nao quis", "nao respondeu", "nulo",
    "nada nao", "nao nada", "nenhum comentario", "sem observacoes", "sem observacao", "nt", "nn",
    # 2021T3–2021T4: o campo traz só "Sim" em 2.458 linhas (aparentemente a resposta a "deseja comentar?",
    # sem o texto). Não ocorre em 2023 — não altera a classificação daquele ano.
    "sim",
    # "nada a informar" e variantes: preenchimento padrão do entrevistador (1.810 linhas, quase todas SBCY em
    # 2021–2022; nenhuma em 2023). Antes caíam como substantivas e inflavam o tema "Informação" em 2022.
    "nada a informar", "nada informar", "nao informar", "nada a informar sobre",
}

# Elogio/avaliação genérica curta: válido para sentimento, sem conteúdo temático
PAL_GENERICAS = {
    "tudo", "muito", "bom", "boa", "bem", "otimo", "otima", "excelente", "perfeito", "perfeita", "ok",
    "tranquilo", "tranquila", "certo", "esta", "ta", "e", "foi", "o", "a", "aeroporto", "satisfeito",
    "satisfeita", "satisfatorio", "satisfatoria", "parabens", "gostei", "gostou", "maravilhoso",
    "maravilhosa", "estou", "legal", "show", "top", "sim", "bonito", "lindo", "estava", "sem",
    "reclamacoes", "reclamacao", "nada", "reclamar", "a", "de", "do", "um", "uma", "so", "elogios",
    "continue", "assim", "muita", "bastante", "super", "tudo", "ideal", "razoavel", "regular", "normal",
    "adorei", "amei", "incrivel", "agradavel", "positivo", "positiva", "melhor", "sempre", "ruim",
    "pessimo", "pessima", "horrivel", "pode", "melhorar", "mais", "ou", "menos", "mais", "gosto",
}

RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
RE_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
RE_NUM_LONGO = re.compile(r"\b\d[\d .\-/()]{6,}\d\b")  # telefones, CPF, protocolos, nº de voo longos
RE_NOME = re.compile(r"\b((?:meu nome|me chamo|sou o|sou a|funcionári[oa]|atendente|senhor[a]?|sr\.?|sra\.?)\s+)"
                     r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*)")


def anonimizar(s: str) -> str:
    s = RE_EMAIL.sub("[EMAIL]", s)
    s = RE_URL.sub("[URL]", s)
    s = RE_NUM_LONGO.sub("[NÚMERO]", s)
    s = RE_NOME.sub(lambda m: m.group(1) + "[NOME]", s)
    return s


def classificar(s) -> str:
    """'vazio' | 'ruido' | 'generico' | 'substantivo'."""
    if s is None or str(s).strip() in ("", "None", "nan"):
        return "vazio"
    n = normalizar(s)
    toks = n.split()
    if n in ("ok", "bom", "boa", "top"):
        return "generico"
    if not n or n in RUIDO_EXATO or n.isdigit() or len(n) <= 2 or not re.search(r"[a-z]", n):
        return "ruido"
    if len(toks) <= 6 and all(t in PAL_GENERICAS for t in toks):
        return "generico"
    return "substantivo"


def fmt(x, nd=2):
    """Número no padrão pt-BR para LaTeX (vírgula decimal)."""
    return f"{x:.{nd}f}".replace(".", "{,}")

# Stopwords PT-BR (lista enxuta; usada só na representação c-TF-IDF dos tópicos, não nos embeddings)
STOP_PT = sorted(set("""
a à ao aos as às até com como da das de dela dele deles do dos e é ela elas ele eles em entre era essa esse
esta está estão estava este eu foi for há isso isto já la lá lhe mais mas me mesmo meu minha muito muita muitos
muitas na nas não nao nem no nos nós o os ou para pela pelas pelo pelos por pra pro qual quando que quem se sem
ser seu sua são só também te tem têm ter tinha tive to tá ta um uma umas uns vai vez você vc aqui ai aí então
ainda bem bom boa acho achei gostaria deveria poderia precisa ficar fica estar sendo seria sobre todo toda todos
todas tudo coisa coisas algum alguma pouco pouca poucos poucas mt mto passageiro passageira relata menciona sugere
diz dizer mostra aeroporto aeroportos q pq porque ja so esta estao sao nas nos de do da
""".split()))
