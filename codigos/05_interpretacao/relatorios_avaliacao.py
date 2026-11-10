"""
Métricas automáticas de avaliação dos relatórios gerados por LLM.

(a) Fidelidade numérica: extrai todos os números do relatório e confronta com
    os números presentes no prompt (dados de entrada). Cada número recebe uma
    das classes:
      - estrutural: numeração de seções/listas ("## 3.", "1. ...") → ignorado;
      - exato:      igual a um número do prompt;
      - arredondado: igual a um número do prompt arredondado às casas exibidas;
      - derivado:   diferença entre dois números do prompt (ex.: 4,38 − 4,21);
      - inventado:  nenhuma das anteriores.
    Fidelidade = (exato + arredondado + derivado) / (total não estrutural).
(b) Cobertura das 5 prioridades: cada atributo conta como coberto se todos os
    seus termos significativos (radicais de 5 letras, sem acento) aparecem no texto.
(c) Estrutura: seções exigidas pelo prompt, frase de encerramento, nº de palavras.
(d) Erros de PT-BR: LanguageTool (pt-BR) sobre o texto sem marcação Markdown.
"""
import re
import unicodedata

SECOES = {
    "sumario": r"sum[aá]rio",
    "diagnostico": r"diagn[oó]stico",
    "prioridades": r"prioridades",
    "recomendacoes": r"recomenda[cç]",
    "riscos": r"riscos",
    "kpis": r"indicadores|kpi",
}
FRASE_FINAL = "análise gerada automaticamente"
STOP = {"de", "da", "do", "das", "dos", "e", "no", "na", "nos", "nas", "a", "o", "em"}

_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])")


def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _interpretacoes(tok: str):
    """Valores plausíveis de um token numérico (pt-BR e en), com nº de casas."""
    out = []
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", tok):          # 6.054 ou 1.234,5
        inteiro, _, dec = tok.partition(",")
        out.append((float(inteiro.replace(".", "") + ("." + dec if dec else "")), len(dec)))
    if "," in tok and "." not in tok:
        a, b = tok.split(",")
        out.append((float(a + "." + b), len(b)))                   # 4,21
        if len(b) == 3:
            out.append((float(a + b), 0))                          # 6,054 (milhar en)
    if "." in tok and "," not in tok:
        a, b = tok.split(".")
        out.append((float(tok), len(b)))                           # 0.176
    if tok.isdigit():
        out.append((float(tok), 0))
    return out


def extrair_numeros(texto: str):
    """Lista de (token, [(valor, casas)], estrutural, contexto)."""
    res = []
    for linha in texto.splitlines():
        m_estr = re.match(r"^\s*(?:#+\s*|\*\*)?\s*(\d{1,2})[.)]\s", linha)
        for m in _NUM.finditer(linha):
            tok = m.group(1)
            estrutural = bool(m_estr and m.start(1) == m_estr.start(1))
            ctx = linha[max(0, m.start() - 40): m.end() + 40].strip()
            res.append((tok, _interpretacoes(tok), estrutural, ctx))
    return res


def valores_prompt(prompt: str):
    """Números do prompt, lidos pelo formato em que o próprio prompt os escreve
    (vírgula decimal no texto, ponto decimal nos valores SHAP, inteiros sem milhar)."""
    vals = set()
    for m in _NUM.finditer(prompt):
        vals.add(float(m.group(1).replace(",", ".")))
    return sorted(vals)


def classificar(v, casas, permitidos, derivados):
    tol = 1e-9
    if any(abs(v - a) < tol for a in permitidos):
        return "exato"
    meio = 0.5 * 10 ** (-casas) + 1e-9
    # arredondamento: só aceita se o texto exibe menos precisão que o valor do prompt
    if any(abs(v - a) <= meio and (casas > 0 or a >= 10) for a in permitidos):
        return "arredondado"
    if any(abs(v - d) <= meio for d in derivados):
        return "derivado"
    return "inventado"


def fidelidade_numerica(texto: str, prompt: str) -> dict:
    permitidos = valores_prompt(prompt)
    # derivados: diferenças entre valores não inteiros (ex.: 4,38 − 4,21; 100 − 88,7)
    # e razões entre valores SHAP (ex.: "quase o dobro")
    frac = [a for a in permitidos if a != int(a)] + [100.0]
    derivados = {round(abs(a - b), 6) for a in frac for b in frac if a != b}
    shap = [a for a in permitidos if 0 < a < 1]
    derivados |= {round(a / b, 6) for a in shap for b in shap if a > b}
    cont = {"exato": 0, "arredondado": 0, "derivado": 0, "inventado": 0, "estrutural": 0}
    inventados = []
    for tok, interps, estrutural, ctx in extrair_numeros(texto):
        if estrutural:
            cont["estrutural"] += 1
            continue
        ordem = ["exato", "arredondado", "derivado", "inventado"]
        classes = [classificar(v, c, permitidos, derivados) for v, c in interps] or ["inventado"]
        cl = min(classes, key=ordem.index)
        cont[cl] += 1
        if cl == "inventado":
            inventados.append({"token": tok, "contexto": ctx})
    total = sum(cont[k] for k in ("exato", "arredondado", "derivado", "inventado"))
    return {**{f"num_{k}": v for k, v in cont.items()},
            "num_total": total,
            "fidelidade": (total - cont["inventado"]) / total if total else float("nan"),
            "inventados": inventados}


def radicais(feature: str):
    toks = re.findall(r"[a-z]+", sem_acento(feature))
    return [t[:5] for t in toks if t not in STOP and len(t) >= 4]


def cobertura_prioridades(texto: str, features) -> dict:
    t = sem_acento(texto)
    cobertos = [f for f in features if all(r in t for r in radicais(f))]
    return {"cobertura": len(cobertos) / len(features),
            "n_cobertos": len(cobertos),
            "nao_cobertos": [f for f in features if f not in cobertos]}


def estrutura(texto: str) -> dict:
    titulos = "\n".join(l for l in texto.splitlines()
                        if re.match(r"^\s*(#+|\*\*|\d+\.)", l))
    tit = sem_acento(titulos)
    secoes = {k: bool(re.search(sem_acento(p), tit)) for k, p in SECOES.items()}
    palavras = len(re.findall(r"\w+", limpar_markdown(texto)))
    return {"n_secoes": sum(secoes.values()),
            "frase_final": FRASE_FINAL in texto.lower(),
            "palavras": palavras,
            "extensao_ok": 700 <= palavras <= 1000}


def limpar_markdown(texto: str) -> str:
    t = re.sub(r"^\s*#+\s*", "", texto, flags=re.M)
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.M)
    t = t.replace("**", "").replace("__", "")
    return t


# Regras do LanguageTool desconsideradas por serem de tipografia/estilo, não erro
REGRAS_IGNORADAS = {"WHITESPACE_RULE", "UNPAIRED_BRACKETS", "PUNCTUATION_PARAGRAPH_END",
                    "DOUBLE_PUNCTUATION", "TYPOGRAPHY", "PT_WORDINESS_REPLACE",
                    "PT_BARBARISMS_REPLACE", "PT_CLICHE_REPLACE", "PT_REDUNDANCY_REPLACE",
                    "PARONYM_POLITICA_123", "UPPERCASE_SENTENCE_START", "PT_WORD_REPEAT_RULE",
                    "PT_WORD_REPEAT_BEGINNING_RULE", "PT_AGREEMENT_REPLACE",
                    "PT_MULTI_REPLACE"}
CATS_IGNORADAS = {"TYPOGRAPHY", "STYLE", "REDUNDANCY", "REPETITIONS_STYLE", "CASING"}

_lt = None


def ferramenta_lt():
    global _lt
    if _lt is None:
        import language_tool_python
        _lt = language_tool_python.LanguageTool("pt-BR")
    return _lt


def erros_ptbr(texto: str, prompt: str = "") -> dict:
    """Conta erros ortográficos e gramaticais (por 1.000 palavras).

    Não contam como erro ortográfico: siglas em caixa alta (SBGR, SHAP, ANAC)
    e palavras que já aparecem no prompt (nomes próprios, concessionárias)."""
    lt = ferramenta_lt()
    t = limpar_markdown(texto)
    ok = {sem_acento(x) for x in re.findall(r"\w+", prompt)}
    orto, gram, exemplos = 0, 0, []
    for m in lt.check(t):
        if m.rule_id in REGRAS_IGNORADAS or m.category in CATS_IGNORADAS:
            continue
        trecho = t[m.offset: m.offset + m.error_length]
        eh_orto = "MORFOLOGIK" in m.rule_id or m.category == "TYPOS"
        if eh_orto and (sem_acento(trecho) in ok or re.fullmatch(r"[A-Z0-9]{2,6}s?", trecho)):
            continue
        if eh_orto:
            orto += 1
        else:
            gram += 1
        if len(exemplos) < 15:
            exemplos.append({"regra": m.rule_id, "trecho": trecho,
                             "sugestao": (m.replacements or [""])[0]})
    palavras = max(1, len(re.findall(r"\w+", t)))
    return {"erros_orto": orto, "erros_gram": gram,
            "erros_por_mil": 1000 * (orto + gram) / palavras,
            "erros_exemplos": exemplos}


def associacao_shap(texto: str, top_atributos) -> dict:
    """Verifica se cada valor SHAP citado aparece junto do atributo a que pertence.

    O texto é dividido em blocos (parágrafos); cada bloco herda o último título
    (cabeçalho Markdown ou linha inteira em negrito; "**Rótulo:** texto" não é título).
    Um valor SHAP citado (exato ou arredondado) é 'correto' se os radicais do
    atributo dono do valor aparecem no bloco ou no seu título."""
    ok = total = 0
    titulo = ""
    for bloco in re.split(r"\n\s*\n", texto):
        linhas = bloco.strip().splitlines()
        if not linhas:
            continue
        for ln in linhas:   # título = cabeçalho Markdown ou linha inteira em negrito
            if re.match(r"^\s*#+\s", ln) or re.fullmatch(r"\s*(\d+\.\s*)?\*\*[^*]+\*\*:?\s*", ln):
                titulo = ln
        alvo = sem_acento(titulo + " " + bloco)
        for tok, interps, _, _ in extrair_numeros(bloco):
            for a in top_atributos:
                s = a["shap_abs"]
                if any(c > 0 and abs(v - s) <= 0.5 * 10 ** (-c) + 1e-9 and v < 1
                       for v, c in interps):
                    total += 1
                    ok += all(r in alvo for r in radicais(a["feature"]))
                    break
    return {"shap_citados": total, "shap_assoc_ok": ok,
            "shap_assoc": ok / total if total else float("nan")}
