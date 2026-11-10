"""Capturas da interface (Streamlit já rodando em localhost:8501) para a defesa do TG-2.
Salva em apresentacoes/img_png/ (não toca em midias/)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "img_png"
URL = "http://localhost:8501"


def esperar_render(page, ms=2500):
    page.wait_for_selector('[data-testid="stApp"]')
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')?.innerText?.includes('Running')",
        timeout=90000)
    page.wait_for_timeout(ms)


def escolher_aeroporto(page, nome):
    sb = page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first
    sb.click()
    page.wait_for_timeout(800)
    page.keyboard.type(nome[:6])
    page.wait_for_timeout(800)
    page.get_by_role("option").filter(has_text=nome).first.click()
    esperar_render(page, 3500)


def aba(page, rotulo):
    t = page.get_by_role("tab", name=rotulo)
    t.click()
    esperar_render(page, 2500)
    t.evaluate("e => e.closest('[data-testid=\"stTabs\"]').scrollIntoView({block: 'start'})")
    page.mouse.move(5, 5)
    page.wait_for_timeout(1000)


def matriz(page, sufixo):
    """Recorte da aba Prioridades: o bloco com a matriz e a coluna 'Onde agir primeiro'."""
    painel = page.locator('[data-testid="stTabs"] [role="tabpanel"]').first
    bloco = painel.locator('[data-testid="stHorizontalBlock"]').first
    bloco.screenshot(path=OUT / f"defesa_matriz_{sufixo}.png")
    grafico = painel.locator('[data-testid="stVegaLiteChart"], .stVegaLiteChart').first
    if grafico.count():
        grafico.screenshot(path=OUT / f"defesa_matriz_{sufixo}_grafico.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        nav = p.chromium.launch()
        page = nav.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(URL)
        esperar_render(page, 5000)

        if "--so-sbfl" not in sys.argv:
            escolher_aeroporto(page, "Guarulhos")
            page.screenshot(path=OUT / "defesa_tela_inicial_SBGR.png")
            aba(page, "Prioridades (SHAP)")
            page.screenshot(path=OUT / "defesa_prioridades_SBGR.png")
            matriz(page, "SBGR")
            aba(page, "Relatório LLM")
            page.screenshot(path=OUT / "defesa_relatorio_llm_SBGR.png")
            if page.get_by_role("tab", name="Comentários").count():
                aba(page, "Comentários")
                page.screenshot(path=OUT / "defesa_comentarios_SBGR.png")
            page.evaluate("window.scrollTo(0, 0)")
        escolher_aeroporto(page, "Florianópolis")
        aba(page, "Prioridades (SHAP)")
        page.screenshot(path=OUT / "defesa_prioridades_SBFL.png")
        matriz(page, "SBFL")
        nav.close()
    print("ok", sorted(x.name for x in OUT.glob("defesa_*")))


if __name__ == "__main__":
    sys.exit(main())
