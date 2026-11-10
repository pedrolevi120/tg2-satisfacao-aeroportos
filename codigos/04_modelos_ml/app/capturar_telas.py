"""
Captura as telas do MVP para o TG (Playwright, viewport 1440x900, escala 2).

Sobe o Streamlit em modo headless, navega pelas abas e salva em midias/imagens/:
  mvp_01_selecao.png  mvp_02_shap.png  mvp_03_relatorio.png
  mvp_04_historico.png  mvp_05_comentarios.png (se a aba existir)  mvp_06_evolucao.png
Encerra o Streamlit ao final.

Uso: python codigos/04_modelos_ml/app/capturar_telas.py [--aeroporto SBGR]
"""
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
APP = Path(__file__).resolve().parent / "app.py"
OUT = ROOT / "midias" / "imagens"
PORTA = 8599
URL = f"http://localhost:{PORTA}"


def esperar_servidor(timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Streamlit não respondeu")


def esperar_render(page, ms=2500):
    page.wait_for_selector('[data-testid="stApp"]')
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')?.innerText?.includes('Running')",
        timeout=60000)
    page.wait_for_timeout(ms)


def aba(page, rotulo):
    t = page.get_by_role("tab", name=rotulo)
    t.click()
    esperar_render(page, 2000)
    # rola até a barra de abas, para o conteúdo da aba ocupar a tela
    t.evaluate("e => e.closest('[data-testid=\"stTabs\"]').scrollIntoView({block: 'start'})")
    page.mouse.move(200, 890)   # tira o mouse de cima de gráficos (evita a barra de hover)
    page.wait_for_timeout(800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aeroporto", default="SBGR")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true",
         "--server.port", str(PORTA), "--browser.gatherUsageStats", "false",
         "--theme.base", "light", "--client.toolbarMode", "minimal"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        esperar_servidor()
        with sync_playwright() as p:
            nav = p.chromium.launch()
            page = nav.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
            page.goto(URL)
            esperar_render(page, 4000)

            # Seleciona o aeroporto no primeiro selectbox da sidebar
            sb = page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first
            sb.click()
            page.wait_for_timeout(800)
            page.get_by_role("option", name=args.aeroporto).first.click()
            esperar_render(page)

            # 01 — seleção: dropdown de aeroportos aberto
            sb.click()
            page.wait_for_timeout(1000)
            page.screenshot(path=OUT / "mvp_01_selecao.png")
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)

            aba(page, "Prioridades (SHAP)")
            page.screenshot(path=OUT / "mvp_02_shap.png")
            aba(page, "Evolução da satisfação")
            page.screenshot(path=OUT / "mvp_06_evolucao.png")
            aba(page, "Relatório LLM")
            page.screenshot(path=OUT / "mvp_03_relatorio.png")
            aba(page, "Histórico")
            page.screenshot(path=OUT / "mvp_04_historico.png")
            if page.get_by_role("tab", name="Comentários").count():
                aba(page, "Comentários")
                page.screenshot(path=OUT / "mvp_05_comentarios.png")
            nav.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("Telas salvas em", OUT)


if __name__ == "__main__":
    main()
