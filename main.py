"""
Ponto de Entrada Principal da Aplicação
Sistema de Otimização e Previsão de Ativos com Algoritmo Genético e Yahoo Finance
Suporta Interface Web Moderna (padrão) e Interface Desktop (Tkinter).
"""

import argparse
import logging
import sys
import threading
import time
import webbrowser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def start_web_interface(port: int = 5000, open_browser: bool = True):
    """Inicializa o servidor web Flask e abre o navegador."""
    url = f"http://127.0.0.1:{port}"
    logger.info(f"Iniciando Interface Web em {url}...")

    if open_browser:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open_new_tab(url)
            except Exception as ex:
                logger.warning(f"Não foi possível abrir o navegador automaticamente: {ex}")
        threading.Thread(target=_open, daemon=True).start()

    from web_app import run_web_server
    run_web_server(host="127.0.0.1", port=port, debug=False)


def start_gui_interface():
    """Inicializa a interface gráfica desktop (Tkinter)."""
    logger.info("Iniciando Interface Gráfica Desktop (Tkinter)...")
    try:
        from gui import ModernFinancialGUI
        app = ModernFinancialGUI()
        app.mainloop()
    except ImportError as e:
        logger.error(f"Dependência ausente: {e}")
        print("\n[ERRO] Dependências necessárias não encontradas.")
        print("Por favor, instale as dependências executando:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro fatal na aplicação: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Função principal com parsing de argumentos."""
    parser = argparse.ArgumentParser(
        description="AI Asset Predictor - Algoritmo Genético & Previsão de Séries Temporais"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Inicia a interface web moderna no navegador (padrão)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Inicia a interface desktop baseada em Tkinter"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Porta para o servidor web (padrão: 5000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Não abre o navegador automaticamente ao iniciar o servidor web"
    )

    args = parser.parse_args()

    if args.gui:
        start_gui_interface()
    else:
        # Por padrão, inicia a interface web moderna
        start_web_interface(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
