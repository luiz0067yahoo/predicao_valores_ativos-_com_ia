"""
Ponto de Entrada Principal da Aplicação
Sistema de Otimização e Previsão de Ativos com Algoritmo Genético e Yahoo Finance
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Inicializa a interface gráfica moderna."""
    logger.info("Iniciando AI Asset Predictor...")
    try:
        from gui import ModernFinancialGUI
        app = ModernFinancialGUI()
        app.mainloop()
    except ImportError as e:
        logger.error(f"Dependência ausente: {e}")
        print("\n[ERRO] Algumas bibliotecas necessárias não estão instaladas.")
        print("Por favor, instale as dependências executando:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro fatal na aplicação: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
