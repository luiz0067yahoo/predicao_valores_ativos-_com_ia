"""
Testes unitários para persistência do banco de dados local em arquivos Excel (.xls / .xlsx)
e consulta inteligente de intervalos em cache (tests/test_db_excel_cache.py).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from config import DB_DIR
from data_fetcher import DataFetcher


def test_obter_caminho_arquivo_excel():
    """Verifica se os nomes de arquivos canônicos são mapeados corretamente."""
    caminho_btc = DataFetcher.obter_caminho_arquivo_excel("BTC-USD")
    assert caminho_btc.name == "bitcoin.xls"
    assert caminho_btc.parent == DB_DIR

    caminho_dolar = DataFetcher.obter_caminho_arquivo_excel("USDBRL=X")
    assert caminho_dolar.name == "dolar_usd_brl.xls"

    caminho_ibov = DataFetcher.obter_caminho_arquivo_excel("^BVSP")
    assert caminho_ibov.name == "ibovespa.xls"

    caminho_custom = DataFetcher.obter_caminho_arquivo_excel("PETR4.SA")
    assert caminho_custom.name == "petr4_sa.xls"


def test_carregamento_direto_excel_quando_intervalo_coberto(tmp_path):
    """
    Verifica se, quando o arquivo .xls na pasta db/ já contém o intervalo solicitado,
    o sistema carrega direto do Excel sem chamar a API externa do Yahoo Finance.
    """
    ticker_teste = "TEST_ASSET"
    caminho_excel = tmp_path / "test_asset.xls"

    datas = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    df_falso = pd.DataFrame({
        "Open": np.linspace(10, 20, len(datas)),
        "High": np.linspace(11, 21, len(datas)),
        "Low": np.linspace(9, 19, len(datas)),
        "Close": np.linspace(10, 20, len(datas)),
        "Volume": np.random.randint(100, 1000, len(datas))
    }, index=datas)

    df_falso.to_excel(caminho_excel, engine="openpyxl")
    assert caminho_excel.exists()

    fetcher = DataFetcher(use_cache=True)
    fetcher.db_dir = tmp_path

    with patch.object(DataFetcher, "obter_caminho_arquivo_excel", return_value=caminho_excel), \
         patch("yfinance.Ticker") as mock_yf:

        # Requisita um intervalo contido nos dados salvos (Março a Junho de 2024)
        df_resultado = fetcher.fetch_asset_data(
            ticker=ticker_teste,
            start_date="2024-03-01",
            end_date="2024-06-30"
        )

        # O yfinance.Ticker NÃO deve ter sido chamado porque os dados vieram do Excel
        mock_yf.assert_not_called()
        assert not df_resultado.empty
        assert df_resultado.index.min() >= pd.Timestamp("2024-03-01")
        assert df_resultado.index.max() <= pd.Timestamp("2024-06-30")


def test_atualizacao_e_salvamento_excel_quando_nao_coberto(tmp_path):
    """
    Verifica se, quando o intervalo não está em cache ou o arquivo não existe,
    o sistema busca novos dados e salva na pasta db/.
    """
    ticker_teste = "TEST_NOVOS_DADOS"
    caminho_excel = tmp_path / "test_novos_dados.xls"

    datas_novas = pd.date_range("2025-01-01", periods=60, freq="B")
    df_novo = pd.DataFrame({
        "Open": np.linspace(50, 60, 60),
        "High": np.linspace(51, 61, 60),
        "Low": np.linspace(49, 59, 60),
        "Close": np.linspace(50, 60, 60),
        "Volume": np.random.randint(1000, 5000, 60)
    }, index=datas_novas)

    fetcher = DataFetcher(use_cache=True)
    fetcher.db_dir = tmp_path

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = df_novo

    with patch.object(DataFetcher, "obter_caminho_arquivo_excel", return_value=caminho_excel), \
         patch("yfinance.Ticker", return_value=mock_ticker_instance):

        df_obtido = fetcher.fetch_asset_data(
            ticker=ticker_teste,
            start_date="2025-01-01",
            end_date="2025-03-31"
        )

        assert caminho_excel.exists()
        df_lido_do_disco = pd.read_excel(caminho_excel, index_col=0, engine="openpyxl")
        assert len(df_lido_do_disco) == 60
        assert "Close" in df_lido_do_disco.columns
