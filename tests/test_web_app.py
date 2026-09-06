"""
Testes unitários e de integração para o servidor web Flask (web_app.py).
"""

import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from web_app import app, _parse_train_params, LAST_RESULT


@pytest.fixture
def client():
    """Fixture para cliente de testes do Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verifica se a rota raiz entrega a página HTML do dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "AI ASSET PREDICTOR" in html
    assert "chart-forecast-canvas" in html
    assert "Parâmetros do Modelo" in html


def test_api_config(client):
    """Verifica se o endpoint /api/config retorna as definições esperadas."""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.get_json()
    assert "assets" in data
    assert "periods" in data
    assert "default_ga" in data
    assert "Dólar (USD/BRL)" in data["assets"]
    assert "population_size" in data["default_ga"]


def test_api_history(client):
    """Verifica se o endpoint /api/history retorna estrutura correta."""
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.get_json()
    assert "records" in data
    assert isinstance(data["records"], list)


def test_api_reports(client):
    """Verifica listagem de relatórios."""
    response = client.get("/api/reports")
    assert response.status_code == 200
    data = response.get_json()
    assert "reports" in data
    assert isinstance(data["reports"], list)


def test_parse_train_params_valid():
    """Testa extração de parâmetros válidos."""
    params = {
        "asset": "Dólar (USD/BRL)",
        "period": "1 Ano",
        "population_size": "50",
        "generations": "20",
        "mutation_rate": "15",
        "forecast_horizon": "5"
    }
    parsed = _parse_train_params(params)
    assert parsed["ticker"] == "USDBRL=X"
    assert parsed["currency"] == "BRL"
    assert parsed["pop_size"] == 50
    assert parsed["generations"] == 20
    assert parsed["mutation_rate"] == 0.15
    assert parsed["horizon"] == 5


def test_parse_train_params_custom_without_ticker():
    """Testa se rejeita ticker customizado vazio."""
    params = {
        "asset": "Personalizado",
        "custom_ticker": "",
        "period": "1y"
    }
    with pytest.raises(ValueError, match="informe o código do Ticker"):
        _parse_train_params(params)


def test_export_docx_no_prediction(client):
    """Verifica que exportar sem predição recente retorna erro 400."""
    LAST_RESULT["prediction_results"] = None
    LAST_RESULT["ga_history"] = None
    response = client.post("/api/export-docx")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False


def test_predict_sync_mocked(client):
    """Testa rota /api/predict com simulação dos dados e motor genético."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    raw_df = pd.DataFrame({
        "Open": np.linspace(5.0, 5.5, 60),
        "High": np.linspace(5.1, 5.6, 60),
        "Low": np.linspace(4.9, 5.4, 60),
        "Close": np.linspace(5.0, 5.5, 60),
        "Volume": np.random.randint(100, 1000, 60)
    }, index=dates)

    history_df = pd.DataFrame({
        "Preco_Real": np.linspace(5.0, 5.5, 30),
        "Preco_Previsto_IA": np.linspace(5.02, 5.48, 30)
    }, index=dates[-30:])

    forecast_dates = pd.date_range("2025-03-03", periods=3, freq="D")
    forecast_df = pd.DataFrame({
        "Preco_Projetado": [5.55, 5.60, 5.65],
        "Limite_Inferior": [5.45, 5.50, 5.55],
        "Limite_Superior": [5.65, 5.70, 5.75]
    }, index=forecast_dates)

    mock_results = {
        "metrics": {
            "ultimo_preco_real": 5.50,
            "preco_projetado_final": 5.65,
            "variacao_esperada_pct": 2.73,
            "tendencia_esperada": "Alta Estimada",
            "rmse": 0.015,
            "mae": 0.012,
            "mape": 0.25,
            "r2": 0.96,
            "acuracia_direcional": 85.0
        },
        "history_df": history_df,
        "forecast_df": forecast_df
    }

    with patch("web_app.DataFetcher") as mock_df_cls, \
         patch("web_app.GeneticEngine") as mock_ge_cls, \
         patch("web_app.ModelPredictor") as mock_mp_cls:

        # Mock DataFetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_asset_data.return_value = raw_df
        mock_fetcher.prepare_lagged_features.return_value = (
            np.random.rand(30, 15),
            np.random.rand(30),
            dates[-30:],
            (0.0, 1.0)
        )
        mock_df_cls.return_value = mock_fetcher

        # Mock GeneticEngine
        mock_engine = MagicMock()
        mock_engine.evolve.return_value = np.ones(15)
        mock_engine.history = {
            "generation": [1, 2],
            "best_fitness": [850.0, 920.0],
            "avg_fitness": [400.0, 550.0],
            "best_rmse": [0.03, 0.015]
        }
        mock_ge_cls.return_value = mock_engine

        # Mock ModelPredictor
        mock_predictor = MagicMock()
        mock_predictor.evaluate_and_predict.return_value = mock_results
        mock_mp_cls.return_value = mock_predictor

        payload = {
            "asset": "Dólar (USD/BRL)",
            "period": "1 Mês",
            "population_size": 20,
            "generations": 5,
            "mutation_rate": 10,
            "forecast_horizon": 3
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["ticker"] == "USDBRL=X"
        assert len(data["forecast"]) == 3
        assert data["metrics"]["tendencia_esperada"] == "Alta Estimada"
