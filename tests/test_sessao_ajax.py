"""
Testes unitários para as rotas AJAX com Sessão e Exportação de PDF/DOCX em web_app.py.
"""

import json
import time
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from web_app import app, ULTIMO_RESULTADO


@pytest.fixture
def cliente_teste():
    """Cliente Flask de teste com suporte a cookies/sessão."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_algoritmos_catalogo(cliente_teste):
    """Verifica se o endpoint /api/algoritmos retorna todos os 11 modelos."""
    resposta = cliente_teste.get("/api/algoritmos")
    assert resposta.status_code == 200
    dados = resposta.get_json()
    assert len(dados["algoritmos"]) == 13
    identificadores = [a["identificador"] for a in dados["algoritmos"]]
    assert "xgboost" in identificadores
    assert "lstm" in identificadores
    assert "transformer" in identificadores
    assert "mirofish" in identificadores
    assert "prophet" in identificadores
    assert "rede_neural" in identificadores
    assert "logica_fuzzy" in identificadores


def test_iniciar_treinamento_ajax_e_progresso_sessao(cliente_teste):
    """Testa o disparo assíncrono via POST e a consulta de progresso pela sessão."""
    datas = pd.date_range("2025-01-01", periods=60, freq="B")
    df_falso = pd.DataFrame({
        "Open": np.linspace(5.0, 5.5, 60),
        "High": np.linspace(5.1, 5.6, 60),
        "Low": np.linspace(4.9, 5.4, 60),
        "Close": np.linspace(5.0, 5.5, 60),
        "Volume": np.random.randint(1000, 5000, 60)
    }, index=datas)

    with patch("web_app.DataFetcher.fetch_asset_data", return_value=df_falso):
        corpo = {
            "asset": "Dólar (USD/BRL)",
            "period": "1 Mês",
            "algoritmo": "regressao_linear",
            "horizonte_projecao": 3
        }
        resp_inicio = cliente_teste.post(
            "/api/iniciar-treinamento",
            data=json.dumps(corpo),
            content_type="application/json"
        )
        assert resp_inicio.status_code == 200
        dados_inicio = resp_inicio.get_json()
        assert dados_inicio["sucesso"] is True
        id_tarefa = dados_inicio["id_tarefa"]
        assert id_tarefa != ""

        # Aguarda brevemente a thread finalizar o modelo leve de regressão linear
        time.sleep(1.0)

        # Consulta progresso pela sessão
        resp_poll = cliente_teste.get(f"/api/progresso-sessao?id_tarefa={id_tarefa}")
        assert resp_poll.status_code == 200
        dados_poll = resp_poll.get_json()
        assert "progresso" in dados_poll
        assert "concluido" in dados_poll
        assert dados_poll["concluido"] is True
        assert dados_poll["resultado"] is not None
        assert dados_poll["resultado"]["algoritmo"] == "regressao_linear"


def test_exportacao_pdf_e_docx_com_resultado(cliente_teste):
    """Verifica download dos endpoints /api/export-pdf e /api/export-docx."""
    datas_hist = pd.date_range("2025-01-01", periods=30, freq="B")
    datas_proj = pd.date_range("2025-02-12", periods=3, freq="B")

    df_hist = pd.DataFrame({
        "Preco_Real": np.linspace(5.0, 5.5, 30),
        "Preco_Previsto_IA": np.linspace(5.02, 5.48, 30)
    }, index=datas_hist)

    df_proj = pd.DataFrame({
        "Preco_Projetado": [5.55, 5.58, 5.60],
        "Limite_Inferior": [5.40, 5.42, 5.45],
        "Limite_Superior": [5.70, 5.72, 5.75]
    }, index=datas_proj)

    # Injeta estado simulado
    ULTIMO_RESULTADO["prediction_results"] = {
        "metrics": {
            "ultimo_preco_real": 5.50,
            "preco_projetado_final": 5.60,
            "variacao_esperada_pct": 1.82,
            "tendencia_esperada": "Alta Estimada",
            "rmse": 0.02,
            "mae": 0.015,
            "mape": 0.3,
            "r2": 0.95,
            "acuracia_direcional": 80.0
        },
        "history_df": df_hist,
        "forecast_df": df_proj
    }
    ULTIMO_RESULTADO["ga_history"] = {
        "generations": [1, 2, 3],
        "best_fitness": [100, 200, 300],
        "avg_fitness": [50, 100, 150]
    }
    ULTIMO_RESULTADO["asset_name"] = "Dólar (USD/BRL)"
    ULTIMO_RESULTADO["ticker"] = "USDBRL=X"
    ULTIMO_RESULTADO["currency"] = "BRL"
    ULTIMO_RESULTADO["period_str"] = "1 Ano"
    ULTIMO_RESULTADO["algoritmo"] = "xgboost"
    ULTIMO_RESULTADO["nome_algoritmo"] = "XGBoost"
    ULTIMO_RESULTADO["hiperparametros"] = {"numero_estimadores": 50}
    ULTIMO_RESULTADO["nome_base_arquivo"] = "dolar_usd_brl"
    ULTIMO_RESULTADO["data_inicio_formatada"] = "01-01-2025"
    ULTIMO_RESULTADO["data_fim_formatada"] = "11-02-2025"

    # Testa PDF
    resp_pdf = cliente_teste.get("/api/export-pdf")
    assert resp_pdf.status_code == 200
    assert "application/pdf" in resp_pdf.content_type
    assert "dolar_usd_brl 01-01-2025 a 11-02-2025.pdf" in resp_pdf.headers.get("Content-Disposition", "")

    # Testa DOCX
    resp_docx = cliente_teste.get("/api/export-docx")
    assert resp_docx.status_code == 200
    assert "openxmlformats" in resp_docx.content_type
    assert "dolar_usd_brl 01-01-2025 a 11-02-2025.docx" in resp_docx.headers.get("Content-Disposition", "")

    # Testa cenário Bitcoin com datas explícitas (exemplo do usuário: bitcoin 30-12-2024 a 30-12-2025.pdf)
    ULTIMO_RESULTADO["ticker"] = "BTC-USD"
    ULTIMO_RESULTADO["nome_base_arquivo"] = "bitcoin"
    ULTIMO_RESULTADO["data_inicio_formatada"] = "30-12-2024"
    ULTIMO_RESULTADO["data_fim_formatada"] = "30-12-2025"

    resp_btc_pdf = cliente_teste.get("/api/export-pdf")
    assert resp_btc_pdf.status_code == 200
    assert "bitcoin 30-12-2024 a 30-12-2025.pdf" in resp_btc_pdf.headers.get("Content-Disposition", "")

    resp_btc_docx = cliente_teste.get("/api/export-docx")
    assert resp_btc_docx.status_code == 200
    assert "bitcoin 30-12-2024 a 30-12-2025.docx" in resp_btc_docx.headers.get("Content-Disposition", "")
