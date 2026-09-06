"""
Testes unitários e de integração para o Simulador de Investimentos e métricas.
"""

import numpy as np
import pandas as pd
import pytest
from mapp.simulator.investment_simulator import InvestmentSimulator
from mapp.backtesting.metrics import CalculadorMetricas
from mapp.horizon import NormalizadorHorizonte
from web_app import app


@pytest.fixture
def test_client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_df():
    np.random.seed(99)
    n = 150
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    close = 50 + np.cumsum(np.random.normal(0.05, 0.8, n))
    return pd.DataFrame({
        'Open': close,
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Volume': np.random.uniform(1e4, 5e4, n)
    }, index=dates)


def test_calculador_metricas():
    retornos = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.03])
    equity = np.array([10000, 10100, 10049.5, 10250.49, 10404.25, 10300.2, 10609.2])
    y_true = np.array([10.0, 10.5, 11.0, 10.8])
    y_pred = np.array([10.1, 10.4, 10.9, 11.0])

    metricas = CalculadorMetricas.calcular_todas(equity, retornos, y_true, y_pred)
    assert metricas.retorno_total_pct > 0
    assert metricas.sharpe is not None
    assert metricas.max_drawdown_pct >= 0
    assert metricas.rmse > 0


def test_investment_simulator_run(sample_df):
    simulador = InvestmentSimulator(capital_inicial=10000.0)
    horizonte = NormalizadorHorizonte.normalizar(5, 'dias')
    resultado = simulador.simular_multiplos_algoritmos(
        df=sample_df,
        algoritmos_selecionados=['regressao_linear', 'random_forest'],
        horizonte=horizonte
    )

    assert resultado['success'] is True
    assert 'comparison_table' in resultado
    assert len(resultado['comparison_table']) == 2
    assert 'equity_curves' in resultado
    assert 'highlights' in resultado


def test_simulator_web_routes(test_client):
    # GET na página do simulador
    resp = test_client.get('/portfolio-simulator')
    assert resp.status_code == 200
    assert b'Simulador de Investimento' in resp.data
    assert b'M.A.P.P.' in resp.data

    # POST com parâmetros inválidos
    resp_bad = test_client.post('/api/simulator/run', json={})
    assert resp_bad.status_code == 400

    # GET progresso de simulação inexistente
    resp_prog = test_client.get('/api/simulator/progress?id_simulacao=inexistente')
    assert resp_prog.status_code == 404
