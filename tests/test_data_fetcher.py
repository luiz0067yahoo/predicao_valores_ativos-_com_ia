"""
Testes Unitários para a Ingestão de Dados e Engenharia de Atributos
"""

import pytest
import numpy as np
import pandas as pd
from data_fetcher import DataFetcher


def test_technical_indicators_calculation():
    # Criação de um DataFrame simulado de preços
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    # Passeio aleatório para simular cotação
    returns = np.random.normal(0.001, 0.02, size=len(dates))
    price_series = 100.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "Open": price_series * 0.99,
        "High": price_series * 1.01,
        "Low": price_series * 0.98,
        "Close": price_series,
        "Volume": np.random.randint(1000, 50000, size=len(dates))
    }, index=dates)

    indicators_df = DataFetcher.calculate_technical_indicators(df)

    expected_cols = [
        "SMA_5", "SMA_10", "SMA_20", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "RSI_14", "BB_Upper", "BB_Lower", "Volatility_10"
    ]

    for col in expected_cols:
        assert col in indicators_df.columns
        assert not indicators_df[col].isna().any()

    # RSI deve estar contido no intervalo [0, 100]
    assert indicators_df["RSI_14"].min() >= 0.0
    assert indicators_df["RSI_14"].max() <= 100.0


def test_prepare_lagged_features():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    prices = 50.0 + np.cumsum(np.random.normal(0, 1, size=len(dates)))

    df = pd.DataFrame({
        "Close": prices,
        "Volume": np.random.randint(100, 1000, size=len(dates))
    }, index=dates)

    lookback = 7
    X, y, target_index, scaler_params = DataFetcher.prepare_lagged_features(df, lookback=lookback)

    assert len(X) > 0
    assert len(X) == len(y)
    assert len(target_index) == len(X)
    # Quantidade de colunas: lookback lags + 4 indicadores + 1 bias = lookback + 5
    assert X.shape[1] == lookback + 5
    assert "last_close" in scaler_params
