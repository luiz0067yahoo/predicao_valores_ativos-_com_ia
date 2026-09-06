"""
Testes Unitários para Inferência e Preditor de Modelo
"""

import pytest
import numpy as np
import pandas as pd
from genetic_engine import Individual
from model_predictor import ModelPredictor


def test_model_predictor_evaluation_and_forecast():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.normal(0.2, 1.5, size=len(dates)))

    df = pd.DataFrame({"Close": prices}, index=dates)

    lookback = 5
    num_samples = len(df) - lookback
    feature_dim = lookback + 5
    X = np.random.randn(num_samples, feature_dim)

    # Indivíduo com genes definidos
    ind = Individual(chromosome_length=feature_dim, genes=np.random.uniform(-0.5, 0.5, size=feature_dim))

    predictor = ModelPredictor(best_individual=ind, lookback_window=lookback)
    results = predictor.evaluate_and_predict(
        df_raw=df,
        X=X,
        dates=dates[lookback:],
        forecast_horizon=7,
        is_crypto=False
    )

    assert "metrics" in results
    assert "history_df" in results
    assert "forecast_df" in results

    metrics = results["metrics"]
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics
    assert "variacao_esperada_pct" in metrics
    assert "tendencia_esperada" in metrics
    assert metrics["horizonte_dias"] == 7

    forecast_df = results["forecast_df"]
    assert len(forecast_df) == 7
    assert "Preco_Projetado" in forecast_df.columns
    assert "Limite_Inferior" in forecast_df.columns
    assert "Limite_Superior" in forecast_df.columns
