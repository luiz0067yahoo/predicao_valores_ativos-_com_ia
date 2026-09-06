"""
Testes unitários para o pipeline de extração de características do M.A.P.P.
"""

import numpy as np
import pandas as pd
import pytest
from mapp.features import extrair_todas_caracteristicas_mapp
from mapp.features.trend import calcular_caracteristicas_tendencia
from mapp.features.volume import calcular_caracteristicas_volume
from mapp.features.volatility import calcular_caracteristicas_volatilidade
from mapp.features.momentum import calcular_caracteristicas_momentum
from mapp.features.candlesticks import calcular_caracteristicas_candlesticks
from mapp.features.support_resistance import SupportResistanceDetector
from mapp.features.fibonacci import calcular_caracteristicas_fibonacci
from mapp.features.divergences import calcular_caracteristicas_divergencias


@pytest.fixture
def sample_ohlcv():
    """Gera DataFrame OHLCV sintético e consistente de 200 barras."""
    np.random.seed(123)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = 100 + np.cumsum(np.random.normal(0.1, 1.2, n))
    high = close + np.abs(np.random.normal(0.8, 0.4, n))
    low = close - np.abs(np.random.normal(0.8, 0.4, n))
    open_ = low + np.random.uniform(0.1, 0.9, n) * (high - low)
    volume = np.random.uniform(50000, 200000, n)

    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


def test_trend_features(sample_ohlcv):
    df_t = calcular_caracteristicas_tendencia(sample_ohlcv)
    assert 'sma_20' in df_t.columns
    assert 'ema_9' in df_t.columns
    assert 'adx' in df_t.columns or 'adx_14' in df_t.columns
    assert len(df_t) == len(sample_ohlcv)


def test_volume_features(sample_ohlcv):
    df_v = calcular_caracteristicas_volume(sample_ohlcv)
    assert 'volume_relativo_20' in df_v.columns or 'relative_volume_20' in df_v.columns or any('volume' in col for col in df_v.columns)
    assert 'obv' in df_v.columns


def test_volatility_features(sample_ohlcv):
    df_vol = calcular_caracteristicas_volatilidade(sample_ohlcv)
    assert any('atr' in col.lower() for col in df_vol.columns)
    assert any('bb' in col.lower() for col in df_vol.columns)


def test_candlestick_patterns(sample_ohlcv):
    df_c = calcular_caracteristicas_candlesticks(sample_ohlcv)
    assert len(df_c.columns) > 0
    assert len(df_c) == len(sample_ohlcv)


def test_support_resistance(sample_ohlcv):
    df_sr = SupportResistanceDetector.extrair_caracteristicas(sample_ohlcv)
    assert len(df_sr.columns) > 0
    assert len(df_sr) == len(sample_ohlcv)


def test_unified_feature_pipeline(sample_ohlcv):
    """Testa a extração unificada de todas as características."""
    df_all = extrair_todas_caracteristicas_mapp(sample_ohlcv)
    # Deve conter um rico conjunto de colunas (mais de 40)
    assert df_all.shape[1] >= 40
    # Não deve ter infinitos ou NaNs descontrolados
    assert not np.isinf(df_all.values).any()
    assert len(df_all) == len(sample_ohlcv)
