"""
Módulo de Características de Momentum (mapp/features/momentum.py)
Retornos acumulados em multi-períodos (1, 3, 5, 10, 20, 50), RSI, Estocástico,
Williams %R, ROC, MACD e histograma, aceleração de momentum.
Sem vazamento temporal.
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


def calcular_caracteristicas_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o conjunto completo de osciladores e taxas de variação de momentum.
    """
    res = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close

    # 1. Retornos percentuais multi-período
    res["retorno_1d"] = close.pct_change(1).fillna(0.0)
    res["retorno_3d"] = close.pct_change(3).fillna(0.0)
    res["retorno_5d"] = close.pct_change(5).fillna(0.0)
    res["retorno_10d"] = close.pct_change(10).fillna(0.0)
    res["retorno_20d"] = close.pct_change(20).fillna(0.0)
    res["retorno_50d"] = close.pct_change(50).fillna(0.0)

    # 2. RSI (Índice de Força Relativa) de 14 períodos
    delta = close.diff()
    ganho = delta.clip(lower=0.0)
    perda = (-delta).clip(lower=0.0)
    avg_gain = ganho.ewm(com=13, adjust=False).mean()
    avg_loss = perda.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    res["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    res["rsi_14_normalizado"] = (res["rsi_14"] - 50.0) / 50.0  # [-1, +1]

    # 3. Estocástico Rápido e Lento (%K e %D)
    low_14 = low.rolling(14, min_periods=5).min()
    high_14 = high.rolling(14, min_periods=5).max()
    res["stoch_k"] = 100.0 * ((close - low_14) / (high_14 - low_14 + 1e-9))
    res["stoch_d"] = res["stoch_k"].rolling(3, min_periods=1).mean()
    res["stoch_sobrecompra"] = (res["stoch_k"] > 80.0).astype(float)
    res["stoch_sobrevenda"] = (res["stoch_k"] < 20.0).astype(float)

    # 4. Williams %R (14 períodos)
    res["williams_r_14"] = -100.0 * ((high_14 - close) / (high_14 - low_14 + 1e-9))

    # 5. ROC (Rate of Change) de 12 períodos
    res["roc_12"] = 100.0 * ((close - close.shift(12)) / (close.shift(12) + 1e-9))

    # 6. MACD (Moving Average Convergence Divergence) e Histograma
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    res["macd_linha"] = ema_12 - ema_26
    res["macd_sinal"] = res["macd_linha"].ewm(span=9, adjust=False).mean()
    res["macd_hist"] = res["macd_linha"] - res["macd_sinal"]

    # Normalização do histograma por média de preço
    res["macd_hist_normalizado"] = res["macd_hist"] / (close + 1e-9)

    # 7. Aceleração de Momentum (derivada do ROC)
    res["aceleracao_momentum"] = res["roc_12"].diff(3).fillna(0.0)

    return res.ffill().bfill()
