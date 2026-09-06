"""
Módulo de Características de Tendência (mapp/features/trend.py)
Médias Móveis (SMA 20/50/100/200, EMA 9/21/50), inclinações, distâncias relativas,
cruzamentos, Higher High, Higher Low, Lower High, Lower Low, ADX e aceleração.
Sem vazamento temporal (computações puramente passadas).
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


def calcular_caracteristicas_tendencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o conjunto completo de indicadores de tendência a partir de OHLCV.
    """
    res = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close

    # Médias Móveis Simples (SMA)
    res["sma_20"] = close.rolling(20, min_periods=5).mean()
    res["sma_50"] = close.rolling(50, min_periods=10).mean()
    res["sma_100"] = close.rolling(100, min_periods=20).mean()
    res["sma_200"] = close.rolling(200, min_periods=30).mean()

    # Médias Móveis Exponenciais (EMA)
    res["ema_9"] = close.ewm(span=9, adjust=False).mean()
    res["ema_21"] = close.ewm(span=21, adjust=False).mean()
    res["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # Distâncias relativas do preço para as médias (%)
    res["dist_sma_20"] = (close - res["sma_20"]) / (res["sma_20"] + 1e-9)
    res["dist_sma_50"] = (close - res["sma_50"]) / (res["sma_50"] + 1e-9)
    res["dist_sma_200"] = (close - res["sma_200"]) / (res["sma_200"] + 1e-9)
    res["dist_ema_9"] = (close - res["ema_9"]) / (res["ema_9"] + 1e-9)
    res["dist_ema_21"] = (close - res["ema_21"]) / (res["ema_21"] + 1e-9)

    # Inclinações das médias (slope normalizado em 5 períodos)
    res["inclinacao_sma_20"] = (res["sma_20"] - res["sma_20"].shift(5)) / (res["sma_20"].shift(5) + 1e-9)
    res["inclinacao_sma_50"] = (res["sma_50"] - res["sma_50"].shift(5)) / (res["sma_50"].shift(5) + 1e-9)
    res["inclinacao_ema_21"] = (res["ema_21"] - res["ema_21"].shift(5)) / (res["ema_21"].shift(5) + 1e-9)

    # Cruzamentos (Golden Cross e Death Cross entre EMA 9 e EMA 21, SMA 50 e SMA 200)
    cruz_9_21 = (res["ema_9"] > res["ema_21"]).astype(float)
    res["cruzamento_ema_9_21"] = cruz_9_21.diff().fillna(0)  # +1 se cruzou para cima, -1 para baixo
    cruz_50_200 = (res["sma_50"] > res["sma_200"]).astype(float)
    res["golden_cross_50_200"] = cruz_50_200.diff().fillna(0)

    # Alinhamento das médias (Bullish Stack: Preço > EMA 9 > EMA 21 > SMA 50 > SMA 200)
    res["alinhamento_alta"] = (
        (close > res["ema_9"]) &
        (res["ema_9"] > res["ema_21"]) &
        (res["ema_21"] > res["sma_50"]) &
        (res["sma_50"] > res["sma_200"])
    ).astype(float)

    res["alinhamento_baixa"] = (
        (close < res["ema_9"]) &
        (res["ema_9"] < res["ema_21"]) &
        (res["ema_21"] < res["sma_50"]) &
        (res["sma_50"] < res["sma_200"])
    ).astype(float)

    # Rompimentos de Máximas e Mínimas (canais Donchian de 20 períodos)
    max_20_passada = high.shift(1).rolling(20, min_periods=5).max()
    min_20_passada = low.shift(1).rolling(20, min_periods=5).min()
    res["rompeu_maxima_20"] = (close > max_20_passada).astype(float)
    res["rompeu_minima_20"] = (close < min_20_passada).astype(float)

    # Estrutura de Higher High (HH), Higher Low (HL), Lower High (LH), Lower Low (LL)
    max_local_5 = high.rolling(5, min_periods=2).max()
    min_local_5 = low.rolling(5, min_periods=2).min()
    res["higher_high"] = (max_local_5 > max_local_5.shift(5)).astype(float)
    res["higher_low"] = (min_local_5 > min_local_5.shift(5)).astype(float)
    res["lower_high"] = (max_local_5 < max_local_5.shift(5)).astype(float)
    res["lower_low"] = (min_local_5 < min_local_5.shift(5)).astype(float)

    # Cálculo do ADX (Average Directional Index) de 14 períodos
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(14, min_periods=5).mean() / (atr14 + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(14, min_periods=5).mean() / (atr14 + 1e-9))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    res["adx_14"] = dx.rolling(14, min_periods=5).mean().fillna(20.0)
    res["tendencia_forte_adx"] = (res["adx_14"] > 25.0).astype(float)

    # Aceleração da tendência (derivada segunda do preço em 10 períodos)
    velocidade = close.diff(5) / (close.shift(5) + 1e-9)
    res["aceleracao_tendencia"] = velocidade.diff(5).fillna(0.0)

    return res.ffill().bfill()
