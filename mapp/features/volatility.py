"""
Módulo de Características de Volatilidade (mapp/features/volatility.py)
ATR, ATR percentual, volatilidade histórica e realizada, Bollinger Band Width,
detecção de compressão (squeeze) e expansão, volatilidade assimétrica.
Sem vazamento temporal.
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


def calcular_caracteristicas_volatilidade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de volatilidade, regimes de compressão e expansão.
    """
    res = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close

    # 1. True Range (TR) e ATR de 14 períodos
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    res["atr_14"] = tr.rolling(14, min_periods=5).mean()
    res["atr_pct"] = res["atr_14"] / (close + 1e-9)

    # 2. Volatilidade Histórica (Retornos diários anualizados em 20 e 60 períodos)
    retornos = close.pct_change().fillna(0.0)
    res["volatilidade_20d"] = retornos.rolling(20, min_periods=5).std() * np.sqrt(252)
    res["volatilidade_60d"] = retornos.rolling(60, min_periods=15).std() * np.sqrt(252)

    # Razão de volatilidade de curto vs médio prazo
    res["razao_volatilidade_curto_medio"] = res["volatilidade_20d"] / (res["volatilidade_60d"] + 1e-9)

    # 3. Bollinger Bands e Band Width
    bb_media_20 = close.rolling(20, min_periods=5).mean()
    bb_desvio_20 = close.rolling(20, min_periods=5).std()
    bb_superior = bb_media_20 + (2.0 * bb_desvio_20)
    bb_inferior = bb_media_20 - (2.0 * bb_desvio_20)

    res["bb_width"] = (bb_superior - bb_inferior) / (bb_media_20 + 1e-9)
    res["bb_percent_b"] = (close - bb_inferior) / (bb_superior - bb_inferior + 1e-9)

    # 4. Detecção de Compressão de Volatilidade (Squeeze de Bollinger dentro de Keltner)
    # Keltner Channel: EMA 20 +/- 1.5 * ATR 14
    kc_media = close.ewm(span=20, adjust=False).mean()
    kc_superior = kc_media + (1.5 * res["atr_14"])
    kc_inferior = kc_media - (1.5 * res["atr_14"])

    # Squeeze ocorre quando a Banda de Bollinger fica totalmente dentro do Canal de Keltner
    compressao_bollinger = (bb_superior < kc_superior) & (bb_inferior > kc_inferior)
    # Ou quando o BB Width atinge o percentil 15 dos últimos 60 dias
    width_min_60 = res["bb_width"].rolling(60, min_periods=20).quantile(0.15)
    res["volatility_compression"] = (compressao_bollinger | (res["bb_width"] <= width_min_60)).astype(float)

    # Expansão ocorre quando sai da compressão com inclinação acentuada do BB Width
    width_diff = res["bb_width"].diff(3)
    res["volatility_expansion"] = ((res["volatility_compression"].shift(2) == 1.0) & (width_diff > 0.02)).astype(float)

    # 5. Volatilidade Assimétrica (Downside Volatility vs Upside Volatility)
    ret_neg = retornos.clip(upper=0.0)
    ret_pos = retornos.clip(lower=0.0)
    downside_std = ret_neg.rolling(20, min_periods=5).std()
    upside_std = ret_pos.rolling(20, min_periods=5).std()
    res["assimetria_volatilidade"] = (downside_std - upside_std) / (downside_std + upside_std + 1e-9)

    return res.ffill().bfill()
