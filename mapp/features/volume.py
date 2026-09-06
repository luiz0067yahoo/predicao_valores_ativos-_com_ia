"""
Módulo de Características de Volume (mapp/features/volume.py)
Volume relativo, Z-Score de volume, OBV, VWAP, fluxo em rompimentos,
relação retorno x volume e divergência preço/volume.
Sem vazamento temporal.
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


def calcular_caracteristicas_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores avançados de volume e fluxo financeiro.
    """
    res = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    # Se o volume for estático ou ausente (ex: forex sintético), gera indicadores neutros
    if volume.nunique() <= 2 or volume.sum() <= 0:
        res["volume_relativo_20"] = 1.0
        res["volume_zscore_20"] = 0.0
        res["obv"] = 0.0
        res["obv_normalizado"] = 0.0
        res["dist_vwap_20"] = 0.0
        res["divergencia_preco_volume"] = 0.0
        res["volume_no_rompimento"] = 0.0
        res["razao_retorno_volume"] = 0.0
        return res

    # 1. Volume Relativo (Volume / Média Móvel de 20 e 50 períodos)
    vol_sma_20 = volume.rolling(20, min_periods=5).mean()
    vol_sma_50 = volume.rolling(50, min_periods=10).mean()
    res["volume_relativo_20"] = volume / (vol_sma_20 + 1e-9)
    res["volume_relativo_50"] = volume / (vol_sma_50 + 1e-9)

    # 2. Z-Score do Volume em 20 períodos
    vol_std_20 = volume.rolling(20, min_periods=5).std()
    res["volume_zscore_20"] = ((volume - vol_sma_20) / (vol_std_20 + 1e-9)).clip(-3.0, 3.0)

    # 3. OBV (On-Balance Volume)
    direcao_preco = np.sign(close.diff().fillna(0))
    obv_bruto = (direcao_preco * volume).cumsum()
    res["obv"] = obv_bruto
    # Normalização por média e desvio para manter estacionariedade
    obv_mean_50 = obv_bruto.rolling(50, min_periods=10).mean()
    obv_std_50 = obv_bruto.rolling(50, min_periods=10).std()
    res["obv_normalizado"] = ((obv_bruto - obv_mean_50) / (obv_std_50 + 1e-9)).clip(-3.0, 3.0).fillna(0.0)

    # 4. VWAP Rolante (20 períodos)
    preco_tipico = (high + low + close) / 3.0
    vol_preco_cum = (preco_tipico * volume).rolling(20, min_periods=5).sum()
    vol_cum = volume.rolling(20, min_periods=5).sum()
    vwap_20 = vol_preco_cum / (vol_cum + 1e-9)
    res["dist_vwap_20"] = (close - vwap_20) / (vwap_20 + 1e-9)

    # 5. Relação Retorno x Volume (Elasticidade do preço por unidade de volume)
    retorno_diario = close.pct_change().fillna(0.0)
    res["razao_retorno_volume"] = retorno_diario / (res["volume_relativo_20"] + 1e-9)

    # 6. Divergência Preço / Volume
    # Caso 1: Preço sobe mas Volume cai (exaustão da alta) -> sinal negativo (-1)
    # Caso 2: Preço cai mas Volume sobe (forte pressão vendedora) -> sinal negativo (-1)
    # Caso 3: Preço sobe e Volume sobe (confirmação da alta) -> sinal positivo (+1)
    # Caso 4: Preço cai e Volume cai (correção fraca) -> sinal positivo para repique (+0.5)
    preco_subindo = (retorno_diario > 0.001)
    preco_caindo = (retorno_diario < -0.001)
    vol_acima_media = (res["volume_relativo_20"] > 1.1)
    vol_abaixo_media = (res["volume_relativo_20"] < 0.9)

    divergencia = np.zeros(len(df))
    divergencia[preco_subindo & vol_acima_media] = 1.0    # Alta confirmada por volume
    divergencia[preco_subindo & vol_abaixo_media] = -0.8  # Alta sem volume (fraqueza)
    divergencia[preco_caindo & vol_acima_media] = -1.0    # Queda com volume pesado
    divergencia[preco_caindo & vol_abaixo_media] = 0.5    # Correção sem pressão vendedora
    res["divergencia_preco_volume"] = pd.Series(divergencia, index=df.index)

    # 7. Volume no Rompimento (volume acima de 1.5x em dias de rompimento de máxima/mínima de 20 dias)
    max_20 = high.shift(1).rolling(20, min_periods=5).max()
    min_20 = low.shift(1).rolling(20, min_periods=5).min()
    rompeu = (close > max_20) | (close < min_20)
    res["volume_no_rompimento"] = (rompeu & (res["volume_relativo_20"] > 1.5)).astype(float)

    return res.ffill().bfill()
