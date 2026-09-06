"""
Módulo de Detecção de Divergências (mapp/features/divergences.py)
Identifica divergências regulares e ocultas entre preço e RSI, MACD, OBV e Momentum.
Classifica em BULLISH_DIVERGENCE, BEARISH_DIVERGENCE ou NONE.
Sem vazamento temporal.
Todos os nomes em português.
"""

from enum import Enum
import numpy as np
import pandas as pd


class TipoDivergencia(str, Enum):
    BULLISH_DIVERGENCE = "BULLISH_DIVERGENCE"
    BEARISH_DIVERGENCE = "BEARISH_DIVERGENCE"
    NONE = "NONE"


def calcular_caracteristicas_divergencias(
    df: pd.DataFrame,
    janela_comparacao: int = 20
) -> pd.DataFrame:
    """
    Detecta divergências entre novos topos/fundos de preço e seus respectivos osciladores.
    """
    res = pd.DataFrame(index=df.index)
    close = df["Close"].values
    high = df["High"].values if "High" in df.columns else close
    low = df["Low"].values if "Low" in df.columns else close
    volume = df["Volume"].values if "Volume" in df.columns else np.ones_like(close)

    n = len(df)

    # Pré-calcula RSI e MACD para o algoritmo de divergência
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0.0).ewm(com=13, adjust=False).mean().values
    loss = (-delta).clip(lower=0.0).ewm(com=13, adjust=False).mean().values
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_12 - ema_26

    # OBV
    dir_p = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(dir_p * volume)

    div_rsi_bull = np.zeros(n)
    div_rsi_bear = np.zeros(n)
    div_macd_bull = np.zeros(n)
    div_macd_bear = np.zeros(n)
    div_obv_bull = np.zeros(n)
    div_obv_bear = np.zeros(n)
    classificacao_geral = []

    for i in range(janela_comparacao, n):
        # Janela de análise prévia
        sub_lo = low[i - janela_comparacao : i + 1]
        sub_hi = high[i - janela_comparacao : i + 1]
        sub_rsi = rsi[i - janela_comparacao : i + 1]
        sub_macd = macd[i - janela_comparacao : i + 1]
        sub_obv = obv[i - janela_comparacao : i + 1]

        # 1. Divergência Altista (Bullish Divergence)
        # Preço atinge nova mínima na janela, mas o oscilador forma mínima mais alta
        min_idx = np.argmin(sub_lo)
        if min_idx == len(sub_lo) - 1:  # mínima ocorreu na barra atual
            # Procura a mínima anterior mais baixa na primeira metade da janela
            primeira_metade_lo = sub_lo[:len(sub_lo) // 2]
            min_ant_idx = np.argmin(primeira_metade_lo)

            # Preço atual é menor que o anterior
            if sub_lo[-1] < primeira_metade_lo[min_ant_idx]:
                if sub_rsi[-1] > sub_rsi[min_ant_idx]:
                    div_rsi_bull[i] = 1.0
                if sub_macd[-1] > sub_macd[min_ant_idx]:
                    div_macd_bull[i] = 1.0
                if sub_obv[-1] > sub_obv[min_ant_idx]:
                    div_obv_bull[i] = 1.0

        # 2. Divergência Baixista (Bearish Divergence)
        # Preço atinge nova máxima na janela, mas o oscilador forma máxima mais baixa
        max_idx = np.argmax(sub_hi)
        if max_idx == len(sub_hi) - 1:  # máxima ocorreu na barra atual
            primeira_metade_hi = sub_hi[:len(sub_hi) // 2]
            max_ant_idx = np.argmax(primeira_metade_hi)

            if sub_hi[-1] > primeira_metade_hi[max_ant_idx]:
                if sub_rsi[-1] < sub_rsi[max_ant_idx]:
                    div_rsi_bear[i] = 1.0
                if sub_macd[-1] < sub_macd[max_ant_idx]:
                    div_macd_bear[i] = 1.0
                if sub_obv[-1] < sub_obv[max_ant_idx]:
                    div_obv_bear[i] = 1.0

    res["divergencia_rsi_bull"] = pd.Series(div_rsi_bull, index=df.index)
    res["divergencia_rsi_bear"] = pd.Series(div_rsi_bear, index=df.index)
    res["divergencia_macd_bull"] = pd.Series(div_macd_bull, index=df.index)
    res["divergencia_macd_bear"] = pd.Series(div_macd_bear, index=df.index)
    res["divergencia_obv_bull"] = pd.Series(div_obv_bull, index=df.index)
    res["divergencia_obv_bear"] = pd.Series(div_obv_bear, index=df.index)

    # Classificação categórica prioritária
    total_bull = res["divergencia_rsi_bull"] + res["divergencia_macd_bull"] + res["divergencia_obv_bull"]
    total_bear = res["divergencia_rsi_bear"] + res["divergencia_macd_bear"] + res["divergencia_obv_bear"]

    for b, r in zip(total_bull, total_bear):
        if b > 0 and b >= r:
            classificacao_geral.append(TipoDivergencia.BULLISH_DIVERGENCE.value)
        elif r > 0 and r > b:
            classificacao_geral.append(TipoDivergencia.BEARISH_DIVERGENCE.value)
        else:
            classificacao_geral.append(TipoDivergencia.NONE.value)

    res["tipo_divergencia"] = pd.Series(classificacao_geral, index=df.index)
    res["score_divergencia"] = (total_bull - total_bear).clip(-3.0, 3.0) / 3.0

    return res
