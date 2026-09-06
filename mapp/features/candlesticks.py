"""
Módulo de Reconhecimento Quantitativo de Candlesticks (mapp/features/candlesticks.py)
Identifica padrões clássicos de 1, 2 e 3 velas e os converte em features numéricas.
Hammer, Inverted Hammer, Shooting Star, Hanging Man, Doji (Dragonfly/Gravestone),
Engolfo de Alta/Baixa, Piercing, Dark Cloud Cover, Morning/Evening Star, Harami.
Sem vazamento temporal.
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


def calcular_caracteristicas_candlesticks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula scores contínuos e binários para padrões de velas japonesas.
    """
    res = pd.DataFrame(index=df.index)
    op = df["Open"] if "Open" in df.columns else df["Close"]
    cl = df["Close"]
    hi = df["High"] if "High" in df.columns else np.maximum(op, cl)
    lo = df["Low"] if "Low" in df.columns else np.minimum(op, cl)

    corpo = (cl - op).abs()
    range_total = (hi - lo).abs() + 1e-9
    sombra_superior = hi - np.maximum(op, cl)
    sombra_inferior = np.minimum(op, cl) - lo

    eh_alta = cl > op
    eh_baixa = cl < op

    # Proporções
    prop_corpo = corpo / range_total
    prop_sombra_sup = sombra_superior / range_total
    prop_sombra_inf = sombra_inferior / range_total

    # 1. Doji (corpo muito estreito <= 10% do range total)
    res["padrao_doji"] = (prop_corpo <= 0.10).astype(float)
    res["padrao_dragonfly_doji"] = ((prop_corpo <= 0.10) & (prop_sombra_inf >= 0.60) & (prop_sombra_sup <= 0.10)).astype(float)
    res["padrao_gravestone_doji"] = ((prop_corpo <= 0.10) & (prop_sombra_sup >= 0.60) & (prop_sombra_inf <= 0.10)).astype(float)

    # 2. Martelo (Hammer) e Enforcado (Hanging Man)
    # Sombra inferior longa (>= 2x corpo), pouca ou nenhuma sombra superior
    martelo_geom = (prop_sombra_inf >= 2.0 * prop_corpo) & (prop_sombra_sup <= 0.15)
    tendencia_baixa_previa = cl.shift(1) < cl.shift(4)
    tendencia_alta_previa = cl.shift(1) > cl.shift(4)

    res["padrao_hammer"] = (martelo_geom & tendencia_baixa_previa).astype(float)
    res["padrao_hanging_man"] = (martelo_geom & tendencia_alta_previa).astype(float)

    # 3. Estrela Cadente (Shooting Star) e Martelo Invertido (Inverted Hammer)
    estrela_geom = (prop_sombra_sup >= 2.0 * prop_corpo) & (prop_sombra_inf <= 0.15)
    res["padrao_shooting_star"] = (estrela_geom & tendencia_alta_previa).astype(float)
    res["padrao_inverted_hammer"] = (estrela_geom & tendencia_baixa_previa).astype(float)

    # 4. Engolfos (Bullish & Bearish Engulfing)
    corpo_ant = corpo.shift(1)
    op_ant = op.shift(1)
    cl_ant = cl.shift(1)

    engolfo_alta = (
        (cl.shift(1) < op.shift(1)) &     # Vela anterior de baixa
        eh_alta &                          # Vela atual de alta
        (op <= cl_ant) &                   # Abertura abaixo ou igual ao fechamento anterior
        (cl >= op_ant) &                   # Fechamento acima da abertura anterior
        (corpo > corpo_ant)                # Corpo atual maior que o anterior
    )
    res["padrao_bullish_engulfing"] = engolfo_alta.astype(float)

    engolfo_baixa = (
        (cl.shift(1) > op.shift(1)) &     # Vela anterior de alta
        eh_baixa &                         # Vela atual de baixa
        (op >= cl_ant) &                   # Abertura acima ou igual ao fechamento anterior
        (cl <= op_ant) &                   # Fechamento abaixo da abertura anterior
        (corpo > corpo_ant)                # Corpo atual maior que o anterior
    )
    res["padrao_bearish_engulfing"] = engolfo_baixa.astype(float)

    # 5. Piercing Line e Dark Cloud Cover
    meio_corpo_ant = (op_ant + cl_ant) / 2.0
    res["padrao_piercing"] = (
        (cl.shift(1) < op.shift(1)) & eh_alta & (op < lo.shift(1)) & (cl > meio_corpo_ant) & (cl < op_ant)
    ).astype(float)

    res["padrao_dark_cloud"] = (
        (cl.shift(1) > op.shift(1)) & eh_baixa & (op > hi.shift(1)) & (cl < meio_corpo_ant) & (cl > op_ant)
    ).astype(float)

    # 6. Harami (Mulher Grávida) de Alta e Baixa
    harami_alta = (
        (cl_ant < op_ant) & eh_alta & (op > cl_ant) & (cl < op_ant) & (corpo < corpo_ant * 0.7)
    )
    res["padrao_harami_alta"] = harami_alta.astype(float)

    harami_baixa = (
        (cl_ant > op_ant) & eh_baixa & (op < cl_ant) & (cl > op_ant) & (corpo < corpo_ant * 0.7)
    )
    res["padrao_harami_baixa"] = harami_baixa.astype(float)

    # 7. Morning Star e Evening Star (3 velas)
    vela1_baixa = cl.shift(2) < op.shift(2)
    vela2_pequena = corpo.shift(1) <= range_total.shift(1) * 0.35
    vela3_alta_forte = eh_alta & (cl > (op.shift(2) + cl.shift(2)) / 2.0)
    res["padrao_morning_star"] = (vela1_baixa & vela2_pequena & vela3_alta_forte).astype(float)

    vela1_alta = cl.shift(2) > op.shift(2)
    vela3_baixa_forte = eh_baixa & (cl < (op.shift(2) + cl.shift(2)) / 2.0)
    res["padrao_evening_star"] = (vela1_alta & vela2_pequena & vela3_baixa_forte).astype(float)

    # 8. Score Consolidado de Candlestick (+1.0 muito altista, -1.0 muito baixista)
    score_altista = (
        res["padrao_hammer"] * 0.7 +
        res["padrao_bullish_engulfing"] * 0.9 +
        res["padrao_morning_star"] * 1.0 +
        res["padrao_piercing"] * 0.6 +
        res["padrao_dragonfly_doji"] * 0.5 +
        res["padrao_harami_alta"] * 0.4
    )
    score_baixista = (
        res["padrao_shooting_star"] * 0.7 +
        res["padrao_bearish_engulfing"] * 0.9 +
        res["padrao_evening_star"] * 1.0 +
        res["padrao_dark_cloud"] * 0.6 +
        res["padrao_gravestone_doji"] * 0.5 +
        res["padrao_harami_baixa"] * 0.4
    )
    res["score_candlestick_direcional"] = (score_altista - score_baixista).clip(-1.0, 1.0)

    return res.fillna(0.0)
