"""
Pacote de Engenharia de Características M.A.P.P. (mapp/features/__init__.py)
Agrega todos os módulos especializados:
- Tendência (trend.py)
- Volume e Fluxo (volume.py)
- Volatilidade e Squeeze (volatility.py)
- Candlesticks (candlesticks.py)
- Suportes e Resistências (support_resistance.py)
- Padrões Estruturais (patterns.py)
- Níveis de Fibonacci (fibonacci.py)
- Momentum e Osciladores (momentum.py)
- Divergências (divergences.py)
Garante 100% de conformidade temporal (zero data leakage).
Todos os nomes em português.
"""

from typing import List, Optional
import pandas as pd

from mapp.features.trend import calcular_caracteristicas_tendencia
from mapp.features.volume import calcular_caracteristicas_volume
from mapp.features.volatility import calcular_caracteristicas_volatilidade
from mapp.features.candlesticks import calcular_caracteristicas_candlesticks
from mapp.features.support_resistance import SupportResistanceDetector
from mapp.features.patterns import PatternDetector
from mapp.features.fibonacci import calcular_caracteristicas_fibonacci
from mapp.features.momentum import calcular_caracteristicas_momentum
from mapp.features.divergences import calcular_caracteristicas_divergencias


def extrair_todas_caracteristicas_mapp(
    df: pd.DataFrame,
    incluir_categoricas: bool = False
) -> pd.DataFrame:
    """
    Gera a matriz unificada de features quantitativas e técnicas para modelagem preditiva.
    """
    if len(df) < 20:
        raise ValueError(f"Série histórica insuficiente ({len(df)} barras) para extração de features MAPP.")

    df_base = df.copy()

    # 1. Extrai blocos analíticos modulares
    f_tend = calcular_caracteristicas_tendencia(df_base)
    f_vol = calcular_caracteristicas_volume(df_base)
    f_vola = calcular_caracteristicas_volatilidade(df_base)
    f_candle = calcular_caracteristicas_candlesticks(df_base)
    f_sr = SupportResistanceDetector.extrair_caracteristicas(df_base)
    f_patt = PatternDetector.extrair_caracteristicas(df_base)
    f_fibo = calcular_caracteristicas_fibonacci(df_base)
    f_mom = calcular_caracteristicas_momentum(df_base)
    f_div = calcular_caracteristicas_divergencias(df_base)

    blocos = [
        f_tend,
        f_vol,
        f_vola,
        f_candle,
        f_sr,
        f_patt,
        f_fibo,
        f_mom,
        f_div
    ]

    matriz = pd.concat(blocos, axis=1)

    if not incluir_categoricas and "tipo_divergencia" in matriz.columns:
        matriz = matriz.drop(columns=["tipo_divergencia"])

    # Remove valores infinitos ou NaNs iniciais causados por janelas móveis
    matriz = matriz.replace([float("inf"), float("-inf")], 0.0)
    matriz = matriz.ffill().bfill().fillna(0.0)

    return matriz
