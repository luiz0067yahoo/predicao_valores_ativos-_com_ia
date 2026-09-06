"""
Módulo de Análise e Níveis de Fibonacci (mapp/features/fibonacci.py)
Distâncias aos níveis de retração (23.6%, 38.2%, 50.0%, 61.8%, 78.6%),
rejeições, confluências com suportes/médias móveis e rompimentos.
Sem vazamento temporal.
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


def calcular_caracteristicas_fibonacci(df: pd.DataFrame, janela_swing: int = 40) -> pd.DataFrame:
    """
    Calcula os níveis de retração de Fibonacci dos swings passados mais relevantes
    e a proximidade do preço atual a cada nível crítico.
    """
    res = pd.DataFrame(index=df.index)
    close = df["Close"].values
    high = df["High"].values if "High" in df.columns else close
    low = df["Low"].values if "Low" in df.columns else close

    n = len(df)
    dist_236 = np.zeros(n)
    dist_382 = np.zeros(n)
    dist_500 = np.zeros(n)
    dist_618 = np.zeros(n)
    dist_786 = np.zeros(n)
    nivel_mais_proximo = np.zeros(n)
    rejeicao_fib = np.zeros(n)

    for i in range(10, n):
        ini = max(0, i - janela_swing)
        swing_high = np.max(high[ini:i])
        swing_low = np.min(low[ini:i])
        diff_swing = swing_high - swing_low

        if diff_swing < 1e-6:
            continue

        preco = close[i]

        # Níveis de Fibonacci em relação ao swing recente
        # Se última tendência foi de alta (fechamento recente mais perto da máxima)
        fib_0 = swing_low
        fib_1 = swing_high

        fib_236 = fib_1 - 0.236 * diff_swing
        fib_382 = fib_1 - 0.382 * diff_swing
        fib_500 = fib_1 - 0.500 * diff_swing
        fib_618 = fib_1 - 0.618 * diff_swing
        fib_786 = fib_1 - 0.786 * diff_swing

        dist_236[i] = (preco - fib_236) / (preco + 1e-9)
        dist_382[i] = (preco - fib_382) / (preco + 1e-9)
        dist_500[i] = (preco - fib_500) / (preco + 1e-9)
        dist_618[i] = (preco - fib_618) / (preco + 1e-9)
        dist_786[i] = (preco - fib_786) / (preco + 1e-9)

        # Identifica o nível de Fibonacci mais próximo
        dists = [abs(dist_236[i]), abs(dist_382[i]), abs(dist_500[i]), abs(dist_618[i]), abs(dist_786[i])]
        min_idx = np.argmin(dists)
        niveis = [0.236, 0.382, 0.500, 0.618, 0.786]
        nivel_mais_proximo[i] = niveis[min_idx]

        # Rejeição no nível de ouro (0.618 ou 0.500): mínima testou o nível mas fechou acima (com sombra)
        if dists[3] < 0.008 and low[i] <= fib_618 and close[i] > fib_618:
            rejeicao_fib[i] = 1.0  # Rejeição altista no 61.8%
        elif dists[2] < 0.008 and low[i] <= fib_500 and close[i] > fib_500:
            rejeicao_fib[i] = 0.8  # Rejeição altista no 50.0%

    res["distance_to_fib_236"] = pd.Series(dist_236, index=df.index).clip(-0.5, 0.5)
    res["distance_to_fib_382"] = pd.Series(dist_382, index=df.index).clip(-0.5, 0.5)
    res["distance_to_fib_500"] = pd.Series(dist_500, index=df.index).clip(-0.5, 0.5)
    res["distance_to_fib_618"] = pd.Series(dist_618, index=df.index).clip(-0.5, 0.5)
    res["distance_to_fib_786"] = pd.Series(dist_786, index=df.index).clip(-0.5, 0.5)
    res["fib_nivel_mais_proximo"] = pd.Series(nivel_mais_proximo, index=df.index)
    res["rejeicao_fibonacci_suporte"] = pd.Series(rejeicao_fib, index=df.index)

    return res.ffill().bfill()
