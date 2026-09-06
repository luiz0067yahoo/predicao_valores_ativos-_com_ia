"""
Módulo de Detecção de Suportes e Resistências (mapp/features/support_resistance.py)
Identifica zonas de suporte e resistência dinâmicas, número de toques/testes,
força dos níveis, distância percentual, rompimentos, falsos rompimentos e retestes.
Sem vazamento temporal.
Todos os nomes em português.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class NivelEstrutural:
    preco: float
    tipo: str  # 'SUPORTE' ou 'RESISTENCIA'
    toques: int
    forca: float  # [0.0, 1.0]
    ultima_barra_toque: int


class SupportResistanceDetector:
    """
    Detector algorítmico de suporte e resistência por agrupamento de pivôs locais.
    """

    @classmethod
    def extrair_caracteristicas(cls, df: pd.DataFrame, janela_lookback: int = 60) -> pd.DataFrame:
        """
        Calcula as features contínuas de suporte e resistência para toda a série temporal.
        """
        res = pd.DataFrame(index=df.index)
        close = df["Close"].values
        high = df["High"].values if "High" in df.columns else close
        low = df["Low"].values if "Low" in df.columns else close

        n = len(df)
        dist_sup_list = []
        dist_res_list = []
        forca_sup_list = []
        forca_res_list = []
        rompimento_list = []
        falso_rompimento_list = []

        # Para cada barra t, usamos estritamente os dados até t (passado)
        for i in range(n):
            if i < 15:
                dist_sup_list.append(0.05)
                dist_res_list.append(0.05)
                forca_sup_list.append(0.5)
                forca_res_list.append(0.5)
                rompimento_list.append(0.0)
                falso_rompimento_list.append(0.0)
                continue

            ini = max(0, i - janela_lookback)
            sub_hi = high[ini:i]
            sub_lo = low[ini:i]
            sub_cl = close[ini:i]
            preco_atual = close[i]

            # Encontra pivôs de alta e baixa locais (janela de 3 barras)
            suportes_candidatos = []
            resistencias_candidatas = []

            for j in range(2, len(sub_cl) - 2):
                if sub_lo[j] <= sub_lo[j - 1] and sub_lo[j] <= sub_lo[j - 2] and sub_lo[j] <= sub_lo[j + 1] and sub_lo[j] <= sub_lo[j + 2]:
                    suportes_candidatos.append(sub_lo[j])
                if sub_hi[j] >= sub_hi[j - 1] and sub_hi[j] >= sub_hi[j - 2] and sub_hi[j] >= sub_hi[j + 1] and sub_hi[j] >= sub_hi[j + 2]:
                    resistencias_candidatas.append(sub_hi[j])

            # Suportes estritamente abaixo do preço atual
            sups_abaixo = [s for s in suportes_candidatos if s < preco_atual]
            res_acima = [r for r in resistencias_candidatas if r > preco_atual]

            # Suporte mais próximo
            if sups_abaixo:
                sup_mais_proximo = max(sups_abaixo)
                dist_sup = (preco_atual - sup_mais_proximo) / (preco_atual + 1e-9)
                # Conta toques próximos (+/- 0.8%)
                toques_sup = sum(1 for s in sups_abaixo if abs(s - sup_mais_proximo) / sup_mais_proximo < 0.008)
                forca_sup = min(1.0, 0.4 + (toques_sup * 0.2))
            else:
                sup_mais_proximo = np.min(sub_lo)
                dist_sup = max(0.001, (preco_atual - sup_mais_proximo) / (preco_atual + 1e-9))
                forca_sup = 0.3

            # Resistência mais próxima
            if res_acima:
                res_mais_proxima = min(res_acima)
                dist_res = (res_mais_proxima - preco_atual) / (preco_atual + 1e-9)
                toques_res = sum(1 for r in res_acima if abs(r - res_mais_proxima) / res_mais_proxima < 0.008)
                forca_res = min(1.0, 0.4 + (toques_res * 0.2))
            else:
                res_mais_proxima = np.max(sub_hi)
                dist_res = max(0.001, (res_mais_proxima - preco_atual) / (preco_atual + 1e-9))
                forca_res = 0.3

            # Rompimento recente (fechamento acima da resistência ou abaixo do suporte)
            rompeu = 0.0
            if preco_atual > np.max(sub_hi[-5:]) and close[i - 1] <= np.max(sub_hi[-5:]):
                rompeu = 1.0  # Rompimento de alta
            elif preco_atual < np.min(sub_lo[-5:]) and close[i - 1] >= np.min(sub_lo[-5:]):
                rompeu = -1.0  # Rompimento de baixa

            # Falso rompimento (rompeu na barra i-1 mas fechou de volta na barra i)
            falso_romp = 0.0
            if i >= 2:
                max_ref = np.max(sub_hi[-10:])
                if high[i - 1] > max_ref and close[i - 1] < max_ref and close[i] < close[i - 1]:
                    falso_romp = -1.0  # Upthrust / Falso rompimento de alta
                min_ref = np.min(sub_lo[-10:])
                if low[i - 1] < min_ref and close[i - 1] > min_ref and close[i] > close[i - 1]:
                    falso_romp = 1.0  # Spring / Falso rompimento de baixa

            dist_sup_list.append(float(np.clip(dist_sup, 0.0, 1.0)))
            dist_res_list.append(float(np.clip(dist_res, 0.0, 1.0)))
            forca_sup_list.append(float(forca_sup))
            forca_res_list.append(float(forca_res))
            rompimento_list.append(float(rompeu))
            falso_rompimento_list.append(float(falso_romp))

        res["distance_to_support"] = pd.Series(dist_sup_list, index=df.index)
        res["distance_to_resistance"] = pd.Series(dist_res_list, index=df.index)
        res["support_strength"] = pd.Series(forca_sup_list, index=df.index)
        res["resistance_strength"] = pd.Series(forca_res_list, index=df.index)
        res["breakout_strength"] = pd.Series(rompimento_list, index=df.index)
        res["false_breakout_signal"] = pd.Series(falso_rompimento_list, index=df.index)

        # Proximidade relativa (Score onde 0 está no suporte e 1 está na resistência)
        range_sup_res = (res["distance_to_support"] + res["distance_to_resistance"] + 1e-9)
        res["posicao_no_canal_suporte_resistencia"] = (res["distance_to_support"] / range_sup_res).clip(0.0, 1.0)

        return res.ffill().bfill()
