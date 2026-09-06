"""
Módulo de Reconhecimento de Padrões Gráficos (mapp/features/patterns.py)
Padrões de Continuação: Triângulos (Ascendente/Descendente/Simétrico), Bandeira (Flag).
Padrões de Reversão: Topo/Fundo Duplo, Topo/Fundo Triplo, OCO (Head & Shoulders).
Estrutura: Break of Structure (BOS) e Change of Character (CHoCH).
Regras matemáticas robustas sem inspeção puramente visual.
Sem vazamento temporal.
Todos os nomes em português.
"""

import numpy as np
import pandas as pd


class PatternDetector:
    """
    Detector matemático de padrões geométricos e estruturais em séries temporais.
    """

    @classmethod
    def extrair_caracteristicas(cls, df: pd.DataFrame, janela: int = 30) -> pd.DataFrame:
        """
        Gera métricas numéricas e scores de probabilidade de cada padrão gráfico.
        """
        res = pd.DataFrame(index=df.index)
        close = df["Close"].values
        high = df["High"].values if "High" in df.columns else close
        low = df["Low"].values if "Low" in df.columns else close

        n = len(df)
        topo_duplo = np.zeros(n)
        fundo_duplo = np.zeros(n)
        triangulo_asc = np.zeros(n)
        triangulo_desc = np.zeros(n)
        triangulo_sim = np.zeros(n)
        bandeira = np.zeros(n)
        head_shoulders = np.zeros(n)
        inv_head_shoulders = np.zeros(n)
        bos_alta = np.zeros(n)
        bos_baixa = np.zeros(n)
        choch = np.zeros(n)

        for i in range(janela, n):
            sub_hi = high[i - janela : i + 1]
            sub_lo = low[i - janela : i + 1]
            sub_cl = close[i - janela : i + 1]

            # 1. Detecção de Topo Duplo e Fundo Duplo (M e W)
            # Dois picos em níveis similares (+/- 1.2%) separados por um vale intermediário
            max1_idx = np.argmax(sub_hi[:janela // 2])
            max2_idx = (janela // 2) + np.argmax(sub_hi[janela // 2:])
            max1 = sub_hi[max1_idx]
            max2 = sub_hi[max2_idx]

            if abs(max1 - max2) / (max1 + 1e-9) < 0.015 and (max2_idx - max1_idx) >= 5:
                vale_intermediario = np.min(sub_lo[max1_idx : max2_idx + 1])
                # Confirma se houve recuo relevante no meio (pelo menos 2.5%)
                if (max1 - vale_intermediario) / (max1 + 1e-9) > 0.025:
                    topo_duplo[i] = 1.0

            min1_idx = np.argmin(sub_lo[:janela // 2])
            min2_idx = (janela // 2) + np.argmin(sub_lo[janela // 2:])
            min1 = sub_lo[min1_idx]
            min2 = sub_lo[min2_idx]

            if abs(min1 - min2) / (min1 + 1e-9) < 0.015 and (min2_idx - min1_idx) >= 5:
                pico_intermediario = np.max(sub_hi[min1_idx : min2_idx + 1])
                if (pico_intermediario - min1) / (min1 + 1e-9) > 0.025:
                    fundo_duplo[i] = 1.0

            # 2. Triângulos (Ascendente, Descendente, Simétrico) via regressão linear de topos e fundos
            x = np.arange(len(sub_cl))
            # Inclinação das máximas e mínimas locais
            p_hi = np.polyfit(x, sub_hi, 1)[0]
            p_lo = np.polyfit(x, sub_lo, 1)[0]

            # Triângulo Ascendente: Topos planos (slope ~ 0) e Fundos ascendentes (slope > 0)
            if abs(p_hi) < 0.002 and p_lo > 0.005:
                triangulo_asc[i] = 1.0
            # Triângulo Descendente: Fundos planos (slope ~ 0) e Topos descendentes (slope < 0)
            elif abs(p_lo) < 0.002 and p_hi < -0.005:
                triangulo_desc[i] = 1.0
            # Triângulo Simétrico: Topos convergindo para baixo e fundos para cima
            elif p_hi < -0.003 and p_lo > 0.003:
                triangulo_sim[i] = 1.0

            # 3. Bandeira (Flag): Impulso forte prévio seguido de canal estreito contra-tendência
            ret_impulso = (sub_cl[10] - sub_cl[0]) / (sub_cl[0] + 1e-9)
            ret_canal = (sub_cl[-1] - sub_cl[10]) / (sub_cl[10] + 1e-9)
            if ret_impulso > 0.05 and -0.03 < ret_canal < 0.0:
                bandeira[i] = 1.0  # Bull Flag
            elif ret_impulso < -0.05 and 0.0 < ret_canal < 0.03:
                bandeira[i] = -1.0  # Bear Flag

            # 4. Head and Shoulders (OCO)
            # Ombro esquerdo < Cabeça > Ombro direito
            terco = len(sub_hi) // 3
            ombro_esq = np.max(sub_hi[:terco])
            cabeca = np.max(sub_hi[terco : 2 * terco])
            ombro_dir = np.max(sub_hi[2 * terco:])
            if cabeca > ombro_esq * 1.02 and cabeca > ombro_dir * 1.02 and abs(ombro_esq - ombro_dir) / (ombro_esq + 1e-9) < 0.03:
                head_shoulders[i] = 1.0

            # OCO Invertido
            o_esq_inv = np.min(sub_lo[:terco])
            cabeca_inv = np.min(sub_lo[terco : 2 * terco])
            o_dir_inv = np.min(sub_lo[2 * terco:])
            if cabeca_inv < o_esq_inv * 0.98 and cabeca_inv < o_dir_inv * 0.98 and abs(o_esq_inv - o_dir_inv) / (o_esq_inv + 1e-9) < 0.03:
                inv_head_shoulders[i] = 1.0

            # 5. Break of Structure (BOS) e Change of Character (CHoCH)
            # BOS: Rompimento do último topo/fundo na direção da tendência
            # CHoCH: Rompimento do último fundo em tendência de alta ou topo em tendência de baixa
            topo_recente = np.max(sub_hi[-15:-2])
            fundo_recente = np.min(sub_lo[-15:-2])

            if close[i] > topo_recente and close[i - 1] <= topo_recente:
                bos_alta[i] = 1.0
            elif close[i] < fundo_recente and close[i - 1] >= fundo_recente:
                bos_baixa[i] = 1.0

            # CHoCH: Inversão estrutural
            if sub_cl[-10] > sub_cl[0] and close[i] < fundo_recente:
                choch[i] = -1.0  # Mudança de caráter para baixa
            elif sub_cl[-10] < sub_cl[0] and close[i] > topo_recente:
                choch[i] = 1.0  # Mudança de caráter para alta

        res["padrao_double_top"] = pd.Series(topo_duplo, index=df.index)
        res["padrao_double_bottom"] = pd.Series(fundo_duplo, index=df.index)
        res["padrao_triangle_asc"] = pd.Series(triangulo_asc, index=df.index)
        res["padrao_triangle_desc"] = pd.Series(triangulo_desc, index=df.index)
        res["padrao_triangle_sym"] = pd.Series(triangulo_sim, index=df.index)
        res["padrao_flag"] = pd.Series(bandeira, index=df.index)
        res["padrao_head_shoulders"] = pd.Series(head_shoulders, index=df.index)
        res["padrao_inv_head_shoulders"] = pd.Series(inv_head_shoulders, index=df.index)
        res["break_of_structure_bull"] = pd.Series(bos_alta, index=df.index)
        res["break_of_structure_bear"] = pd.Series(bos_baixa, index=df.index)
        res["change_of_character"] = pd.Series(choch, index=df.index)

        # Score Composto de Padrões (+1 altista, -1 baixista)
        score_bull = res["padrao_double_bottom"] + res["padrao_triangle_asc"] + (res["padrao_flag"] > 0) + res["padrao_inv_head_shoulders"] + res["break_of_structure_bull"]
        score_bear = res["padrao_double_top"] + res["padrao_triangle_desc"] + (res["padrao_flag"] < 0) + res["padrao_head_shoulders"] + res["break_of_structure_bear"]
        res["score_padroes_estruturais"] = (score_bull - score_bear).clip(-2.0, 2.0) / 2.0

        return res.fillna(0.0)
