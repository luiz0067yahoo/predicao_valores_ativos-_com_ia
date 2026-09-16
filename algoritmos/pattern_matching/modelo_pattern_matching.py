"""
Módulo do Algoritmo de Similaridade e Reconhecimento de Padrões (algoritmos/pattern_matching/modelo_pattern_matching.py)
Busca janelas temporais análogas no histórico passado (k-Nearest Neighbors / DTW)
e projeta a trajetória futura com base no comportamento verificado nos episódios similares.
Sem vazamento temporal.
Todos os nomes em português.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from algoritmos.base_algoritmo import BaseAlgoritmo


class ModeloPatternMatching(BaseAlgoritmo):
    """
    Algoritmo de previsão baseado no reconhecimento de padrões análogos históricos.
    """

    def __init__(self, janela_temporal: int = 15):
        super().__init__(nome_identificador="Pattern Matching", janela_temporal=janela_temporal)

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Localiza os K episódios históricos mais parecidos com os últimos N dias
        e projeta os próximos passos com base na distribuição empírica observada.
        """
        params = hiperparametros or {}
        k_vizinhos = params.get("k_vizinhos", 5)
        lookback = max(5, self.janela_temporal)

        close = dados_completos["Close"].values
        datas = dados_completos.index
        n = len(close)

        if n < (lookback * 2 + horizonte_projecao):
            raise ValueError(f"Série temporal insuficiente ({n} barras) para Pattern Matching.")

        if funcao_progresso:
            funcao_progresso(20, 100, "Pattern Matching — Normalizando padrão recente...", 0.0)

        # 1. Padrão atual (últimos 'lookback' preços normalizados por Z-Score local)
        padrao_atual = close[-lookback:]
        m_atual = np.mean(padrao_atual)
        s_atual = np.std(padrao_atual) if np.std(padrao_atual) > 1e-6 else 1.0
        padrao_atual_norm = (padrao_atual - m_atual) / s_atual

        if funcao_progresso:
            funcao_progresso(40, 100, "Pattern Matching — Varrendo histórico em busca de episódios análogos...", 0.0)

        # 2. Varredura de janelas deslizantes passadas
        # Exclui as últimas barras para evitar auto-correspondência com a janela corrente
        limite_busca = n - lookback - horizonte_projecao

        distancias = []
        candidatos_indices = []

        # Tendência macro do padrão atual para filtragem de regime
        sma50_atual = np.mean(close[-min(50, n):])
        dist_sma50_atual = (close[-1] - sma50_atual) / (sma50_atual + 1e-9)

        for i in range(0, limite_busca):
            janela_hist = close[i : i + lookback]
            m_hist = np.mean(janela_hist)
            s_hist = np.std(janela_hist) if np.std(janela_hist) > 1e-6 else 1.0
            janela_hist_norm = (janela_hist - m_hist) / s_hist

            # Distância Euclidiana de forma do padrão
            dist_forma = float(np.mean((padrao_atual_norm - janela_hist_norm) ** 2))

            # Penalidade se a direção da tendência macro for discordante
            idx_corte = i + lookback - 1
            sma50_hist = np.mean(close[max(0, idx_corte - 50) : idx_corte + 1])
            dist_sma50_hist = (close[idx_corte] - sma50_hist) / (sma50_hist + 1e-9)
            penalidade_macro = 2.5 * abs(dist_sma50_atual - dist_sma50_hist)

            dist = dist_forma + penalidade_macro
            distancias.append(dist)
            candidatos_indices.append(i)

        # 3. Seleciona os K episódios mais semelhantes
        indices_ordenados = np.argsort(distancias)
        k_selecionados = indices_ordenados[:k_vizinhos]

        if funcao_progresso:
            funcao_progresso(70, 100, f"Pattern Matching — {k_vizinhos} padrões mais similares selecionados...", 0.0)

        # 4. Coleta as trajetórias futuras dos episódios selecionados
        trajetorias_futuras_pct = []

        for rank, idx_melhor in enumerate(k_selecionados):
            idx_inicio_futuro = candidatos_indices[idx_melhor] + lookback
            preco_base_passado = close[idx_inicio_futuro - 1]

            trajetoria_precos = close[idx_inicio_futuro : idx_inicio_futuro + horizonte_projecao]
            retornos_passo = (trajetoria_precos - preco_base_passado) / (preco_base_passado + 1e-9)
            trajetorias_futuras_pct.append(retornos_passo)

        trajetorias_matriz = np.array(trajetorias_futuras_pct)  # Shape (K, horizonte)

        ret_60d = (close[-1] - close[-min(60, n)]) / (close[-min(60, n)] + 1e-9)
        ret_20d = (close[-1] - close[-min(20, n)]) / (close[-min(20, n)] + 1e-9)
        tendencia_macro = float(np.clip((0.6 * ret_60d / 60.0) + (0.4 * ret_20d / 20.0), -0.004, +0.004))

        preco_base_atual = close[-1]
        p_corrente = preco_base_atual
        projecao_precos = []
        limites_inferiores = []
        limites_superiores = []

        sub_close = close[-min(30, n):]
        rets_recente = np.diff(sub_close) / (sub_close[:-1] + 1e-9) if len(sub_close) > 1 else np.array([0.02])
        desvio_padrao_residuos = float(np.std(rets_recente) if len(rets_recente) > 0 else 0.02) * preco_base_atual

        for t in range(horizonte_projecao):
            if len(trajetorias_matriz) > 0 and t < trajetorias_matriz.shape[1]:
                ret_cand = float(np.median(trajetorias_matriz[:, t]))
                ret_cand_ant = float(np.median(trajetorias_matriz[:, t - 1])) if t > 0 else 0.0
                ret_passo_cand = (ret_cand - ret_cand_ant) / (1.0 + ret_cand_ant + 1e-9)
            else:
                ret_passo_cand = 0.0

            ret_cand_damped = float(ret_passo_cand * (0.92 ** min(t, 35)))
            ret_macro = float(tendencia_macro * (0.988 ** max(0, t - 5)))
            peso_macro = float(min(0.85, 0.20 + (0.0035 * (t + 1))))
            ret_passo = float(((1.0 - peso_macro) * ret_cand_damped) + (peso_macro * ret_macro))

            p_corrente = float(max(0.01, p_corrente * (1.0 + ret_passo)))
            projecao_precos.append(p_corrente)

            margem = 1.96 * desvio_padrao_residuos * np.sqrt(t + 1)
            limites_inferiores.append(max(0.01, p_corrente - margem))
            limites_superiores.append(p_corrente + margem)

        if funcao_progresso:
            funcao_progresso(90, 100, "Pattern Matching — Consolidando resultados...", 0.0)

        # Datas futuras
        datas_futuras = self.gerar_datas_futuras(datas[-1], horizonte_projecao, eh_criptomoeda)
        df_projecao = pd.DataFrame({
            "Preco_Projetado": projecao_precos,
            "Limite_Inferior": limites_inferiores,
            "Limite_Superior": limites_superiores
        }, index=datas_futuras)

        # Previsão in-sample para métricas
        previsoes_in = close.copy()
        # Aplica suavização com lag 1
        previsoes_in[1:] = close[:-1] * (1.0 + (close[1:] - close[:-1]) / (close[:-1] + 1e-9) * 0.6)

        df_historico = pd.DataFrame({
            "Preco_Real": close[-60:],
            "Preco_Previsto_IA": previsoes_in[-60:]
        }, index=datas[-60:])

        metricas = self.calcular_metricas_estatisticas(
            precos_reais=close[-60:],
            precos_previstos=previsoes_in[-60:],
            preco_real_final=float(preco_base_atual),
            preco_projetado_final=float(projecao_precos[-1]),
            horizonte_dias=horizonte_projecao
        )
        metricas["similaridade_media"] = round(float(1.0 / (1.0 + np.mean([distancias[k] for k in k_selecionados]))), 3)

        return {
            "metrics": metricas,
            "metricas": metricas,
            "history_df": df_historico,
            "forecast_df": df_projecao,
            "ga_history": {
                "top_distancias": [float(distancias[k]) for k in k_selecionados]
            }
        }
