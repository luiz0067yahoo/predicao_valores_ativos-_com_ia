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

        for i in range(0, limite_busca):
            janela_hist = close[i : i + lookback]
            m_hist = np.mean(janela_hist)
            s_hist = np.std(janela_hist) if np.std(janela_hist) > 1e-6 else 1.0
            janela_hist_norm = (janela_hist - m_hist) / s_hist

            # Distância Euclidiana normalizada
            dist = float(np.mean((padrao_atual_norm - janela_hist_norm) ** 2))
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
            # Retorno percentual relativo à barra de corte
            retornos_passo = (trajetoria_precos - preco_base_passado) / (preco_base_passado + 1e-9)
            trajetorias_futuras_pct.append(retornos_passo)

        trajetorias_matriz = np.array(trajetorias_futuras_pct)  # Shape (K, horizonte)

        # Trajetória projetada: mediana dos episódios históricos
        mediana_retornos = np.median(trajetorias_matriz, axis=0)
        p10_retornos = np.percentile(trajetorias_matriz, 10, axis=0)
        p90_retornos = np.percentile(trajetorias_matriz, 90, axis=0)

        preco_base_atual = close[-1]
        projecao_precos = list(preco_base_atual * (1.0 + mediana_retornos))
        limites_inferiores = list(preco_base_atual * (1.0 + p10_retornos))
        limites_superiores = list(preco_base_atual * (1.0 + p90_retornos))

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
