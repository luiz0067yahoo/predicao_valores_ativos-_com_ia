"""
Módulo Base de Algoritmos Preditivos
Define a classe abstrata que padroniza a interface de todos os modelos de IA,
além de funções utilitárias para métricas estatísticas e projeções recursivas.
Todos os nomes de arquivos, classes, variáveis e comentários estão em português.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaseAlgoritmo(ABC):
    """
    Classe base abstrata para todos os modelos preditivos do sistema.
    Assegura que todo algoritmo forneça treinamento, projeção futura e
    cálculo uniforme de intervalos de confiança e métricas.
    """

    def __init__(self, nome_identificador: str, janela_temporal: int = 10):
        """
        Inicializa os atributos fundamentais do algoritmo.

        Parâmetros:
            nome_identificador: Rótulo de exibição do algoritmo (ex: 'XGBoost', 'LSTM').
            janela_temporal: Quantidade de dias passados (lags) observados pelo modelo.
        """
        self.nome_identificador = nome_identificador
        self.janela_temporal = janela_temporal
        self.historico_aprendizado: Dict[str, Any] = {}

    @abstractmethod
    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo:
        1. Pré-processamento e engenharia de características.
        2. Treinamento/ajuste dos pesos do modelo com feedback periódico de progresso.
        3. Avaliação dos dados históricos (in-sample) e projeção futura recursiva.
        4. Cálculo de métricas e intervalos de confiança (95%).

        Parâmetros:
            dados_completos: DataFrame histórico com colunas [Open, High, Low, Close, Volume].
            horizonte_projecao: Número de dias futuros a serem projetados.
            eh_criptomoeda: Se True, considera negociação contínua aos finais de semana.
            funcao_progresso: Callback para informar progresso (passo_atual, total_passos, status, metrica_atual).
            hiperparametros: Dicionário opcional com parâmetros de ajuste fino do modelo.

        Retorno:
            Dicionário com 'metrics', 'history_df', 'forecast_df' e 'ga_history'.
        """
        pass

    @staticmethod
    def gerar_datas_futuras(
        ultima_data: pd.Timestamp,
        quantidade_dias: int,
        eh_criptomoeda: bool = False
    ) -> List[pd.Timestamp]:
        """
        Gera uma lista de datas futuras a partir da última data conhecida da série.

        Parâmetros:
            ultima_data: Timestamp da última linha histórica disponível.
            quantidade_dias: Quantidade de passos à frente para projetar.
            eh_criptomoeda: Se True, inclui sábado e domingo (mercado 24/7).

        Retorno:
            Lista contendo as próximas datas cronológicas.
        """
        datas_futuras: List[pd.Timestamp] = []
        data_corrente = ultima_data

        while len(datas_futuras) < quantidade_dias:
            data_corrente = data_corrente + timedelta(days=1)
            # Mercado tradicional fecha aos finais de semana (5 = Sábado, 6 = Domingo)
            if not eh_criptomoeda and data_corrente.weekday() >= 5:
                continue
            datas_futuras.append(data_corrente)

        return datas_futuras

    @staticmethod
    def calcular_metricas_estatisticas(
        precos_reais: np.ndarray,
        precos_previstos: np.ndarray,
        preco_real_final: float,
        preco_projetado_final: float,
        horizonte_dias: int = 5
    ) -> Dict[str, Any]:
        """
        Calcula as métricas quantitativas de desempenho preditivo do modelo:
        - RMSE (Raiz do Erro Quadrático Médio)
        - MAE (Erro Absoluto Médio)
        - MAPE (Erro Percentual Absoluto Médio)
        - R² (Coeficiente de Determinação)
        - Acurácia Direcional (% de acerto na direção de subida/descida diária)

        Parâmetros:
            precos_reais: Array com os valores reais da série histórica.
            precos_previstos: Array com os valores ajustados pelo modelo.
            preco_real_final: Último preço de fechamento real conhecido.
            preco_projetado_final: Preço estimado no último dia do horizonte futuro.
            horizonte_dias: Quantidade de períodos projetados à frente.

        Retorno:
            Dicionário estruturado com todas as métricas formatadas.
        """
        # Erro residual entre os valores reais e as predições
        residuos = precos_reais - precos_previstos

        # 1. RMSE: penaliza erros de grande magnitude
        rmse = float(np.sqrt(np.mean(residuos ** 2)))

        # 2. MAE: média linear dos erros absolutos
        mae = float(np.mean(np.abs(residuos)))

        # 3. MAPE: erro percentual relativo
        denominador = np.where(np.abs(precos_reais) < 1e-8, 1e-8, np.abs(precos_reais))
        mape = float(np.mean(np.abs(residuos) / denominador) * 100.0)

        # 4. R² (Coeficiente de Determinação): proporção da variância explicada
        variancia_total = float(np.sum((precos_reais - np.mean(precos_reais)) ** 2))
        variancia_residual = float(np.sum(residuos ** 2))
        if variancia_total > 1e-8:
            r2 = float(max(-2.0, 1.0 - (variancia_residual / variancia_total)))
        else:
            r2 = 0.0

        # 5. Acurácia Direcional: assertividade do sentido da variação (subiu ou caiu)
        if len(precos_reais) > 1:
            direcao_real = np.sign(np.diff(precos_reais))
            direcao_prevista = np.sign(np.diff(precos_previstos))
            acertos_direcionais = np.sum(direcao_real == direcao_prevista)
            acuracia_direcional = float((acertos_direcionais / len(direcao_real)) * 100.0)
        else:
            acuracia_direcional = 50.0

        # 6. Variação Percentual Esperada no Horizonte
        variacao_esperada_pct = float(
            ((preco_projetado_final - preco_real_final) / (preco_real_final + 1e-8)) * 100.0
        )

        # 7. Classificação da Tendência
        if variacao_esperada_pct > 0.5:
            tendencia_esperada = "Alta Estimada (Bullish)"
        elif variacao_esperada_pct < -0.5:
            tendencia_esperada = "Baixa Estimada (Bearish)"
        else:
            tendencia_esperada = "Lateralidade / Neutro"

        return {
            "ultimo_preco_real": round(preco_real_final, 4),
            "preco_projetado_final": round(preco_projetado_final, 4),
            "variacao_esperada_pct": round(variacao_esperada_pct, 2),
            "tendencia_esperada": tendencia_esperada,
            "horizonte_dias": int(horizonte_dias),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
            "acuracia_direcional": round(acuracia_direcional, 1)
        }

    @staticmethod
    def calcular_intervalos_confianca(
        precos_projetados: List[float],
        desvio_padrao_residuos: float
    ) -> Tuple[List[float], List[float]]:
        """
        Gera o cone de incerteza estatístico com 95% de confiança empírica (Z = 1.96).
        A incerteza se expande proporcionalmente à raiz quadrada do passo temporal:
        Limite = Projeção ± 1.96 * sigma * sqrt(passo).

        Parâmetros:
            precos_projetados: Lista com os preços calculados para cada passo à frente.
            desvio_padrao_residuos: Dispersão histórica dos erros do modelo.

        Retorno:
            Tupla contendo (limites_inferiores, limites_superiores).
        """
        limites_inferiores: List[float] = []
        limites_superiores: List[float] = []

        desvio_base = max(desvio_padrao_residuos, 1e-4)

        for passo, preco in enumerate(precos_projetados, start=1):
            fator_expansao = 1.96 * desvio_base * np.sqrt(passo)
            # Garante que o limite inferior nunca seja negativo para cotações
            limite_inf = max(0.0, float(preco - fator_expansao))
            limite_sup = float(preco + fator_expansao)
            limites_inferiores.append(limite_inf)
            limites_superiores.append(limite_sup)

        return limites_inferiores, limites_superiores

    def desnormalizar_historico(
        self,
        dados_completos: pd.DataFrame,
        total_pontos: int,
        predicoes_normalizadas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Desnormaliza as predições históricas in-sample usando a média e desvio padrão
        locais de cada janela temporal correspondente.
        Retorna (precos_reais, precos_previstos_reais).
        """
        precos_fechamento = dados_completos["Close"].values
        precos_reais = precos_fechamento[-total_pontos:]

        precos_previstos = []
        for i in range(total_pontos):
            idx_no_fechamento = len(precos_fechamento) - total_pontos + i
            janela = precos_fechamento[idx_no_fechamento - self.janela_temporal : idx_no_fechamento]
            media_janela = float(np.mean(janela))
            desvio_janela = float(np.std(janela) if np.std(janela) > 1e-6 else 1.0)
            pred_real = (predicoes_normalizadas[i] * desvio_janela) + media_janela
            precos_previstos.append(pred_real)

        return np.array(precos_reais, dtype=np.float64), np.array(precos_previstos, dtype=np.float64)

    def projetar_futuro_recursivo(
        self,
        dados_completos: pd.DataFrame,
        funcao_predicao_vetor: Callable[[np.ndarray], float],
        horizonte_projecao: int,
        desvio_padrao_erros: float,
        eh_criptomoeda: bool
    ) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float]]:
        """
        Realiza a projeção recursiva dia a dia calculando indicadores técnicos dinamicamente
        e expandindo o cone de incerteza de 95%.
        """
        precos_recentes = list(dados_completos["Close"].values)
        datas_futuras: List[pd.Timestamp] = []
        previsoes_futuras: List[float] = []
        limites_inferiores: List[float] = []
        limites_superiores: List[float] = []

        data_corrente = pd.to_datetime(dados_completos.index[-1])

        for passo in range(1, horizonte_projecao + 1):
            data_corrente = data_corrente + pd.Timedelta(days=1)
            if not eh_criptomoeda:
                while data_corrente.weekday() >= 5:
                    data_corrente = data_corrente + pd.Timedelta(days=1)
            datas_futuras.append(data_corrente)

            janela_lags = np.array(precos_recentes[-self.janela_temporal:], dtype=np.float64)
            media_lag = float(np.mean(janela_lags))
            desvio_lag = float(np.std(janela_lags) if np.std(janela_lags) > 1e-6 else 1.0)
            lags_normalizados = (janela_lags - media_lag) / desvio_lag

            serie_recente = pd.Series(precos_recentes[-30:])
            delta = serie_recente.diff()
            ganho = float((delta.where(delta > 0, 0.0)).mean())
            perda = float((-delta.where(delta < 0, 0.0)).mean())
            rs = ganho / (perda + 1e-9)
            rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0

            ema12 = float(serie_recente.ewm(span=12).mean().iloc[-1])
            ema26 = float(serie_recente.ewm(span=26).mean().iloc[-1])
            macd = (ema12 - ema26) / (media_lag + 1e-9)
            vol = float(serie_recente.pct_change().std() or 0.01)
            sma5 = float(serie_recente.rolling(5).mean().iloc[-1] if len(serie_recente) >= 5 else media_lag)
            sma20 = float(serie_recente.rolling(20).mean().iloc[-1] if len(serie_recente) >= 20 else media_lag)
            razao_sma = (sma5 / (sma20 + 1e-9)) - 1.0

            vetor_caracteristicas = np.concatenate([
                lags_normalizados,
                np.array([rsi, macd, vol, razao_sma, 1.0])
            ]).reshape(1, -1)

            pred_norm = float(funcao_predicao_vetor(vetor_caracteristicas))
            pred_real = float((pred_norm * desvio_lag) + media_lag)
            precos_recentes.append(pred_real)
            previsoes_futuras.append(pred_real)

            desvio_base = max(desvio_padrao_erros, 1e-4)
            fator_expansao = 1.96 * desvio_base * np.sqrt(passo)
            limite_inf = max(0.0, float(pred_real - fator_expansao))
            limite_sup = float(pred_real + fator_expansao)
            limites_inferiores.append(limite_inf)
            limites_superiores.append(limite_sup)

        return datas_futuras, previsoes_futuras, limites_inferiores, limites_superiores

