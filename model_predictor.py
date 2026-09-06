"""
Módulo de Inferência Preditiva e Projeção Temporal Futura
Reconstrução das previsões em escala original, cálculo de métricas estatísticas e projeção out-of-sample.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from genetic_engine import Individual

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Realiza a previsão de séries temporais utilizando o melhor indivíduo
    otimizado pelo Algoritmo Genético, revertendo as escalas normalizadas
    para os preços reais da moeda do ativo.
    """

    def __init__(self, best_individual: Individual, lookback_window: int = 10):
        self.individual = best_individual
        self.lookback = lookback_window

    def evaluate_and_predict(
        self,
        df_raw: pd.DataFrame,
        X: np.ndarray,
        dates: pd.Index,
        forecast_horizon: int = 5,
        is_crypto: bool = False
    ) -> Dict[str, Any]:
        """
        Executa a inferência histórica e projeta os próximos 'forecast_horizon' períodos no futuro.
        """
        # Obtenção dos preços de fechamento alinhados
        close_prices = df_raw["Close"].values
        # Os alvos reais correspondentes aos pontos de X
        # Como X começa após o período de warm-up dos indicadores + lookback
        total_points = len(X)
        actual_prices = close_prices[-total_points:]
        aligned_dates = dates[-total_points:]

        # 1. Inferência normalizada
        y_pred_norm = self.individual.predict(X)

        # 2. Desnormalização ponto a ponto
        predicted_prices = []
        for i in range(total_points):
            idx_in_close = len(close_prices) - total_points + i
            lag_window = close_prices[idx_in_close - self.lookback : idx_in_close]
            mean_lag = np.mean(lag_window)
            std_lag = np.std(lag_window) if np.std(lag_window) > 1e-6 else 1.0

            pred_real = (y_pred_norm[i] * std_lag) + mean_lag
            predicted_prices.append(pred_real)

        predicted_prices = np.array(predicted_prices, dtype=np.float64)

        # 3. Cálculo de Métricas Estatísticas em escala real
        residuals = actual_prices - predicted_prices
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals / (actual_prices + 1e-9)))) * 100.0

        ss_total = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
        ss_res = np.sum(residuals ** 2)
        r2 = float(1.0 - (ss_res / (ss_total + 1e-9))) if ss_total > 0 else 0.0

        # Direcionalidade (% acerto se sobe ou desce comparado ao fechamento anterior)
        if total_points > 1:
            actual_diff = np.diff(actual_prices)
            pred_diff = predicted_prices[1:] - actual_prices[:-1]
            directional_acc = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff))) * 100.0
        else:
            directional_acc = 100.0

        # Montagem do DataFrame Histórico Comparativo
        df_history = pd.DataFrame({
            "Data": aligned_dates,
            "Preco_Real": actual_prices,
            "Preco_Previsto_IA": predicted_prices,
            "Erro_Absoluto": np.abs(residuals),
            "Erro_Percentual": np.abs(residuals / (actual_prices + 1e-9)) * 100.0
        }).set_index("Data")

        # 4. Projeção Recursiva Out-of-Sample para o Futuro
        future_dates, future_preds, future_lower, future_upper = self._forecast_future(
            df_raw=df_raw,
            last_date=aligned_dates[-1],
            horizon=forecast_horizon,
            rmse_std=rmse,
            is_crypto=is_crypto
        )

        df_forecast = pd.DataFrame({
            "Data": future_dates,
            "Preco_Projetado": future_preds,
            "Limite_Inferior": future_lower,
            "Limite_Superior": future_upper
        }).set_index("Data")

        # Resumo executivo da projeção
        last_real_price = float(actual_prices[-1])
        final_projected_price = float(future_preds[-1])
        variation_pct = ((final_projected_price - last_real_price) / (last_real_price + 1e-9)) * 100.0
        trend = "Alta (Subida)" if variation_pct > 0.05 else ("Baixa (Queda)" if variation_pct < -0.05 else "Neutro / Estavel")

        summary = {
            "ultimo_preco_real": last_real_price,
            "preco_projetado_final": final_projected_price,
            "variacao_esperada_pct": variation_pct,
            "tendencia_esperada": trend,
            "horizonte_dias": forecast_horizon,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "acuracia_direcional": directional_acc
        }

        return {
            "metrics": summary,
            "history_df": df_history,
            "forecast_df": df_forecast
        }

    def _forecast_future(
        self,
        df_raw: pd.DataFrame,
        last_date: pd.Timestamp,
        horizon: int,
        rmse_std: float,
        is_crypto: bool
    ) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float]]:
        """
        Executa projeção recursiva passo a passo no horizonte especificado.
        """
        recent_prices = list(df_raw["Close"].values)
        future_dates: List[pd.Timestamp] = []
        future_preds: List[float] = []
        future_lower: List[float] = []
        future_upper: List[float] = []

        current_date = last_date

        for step in range(1, horizon + 1):
            # Próxima data de mercado
            if is_crypto:
                current_date = current_date + pd.Timedelta(days=1)
            else:
                # Mercado tradicional: pular finais de semana
                current_date = current_date + pd.Timedelta(days=1)
                while current_date.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
                    current_date = current_date + pd.Timedelta(days=1)

            # Extração da janela mais recente
            lags = np.array(recent_prices[-self.lookback:], dtype=np.float64)
            mean_lag = float(np.mean(lags))
            std_lag = float(np.std(lags) if np.std(lags) > 1e-6 else 1.0)
            normalized_lags = (lags - mean_lag) / std_lag

            # Indicadores proxy recentes
            recent_series = pd.Series(recent_prices[-30:])
            delta = recent_series.diff()
            gain = (delta.where(delta > 0, 0.0)).mean()
            loss = (-delta.where(delta < 0, 0.0)).mean()
            rs = gain / (loss + 1e-9)
            rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0

            ema12 = recent_series.ewm(span=12).mean().iloc[-1]
            ema26 = recent_series.ewm(span=26).mean().iloc[-1]
            macd = (ema12 - ema26) / (mean_lag + 1e-9)
            vol = float(recent_series.pct_change().std())
            sma5 = recent_series.rolling(5).mean().iloc[-1]
            sma20 = recent_series.rolling(20).mean().iloc[-1]
            sma_ratio = (sma5 / (sma20 + 1e-9)) - 1.0

            feat_vector = np.concatenate([
                normalized_lags,
                np.array([rsi, macd, vol, sma_ratio, 1.0])
            ])

            # Predição normalizada e desnormalização
            pred_norm = float(np.dot(feat_vector, self.individual.genes))
            pred_real = float((pred_norm * std_lag) + mean_lag)

            # Acumula na série para permitir projeção autorregressiva recursiva
            recent_prices.append(pred_real)

            # Intervalo de incerteza empírica acumulativa (expande com a raiz dos passos)
            uncertainty_margin = 1.96 * rmse_std * np.sqrt(step)

            future_dates.append(current_date)
            future_preds.append(pred_real)
            future_lower.append(max(0.0, pred_real - uncertainty_margin))
            future_upper.append(pred_real + uncertainty_margin)

        return future_dates, future_preds, future_lower, future_upper
