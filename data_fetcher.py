"""
Módulo de Extração de Dados e Engenharia de Atributos (Yahoo Finance)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from config import CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Classe responsável por baixar cotações do Yahoo Finance,
    armazenar em cache local e calcular indicadores técnicos de mercado.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR

    def fetch_asset_data(
        self,
        ticker: str,
        period: Optional[str] = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Baixa o histórico de preços do ativo especificado.
        Se start_date e end_date forem informados, eles terão precedência sobre 'period'.
        """
        logger.info(f"Iniciando download para ticker: {ticker} (Period: {period}, De: {start_date} Até: {end_date})")
        
        try:
            # Baixa dados usando o objeto Ticker para maior robustez
            yf_ticker = yf.Ticker(ticker)
            if start_date and end_date:
                df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                df = yf_ticker.history(period=period or "1y", auto_adjust=True)

            if df is None or df.empty:
                # Tenta fallback via yf.download
                logger.warning(f"Ticker.history retornou vazio para {ticker}. Tentando yf.download...")
                if start_date and end_date:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                else:
                    df = yf.download(ticker, period=period or "1y", progress=False)

            if df is None or df.empty:
                raise ValueError(f"Nenhum dado encontrado para o ticker '{ticker}'. Verifique a conexão ou o código do ativo.")

            # Trata possíveis MultiIndex em colunas em versões recentes do yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Limpeza e padronização do índice de datas
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()
            # Garante que as colunas necessárias existam
            required_cols = ["Close"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Coluna obrigatória '{col}' não está presente nos dados retornados.")

            # Preenchimento de eventuais dados faltantes
            df["Close"] = df["Close"].ffill().bfill()
            if "Open" in df.columns:
                df["Open"] = df["Open"].ffill().bfill()
            if "High" in df.columns:
                df["High"] = df["High"].ffill().bfill()
            if "Low" in df.columns:
                df["Low"] = df["Low"].ffill().bfill()
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].fillna(0)

            logger.info(f"Dados baixados com sucesso: {len(df)} registros para {ticker}.")
            return df

        except Exception as e:
            logger.error(f"Erro ao baixar dados para {ticker}: {e}")
            raise

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos: Médias Móveis, RSI, MACD, Bandas de Bollinger e Volatilidade.
        """
        data = df.copy()
        close = data["Close"]

        # Médias Móveis Simples (SMA) e Exponenciais (EMA)
        data["SMA_5"] = close.rolling(window=5).mean()
        data["SMA_10"] = close.rolling(window=10).mean()
        data["SMA_20"] = close.rolling(window=20).mean()
        data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        data["EMA_26"] = close.ewm(span=26, adjust=False).mean()

        # MACD e Linha de Sinal
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # RSI (Índice de Força Relativa) de 14 períodos
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        data["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Bandas de Bollinger (20 períodos, 2 desvios padrão)
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        data["BB_Upper"] = bb_middle + (2 * bb_std)
        data["BB_Lower"] = bb_middle - (2 * bb_std)
        data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (bb_middle + 1e-9)

        # Volatilidade de 10 dias
        data["Volatility_10"] = close.pct_change().rolling(window=10).std()

        # Retorno diário percentual
        data["Return"] = close.pct_change().fillna(0)

        # Remove linhas com NaNs gerados pelo cálculo de janelas rolantes
        data = data.dropna()
        return data

    @staticmethod
    def prepare_lagged_features(
        df: pd.DataFrame,
        lookback: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index, Dict[str, float]]:
        """
        Cria a matriz de features defasadas (lags) e os vetores alvo para treino e previsão.
        Retorna (X, y, datas, scaler_params).
        """
        data = DataFetcher.calculate_technical_indicators(df)
        prices = data["Close"].values
        dates = data.index

        X_list = []
        y_list = []
        target_dates = []

        # Para cada ponto t, usamos os 'lookback' preços anteriores e os indicadores técnicos
        for i in range(lookback, len(data)):
            lags = prices[i - lookback : i]
            # Normalização local baseada na média dos lags para manter invariância de escala
            mean_val = np.mean(lags)
            std_val = np.std(lags) if np.std(lags) > 1e-6 else 1.0

            normalized_lags = (lags - mean_val) / std_val

            # Atributos técnicos no ponto i-1
            rsi = data["RSI_14"].iloc[i - 1] / 100.0  # escala [0, 1]
            macd = data["MACD"].iloc[i - 1] / (mean_val + 1e-9)
            vol = data["Volatility_10"].iloc[i - 1]
            sma_ratio = data["SMA_5"].iloc[i - 1] / (data["SMA_20"].iloc[i - 1] + 1e-9) - 1.0

            feature_vector = np.concatenate([
                normalized_lags,
                np.array([rsi, macd, vol, sma_ratio, 1.0])  # 1.0 para bias
            ])

            # Alvo normalizado
            target_norm = (prices[i] - mean_val) / std_val

            X_list.append(feature_vector)
            y_list.append(target_norm)
            target_dates.append(dates[i])

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        target_index = pd.Index(target_dates)

        # Salva parâmetros globais para desnormalização de conveniência
        scaler_params = {
            "global_mean": float(np.mean(prices)),
            "global_std": float(np.std(prices)),
            "last_mean": float(np.mean(prices[-lookback:])),
            "last_std": float(np.std(prices[-lookback:]) if np.std(prices[-lookback:]) > 1e-6 else 1.0),
            "last_close": float(prices[-1])
        }

        return X, y, target_index, scaler_params
