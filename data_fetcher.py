"""
Módulo de Extração de Dados e Engenharia de Atributos (Yahoo Finance)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from config import CACHE_DIR, DB_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Classe responsável por baixar cotações do Yahoo Finance,
    armazenar o histórico em arquivos Excel (.xls) dentro da pasta db/
    e consultar se o intervalo da nova pesquisa já está disponível localmente.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR
        self.db_dir = DB_DIR

    @staticmethod
    def obter_caminho_arquivo_excel(ticker: str) -> Path:
        """
        Retorna o caminho canônico do arquivo Excel no diretório db/.
        Exemplos:
        - BTC-USD -> db/bitcoin.xls
        - BTC-BRL -> db/bitcoin_brl.xls
        - USDBRL=X -> db/dolar_usd_brl.xls
        - ^BVSP -> db/ibovespa.xls
        """
        mapa_conhecido = {
            "BTC-USD": "bitcoin.xls",
            "BTC-BRL": "bitcoin_brl.xls",
            "USDBRL=X": "dolar_usd_brl.xls",
            "EURBRL=X": "euro_eur_brl.xls",
            "^BVSP": "ibovespa.xls",
            "ZS=F": "soja_cbot.xls",
            "ZC=F": "milho_cbot.xls",
            "KC=F": "cafe_nybot.xls",
            "BGI=F": "boi_gordo_b3.xls",
            "CL=F": "petroleo_wti.xls",
            "GC=F": "ouro.xls",
            "ETH-USD": "ethereum.xls",
            "IVVB11.SA": "etf_ivvb11.xls",
            "SPXI11.SA": "etf_spxi11.xls",
            "SPXB11.SA": "etf_spxb11.xls",
            "SPXR11.SA": "etf_spxr11.xls",
            "SPBZ11.SA": "etf_spbz11.xls",
            "IVVB11": "etf_ivvb11.xls",
            "SPXI11": "etf_spxi11.xls",
            "SPXB11": "etf_spxb11.xls",
            "SPXR11": "etf_spxr11.xls",
            "SPBZ11": "etf_spbz11.xls"
        }
        if ticker in mapa_conhecido:
            nome_arquivo = mapa_conhecido[ticker]
        else:
            nome_sanitizado = ticker.lower().replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")
            nome_arquivo = f"{nome_sanitizado}.xls"
        return DB_DIR / nome_arquivo

    def _fetch_btc_brl(
        self,
        period: Optional[str] = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        df_existente: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Gera o histórico do par BTC-BRL sintetizado a partir de BTC-USD e USDBRL=X.
        Como o Yahoo Finance descontinuou o ticker direto 'BTC-BRL', o valor é
        sintetizado multiplicando o preço do BTC em dólares pela cotação do Dólar (USD/BRL).
        Os dados de câmbio são interpolados e propagados para cobrir finais de semana e feriados.
        Quando for criar ou atualizar o XLS em db/, baixa todos os valores diários históricos disponíveis.
        """
        logger.info("Sintetizando histórico completo diário de BTC-BRL a partir de BTC-USD e USDBRL=X...")
        try:
            # Baixa todos os valores diários históricos disponíveis (period='max') para salvar o banco completo
            df_btc = self.fetch_asset_data("BTC-USD", period="max")
            df_usd = self.fetch_asset_data("USDBRL=X", period="max")
        except Exception as e_sint:
            logger.warning(f"Erro ao obter componentes para sintetizar BTC-BRL: {e_sint}")
            if df_existente is not None and not df_existente.empty:
                logger.warning("Utilizando histórico prévio do Excel como contingência.")
                return df_existente
            raise ValueError(f"Não foi possível obter dados para sintetizar BTC-BRL: {e_sint}") from e_sint

        if df_btc is None or df_btc.empty:
            if df_existente is not None and not df_existente.empty:
                return df_existente
            raise ValueError("Não foi possível obter cotações de BTC-USD para sintetizar BTC-BRL.")

        if df_usd is None or df_usd.empty:
            if df_existente is not None and not df_existente.empty:
                return df_existente
            raise ValueError("Não foi possível obter cotações de USDBRL=X para sintetizar BTC-BRL.")

        # Alinha as datas de câmbio com as do Bitcoin (que opera 24/7)
        todas_datas = df_btc.index.union(df_usd.index).sort_values()
        usd_alinhado = df_usd.reindex(todas_datas).ffill().bfill().reindex(df_btc.index).ffill().bfill()

        df_brl = pd.DataFrame(index=df_btc.index)
        coluna_preco_usd = usd_alinhado["Close"]
        coluna_open_usd = usd_alinhado["Open"] if "Open" in usd_alinhado.columns else coluna_preco_usd
        coluna_high_usd = usd_alinhado["High"] if "High" in usd_alinhado.columns else coluna_preco_usd
        coluna_low_usd = usd_alinhado["Low"] if "Low" in usd_alinhado.columns else coluna_preco_usd

        df_brl["Close"] = df_btc["Close"] * coluna_preco_usd
        df_brl["Open"] = df_btc["Open"] * coluna_open_usd
        df_brl["High"] = df_btc["High"] * coluna_high_usd
        df_brl["Low"] = df_btc["Low"] * coluna_low_usd
        if "Volume" in df_btc.columns:
            df_brl["Volume"] = df_btc["Volume"] * coluna_preco_usd

        # Coerência de máximas e mínimas nos candles
        df_brl["High"] = df_brl[["High", "Open", "Close"]].max(axis=1)
        df_brl["Low"] = df_brl[["Low", "Open", "Close"]].min(axis=1)

        df_brl = df_brl.ffill().bfill()

        # Mescla com histórico prévio no Excel se houver
        if df_existente is not None and not df_existente.empty:
            colunas_comuns = [c for c in df_brl.columns if c in df_existente.columns]
            df_completo = pd.concat([df_existente[colunas_comuns], df_brl[colunas_comuns]])
            df_completo = df_completo[~df_completo.index.duplicated(keep="last")].sort_index()
        else:
            df_completo = df_brl

        # Salva todos os valores diários em db/bitcoin_brl.xls
        caminho_excel = self.obter_caminho_arquivo_excel("BTC-BRL")
        try:
            self.db_dir.mkdir(parents=True, exist_ok=True)
            df_completo.to_excel(caminho_excel, engine="openpyxl")
            logger.info(
                f"Histórico consolidado com todos os valores diários de BTC-BRL salvo em 'db/{caminho_excel.name}' "
                f"({len(df_completo)} registros, de {df_completo.index.min().date()} a {df_completo.index.max().date()})."
            )
        except Exception as e_salvamento:
            logger.warning(f"Não foi possível salvar {caminho_excel}: {e_salvamento}")

        # Retorna a fatia solicitada pelo chamador se houver restrição de datas
        if start_date and end_date:
            data_inicio_req = pd.to_datetime(start_date)
            data_fim_req = pd.to_datetime(end_date)
            fatia = df_completo[
                (df_completo.index >= data_inicio_req) & (df_completo.index <= data_fim_req)
            ]
            if len(fatia) >= 20:
                return fatia
        elif period and period != "max":
            agora = pd.Timestamp.now()
            dias_minimos_periodo = {
                "1mo": 35,
                "3mo": 100,
                "6mo": 190,
                "1y": 375,
                "2y": 740,
                "5y": 1850
            }
            dias_req = dias_minimos_periodo.get(period, 375)
            data_inicio_req = agora - timedelta(days=dias_req)
            fatia = df_completo[df_completo.index >= data_inicio_req]
            if len(fatia) >= 20:
                return fatia

        return df_completo

    def fetch_asset_data(
        self,
        ticker: str,
        period: Optional[str] = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Obtém o histórico de preços do ativo especificado.
        Primeiro consulta se o intervalo solicitado já está baixado no arquivo .xls correspondente
        dentro da pasta db/. Se estiver coberto, carrega direto do Excel. Caso contrário, baixa
        todos os valores diários disponíveis do Yahoo Finance (period='max' / histórico completo),
        mescla com os dados prévios e salva a planilha com todos os valores diários em db/,
        retornando a fatia solicitada.
        """
        # Normaliza tickers de ETFs e ativos da B3 caso venham sem o sufixo .SA
        etfs_b3 = {"IVVB11", "SPXI11", "SPXB11", "SPXR11", "SPBZ11"}
        if ticker.upper() in etfs_b3:
            ticker = f"{ticker.upper()}.SA"

        agora = pd.Timestamp.now()

        if start_date and end_date:
            data_inicio_req = pd.to_datetime(start_date)
            data_fim_req = pd.to_datetime(end_date)
            tolerancia_inicio = timedelta(days=4)
        else:
            # Tolerância para períodos aproximados (ex: 1mo = ~27-31 dias de histórico)
            dias_minimos_periodo = {
                "1mo": 25,
                "3mo": 80,
                "6mo": 170,
                "1y": 350,
                "2y": 700,
                "5y": 1780,
                "max": 3400
            }
            dias_req = dias_minimos_periodo.get(period or "1y", 350)
            data_inicio_req = agora - timedelta(days=dias_req)
            data_fim_req = agora
            tolerancia_inicio = timedelta(days=5)

        caminho_excel = self.obter_caminho_arquivo_excel(ticker)
        df_existente: Optional[pd.DataFrame] = None

        # 1. Consulta se o arquivo Excel já existe e se cobre o intervalo desejado
        if self.use_cache and caminho_excel.exists():
            try:
                df_carregado = pd.read_excel(caminho_excel, index_col=0, engine="openpyxl")
                df_carregado.index = pd.to_datetime(df_carregado.index)
                if df_carregado.index.tz is not None:
                    df_carregado.index = df_carregado.index.tz_localize(None)
                df_carregado = df_carregado.sort_index()

                if not df_carregado.empty and len(df_carregado) >= 20:
                    df_existente = df_carregado
                    data_min_excel = df_carregado.index.min()
                    data_max_excel = df_carregado.index.max()

                    # Tolerância para finais de semana, feriados e períodos
                    tolerancia_recente = agora - timedelta(days=5)
                    coberto_inicio = data_min_excel <= (data_inicio_req + tolerancia_inicio)
                    coberto_fim = (data_fim_req <= data_max_excel) or (data_max_excel >= tolerancia_recente)

                    if coberto_inicio and coberto_fim:
                        logger.info(
                            f"Consulta atendida diretamente do Excel db/{caminho_excel.name} "
                            f"(cobertura: {data_min_excel.date()} até {data_max_excel.date()})."
                        )
                        fatia = df_carregado[
                            (df_carregado.index >= data_inicio_req) & (df_carregado.index <= data_fim_req)
                        ]
                        if len(fatia) >= 20:
                            return fatia
                        # Caso a fatia resulte em poucos pontos, usa todo o histórico contido
                        return df_carregado

            except Exception as e_leitura:
                logger.warning(f"Erro ao ler cache Excel em {caminho_excel}: {e_leitura}. Baixando novos dados...")

        # 2. Se for BTC-BRL, sintetiza a partir de BTC-USD e USDBRL=X (pois o Yahoo delistou o par direto)
        if ticker == "BTC-BRL":
            return self._fetch_btc_brl(
                period=period,
                start_date=start_date,
                end_date=end_date,
                df_existente=df_existente
            )

        # 3. Se não estiver coberto ou ao criar o XLS em db/, baixa TODOS os valores diários (period='max' / histórico completo)
        logger.info(f"Baixando todos os valores diários de {ticker} no Yahoo Finance para criar/atualizar db/{caminho_excel.name}...")
        try:
            yf_ticker = yf.Ticker(ticker)

            # Baixa todos os valores diários históricos disponíveis
            df_novo = yf_ticker.history(period="max", interval="1d", auto_adjust=True)

            if df_novo is None or df_novo.empty:
                logger.warning(f"Ticker.history(period='max') retornou vazio para {ticker}. Tentando yf.download(period='max')...")
                df_novo = yf.download(ticker, period="max", interval="1d", progress=False)

            # Fallback com parâmetros de data explícitos se period='max' falhar
            if (df_novo is None or df_novo.empty) and start_date and end_date:
                logger.warning(f"Tentando download com datas explícitas ({start_date} a {end_date})...")
                df_novo = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
                if df_novo is None or df_novo.empty:
                    df_novo = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df_novo is None or df_novo.empty:
                # Se falhar o download online mas tivermos algum dado em cache Excel, usa o Excel como fallback
                if df_existente is not None and not df_existente.empty:
                    logger.warning(f"Falha no download online para {ticker}. Utilizando dados prévios do Excel.")
                    return df_existente
                raise ValueError(f"Nenhum dado encontrado para o ticker '{ticker}'. Verifique a conexão ou o código do ativo.")

            if isinstance(df_novo.columns, pd.MultiIndex):
                df_novo.columns = df_novo.columns.get_level_values(0)

            df_novo.index = pd.to_datetime(df_novo.index)
            if df_novo.index.tz is not None:
                df_novo.index = df_novo.index.tz_localize(None)

            df_novo = df_novo.sort_index()

            for col in ["Close"]:
                if col not in df_novo.columns:
                    raise ValueError(f"Coluna obrigatória '{col}' não está presente nos dados retornados.")

            df_novo["Close"] = df_novo["Close"].ffill().bfill()
            for c in ["Open", "High", "Low"]:
                if c in df_novo.columns:
                    df_novo[c] = df_novo[c].ffill().bfill()
            if "Volume" in df_novo.columns:
                df_novo["Volume"] = df_novo["Volume"].fillna(0)

            # 4. Mescla com os dados existentes no arquivo Excel (se houver) para consolidar histórico completo
            if df_existente is not None and not df_existente.empty:
                colunas_comuns = [c for c in df_novo.columns if c in df_existente.columns]
                df_completo = pd.concat([df_existente[colunas_comuns], df_novo[colunas_comuns]])
                df_completo = df_completo[~df_completo.index.duplicated(keep="last")].sort_index()
            else:
                df_completo = df_novo

            # 5. Salva TODOS os valores diários no arquivo .xls na pasta db/
            try:
                self.db_dir.mkdir(parents=True, exist_ok=True)
                df_completo.to_excel(caminho_excel, engine="openpyxl")
                logger.info(
                    f"Histórico com todos os valores diários de {ticker} salvo em 'db/{caminho_excel.name}' "
                    f"({len(df_completo)} registros diários, de {df_completo.index.min().date()} a {df_completo.index.max().date()})."
                )
            except Exception as e_salvamento:
                logger.warning(f"Não foi possível salvar {caminho_excel}: {e_salvamento}")

            # 6. Retorna a fatia solicitada pelo chamador
            if start_date and end_date:
                fatia = df_completo[
                    (df_completo.index >= data_inicio_req) & (df_completo.index <= data_fim_req)
                ]
                if len(fatia) >= 20:
                    return fatia
            elif period and period != "max":
                fatia = df_completo[
                    (df_completo.index >= data_inicio_req) & (df_completo.index <= data_fim_req)
                ]
                if len(fatia) >= 20:
                    return fatia

            return df_completo

        except Exception as e:
            logger.error(f"Erro ao obter dados para {ticker}: {e}")
            if df_existente is not None and not df_existente.empty:
                logger.warning("Recorrendo aos dados disponíveis em cache local Excel.")
                return df_existente
            raise

    def atualizar_todos_ativos_db(self) -> Dict[str, Path]:
        """
        Percorre todos os ativos mapeados no sistema, baixa todos os valores diários
        históricos disponíveis via Yahoo Finance e salva/atualiza as planilhas .xls no diretório db/.
        Retorna um dicionário mapeando ticker para o caminho do arquivo gerado.
        """
        from config import ASSETS
        arquivos_atualizados = {}
        for nome_ativo, info in ASSETS.items():
            ticker = info["ticker"]
            logger.info(f"Atualizando banco diário completo para {nome_ativo} ({ticker})...")
            try:
                # Força download completo de todos os valores diários
                caminho = self.obter_caminho_arquivo_excel(ticker)
                self.fetch_asset_data(ticker=ticker, period="max")
                arquivos_atualizados[ticker] = caminho
            except Exception as e_ativo:
                logger.error(f"Erro ao atualizar {nome_ativo} ({ticker}): {e_ativo}")
        return arquivos_atualizados

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
