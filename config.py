"""
Módulo de Configurações Gerais e Mapeamento de Ativos
Sistema de Previsão de Preços com Algoritmo Genético e Yahoo Finance
"""

import os
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Criação automática dos diretórios
for directory in [OUTPUT_DIR, REPORTS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Mapeamento de Ativos Financeiros e Commodities no Yahoo Finance
ASSETS = {
    "Dólar (USD/BRL)": {
        "ticker": "USDBRL=X",
        "description": "Cotação do Dólar Americano em Reais",
        "category": "Câmbio",
        "currency": "BRL"
    },
    "Euro (EUR/BRL)": {
        "ticker": "EURBRL=X",
        "description": "Cotação do Euro em Reais",
        "category": "Câmbio",
        "currency": "BRL"
    },
    "Bitcoin (BTC/USD)": {
        "ticker": "BTC-USD",
        "description": "Cotação do Bitcoin em Dólares",
        "category": "Criptomoedas",
        "currency": "USD"
    },
    "Bitcoin (BTC/BRL)": {
        "ticker": "BTC-BRL",
        "description": "Cotação do Bitcoin em Reais",
        "category": "Criptomoedas",
        "currency": "BRL"
    },
    "Ibovespa (IBOV)": {
        "ticker": "^BVSP",
        "description": "Índice Bovespa (Referência do Mercado Brasileiro)",
        "category": "Índices",
        "currency": "BRL"
    },
    "Soja (Soybeans Futures)": {
        "ticker": "ZS=F",
        "description": "Contratos Futuros de Soja na CBOT (Chicago)",
        "category": "Commodities Agrícolas",
        "currency": "USD"
    },
    "Milho (Corn Futures)": {
        "ticker": "ZC=F",
        "description": "Contratos Futuros de Milho na CBOT (Chicago)",
        "category": "Commodities Agrícolas",
        "currency": "USD"
    },
    "Café (Coffee Futures)": {
        "ticker": "KC=F",
        "description": "Contratos Futuros de Café Arábica na ICE",
        "category": "Commodities Agrícolas",
        "currency": "USD"
    },
    "Boi Gordo (Live Cattle)": {
        "ticker": "LE=F",
        "description": "Contratos Futuros de Gado Vivo / Boi Gordo na CME",
        "category": "Pecuária",
        "currency": "USD"
    },
    "Feeder Cattle (Gado de Engorda)": {
        "ticker": "GF=F",
        "description": "Contratos Futuros de Gado de Recria/Engorda",
        "category": "Pecuária",
        "currency": "USD"
    }
}

# Períodos rápidos pré-configurados
PERIOD_CHOICES = {
    "1 Mês": "1mo",
    "3 Meses": "3mo",
    "6 Meses": "6mo",
    "1 Ano": "1y",
    "2 Anos": "2y",
    "5 Anos": "5y",
    "Personalizado": "custom"
}

# Parâmetros padrão do Algoritmo Genético
DEFAULT_GA_CONFIG = {
    "population_size": 60,       # Tamanho da população de cromossomos
    "generations": 35,           # Número de gerações evolutivas
    "crossover_rate": 0.85,      # Probabilidade de recombinação (crossover)
    "mutation_rate": 0.15,       # Probabilidade de mutação gênica
    "elitism_ratio": 0.08,       # Porcentagem dos melhores indivíduos preservados
    "tournament_size": 4,        # Tamanho do torneio para seleção
    "lookback_window": 10,       # Quantidade de lags temporais passados (dias)
    "forecast_horizon": 5        # Quantidade de passos futuros a projetar (dias)
}

# Caminho do arquivo de logs e ensaios
TRIALS_LOG_FILE = LOGS_DIR / "historico_ensaios.csv"
