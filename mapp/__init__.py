"""
Ecossistema M.A.P.P. — Market Analysis Pattern Prediction (mapp/__init__.py)
Plataforma quantitativa modular para:
- Regime de Mercado (MarketRegimeDetector)
- Normalização Inteligente de Horizonte (ForecastHorizon)
- Cronometragem e Estimativa de Tempo Real (ProgressTracker)
- Engenharia de Atributos sem Vazamento Temporal (mapp.features)
- Motor de Backtesting Walk-Forward (BacktestEngine)
- Simulador de Investimentos Multi-Algoritmo (InvestmentSimulator)
- Otimização Multiobjetivo de Hiperparâmetros (HyperparameterOptimizer)
"""

from mapp.horizon import ForecastHorizon, NormalizadorHorizonte
from mapp.tracker import ProgressTracker, EstadoProgresso
from mapp.market_regime import MarketRegimeDetector, TipoRegimeMercado, ResultadoRegimeMercado
from mapp.backtesting import BacktestEngine, CalculadorMetricas, MetricasBacktesting
from mapp.simulator import InvestmentSimulator, HyperparameterOptimizer

__all__ = [
    "ForecastHorizon",
    "NormalizadorHorizonte",
    "ProgressTracker",
    "EstadoProgresso",
    "MarketRegimeDetector",
    "TipoRegimeMercado",
    "ResultadoRegimeMercado",
    "BacktestEngine",
    "CalculadorMetricas",
    "MetricasBacktesting",
    "InvestmentSimulator",
    "HyperparameterOptimizer"
]
