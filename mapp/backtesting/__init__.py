"""
Pacote de Backtesting e Avaliação de Desempenho (mapp/backtesting/__init__.py)
Exporta o motor BacktestEngine e o CalculadorMetricas.
"""

from mapp.backtesting.engine import BacktestEngine
from mapp.backtesting.metrics import CalculadorMetricas, MetricasBacktesting

__all__ = ["BacktestEngine", "CalculadorMetricas", "MetricasBacktesting"]
