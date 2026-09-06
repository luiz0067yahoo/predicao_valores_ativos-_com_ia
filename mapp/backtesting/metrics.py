"""
Módulo de Métricas Estatísticas e Financeiras de Backtesting (mapp/backtesting/metrics.py)
Calcula com rigor: Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate,
Profit Factor, Calmar Ratio, RMSE, MAE, MAPE, R² e Acurácia Direcional.
Sem métricas artificiais.
Todos os nomes em português.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd


@dataclass
class MetricasBacktesting:
    """Conjunto unificado de métricas de desempenho e risco financeiro."""
    capital_inicial: float
    capital_final: float
    lucro_total: float
    retorno_total_pct: float
    retorno_anualizado_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    calmar_ratio: float
    total_operacoes: int
    operacoes_vencedoras: int
    operacoes_perdedoras: int
    rmse: float
    mae: float
    mape_pct: float
    r2: float
    acuracia_direcional_pct: float

    @property
    def sharpe(self) -> float:
        return self.sharpe_ratio

    def para_dicionario(self) -> Dict[str, Any]:
        """Serializa em formato JSON amigável com arredondamentos elegantes."""
        return {
            "initial_capital": round(self.capital_inicial, 2),
            "final_capital": round(self.capital_final, 2),
            "total_profit": round(self.lucro_total, 2),
            "total_return_pct": round(self.retorno_total_pct, 2),
            "annualized_return_pct": round(self.retorno_anualizado_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "total_trades": self.total_operacoes,
            "winning_trades": self.operacoes_vencedoras,
            "losing_trades": self.operacoes_perdedoras,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mape_pct": round(self.mape_pct, 2),
            "r2": round(self.r2, 4),
            "directional_accuracy_pct": round(self.acuracia_direcional_pct, 1)
        }


class CalculadorMetricas:
    """
    Motor matemático de avaliação financeira e estatística de estratégias de investimento.
    """

    @classmethod
    def calcular_todas(
        cls,
        curva_capital: Any,
        retornos_operacoes: Any,
        precos_reais: np.ndarray,
        precos_previstos: np.ndarray,
        taxa_livre_risco_anual: float = 0.1075
    ) -> MetricasBacktesting:
        """Alias flexível aceitando np.ndarray ou pd.Series."""
        if not isinstance(curva_capital, pd.Series):
            curva_capital = pd.Series(curva_capital)
        if isinstance(retornos_operacoes, np.ndarray):
            retornos_operacoes = list(retornos_operacoes)
        return cls.calcular_metricas_completas(
            curva_capital=curva_capital,
            retornos_operacoes=retornos_operacoes,
            precos_reais=np.asarray(precos_reais),
            precos_previstos=np.asarray(precos_previstos),
            taxa_livre_risco_anual=taxa_livre_risco_anual
        )

    @classmethod
    def calcular_metricas_completas(
        cls,
        curva_capital: pd.Series,
        retornos_operacoes: List[float],
        precos_reais: np.ndarray,
        precos_previstos: np.ndarray,
        taxa_livre_risco_anual: float = 0.1075  # Selic média aproximada 10.75%
    ) -> MetricasBacktesting:
        """
        Calcula as métricas financeiras e de acurácia preditiva.
        """
        if not isinstance(curva_capital, pd.Series):
            curva_capital = pd.Series(curva_capital)
        if len(curva_capital) == 0:
            return cls._metricas_vazias()

        cap_ini = float(curva_capital.iloc[0])
        cap_fim = float(curva_capital.iloc[-1])
        lucro = cap_fim - cap_ini
        retorno_total = (lucro / (cap_ini + 1e-9)) * 100.0

        n_barras = len(curva_capital)
        anos = max(0.08, n_barras / 252.0)
        retorno_anualizado = (((cap_fim / (cap_ini + 1e-9)) ** (1.0 / anos)) - 1.0) * 100.0

        # Retornos diários do portfólio
        retornos_diarios = curva_capital.pct_change().dropna().values
        if len(retornos_diarios) > 2 and np.std(retornos_diarios) > 1e-6:
            rf_diaria = (1.0 + taxa_livre_risco_anual) ** (1.0 / 252.0) - 1.0
            excesso = retornos_diarios - rf_diaria
            sharpe = float((np.mean(excesso) / (np.std(retornos_diarios) + 1e-9)) * np.sqrt(252))

            # Sortino (apenas desvio dos retornos negativos)
            ret_neg = excesso[excesso < 0]
            if len(ret_neg) > 1 and np.std(ret_neg) > 1e-6:
                sortino = float((np.mean(excesso) / (np.std(ret_neg) + 1e-9)) * np.sqrt(252))
            else:
                sortino = sharpe
        else:
            sharpe = 0.0
            sortino = 0.0

        # Maximum Drawdown
        picos = curva_capital.cummax()
        drawdowns = (curva_capital - picos) / (picos + 1e-9)
        max_dd = float(abs(drawdowns.min())) * 100.0

        # Calmar Ratio
        calmar = float(retorno_anualizado / (max_dd + 1e-9)) if max_dd > 0.01 else 0.0

        # Métricas de Operações (Trades)
        total_ops = len(retornos_operacoes)
        if total_ops > 0:
            ganhos = [r for r in retornos_operacoes if r > 0]
            perdas = [r for r in retornos_operacoes if r <= 0]
            vencedoras = len(ganhos)
            perdedoras = len(perdas)
            win_rate = (vencedoras / total_ops) * 100.0

            soma_ganhos = sum(ganhos) if ganhos else 0.0
            soma_perdas = abs(sum(perdas)) if perdas else 1e-9
            profit_factor = float(soma_ganhos / soma_perdas) if soma_perdas > 1e-9 else 2.0
        else:
            vencedoras = 0
            perdedoras = 0
            win_rate = 0.0
            profit_factor = 1.0

        # Métricas Estatísticas do Modelo Preditivo
        if len(precos_reais) > 0 and len(precos_previstos) == len(precos_reais):
            erros = precos_reais - precos_previstos
            rmse = float(np.sqrt(np.mean(erros ** 2)))
            mae = float(np.mean(np.abs(erros)))
            mape = float(np.mean(np.abs(erros / (precos_reais + 1e-9)))) * 100.0

            # R²
            var_total = np.sum((precos_reais - np.mean(precos_reais)) ** 2)
            var_resid = np.sum(erros ** 2)
            r2 = float(1.0 - (var_resid / (var_total + 1e-9))) if var_total > 1e-9 else 0.0

            # Acurácia Direcional
            if len(precos_reais) >= 2:
                dir_real = np.sign(np.diff(precos_reais))
                dir_prev = np.sign(np.diff(precos_previstos))
                acuracia_dir = float(np.mean(dir_real == dir_prev) * 100.0)
            else:
                acuracia_dir = 50.0
        else:
            rmse = 0.0
            mae = 0.0
            mape = 0.0
            r2 = 0.0
            acuracia_dir = 50.0

        return MetricasBacktesting(
            capital_inicial=cap_ini,
            capital_final=cap_fim,
            lucro_total=lucro,
            retorno_total_pct=retorno_total,
            retorno_anualizado_pct=retorno_anualizado,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
            profit_factor=min(10.0, profit_factor),
            calmar_ratio=min(10.0, calmar),
            total_operacoes=total_ops,
            operacoes_vencedoras=vencedoras,
            operacoes_perdedoras=perdedoras,
            rmse=rmse,
            mae=mae,
            mape_pct=min(500.0, mape),
            r2=max(-1.0, min(1.0, r2)),
            acuracia_direcional_pct=acuracia_dir
        )

    @classmethod
    def _metricas_vazias(cls) -> MetricasBacktesting:
        return MetricasBacktesting(
            capital_inicial=10000.0,
            capital_final=10000.0,
            lucro_total=0.0,
            retorno_total_pct=0.0,
            retorno_anualizado_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            profit_factor=1.0,
            calmar_ratio=0.0,
            total_operacoes=0,
            operacoes_vencedoras=0,
            operacoes_perdedoras=0,
            rmse=0.0,
            mae=0.0,
            mape_pct=0.0,
            r2=0.0,
            acuracia_direcional_pct=50.0
        )
