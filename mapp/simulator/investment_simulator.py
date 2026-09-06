"""
Módulo do Simulador de Investimento Multi-Algoritmo (mapp/simulator/investment_simulator.py)
Executa simulações comparativas entre múltiplos algoritmos de IA sob exatamente
as mesmas condições de mercado (ativo, período, capital inicial, aporte periódico,
taxas e slippage).
Calcula tempo decorrido real, restante e total estimado via ProgressTracker.
Ranqueia os algoritmos por múltiplos critérios de risco e estabilidade.
Todos os nomes em português.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from mapp.backtesting.engine import BacktestEngine
from mapp.horizon import ForecastHorizon, NormalizadorHorizonte
from mapp.tracker import ProgressTracker


class InvestmentSimulator:
    """
    Simulador de carteira de investimentos com comparação simultânea de múltiplos modelos.
    """

    def __init__(
        self,
        capital_inicial: float = 10000.0,
        aporte_periodico: float = 0.0,
        moeda: str = "BRL",
        taxa_corretagem_pct: float = 0.05,
        slippage_pct: float = 0.02
    ):
        self.capital_inicial = capital_inicial
        self.aporte_periodico = aporte_periodico
        self.moeda = moeda
        self.taxa_corretagem_pct = taxa_corretagem_pct
        self.slippage_pct = slippage_pct

    def simular_multiplos_algoritmos(
        self,
        df: pd.DataFrame,
        algoritmos_selecionados: List[str],
        horizonte: ForecastHorizon,
        parametros_por_algoritmo: Optional[Dict[str, Dict[str, Any]]] = None,
        callback_progresso: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executa a simulação para cada algoritmo sob condições estritamente idênticas.
        """
        if not algoritmos_selecionados:
            algoritmos_selecionados = ["mapp", "xgboost", "random_forest"]

        params_map = parametros_por_algoritmo or {}
        total_modelos = len(algoritmos_selecionados)

        tracker = ProgressTracker(total_passos=total_modelos * 10, nome_tarefa="Simulação de Carteira")
        tracker.iniciar()

        engine = BacktestEngine(
            capital_inicial=self.capital_inicial,
            taxa_corretagem_pct=self.taxa_corretagem_pct,
            slippage_pct=self.slippage_pct
        )

        resultados_modelos: Dict[str, Dict[str, Any]] = {}
        passos_concluidos = 0

        for idx, algo_id in enumerate(algoritmos_selecionados):
            # Notifica início do modelo atual
            estado = tracker.atualizar(
                passo_atual=passos_concluidos,
                mensagem=f"Iniciando simulação com {algo_id.upper()} ({idx + 1}/{total_modelos})...",
                algoritmo_atual=algo_id
            )
            if callback_progresso:
                callback_progresso(estado.para_dicionario())

            # Instancia o modelo dinamicamente via fábrica
            from algoritmos.fabrica_algoritmos import FabricaAlgoritmos
            instancia = FabricaAlgoritmos.obter_instancia(algo_id, janela_temporal=10)
            p_algo = params_map.get(algo_id, {})

            # Define função preditiva para o backtest walk-forward
            def funcao_pred_wrapper(sub_df: pd.DataFrame, h_passos: int) -> float:
                res = instancia.treinar_e_projetar(
                    dados_completos=sub_df,
                    horizonte_projecao=h_passos,
                    eh_criptomoeda=horizonte.eh_criptomoeda,
                    funcao_progresso=None,
                    hiperparametros=p_algo
                )
                return float(res["forecast_df"]["Preco_Projetado"].iloc[-1])

            # Executa simulação
            res_backtest = engine.executar_simulacao_walk_forward(
                df=df,
                funcao_predicao=funcao_pred_wrapper,
                horizonte_dias=horizonte.periodos_normalizados,
                janela_treino_minima=max(35, len(df) // 4),
                passo_rebalanceamento=max(3, horizonte.periodos_normalizados)
            )

            passos_concluidos += 10
            estado = tracker.atualizar(
                passo_atual=passos_concluidos,
                mensagem=f"Concluída simulação do modelo {algo_id.upper()}.",
                algoritmo_atual=algo_id
            )
            if callback_progresso:
                callback_progresso(estado.para_dicionario())

            resultados_modelos[algo_id] = res_backtest

        # Consolidação do Relatório Comparativo e Ranqueamento Multicritério
        tabela_comparativa = []
        curvas_capital_combinadas: Dict[str, List[Dict[str, Any]]] = {}

        for algo_id, res in resultados_modelos.items():
            m = res["metrics"]
            # Score de Equilíbrio: Retorno + (Sharpe * 10) - (Drawdown * 0.5)
            score_geral = m["total_return_pct"] + (m["sharpe_ratio"] * 10.0) - (m["max_drawdown_pct"] * 0.5)

            tabela_comparativa.append({
                "algoritmo": algo_id,
                "nome_exibicao": algo_id.replace("_", " ").title(),
                "capital_inicial": m["initial_capital"],
                "capital_final": m["final_capital"],
                "lucro_total": m["total_profit"],
                "retorno_pct": m["total_return_pct"],
                "sharpe": m["sharpe_ratio"],
                "sortino": m["sortino_ratio"],
                "max_drawdown": m["max_drawdown_pct"],
                "win_rate": m["win_rate_pct"],
                "profit_factor": m["profit_factor"],
                "calmar": m["calmar_ratio"],
                "rmse": m["rmse"],
                "score_multicriterio": round(score_geral, 2)
            })

            curvas_capital_combinadas[algo_id] = res["equity_curve"]

        # Ordena pelo score multicritério balanceado (não apenas por lucro bruto)
        tabela_comparativa.sort(key=lambda x: x["score_multicriterio"], reverse=True)

        # Identifica destaques
        melhor_retorno = max(tabela_comparativa, key=lambda x: x["retorno_pct"])
        melhor_sharpe = max(tabela_comparativa, key=lambda x: x["sharpe"])
        menor_drawdown = min(tabela_comparativa, key=lambda x: x["max_drawdown"])
        melhor_geral = tabela_comparativa[0]

        relatorio_destaques = {
            "best_overall": melhor_geral["nome_exibicao"],
            "best_return": f"{melhor_retorno['nome_exibicao']} (+{melhor_retorno['retorno_pct']}%)",
            "best_sharpe": f"{melhor_sharpe['nome_exibicao']} (Sharpe {melhor_sharpe['sharpe']})",
            "lowest_drawdown": f"{menor_drawdown['nome_exibicao']} ({menor_drawdown['max_drawdown']}%)",
            "total_execution_time": tracker.obter_estado().tempo_decorrido_str
        }

        return {
            "success": True,
            "horizon": {
                "value": horizonte.valor,
                "unit": horizonte.unidade,
                "periods": horizonte.periodos_normalizados,
                "target_date": str(horizonte.data_alvo.date())
            },
            "comparison_table": tabela_comparativa,
            "equity_curves": curvas_capital_combinadas,
            "highlights": relatorio_destaques,
            "detailed_results": resultados_modelos,
            "tracker_final": tracker.obter_estado().para_dicionario()
        }
