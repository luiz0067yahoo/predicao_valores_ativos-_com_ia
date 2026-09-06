"""
Módulo de Otimização Automática de Hiperparâmetros (mapp/simulator/optimizer.py)
Aplica busca em grade (Grid Search) e busca aleatória (Random Search)
avaliando sempre fora da amostra (Out-of-Sample) para evitar overfitting.
Função objetivo balanceada: Retorno + Sharpe - Drawdown - Erro.
Todos os nomes em português.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import itertools
import numpy as np
import pandas as pd

from mapp.backtesting.engine import BacktestEngine


class HyperparameterOptimizer:
    """
    Otimizador multiobjetivo de parâmetros de modelos preditivos.
    """

    @classmethod
    def otimizar_algoritmo(
        cls,
        algoritmo_id: str,
        df: pd.DataFrame,
        grade_parametros: Dict[str, List[Any]],
        horizonte_passos: int = 5,
        max_iteracoes: int = 12,
        callback_progresso: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executa busca otimizada de hiperparâmetros com validação fora da amostra.
        """
        # Divide em treino (70%) e validação out-of-sample (30%)
        n = len(df)
        corte = int(n * 0.70)
        df_treino = df.iloc[:corte]
        df_validacao = df.iloc[corte:]

        # Gera combinações de parâmetros
        chaves = list(grade_parametros.keys())
        valores = list(grade_parametros.values())
        todas_combinacoes = [dict(zip(chaves, combo)) for combo in itertools.product(*valores)]

        # Se houver muitas combinações, amostra aleatoriamente até max_iteracoes
        if len(todas_combinacoes) > max_iteracoes:
            np.random.seed(42)
            indices_amostrados = np.random.choice(len(todas_combinacoes), size=max_iteracoes, replace=False)
            combinacoes_teste = [todas_combinacoes[i] for i in indices_amostrados]
        else:
            combinacoes_teste = todas_combinacoes

        melhor_score = -float("inf")
        melhores_params: Dict[str, Any] = {}
        historico_tentativas: List[Dict[str, Any]] = []

        total = len(combinacoes_teste)
        engine = BacktestEngine(capital_inicial=10000.0)

        for idx, params in enumerate(combinacoes_teste):
            if callback_progresso:
                pct = (idx / max(1, total)) * 100.0
                callback_progresso(pct, f"Otimizando {algoritmo_id.upper()} (tentativa {idx + 1}/{total})...")

            from algoritmos.fabrica_algoritmos import FabricaAlgoritmos
            instancia = FabricaAlgoritmos.obter_instancia(algoritmo_id, janela_temporal=10)

            def pred_func(sub_df: pd.DataFrame, h: int) -> float:
                r = instancia.treinar_e_projetar(
                    dados_completos=sub_df,
                    horizonte_projecao=h,
                    hiperparametros=params
                )
                return float(r["forecast_df"]["Preco_Projetado"].iloc[-1])

            try:
                res_oos = engine.executar_simulacao_walk_forward(
                    df=df_validacao,
                    funcao_predicao=pred_func,
                    horizonte_dias=horizonte_passos,
                    janela_treino_minima=max(15, len(df_validacao) // 3),
                    passo_rebalanceamento=horizonte_passos
                )
                m = res_oos["metrics"]
                # Função objetivo multiobjetivo rigorosa:
                score = (m["total_return_pct"] * 1.0) + (m["sharpe_ratio"] * 8.0) - (m["max_drawdown_pct"] * 0.6) - (m["rmse"] * 50.0)
            except Exception:
                score = -100.0
                m = {}

            historico_tentativas.append({
                "params": params,
                "score": round(score, 2),
                "metrics": m
            })

            if score > melhor_score:
                melhor_score = score
                melhores_params = params

        return {
            "algorithm": algoritmo_id,
            "best_params": melhores_params,
            "best_score": round(melhor_score, 2),
            "trials": historico_tentativas
        }
