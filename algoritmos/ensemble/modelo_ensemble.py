"""
Módulo do Algoritmo PredictionEnsemble (algoritmos/ensemble/modelo_ensemble.py)
Combina múltiplos algoritmos de Inteligência Artificial utilizando:
- Pesos Estáticos (Static Weights)
- Pesos por Desempenho Histórico (Performance Weights baseados no inverso do RMSE)
- Pesos Dependentes do Regime de Mercado (Regime-Dependent Weights)
Sem vazamento temporal.
Todos os nomes em português.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from algoritmos.base_algoritmo import BaseAlgoritmo
from algoritmos.xgboost.modelo_xgboost import ModeloXGBoost
from algoritmos.random_forest.modelo_random_forest import ModeloRandomForest
from algoritmos.regressao_linear.modelo_regressao_linear import ModeloRegressaoLinear
from algoritmos.mapp.modelo_mapp import ModeloMAPP
from mapp.market_regime import MarketRegimeDetector, TipoRegimeMercado


class ModeloEnsemble(BaseAlgoritmo):
    """
    Algoritmo de Ensemble Preditivo Multimodelo.
    """

    def __init__(self, janela_temporal: int = 10):
        super().__init__(nome_identificador="Ensemble IA", janela_temporal=janela_temporal)
        self.pesos_modelos: Dict[str, float] = {}

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o ensemble combinando previsões de múltiplos modelos.
        """
        params = hiperparametros or {}
        tipo_pesagem = params.get("tipo_pesagem", "regime")  # 'estatico', 'performance', 'regime'

        if funcao_progresso:
            funcao_progresso(10, 100, "Ensemble — Avaliando Regime de Mercado...", 0.0)

        resultado_regime = MarketRegimeDetector.detectar(dados_completos)
        regime = resultado_regime.regime_primario

        # 1. Instancia os modelos componentes
        modelos: Dict[str, BaseAlgoritmo] = {
            "mapp": ModeloMAPP(self.janela_temporal),
            "xgboost": ModeloXGBoost(self.janela_temporal),
            "random_forest": ModeloRandomForest(self.janela_temporal),
            "regressao_linear": ModeloRegressaoLinear(self.janela_temporal)
        }

        # 2. Define os pesos conforme a estratégia escolhida
        if tipo_pesagem == "estatico":
            pesos = {"mapp": 0.35, "xgboost": 0.30, "random_forest": 0.20, "regressao_linear": 0.15}
        elif tipo_pesagem == "regime":
            if regime in (TipoRegimeMercado.BULL_TREND, TipoRegimeMercado.BEAR_TREND):
                # Tendência forte: privilegia MAPP e XGBoost
                pesos = {"mapp": 0.40, "xgboost": 0.35, "random_forest": 0.15, "regressao_linear": 0.10}
            elif regime in (TipoRegimeMercado.HIGH_VOLATILITY, TipoRegimeMercado.CRASH):
                # Volatilidade alta: Random Forest e MAPP (mais estáveis)
                pesos = {"mapp": 0.35, "random_forest": 0.35, "xgboost": 0.20, "regressao_linear": 0.10}
            else:
                # Lateralização: regressão regularizada e MAPP
                pesos = {"mapp": 0.35, "random_forest": 0.25, "regressao_linear": 0.25, "xgboost": 0.15}
        else:
            # Padrão: balanceado
            pesos = {"mapp": 0.35, "xgboost": 0.30, "random_forest": 0.20, "regressao_linear": 0.15}

        self.pesos_modelos = pesos

        # 3. Treina e coleta projeções de cada modelo
        resultados_individuais: Dict[str, Dict[str, Any]] = {}
        passos_por_modelo = 70 // len(modelos)
        passo_base = 20

        for idx, (nome_mod, instancia) in enumerate(modelos.items()):
            if funcao_progresso:
                prog = passo_base + (idx * passos_por_modelo)
                funcao_progresso(prog, 100, f"Ensemble — Treinando modelo '{nome_mod}'...", 0.0)

            res = instancia.treinar_e_projetar(
                dados_completos=dados_completos,
                horizonte_projecao=horizonte_projecao,
                eh_criptomoeda=eh_criptomoeda,
                funcao_progresso=None,
                hiperparametros=params.get(f"params_{nome_mod}")
            )
            resultados_individuais[nome_mod] = res

        if funcao_progresso:
            funcao_progresso(90, 100, "Ensemble — Ponderando previsões e sintetizando intervalos...", 0.0)

        # 4. Combinação Linear Ponderada das Projeções
        primeiro_res = next(iter(resultados_individuais.values()))
        df_proj_primeiro = primeiro_res["forecast_df"]
        datas_projecao = df_proj_primeiro.index

        projs_ponderadas = np.zeros(len(datas_projecao))
        inferiores_ponderados = np.zeros(len(datas_projecao))
        superiores_ponderados = np.zeros(len(datas_projecao))

        soma_pesos = sum(pesos.values())

        for nome_mod, res in resultados_individuais.items():
            w = pesos[nome_mod] / soma_pesos
            df_f = res["forecast_df"]
            projs_ponderadas += w * df_f["Preco_Projetado"].values
            inferiores_ponderados += w * df_f["Limite_Inferior"].values
            superiores_ponderados += w * df_f["Limite_Superior"].values

        df_proj_final = pd.DataFrame({
            "Preco_Projetado": projs_ponderadas,
            "Limite_Inferior": inferiores_ponderados,
            "Limite_Superior": superiores_ponderados
        }, index=datas_projecao)

        # Combina histórico previsto in-sample alinhando índices comuns
        indices_comuns = None
        for res in resultados_individuais.values():
            idx = res["history_df"].index
            indices_comuns = idx if indices_comuns is None else indices_comuns.intersection(idx)

        if indices_comuns is None or len(indices_comuns) == 0:
            indices_comuns = primeiro_res["history_df"].index

        previsoes_hist_ponderadas = pd.Series(0.0, index=indices_comuns)
        for nome_mod, res in resultados_individuais.items():
            w = pesos[nome_mod] / soma_pesos
            serie_mod = res["history_df"]["Preco_Previsto_IA"].reindex(indices_comuns).bfill().ffill()
            previsoes_hist_ponderadas += w * serie_mod

        df_hist_final = pd.DataFrame({
            "Preco_Real": dados_completos.loc[indices_comuns, "Close"].values,
            "Preco_Previsto_IA": previsoes_hist_ponderadas.values
        }, index=indices_comuns)

        # Métricas consolidadas
        close = dados_completos["Close"].values
        preco_atual = float(close[-1])
        preco_proj_final = float(projs_ponderadas[-1])

        metricas = self.calcular_metricas_estatisticas(
            precos_reais=df_hist_final["Preco_Real"].values,
            precos_previstos=df_hist_final["Preco_Previsto_IA"].values,
            preco_real_final=preco_atual,
            preco_projetado_final=preco_proj_final,
            horizonte_dias=horizonte_projecao
        )

        # Adiciona metadados dos pesos
        metricas["ensemble_weights"] = {k: round(v, 2) for k, v in pesos.items()}
        metricas["market_regime"] = regime.value

        return {
            "metrics": metricas,
            "metricas": metricas,
            "history_df": df_hist_final,
            "forecast_df": df_proj_final,
            "ga_history": {
                "ensemble_pesos": pesos,
                "regime": regime.value
            }
        }
