"""
Modelo de Algoritmo Genético Evolutivo para Previsão de Séries Temporais Financeiras
Utiliza cromossomos em ponto flutuante, seleção por torneio estocástico, crossover BLX-alfa,
mutação gaussiana adaptativa e elitismo para otimizar pesos autorregressivos e indicadores técnicos.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher
from genetic_engine import GeneticEngine
from model_predictor import ModelPredictor


class ModeloAlgoritmoGenetico(BaseAlgoritmo):
    """
    Controlador do Algoritmo Genético evolutivo para modelagem de séries temporais.
    Destaques:
    - Otimização global estocástica para evasão de mínimos locais.
    - Função de aptidão multi-critério penalizando RMSE, MAE e bonificando acurácia direcional.
    - Projeção recursiva autorregressiva com cone de 95% de confiança.
    """

    def __init__(
        self,
        janela_temporal: int = 10,
        tamanho_populacao: int = 60,
        numero_geracoes: int = 35,
        taxa_mutacao: float = 0.15,
        taxa_crossover: float = 0.85
    ):
        """
        Inicializa o algoritmo genético.

        Parâmetros:
            janela_temporal: Defasagens passadas avaliadas pelos genes do cromossomo.
            tamanho_populacao: Quantidade de indivíduos na população.
            numero_geracoes: Número de ciclos evolutivos de seleção e recombinação.
            taxa_mutacao: Probabilidade de perturbação gaussiana dos genes.
            taxa_crossover: Probabilidade de recombinação entre progenitores.
        """
        super().__init__(nome_identificador="Algoritmo Genético", janela_temporal=janela_temporal)
        self.tamanho_populacao = tamanho_populacao
        self.numero_geracoes = numero_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa a evolução dos cromossomos e gera as projeções futuras.

        Parâmetros:
            dados_completos: DataFrame histórico com cotações.
            horizonte_projecao: Dias futuros a projetar.
            eh_criptomoeda: Flag para calendário 24/7.
            funcao_progresso: Callback de atualização por geração.
            hiperparametros: Dicionário opcional com sobrescrita de parâmetros.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        populacao = int(config.get("tamanho_populacao", self.tamanho_populacao))
        geracoes = int(config.get("numero_geracoes", self.numero_geracoes))
        mutacao = float(config.get("taxa_mutacao", self.taxa_mutacao))
        crossover = float(config.get("taxa_crossover", self.taxa_crossover))

        if funcao_progresso:
            funcao_progresso(5, 100, "Algoritmo Genético: Calculando defasagens e osciladores...", 0.0)

        # 1. Obtenção de matriz de defasagens
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        # 2. Inicialização do motor genético
        motor_ga = GeneticEngine(
            population_size=populacao,
            generations=geracoes,
            crossover_rate=crossover,
            mutation_rate=mutacao,
            elitism_ratio=0.08
        )

        def callback_evolutivo(gen_atual: int, max_gen: int, melhor_fit: float, medio_fit: float, melhor_rmse: float):
            if funcao_progresso:
                pct = 10 + int((gen_atual / max_gen) * 75)
                msg = f"GA: Geração {gen_atual}/{max_gen} | Fitness: {melhor_fit:.1f} | RMSE: {melhor_rmse:.4f}"
                funcao_progresso(pct, 100, msg, melhor_rmse)

        # 3. Evolução dos cromossomos
        melhor_cromossomo = motor_ga.evolve(matriz_atributos, vetor_alvo, callback=callback_evolutivo)
        self.historico_aprendizado = motor_ga.history

        if funcao_progresso:
            funcao_progresso(88, 100, "Algoritmo Genético: Calculando projeção recursiva e intervalos...", 0.0)

        # 4. Avaliação e projeção
        preditor = ModelPredictor(best_individual=melhor_cromossomo, lookback_window=self.janela_temporal)
        resultados = preditor.evaluate_and_predict(
            df_raw=dados_completos,
            X=matriz_atributos,
            dates=datas_alvo,
            forecast_horizon=horizonte_projecao,
            is_crypto=eh_criptomoeda
        )

        if funcao_progresso:
            funcao_progresso(100, 100, f"GA: Concluído! Tendência: {resultados['metrics']['tendencia_esperada']}", resultados["metrics"]["rmse"])

        return {
            "metrics": resultados["metrics"],
            "history_df": resultados["history_df"],
            "forecast_df": resultados["forecast_df"],
            "ga_history": {
                "generations": self.historico_aprendizado["generation"],
                "best_fitness": self.historico_aprendizado["best_fitness"],
                "avg_fitness": self.historico_aprendizado["avg_fitness"]
            }
        }
