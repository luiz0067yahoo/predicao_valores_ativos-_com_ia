"""
Modelo Random Forest para Previsão de Séries Temporais Financeiras
Utiliza um conjunto (ensemble) de árvores de decisão com amostragem bootstrap
para capturar dinâmicas não-lineares com alta robustez contra ruídos de mercado.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class ModeloRandomForest(BaseAlgoritmo):
    """
    Implementação de Random Forest Regressor adaptada para projeção de séries temporais.
    Destaques:
    - Agregação por média de múltiplas árvores não correlacionadas (bagging).
    - Alta resiliência a outliers e ruído estocástico de cotações.
    - Projeção recursiva multi-passo com banda de confiança de 95%.
    """

    def __init__(self, janela_temporal: int = 10, numero_arvores: int = 150):
        """
        Inicializa o modelo Random Forest.

        Parâmetros:
            janela_temporal: Quantidade de defasagens passadas observadas.
            numero_arvores: Número total de árvores no conjunto (n_estimators).
        """
        super().__init__(nome_identificador="Random Forest", janela_temporal=janela_temporal)
        self.numero_arvores = numero_arvores
        self.modelo: Optional[RandomForestRegressor] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa treinamento e inferência recursiva com Random Forest.

        Parâmetros:
            dados_completos: DataFrame histórico.
            horizonte_projecao: Horizonte futuro em dias.
            eh_criptomoeda: Indica se mercado opera 24/7.
            funcao_progresso: Callback de progresso.
            hiperparametros: Hiperparâmetros customizados.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        n_arvores = int(config.get("numero_arvores", self.numero_arvores))
        profundidade = config.get("profundidade_maxima", None)
        if profundidade is not None:
            profundidade = int(profundidade)

        if funcao_progresso:
            funcao_progresso(10, 100, "Random Forest: Calculando atributos e indicadores técnicos...", 0.0)

        # 1. Preparação de matriz de atributos defasados
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        if funcao_progresso:
            funcao_progresso(30, 100, f"Random Forest: Treinando ensemble com {n_arvores} árvores...", 0.0)

        # 2. Instanciação e treino
        self.modelo = RandomForestRegressor(
            n_estimators=n_arvores,
            max_depth=profundidade,
            random_state=42,
            n_jobs=-1
        )
        self.modelo.fit(matriz_atributos, vetor_alvo)

        if funcao_progresso:
            funcao_progresso(70, 100, "Random Forest: Gerando projeção recursiva e intervalos estatísticos...", 0.0)

        # 3. Predição in-sample
        predicoes_norm = self.modelo.predict(matriz_atributos)
        precos_reais, precos_previstos_reais = self.desnormalizar_historico(
            dados_completos, len(matriz_atributos), predicoes_norm
        )

        residuos = precos_reais - precos_previstos_reais
        desvio_padrao_erros = float(np.std(residuos))

        # 4. Projeção recursiva autorregressiva
        datas_futuras, previsoes_futuras_reais, limites_inf, limites_sup = self.projetar_futuro_recursivo(
            dados_completos=dados_completos,
            funcao_predicao_vetor=lambda v: float(self.modelo.predict(v)[0]),
            horizonte_projecao=horizonte_projecao,
            desvio_padrao_erros=desvio_padrao_erros,
            eh_criptomoeda=eh_criptomoeda
        )

        # 7. DataFrames de saída
        df_historico_saida = pd.DataFrame(
            {"Preco_Real": precos_reais, "Preco_Previsto_IA": precos_previstos_reais},
            index=datas_alvo
        )

        df_projecao_saida = pd.DataFrame(
            {
                "Preco_Projetado": previsoes_futuras_reais,
                "Limite_Inferior": limites_inf,
                "Limite_Superior": limites_sup
            },
            index=datas_futuras
        )

        # 8. Métricas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=precos_reais,
            precos_previstos=precos_previstos_reais,
            preco_real_final=float(precos_reais[-1]),
            preco_projetado_final=float(previsoes_futuras_reais[-1]),
            horizonte_dias=horizonte_projecao
        )

        importancias = [float(v) for v in self.modelo.feature_importances_[:15]]
        self.historico_aprendizado = {
            "generations": list(range(1, len(importancias) + 1)),
            "best_fitness": [round(v * 1000, 2) for v in importancias],
            "avg_fitness": [round(np.mean(importancias) * 1000, 2)] * len(importancias)
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"Random Forest: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
