"""
Modelo de Regressão Linear para Previsão de Séries Temporais Financeiras
Atua como baseline paramétrico rápido com regularização Ridge (L2) para
evitar problemas de multicolinearidade entre defasagens temporais contíguas.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class ModeloRegressaoLinear(BaseAlgoritmo):
    """
    Implementação de Regressão Linear com penalização L2 (Ridge).
    Destaques:
    - Execução instantânea (solução em forma fechada).
    - Modelo de referência baseline indispensável para avaliar ganho de IA complexa.
    - Coeficientes lineares interpretáveis diretamente.
    """

    def __init__(self, janela_temporal: int = 10, forca_regularizacao: float = 1.0):
        """
        Inicializa o modelo de regressão linear.

        Parâmetros:
            janela_temporal: Quantidade de lags temporais passados.
            forca_regularizacao: Parâmetro alfa de regularização Ridge (L2).
        """
        super().__init__(nome_identificador="Regressão Linear", janela_temporal=janela_temporal)
        self.forca_regularizacao = forca_regularizacao
        self.modelo: Optional[Ridge] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ajusta a reta/hiperplano multidimensional aos dados e calcula a projeção.

        Parâmetros:
            dados_completos: DataFrame com histórico de cotações.
            horizonte_projecao: Horizonte em dias futuros.
            eh_criptomoeda: Flag de operação de final de semana.
            funcao_progresso: Callback de progresso.
            hiperparametros: Configurações adicionais.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        alfa_ridge = float(config.get("forca_regularizacao", self.forca_regularizacao))

        if funcao_progresso:
            funcao_progresso(15, 100, "Regressão Linear: Estruturando variáveis explicativas...", 0.0)

        # 1. Preparação dos dados
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        if funcao_progresso:
            funcao_progresso(40, 100, f"Regressão Linear: Ajustando coeficientes analíticos (alfa={alfa_ridge})...", 0.0)

        # 2. Treinamento Ridge
        self.modelo = Ridge(alpha=alfa_ridge, random_state=42)
        self.modelo.fit(matriz_atributos, vetor_alvo)

        if funcao_progresso:
            funcao_progresso(75, 100, "Regressão Linear: Computando resíduos e bandas de confiança...", 0.0)

        # 3. Predição histórica in-sample
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

        # 8. Métricas estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=precos_reais,
            precos_previstos=precos_previstos_reais,
            preco_real_final=float(precos_reais[-1]),
            preco_projetado_final=float(previsoes_futuras_reais[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Coeficientes absolutos para gráfico
        coefs = [float(abs(c)) for c in self.modelo.coef_[:15]]
        self.historico_aprendizado = {
            "generations": list(range(1, len(coefs) + 1)),
            "best_fitness": [round(v * 100, 2) for v in coefs],
            "avg_fitness": [round(np.mean(coefs) * 100, 2)] * len(coefs)
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"Regressão Linear: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
