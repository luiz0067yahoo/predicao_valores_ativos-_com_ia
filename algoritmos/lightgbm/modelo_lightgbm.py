"""
Modelo LightGBM para Previsão de Séries Temporais Financeiras
Utiliza Light Gradient Boosting Machine com crescimento em folha (leaf-wise),
proporcionando alto desempenho computacional e excelente precisão preditiva.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class ModeloLightGBM(BaseAlgoritmo):
    """
    Implementação do estimador LightGBM para séries temporais.
    Destaques:
    - Otimização baseada em histogramas para máxima velocidade.
    - Eficiência superior em memória com excelente capacidade de generalização.
    - Projeção recursiva autorregressiva com intervalo de confiança empírico (95%).
    """

    def __init__(self, janela_temporal: int = 10, numero_estimadores: int = 100, taxa_aprendizado: float = 0.05):
        """
        Inicializa o modelo LightGBM.

        Parâmetros:
            janela_temporal: Quantidade de dias passados observados.
            numero_estimadores: Quantidade de árvores boosting.
            taxa_aprendizado: Taxa de aprendizado da descida do gradiente.
        """
        super().__init__(nome_identificador="LightGBM", janela_temporal=janela_temporal)
        self.numero_estimadores = numero_estimadores
        self.taxa_aprendizado = taxa_aprendizado
        self.modelo: Optional[LGBMRegressor] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o pipeline de ajuste e projeção do LightGBM.

        Parâmetros:
            dados_completos: DataFrame com histórico de cotações.
            horizonte_projecao: Dias futuros para projeção.
            eh_criptomoeda: Identificador de mercado de criptoativos.
            funcao_progresso: Callback de monitoramento em tempo real.
            hiperparametros: Dicionário com ajustes opcionais.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        n_estimadores = int(config.get("numero_estimadores", self.numero_estimadores))
        taxa_lr = float(config.get("taxa_aprendizado", self.taxa_aprendizado))
        numero_folhas = int(config.get("numero_folhas", 31))

        if funcao_progresso:
            funcao_progresso(10, 100, "LightGBM: Processando matriz de defasagens e indicadores...", 0.0)

        # 1. Obtenção de matriz de defasagens (lags)
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        if funcao_progresso:
            funcao_progresso(30, 100, f"LightGBM: Ajustando {n_estimadores} árvores com algoritmo histograma...", 0.0)

        # 2. Configuração do modelo LightGBM
        self.modelo = LGBMRegressor(
            n_estimators=n_estimadores,
            learning_rate=taxa_lr,
            num_leaves=numero_folhas,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

        # Treina com os dados normalizados
        self.modelo.fit(matriz_atributos, vetor_alvo)

        if funcao_progresso:
            funcao_progresso(70, 100, "LightGBM: Calculando projeção futura recursiva e incerteza...", 0.0)

        # 3. Predição histórica (in-sample)
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

        # 7. Estruturação dos DataFrames
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

        importancias = [float(v) for v in self.modelo.feature_importances_[:15]]
        self.historico_aprendizado = {
            "generations": list(range(1, len(importancias) + 1)),
            "best_fitness": [round(v, 2) for v in importancias],
            "avg_fitness": [round(np.mean(importancias), 2)] * len(importancias)
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"LightGBM: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
