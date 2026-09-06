"""
Modelo Prophet para Previsão de Séries Temporais Financeiras
Baseado no modelo aditivo de séries temporais desenvolvido pelo time de Data Science do Facebook/Meta,
decompõe a cotação em componentes não-lineares de tendência, sazonalidade e efeitos de feriados.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from prophet import Prophet

from algoritmos.base_algoritmo import BaseAlgoritmo


class ModeloProphet(BaseAlgoritmo):
    """
    Implementação do estimador Prophet para decomposição e previsão de ativos.
    Destaques:
    - Curva de tendência por regressão segmentada com pontos de quebra (changepoints).
    - Modelagem de sazonalidade periódica via séries de Fourier.
    - Geração nativa de intervalos de incerteza posterior via amostragem de Monte Carlo.
    """

    def __init__(self, flexibilidade_tendencia: float = 0.05, flexibilidade_sazonal: float = 10.0, janela_temporal: int = 10):
        """
        Inicializa o modelo Prophet.

        Parâmetros:
            flexibilidade_tendencia: changepoint_prior_scale (ajusta a rigidez da curva de tendência).
            flexibilidade_sazonal: seasonality_prior_scale (ajusta a amplitude da sazonalidade).
            janela_temporal: Quantidade de dias passados para observação.
        """
        super().__init__(nome_identificador="Prophet", janela_temporal=janela_temporal)
        self.flexibilidade_tendencia = flexibilidade_tendencia
        self.flexibilidade_sazonal = flexibilidade_sazonal
        self.modelo: Optional[Prophet] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ajusta os componentes de tendência e sazonalidade e projeta o horizonte futuro.

        Parâmetros:
            dados_completos: DataFrame com série histórica contendo 'Close'.
            horizonte_projecao: Dias futuros para estimativa.
            eh_criptomoeda: Se True, considera finais de semana.
            funcao_progresso: Callback de progresso.
            hiperparametros: Configurações de sensibilidade da curva.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        escala_changepoint = float(config.get("flexibilidade_tendencia", self.flexibilidade_tendencia))
        escala_sazonal = float(config.get("flexibilidade_sazonal", self.flexibilidade_sazonal))

        if funcao_progresso:
            funcao_progresso(15, 100, "Prophet: Formatando série cronológica [ds, y]...", 0.0)

        # 1. Estruturação no formato obrigatório do Prophet: colunas 'ds' e 'y'
        serie_fechamento = dados_completos["Close"].dropna().astype(float)
        df_prophet = pd.DataFrame({
            "ds": pd.to_datetime(serie_fechamento.index).tz_localize(None),
            "y": serie_fechamento.values
        })

        if len(df_prophet) < 30:
            raise ValueError("Série histórica com registros insuficientes para estimativa do Prophet.")

        if funcao_progresso:
            funcao_progresso(35, 100, f"Prophet: Otimizando changepoints (escala={escala_changepoint})...", 0.0)

        # 2. Inicialização e treinamento do Prophet
        self.modelo = Prophet(
            changepoint_prior_scale=escala_changepoint,
            seasonality_prior_scale=escala_sazonal,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=len(df_prophet) > 180,
            interval_width=0.95
        )

        # Suprime logs verbosos do Stan/Prophet
        self.modelo.fit(df_prophet)

        if funcao_progresso:
            funcao_progresso(70, 100, "Prophet: Amostrando incerteza posterior de Monte Carlo...", 0.0)

        # 3. Geração de datas futuras
        ultima_data_real = pd.to_datetime(df_prophet["ds"].iloc[-1])
        datas_futuras = self.gerar_datas_futuras(
            ultima_data_real, horizonte_projecao, eh_criptomoeda=eh_criptomoeda
        )

        # Cria DataFrame unificado para previsão (histórico + futuro)
        datas_unificadas = list(df_prophet["ds"]) + [d.tz_localize(None) for d in datas_futuras]
        df_futuro_prophet = pd.DataFrame({"ds": datas_unificadas})

        # 4. Inferência e decomposição
        previsoes_prophet = self.modelo.predict(df_futuro_prophet)

        # Separa a porção histórica da porção futura
        tamanho_historico = len(df_prophet)
        precos_reais = df_prophet["y"].values
        precos_previstos_in_sample = previsoes_prophet["yhat"].iloc[:tamanho_historico].values

        precos_projetados = list(previsoes_prophet["yhat"].iloc[tamanho_historico:].values)
        limites_inferiores = list(previsoes_prophet["yhat_lower"].iloc[tamanho_historico:].values)
        limites_superiores = list(previsoes_prophet["yhat_upper"].iloc[tamanho_historico:].values)

        # 5. DataFrames de saída
        datas_alvo_originais = list(serie_fechamento.index)
        df_historico_saida = pd.DataFrame(
            {
                "Preco_Real": precos_reais,
                "Preco_Previsto_IA": precos_previstos_in_sample
            },
            index=datas_alvo_originais
        )

        df_projecao_saida = pd.DataFrame(
            {
                "Preco_Projetado": precos_projetados,
                "Limite_Inferior": [max(0.0, float(v)) for v in limites_inferiores],
                "Limite_Superior": [float(v) for v in limites_superiores]
            },
            index=datas_futuras
        )

        # 6. Métricas estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=precos_reais,
            precos_previstos=precos_previstos_in_sample,
            preco_real_final=float(precos_reais[-1]),
            preco_projetado_final=float(precos_projetados[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Extrai resíduos absolutos para o gráfico
        residuos = [float(abs(r)) for r in (precos_reais - precos_previstos_in_sample)[-15:]]
        self.historico_aprendizado = {
            "generations": list(range(1, len(residuos) + 1)),
            "best_fitness": [round(1.0 / (r + 1e-4), 2) for r in residuos],
            "avg_fitness": [round(1.0 / (np.mean(residuos) + 1e-4), 2)] * len(residuos)
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"Prophet: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
