"""
Modelo Estatístico ARIMA / SARIMA para Previsão de Séries Temporais Financeiras
Utiliza modelagem autorregressiva integrada de médias móveis (AutoRegressive Integrated Moving Average)
via statsmodels para capturar tendências estocásticas e persistência de choques temporais.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from algoritmos.base_algoritmo import BaseAlgoritmo


class ModeloARIMA(BaseAlgoritmo):
    """
    Implementação do modelo estatístico clássico ARIMA(p, d, q).
    Destaques:
    - Diferenciação de primeira ordem (d=1) para estabilização de estacionariedade.
    - Componentes autorregressivos (p) e de médias móveis de ruído branco (q).
    - Obtenção analítica de intervalos de confiança teóricos.
    """

    def __init__(self, ordem_p: int = 2, ordem_d: int = 1, ordem_q: int = 2, janela_temporal: int = 10):
        """
        Inicializa o modelo ARIMA.

        Parâmetros:
            ordem_p: Ordem da defasagem autorregressiva (AR).
            ordem_d: Grau de diferenciação para estacionariedade (I).
            ordem_q: Ordem da janela de médias móveis de erro (MA).
            janela_temporal: Quantidade de dias passados para observação.
        """
        super().__init__(nome_identificador="ARIMA/SARIMA", janela_temporal=janela_temporal)
        self.ordem = (ordem_p, ordem_d, ordem_q)
        self.modelo_ajustado = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ajusta a equação diferencial estocástica aos dados e projeta a trajetória futura.

        Parâmetros:
            dados_completos: DataFrame com histórico de cotações contendo 'Close'.
            horizonte_projecao: Passos à frente para estimativa.
            eh_criptomoeda: Identificador de mercado contínuo.
            funcao_progresso: Callback de monitoramento.
            hiperparametros: Dicionário opcional com nova ordem (p, d, q).

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        p = int(config.get("ordem_p", self.ordem[0]))
        d = int(config.get("ordem_d", self.ordem[1]))
        q = int(config.get("ordem_q", self.ordem[2]))
        ordem_final = (p, d, q)

        if funcao_progresso:
            funcao_progresso(15, 100, f"ARIMA: Inicializando estimação por máxima verossimilhança {ordem_final}...", 0.0)

        # 1. Extração da série temporal de preços de fechamento
        serie_precos = dados_completos["Close"].dropna().astype(float)
        datas_alvo = list(serie_precos.index)
        valores_reais = serie_precos.values

        if len(valores_reais) < 25:
            raise ValueError("Série temporal com registros insuficientes para ajuste estável do ARIMA.")

        if funcao_progresso:
            funcao_progresso(40, 100, "ARIMA: Otimizando parâmetros autorregressivos e médias móveis...", 0.0)

        # 2. Ajuste do modelo ARIMA
        try:
            modelo_arima = ARIMA(valores_reais, order=ordem_final)
            self.modelo_ajustado = modelo_arima.fit()
        except Exception:
            # Fallback seguro para ordem simplificada em caso de não-convergência da matriz hessiana
            modelo_arima = ARIMA(valores_reais, order=(1, 1, 1))
            self.modelo_ajustado = modelo_arima.fit()

        if funcao_progresso:
            funcao_progresso(70, 100, "ARIMA: Extraindo predições in-sample e calculando cone de variância...", 0.0)

        # 3. Predição in-sample
        precos_previstos_in_sample = self.modelo_ajustado.fittedvalues
        # Ajusta o primeiro ponto para evitar o salto da diferenciação inicial
        precos_previstos_in_sample[0] = valores_reais[0]

        # 4. Projeção futura analítica com intervalo de confiança de 95%
        previsao_objeto = self.modelo_ajustado.get_forecast(steps=horizonte_projecao)
        previsoes_futuras = list(previsao_objeto.predicted_mean)
        intervalos_confianca = previsao_objeto.conf_int(alpha=0.05)

        limites_inferiores = [max(0.0, float(intervalos_confianca[i, 0])) for i in range(horizonte_projecao)]
        limites_superiores = [float(intervalos_confianca[i, 1]) for i in range(horizonte_projecao)]

        # 5. Geração de datas futuras
        ultima_data = pd.to_datetime(datas_alvo[-1])
        datas_futuras = self.gerar_datas_futuras(ultima_data, horizonte_projecao, eh_criptomoeda=eh_criptomoeda)

        # 6. Estruturação dos DataFrames
        df_historico_saida = pd.DataFrame(
            {
                "Preco_Real": valores_reais,
                "Preco_Previsto_IA": precos_previstos_in_sample
            },
            index=datas_alvo
        )

        df_projecao_saida = pd.DataFrame(
            {
                "Preco_Projetado": previsoes_futuras,
                "Limite_Inferior": limites_inferiores,
                "Limite_Superior": limites_superiores
            },
            index=datas_futuras
        )

        # 7. Métricas estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=valores_reais,
            precos_previstos=precos_previstos_in_sample,
            preco_real_final=float(valores_reais[-1]),
            preco_projetado_final=float(previsoes_futuras[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Resíduos do modelo para gráfico
        residuos_abs = [float(abs(r)) for r in self.modelo_ajustado.resid[-15:]]
        self.historico_aprendizado = {
            "generations": list(range(1, len(residuos_abs) + 1)),
            "best_fitness": [round(1.0 / (r + 1e-4), 2) for r in residuos_abs],
            "avg_fitness": [round(1.0 / (np.mean(residuos_abs) + 1e-4), 2)] * len(residuos_abs)
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"ARIMA: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
