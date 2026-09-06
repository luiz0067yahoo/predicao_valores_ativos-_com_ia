"""
Módulo do Modelo Preditivo baseado em Lógica Fuzzy (Sistema de Inferência Difusa Takagi-Sugeno)
Implementa particionamento difuso do espaço de características financeiras com
funções de pertinência gaussianas, avaliação contínua de regras SE-ENTÃO,
ajuste adaptativo de consequentes por mínimos quadrados e defuzzificação por Centroide.
Todos os nomes de classes, métodos, variáveis e comentários estão em português.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class _ConjuntoFuzzyGaussiano:
    """
    Representa um termo linguístico difuso com função de pertinência gaussiana:
    mu(x) = exp( - (x - centro)^2 / (2 * largura^2) )
    """

    def __init__(self, nome: str, centro: float, largura: float):
        """
        Inicializa o conjunto difuso.

        Parâmetros:
            nome: Rótulo linguístico (ex: 'Forte Baixa', 'Neutro', 'Forte Alta').
            centro: Ponto central de máxima pertinência (mu = 1.0).
            largura: Dispersão (desvio padrão) da curva de pertinência.
        """
        self.nome = nome
        self.centro = centro
        self.largura = max(largura, 1e-4)

    def calcular_pertinencia(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula o grau de pertinência no intervalo [0.0, 1.0].
        """
        expoente = -0.5 * ((x - self.centro) / self.largura) ** 2
        return np.exp(expoente)


class _SistemaInferenciaFuzzyTSK:
    """
    Sistema de Inferência Difusa Takagi-Sugeno-Kang (TSK) de 1ª ordem adaptativo.
    1. Fuzzificação: mapeia os atributos contínuos em graus de pertinência.
    2. Ativação de Regras: computa o grau de disparo (firing strength) de cada regra.
    3. Consequentes: gera saídas lineares ponderadas pelo grau de pertinência (Centroide).
    """

    def __init__(self, numero_regras: int = 5, largura_base: float = 1.0, forca_regularizacao: float = 0.5):
        """
        Inicializa o sistema fuzzy.

        Parâmetros:
            numero_regras: Quantidade de conjuntos difusos particionados ao longo da distribuição.
            largura_base: Multiplicador de amplitude das curvas gaussianas.
            forca_regularizacao: Parâmetro de penalização Ridge na calibração dos consequentes.
        """
        self.numero_regras = numero_regras
        self.largura_base = largura_base
        self.forca_regularizacao = forca_regularizacao
        self.conjuntos: List[_ConjuntoFuzzyGaussiano] = []
        self.estimador_consequentes = Ridge(alpha=forca_regularizacao, random_state=42)
        self.centros: np.ndarray = np.array([])

    def calibrar_particoes(self, matriz_x: np.ndarray):
        """
        Posiciona os centros das funções de pertinência gaussianas de acordo com os
        quantis da distribuição dos dados históricos, garantindo cobertura uniforme do domínio.
        """
        # Utiliza a primeira componente principal ou média dos atributos como proxy da dinâmica
        sinal_guia = np.mean(matriz_x[:, :5], axis=1)
        quantis = np.linspace(0.05, 0.95, self.numero_regras)
        self.centros = np.quantile(sinal_guia, quantis)

        largura_media = float(np.std(sinal_guia) * self.largura_base / np.sqrt(self.numero_regras))
        largura_media = max(largura_media, 0.2)

        rotulos = [
            "Forte Queda (Bearish Forte)",
            "Queda Moderada",
            "Lateral / Estabilidade",
            "Alta Moderada",
            "Forte Alta (Bullish Forte)"
        ]

        self.conjuntos = []
        for i, c in enumerate(self.centros):
            nome_rotulo = rotulos[i] if i < len(rotulos) else f"Regime Difuso {i + 1}"
            self.conjuntos.append(_ConjuntoFuzzyGaussiano(nome=nome_rotulo, centro=float(c), largura=largura_media))

    def transformar_matriz_disparo(self, matriz_x: np.ndarray) -> np.ndarray:
        """
        Converte cada amostra de entrada em um vetor de graus de disparo normalizados
        combinados com os próprios atributos (formulação Takagi-Sugeno).
        """
        sinal_guia = np.mean(matriz_x[:, :5], axis=1)
        graus_disparo = []

        # Calcula a pertinência de cada amostra em relação a cada regra fuzzy
        for conjunto in self.conjuntos:
            pertinencia = conjunto.calcular_pertinencia(sinal_guia)
            graus_disparo.append(pertinencia)

        matriz_disparo = np.column_stack(graus_disparo)
        # Normalização dos pesos das regras (soma = 1.0 para cada amostra)
        soma_disparos = np.sum(matriz_disparo, axis=1, keepdims=True) + 1e-8
        matriz_disparo_norm = matriz_disparo / soma_disparos

        # Expande os termos para a consequência de Takagi-Sugeno: w_bar_k * [X, 1]
        blocos_consequentes = [matriz_disparo_norm]
        for k in range(self.numero_regras):
            peso_regra = matriz_disparo_norm[:, k : k + 1]
            interacao = peso_regra * matriz_x[:, :5]
            blocos_consequentes.append(interacao)

        matriz_regras_expandida = np.hstack(blocos_consequentes)
        return matriz_regras_expandida

    def ajustar(self, matriz_x: np.ndarray, vetor_y: np.ndarray):
        """
        Ajusta os coeficientes dos consequentes de todas as regras fuzzy usando Ridge Regression.
        """
        self.calibrar_particoes(matriz_x)
        matriz_expandida = self.transformar_matriz_disparo(matriz_x)
        self.estimador_consequentes.fit(matriz_expandida, vetor_y)

    def predizer(self, matriz_x: np.ndarray) -> np.ndarray:
        """
        Executa a defuzzificação analítica para produzir a previsão contínua.
        """
        matriz_expandida = self.transformar_matriz_disparo(matriz_x)
        return self.estimador_consequentes.predict(matriz_expandida)


class ModeloLogicaFuzzy(BaseAlgoritmo):
    """
    Estimador de Inteligência Artificial baseado em Lógica Fuzzy / Neuro-Fuzzy (TSK).
    Destaques:
    - Fuzzificação de osciladores de mercado e momentum em termos linguísticos.
    - Avaliação simultânea de múltiplos regimes de mercado (sobrecompra, sobrevenda, tendência).
    - Defuzzificação por Centro de Gravidade suave, resistente a ruídos e outliers.
    """

    def __init__(
        self,
        numero_conjuntos_fuzzy: int = 5,
        largura_pertinencia: float = 1.0,
        forca_regularizacao: float = 0.5,
        janela_temporal: int = 10
    ):
        """
        Inicializa o modelo de Lógica Fuzzy.

        Parâmetros:
            numero_conjuntos_fuzzy: Quantidade de regras/conjuntos linguísticos (default: 5).
            largura_pertinencia: Dispersão das funções gaussianas (default: 1.0).
            forca_regularizacao: Parâmetro de suavização dos consequentes (default: 0.5).
            janela_temporal: Janela de lags observados na série temporal.
        """
        super().__init__(nome_identificador="Lógica Fuzzy (TSK)", janela_temporal=janela_temporal)
        self.numero_conjuntos_fuzzy = numero_conjuntos_fuzzy
        self.largura_pertinencia = largura_pertinencia
        self.forca_regularizacao = forca_regularizacao
        self.sistema_fuzzy: Optional[_SistemaInferenciaFuzzyTSK] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo:
        1. Fuzzificação das defasagens e osciladores técnicos.
        2. Calibração da base de regras Takagi-Sugeno.
        3. Avaliação histórica in-sample e projeção recursiva futura.
        4. Apuração das métricas estatísticas e intervalos de 95% de confiança.
        """
        config = hiperparametros or {}
        n_conjuntos = int(config.get("numero_conjuntos_fuzzy", self.numero_conjuntos_fuzzy))
        largura = float(config.get("largura_pertinencia", self.largura_pertinencia))
        regularizacao = float(config.get("forca_regularizacao", self.forca_regularizacao))

        if funcao_progresso:
            funcao_progresso(10, 100, "Lógica Fuzzy: Fuzzificando indicadores e variáveis de estado...", 0.0)

        # 1. Obtenção de matriz de defasagens
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        if funcao_progresso:
            funcao_progresso(35, 100, f"Lógica Fuzzy: Calibrando {n_conjuntos} regras de inferência Takagi-Sugeno...", 0.0)

        # 2. Inicialização e calibração do sistema difuso
        self.sistema_fuzzy = _SistemaInferenciaFuzzyTSK(
            numero_regras=n_conjuntos,
            largura_base=largura,
            forca_regularizacao=regularizacao
        )
        self.sistema_fuzzy.ajustar(matriz_atributos, vetor_alvo)

        if funcao_progresso:
            funcao_progresso(70, 100, "Lógica Fuzzy: Defuzzificando previsões e gerando projeção futura...", 0.0)

        # 3. Predição histórica in-sample
        predicoes_normalizadas = self.sistema_fuzzy.predizer(matriz_atributos)
        precos_reais, precos_previstos_reais = self.desnormalizar_historico(
            dados_completos, len(matriz_atributos), predicoes_normalizadas
        )

        residuos = precos_reais - precos_previstos_reais
        desvio_padrao_erros = float(np.std(residuos))

        # 4. Projeção recursiva autorregressiva
        def prever_vetor_fuzzy(v: np.ndarray) -> float:
            return float(self.sistema_fuzzy.predizer(v)[0])

        datas_futuras, previsoes_futuras_reais, limites_inf, limites_sup = self.projetar_futuro_recursivo(
            dados_completos=dados_completos,
            funcao_predicao_vetor=prever_vetor_fuzzy,
            horizonte_projecao=horizonte_projecao,
            desvio_padrao_erros=desvio_padrao_erros,
            eh_criptomoeda=eh_criptomoeda
        )

        # 5. DataFrames de saída
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

        # 6. Apuração de métricas estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=precos_reais,
            precos_previstos=precos_previstos_reais,
            preco_real_final=float(precos_reais[-1]),
            preco_projetado_final=float(previsoes_futuras_reais[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Graus de ativação médios das regras fuzzy para exibição visual
        matriz_disparo = self.sistema_fuzzy.transformar_matriz_disparo(matriz_atributos)
        importancia_regras = [float(np.mean(matriz_disparo[:, k])) for k in range(min(15, matriz_disparo.shape[1]))]

        self.historico_aprendizado = {
            "generations": list(range(1, len(importancia_regras) + 1)),
            "best_fitness": [round(v * 1000, 2) for v in importancia_regras],
            "avg_fitness": [round(np.mean(importancia_regras) * 1000, 2)] * len(importancia_regras)
        }

        if funcao_progresso:
            funcao_progresso(
                100,
                100,
                f"Lógica Fuzzy: Concluído! Tendência: {metricas['tendencia_esperada']}",
                metricas["rmse"]
            )

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
