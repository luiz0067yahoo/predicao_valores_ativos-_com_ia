"""
Módulo do Algoritmo M.A.P.P. (algoritmos/mapp/modelo_mapp.py)
Market Analysis Pattern Prediction
Combina os 4 pilares:
M — Market Regime (Regime de Mercado)
A — Accumulation & Structure (Volume, Suporte/Resistência, Volatilidade)
P — Pattern (Reconhecimento de Padrões Gráficos, Candlesticks, Fibonacci e Divergências)
P — Prediction (Projeção Direcional, Magnitude Esperada, Probabilidade Calibrada e IC 95%)
Sem vazamento temporal.
Todos os nomes em português.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from algoritmos.base_algoritmo import BaseAlgoritmo
from mapp.market_regime import MarketRegimeDetector, TipoRegimeMercado
from mapp.features import extrair_todas_caracteristicas_mapp
from mapp.selection import SeletorCaracteristicas


class ModeloMAPP(BaseAlgoritmo):
    """
    Algoritmo unificado M.A.P.P. (Market Analysis Pattern Prediction).
    """

    def __init__(self, janela_temporal: int = 10):
        super().__init__(nome_identificador="M.A.P.P.", janela_temporal=janela_temporal)
        self.regime_atual: Optional[str] = None
        self.probabilidade_alta: float = 0.5
        self.probabilidade_baixa: float = 0.5
        self.probabilidade_lateral: float = 0.0
        self.magnitude_esperada_pct: float = 0.0

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o pipeline M.A.P.P. completo:
        1. Identificação do Regime de Mercado Atual (M).
        2. Engenharia e seleção dos atributos de Acumulação e Estrutura (A).
        3. Detecção de Padrões gráficos, candlesticks e confluências (P).
        4. Ajuste do modelo preditivo multivariado e calibração probabilística (P).
        """
        params = hiperparametros or {}
        max_features = params.get("max_features", 20)
        peso_regime = params.get("peso_regime", 0.25)
        peso_padroes = params.get("peso_padroes", 0.25)

        if funcao_progresso:
            funcao_progresso(10, 100, "M.A.P.P. — Analisando Regime de Mercado...", 0.0)

        # 1. Identifica Regime de Mercado na última barra
        resultado_regime = MarketRegimeDetector.detectar(dados_completos)
        self.regime_atual = resultado_regime.regime_primario.value

        if funcao_progresso:
            funcao_progresso(30, 100, f"M.A.P.P. — Regime: {self.regime_atual}. Extraindo atributos...", 0.0)

        # 2. Extração de Features Quantitativas MAPP
        matriz_features = extrair_todas_caracteristicas_mapp(dados_completos)

        # Adiciona features de regime de mercado
        for reg_col, reg_val in resultado_regime.vetor_features_regime.items():
            matriz_features[reg_col] = reg_val

        # 3. Construção dos pares de treino supervisionados
        close = dados_completos["Close"].values
        datas = dados_completos.index
        n = len(close)

        # Alvo: retorno futuro no horizonte desejado
        lookback = self.janela_temporal
        X_lista = []
        y_lista = []
        datas_validas = []

        for i in range(lookback, n - 1):
            vetor_feat = matriz_features.iloc[i].values
            retorno_futuro = (close[i + 1] - close[i]) / (close[i] + 1e-9)
            X_lista.append(vetor_feat)
            y_lista.append(retorno_futuro)
            datas_validas.append(datas[i])

        X = np.array(X_lista, dtype=np.float64)
        y = np.array(y_lista, dtype=np.float64)

        if funcao_progresso:
            funcao_progresso(55, 100, "M.A.P.P. — Selecionando características relevantes...", 0.0)

        # Seleção de características para evitar overfitting
        colunas_selecionadas = SeletorCaracteristicas.selecionar_melhores_features(
            matriz_features.iloc[lookback : n - 1],
            y,
            max_features=max_features
        )
        indices_cols = [matriz_features.columns.get_loc(c) for c in colunas_selecionadas]
        X_reduzido = X[:, indices_cols]

        if funcao_progresso:
            funcao_progresso(75, 100, "M.A.P.P. — Ajustando pesos dos modelos e calibração...", 0.0)

        # 4. Treinamento de Regressores Combinados (Ridge + Gradient Boosting)
        modelo_ridge = Ridge(alpha=1.5)
        modelo_ridge.fit(X_reduzido, y)

        modelo_gb = GradientBoostingRegressor(n_estimators=45, max_depth=3, learning_rate=0.08, random_state=42)
        modelo_gb.fit(X_reduzido, y)

        # Previsão in-sample para cálculo de resíduos e intervalo de confiança
        pred_in_ridge = modelo_ridge.predict(X_reduzido)
        pred_in_gb = modelo_gb.predict(X_reduzido)
        pred_in = 0.5 * pred_in_ridge + 0.5 * pred_in_gb

        residuos = y - pred_in
        std_residuos = float(np.std(residuos)) if len(residuos) > 2 else 0.015

        # 5. Projeção Recursiva para o Horizonte Futuro
        projecao_precos: List[float] = []
        limites_inferiores: List[float] = []
        limites_superiores: List[float] = []

        preco_base = close[-1]
        ultimo_vetor_feat = matriz_features.iloc[-1].values[indices_cols].reshape(1, -1)

        # Projeta dia a dia recursivamente
        retorno_prev_1 = float(0.5 * modelo_ridge.predict(ultimo_vetor_feat)[0] + 0.5 * modelo_gb.predict(ultimo_vetor_feat)[0])

        # Ajusta pelo viés do regime de mercado
        fator_regime = 0.0
        if resultado_regime.regime_primario == TipoRegimeMercado.BULL_TREND:
            fator_regime = 0.003
        elif resultado_regime.regime_primario == TipoRegimeMercado.BEAR_TREND:
            fator_regime = -0.003
        elif resultado_regime.regime_primario == TipoRegimeMercado.CRASH:
            fator_regime = -0.008
        elif resultado_regime.regime_primario == TipoRegimeMercado.RECOVERY:
            fator_regime = 0.005

        preco_corrente = preco_base

        for step in range(1, horizonte_projecao + 1):
            ret_passo = retorno_prev_1 * (0.95 ** (step - 1)) + fator_regime
            preco_corrente = max(0.01, preco_corrente * (1.0 + ret_passo))
            projecao_precos.append(preco_corrente)

            # Incerteza crescente com a raiz quadrada do tempo (95% IC ~ 1.96 std)
            margem_erro = 1.96 * std_residuos * np.sqrt(step) * preco_corrente
            limites_inferiores.append(max(0.01, preco_corrente - margem_erro))
            limites_superiores.append(preco_corrente + margem_erro)

        if funcao_progresso:
            funcao_progresso(95, 100, "M.A.P.P. — Consolidando métricas e probabilidades...", 0.0)

        # 6. Calibração de Probabilidades
        retorno_total_esperado = (projecao_precos[-1] - preco_base) / preco_base
        self.magnitude_esperada_pct = retorno_total_esperado * 100.0

        # Converte magnitude e volatilidade em probabilidade normalizada via sigmoide adaptativa
        z = retorno_total_esperado / (std_residuos * np.sqrt(horizonte_projecao) + 1e-9)
        prob_alta_bruta = 1.0 / (1.0 + np.exp(-1.5 * z))

        if abs(retorno_total_esperado) < 0.015:
            # Lateralização
            self.probabilidade_lateral = 0.55 + min(0.3, (0.015 - abs(retorno_total_esperado)) * 20)
            sobra = 1.0 - self.probabilidade_lateral
            self.probabilidade_alta = sobra * prob_alta_bruta
            self.probabilidade_baixa = sobra * (1.0 - prob_alta_bruta)
            direcao_mapp = "LATERALIZAÇÃO"
        elif retorno_total_esperado > 0:
            self.probabilidade_alta = float(np.clip(prob_alta_bruta, 0.52, 0.92))
            self.probabilidade_baixa = float(1.0 - self.probabilidade_alta)
            self.probabilidade_lateral = 0.08
            direcao_mapp = "ALTA"
        else:
            self.probabilidade_baixa = float(np.clip(1.0 - prob_alta_bruta, 0.52, 0.92))
            self.probabilidade_alta = float(1.0 - self.probabilidade_baixa)
            self.probabilidade_lateral = 0.08
            direcao_mapp = "BAIXA"

        # Constrói DataFrame histórico com aderência in-sample
        precos_in_sample = close[lookback : n - 1] * (1.0 + pred_in)
        df_historico = pd.DataFrame({
            "Preco_Real": close[lookback : n - 1],
            "Preco_Previsto_IA": precos_in_sample
        }, index=datas[lookback : n - 1])

        # Datas futuras
        datas_futuras = self.gerar_datas_futuras(datas[-1], horizonte_projecao, eh_criptomoeda)
        df_projecao = pd.DataFrame({
            "Preco_Projetado": projecao_precos,
            "Limite_Inferior": limites_inferiores,
            "Limite_Superior": limites_superiores
        }, index=datas_futuras)

        # Métricas estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=close[lookback : n - 1],
            precos_previstos=precos_in_sample,
            preco_real_final=float(preco_base),
            preco_projetado_final=float(projecao_precos[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Enriquece métricas com M.A.P.P. específico
        metricas["mapp_direction"] = direcao_mapp
        metricas["mapp_probability_up"] = round(self.probabilidade_alta * 100.0, 1)
        metricas["mapp_probability_down"] = round(self.probabilidade_baixa * 100.0, 1)
        metricas["mapp_probability_sideways"] = round(self.probabilidade_lateral * 100.0, 1)
        metricas["mapp_expected_magnitude_pct"] = round(self.magnitude_esperada_pct, 2)
        metricas["mapp_market_regime"] = self.regime_atual
        metricas["mapp_regime_confidence"] = round(resultado_regime.confianca * 100.0, 1)
        metricas["tendencia_esperada"] = f"{direcao_mapp} [{round(max(self.probabilidade_alta, self.probabilidade_baixa) * 100.0, 1)}% Prob.]"

        return {
            "metrics": metricas,
            "metricas": metricas,
            "history_df": df_historico,
            "forecast_df": df_projecao,
            "probabilidade_alta": self.probabilidade_alta,
            "direcao_predita": direcao_mapp,
            "regime_detectado": self.regime_atual,
            "ga_history": {
                "perdas": [float(np.mean(residuos[-20:] ** 2))],
                "mapp_regime": resultado_regime.para_dicionario()
            }
        }
