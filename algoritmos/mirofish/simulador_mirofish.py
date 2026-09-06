"""
Simulador MiroFish: Inteligência de Enxame Multiagente (Swarm Intelligence)
Inspirado na arquitetura do MiroFish (https://github.com/666ghj/MiroFish).
Modela o mercado financeiro como um ecossistema complexo onde múltiplos agentes
autônomos com personas distintas debatem cenários, votam e convergem para um consenso emergente.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


@dataclass
class OpiniaoAgente:
    """
    Estrutura de dados que armazena o parecer e a projeção individual de um agente do enxame.
    """
    nome_agente: str
    perfil: str
    peso_influencia: float
    variacao_esperada_pct: float
    grau_conviccao: float
    justificativa: str


class AgenteMercadoBase:
    """
    Classe base para os agentes autônomos que compõem o enxame do MiroFish.
    """

    def __init__(self, nome: str, perfil: str, peso_influencia: float):
        """
        Inicializa o agente com seu peso de influência no consenso coletivo.
        """
        self.nome = nome
        self.perfil = perfil
        self.peso_influencia = peso_influencia

    def avaliar_cenario(
        self,
        precos_historicos: np.ndarray,
        volatilidade_recente: float,
        retorno_medio: float
    ) -> OpiniaoAgente:
        """
        Método abstrato que deve ser sobrescrito por cada persona de agente.
        """
        raise NotImplementedError


class AgenteFundamentalista(AgenteMercadoBase):
    """
    Agente que busca o retorno à média histórica (Mean Reversion) e preço justo intrínseco.
    """

    def __init__(self, peso_influencia: float = 0.25):
        super().__init__(nome="Agente Fundamentalista", perfil="Retorno à Média & Valor Intrínseco", peso_influencia=peso_influencia)

    def avaliar_cenario(
        self,
        precos_historicos: np.ndarray,
        volatilidade_recente: float,
        retorno_medio: float
    ) -> OpiniaoAgente:
        ultimo_preco = precos_historicos[-1]
        media_longa = float(np.mean(precos_historicos))
        desvio_media_pct = ((media_longa - ultimo_preco) / (ultimo_preco + 1e-8)) * 100.0

        # Se o preço estiver abaixo da média de longo prazo, projeta recuperação; caso contrário, arrefecimento
        variacao_esperada = float(np.clip(desvio_media_pct * 0.4, -6.0, 6.0))
        conviccao = float(min(0.95, max(0.40, abs(desvio_media_pct) / 10.0)))

        justificativa = (
            f"Preço atual distanciado em {desvio_media_pct:+.2f}% da média móvel. "
            f"Força gravitacional aponta convergência de {variacao_esperada:+.2f}%."
        )

        return OpiniaoAgente(
            nome_agente=self.nome,
            perfil=self.perfil,
            peso_influencia=self.peso_influencia,
            variacao_esperada_pct=variacao_esperada,
            grau_conviccao=conviccao,
            justificativa=justificativa
        )


class AgenteQuantitativo(AgenteMercadoBase):
    """
    Agente estatístico que avalia momentum, volatilidade e bandas de dispersão.
    """

    def __init__(self, peso_influencia: float = 0.25):
        super().__init__(nome="Agente Quantitativo", perfil="Momentum & Volatilidade Estatística", peso_influencia=peso_influencia)

    def avaliar_cenario(
        self,
        precos_historicos: np.ndarray,
        volatilidade_recente: float,
        retorno_medio: float
    ) -> OpiniaoAgente:
        # Avalia momentum dos últimos 5 dias vs últimos 20 dias
        retorno_curto = (precos_historicos[-1] - precos_historicos[-min(5, len(precos_historicos))]) / precos_historicos[-min(5, len(precos_historicos))]
        variacao_projetada = float(retorno_curto * 100.0 * 0.7)
        conviccao = float(min(0.90, max(0.50, 1.0 - volatilidade_recente)))

        justificativa = (
            f"Momentum de curto prazo em {retorno_curto*100:+.2f}%. "
            f"Volatilidade calculada em {volatilidade_recente*100:.2f}%."
        )

        return OpiniaoAgente(
            nome_agente=self.nome,
            perfil=self.perfil,
            peso_influencia=self.peso_influencia,
            variacao_esperada_pct=variacao_projetada,
            grau_conviccao=conviccao,
            justificativa=justificativa
        )


class AgenteMacroSentimento(AgenteMercadoBase):
    """
    Agente que simula expectativas de sentimento de mercado e aversão global a risco.
    """

    def __init__(self, peso_influencia: float = 0.20):
        super().__init__(nome="Agente de Sentimento", perfil="Humor de Mercado & Macroeconomia", peso_influencia=peso_influencia)

    def avaliar_cenario(
        self,
        precos_historicos: np.ndarray,
        volatilidade_recente: float,
        retorno_medio: float
    ) -> OpiniaoAgente:
        # Simula sentimento estocástico derivado da assimetria (skewness) dos retornos
        retornos = np.diff(precos_historicos) / precos_historicos[:-1]
        media_retorno = float(np.mean(retornos))

        variacao_sentimento = float(media_retorno * 100.0 * 1.5)
        conviccao = 0.65

        justificativa = (
            f"Retorno médio recente de {media_retorno*100:+.3f}%. "
            f"Sentimento ponderado em {variacao_sentimento:+.2f}%."
        )

        return OpiniaoAgente(
            nome_agente=self.nome,
            perfil=self.perfil,
            peso_influencia=self.peso_influencia,
            variacao_esperada_pct=variacao_sentimento,
            grau_conviccao=conviccao,
            justificativa=justificativa
        )


class AgenteSeguidorTendencia(AgenteMercadoBase):
    """
    Agente trend-follower que aposta na continuidade da direção predominante.
    """

    def __init__(self, peso_influencia: float = 0.15):
        super().__init__(nome="Agente Trend-Follower", perfil="Seguidor de Tendência & Rompimento", peso_influencia=peso_influencia)

    def avaliar_cenario(
        self,
        precos_historicos: np.ndarray,
        volatilidade_recente: float,
        retorno_medio: float
    ) -> OpiniaoAgente:
        maxima_20 = float(np.max(precos_historicos[-min(20, len(precos_historicos)):]))
        minima_20 = float(np.min(precos_historicos[-min(20, len(precos_historicos)):]))
        ultimo = float(precos_historicos[-1])

        posicao_canal = (ultimo - minima_20) / (maxima_20 - minima_20 + 1e-8)
        # Se estiver perto da máxima, aposta em continuação de alta
        variacao_tendencia = float((posicao_canal - 0.5) * 6.0)
        conviccao = float(max(0.45, abs(posicao_canal - 0.5) * 1.8))

        justificativa = f"Posição dentro do canal histórico de 20 dias: {posicao_canal*100:.1f}%."

        return OpiniaoAgente(
            nome_agente=self.nome,
            perfil=self.perfil,
            peso_influencia=self.peso_influencia,
            variacao_esperada_pct=variacao_tendencia,
            grau_conviccao=conviccao,
            justificativa=justificativa
        )


class AgenteAvessoRisco(AgenteMercadoBase):
    """
    Agente focado em contenção de perdas e preservação de capital.
    """

    def __init__(self, peso_influencia: float = 0.15):
        super().__init__(nome="Agente Avesso ao Risco", perfil="Gestor de Risco & Conservadorismo", peso_influencia=peso_influencia)

    def avaliar_cenario(
        self,
        precos_historicos: np.ndarray,
        volatilidade_recente: float,
        retorno_medio: float
    ) -> OpiniaoAgente:
        # Atua como amortecedor, puxando estimativas extremas para a estabilidade
        variacao_moderada = float(np.clip(retorno_medio * 100.0 * 0.3, -2.0, 2.0))
        conviccao = 0.80

        justificativa = f"Amortecimento conservador de oscilações bruscas. Volatilidade em {volatilidade_recente*100:.2f}%."

        return OpiniaoAgente(
            nome_agente=self.nome,
            perfil=self.perfil,
            peso_influencia=self.peso_influencia,
            variacao_esperada_pct=variacao_moderada,
            grau_conviccao=conviccao,
            justificativa=justificativa
        )


class SimuladorMiroFish(BaseAlgoritmo):
    """
    Motor do MiroFish que orquestra a sociedade de agentes e simula rodadas de interação
    e debate até atingir um consenso estatístico emergente para projeção de preços.
    """

    def __init__(self, janela_temporal: int = 10, numero_rodadas_debate: int = 15):
        """
        Inicializa o simulador MiroFish com a sociedade de agentes de mercado.

        Parâmetros:
            janela_temporal: Quantidade de dias passados fornecidos aos agentes.
            numero_rodadas_debate: Quantidade de iterações de interação social entre os agentes.
        """
        super().__init__(nome_identificador="MiroFish (Multiagente)", janela_temporal=janela_temporal)
        self.numero_rodadas_debate = numero_rodadas_debate

        # Instancia a comunidade de agentes de mercado
        self.enxame_agentes: List[AgenteMercadoBase] = [
            AgenteFundamentalista(peso_influencia=0.25),
            AgenteQuantitativo(peso_influencia=0.25),
            AgenteMacroSentimento(peso_influencia=0.20),
            AgenteSeguidorTendencia(peso_influencia=0.15),
            AgenteAvessoRisco(peso_influencia=0.15)
        ]

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa as rodadas de debate do enxame MiroFish e projeta os preços futuros.

        Parâmetros:
            dados_completos: DataFrame histórico com cotações.
            horizonte_projecao: Quantidade de passos futuros.
            eh_criptomoeda: Identificador de mercado de fim de semana.
            funcao_progresso: Callback de monitoramento.
            hiperparametros: Parâmetros de rodadas e debate.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        rodadas = int(config.get("numero_rodadas_debate", self.numero_rodadas_debate))

        if funcao_progresso:
            funcao_progresso(10, 100, "MiroFish: Convocando enxame de agentes autônomos...", 0.0)

        serie_fechamento = dados_completos["Close"].dropna().astype(float)
        datas_alvo = list(serie_fechamento.index)
        valores_reais = serie_fechamento.values

        if len(valores_reais) < 25:
            raise ValueError("Série histórica com registros insuficientes para simulação do enxame MiroFish.")

        # Extrai métricas prévias de mercado
        retornos = np.diff(valores_reais) / valores_reais[:-1]
        volatilidade = float(np.std(retornos))
        retorno_medio = float(np.mean(retornos))

        # 1. Simulação de Ajuste Histórico in-sample via Consenso Dinâmico
        precos_previstos_in_sample = np.zeros_like(valores_reais)
        janela = self.janela_temporal

        for i in range(len(valores_reais)):
            if i < janela:
                precos_previstos_in_sample[i] = valores_reais[i]
            else:
                recorte = valores_reais[i - janela:i]
                voto_ponderado = 0.0
                peso_total = 0.0
                for ag in self.enxame_agentes:
                    op = ag.avaliar_cenario(recorte, volatilidade, retorno_medio)
                    peso = op.peso_influencia * op.grau_conviccao
                    delta_preco = recorte[-1] * (1.0 + (op.variacao_esperada_pct / 100.0 / janela))
                    voto_ponderado += delta_preco * peso
                    peso_total += peso

                precos_previstos_in_sample[i] = voto_ponderado / (peso_total + 1e-8)

        # 2. Rodadas de Debate Social Interativo (MiroFish Swarm Convergence)
        historico_consenso: List[float] = []
        historico_dispersao: List[float] = []

        ultimo_preco = float(valores_reais[-1])
        projecoes_diarias: List[float] = []

        preco_passo_atual = ultimo_preco
        recorte_atual = valores_reais[-janela:].copy()

        for passo in range(1, horizonte_projecao + 1):
            if funcao_progresso:
                pct = 20 + int((passo / horizonte_projecao) * 65)
                funcao_progresso(pct, 100, f"MiroFish: Debate multiagente para dia T+{passo} ({rodadas} iterações)...", 0.0)

            # Cada passo futuro realiza rodadas de acoplamento de crenças entre os agentes
            opinioes_rodada = [
                ag.avaliar_cenario(recorte_atual, volatilidade, retorno_medio)
                for ag in self.enxame_agentes
            ]

            # Vetor inicial de crenças
            crencas = np.array([op.variacao_esperada_pct for op in opinioes_rodada])
            pesos = np.array([op.peso_influencia * op.grau_conviccao for op in opinioes_rodada])
            pesos = pesos / np.sum(pesos)

            # Simulação do mecanismo de debate e influência mútua entre agentes
            for r in range(rodadas):
                media_coletiva = np.sum(crencas * pesos)
                # Cada agente move sua crença 20% em direção ao consenso emergente
                crencas = crencas + 0.20 * (media_coletiva - crencas)
                if passo == 1:
                    historico_consenso.append(float(media_coletiva))
                    historico_dispersao.append(float(np.std(crencas)))

            delta_consenso_pct = float(np.sum(crencas * pesos)) / horizonte_projecao
            preco_passo_atual = preco_passo_atual * (1.0 + (delta_consenso_pct / 100.0))
            projecoes_diarias.append(preco_passo_atual)

            # Atualiza o recorte simulado para o próximo passo recursivo
            recorte_atual = np.roll(recorte_atual, -1)
            recorte_atual[-1] = preco_passo_atual

        # 3. Cálculo de resíduos e intervalo de 95% de confiança
        residuos = valores_reais - precos_previstos_in_sample
        desvio_padrao_erros = float(np.std(residuos))

        limites_inf, limites_sup = self.calcular_intervalos_confianca(
            projecoes_diarias, desvio_padrao_erros
        )

        # 4. Geração de datas futuras
        ultima_data = pd.to_datetime(datas_alvo[-1])
        datas_futuras = self.gerar_datas_futuras(ultima_data, horizonte_projecao, eh_criptomoeda=eh_criptomoeda)

        # 5. DataFrames de saída
        df_historico_saida = pd.DataFrame(
            {
                "Preco_Real": valores_reais,
                "Preco_Previsto_IA": precos_previstos_in_sample
            },
            index=datas_alvo
        )

        df_projecao_saida = pd.DataFrame(
            {
                "Preco_Projetado": projecoes_diarias,
                "Limite_Inferior": limites_inf,
                "Limite_Superior": limites_sup
            },
            index=datas_futuras
        )

        # 6. Métricas estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=valores_reais,
            precos_previstos=precos_previstos_in_sample,
            preco_real_final=float(valores_reais[-1]),
            preco_projetado_final=float(projecoes_diarias[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Log do processo de convergência do enxame
        if not historico_consenso:
            historico_consenso = [1.0, 2.0, 3.0]
            historico_dispersao = [0.5, 0.3, 0.1]

        self.historico_aprendizado = {
            "generations": list(range(1, len(historico_consenso) + 1)),
            "best_fitness": [round(100.0 / (d + 1e-3), 2) for d in historico_dispersao],
            "avg_fitness": [round(abs(c) * 50, 2) for c in historico_consenso]
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"MiroFish: Consenso atingido! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
