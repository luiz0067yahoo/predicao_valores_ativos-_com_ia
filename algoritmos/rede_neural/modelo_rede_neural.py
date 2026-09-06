"""
Módulo do Modelo Preditivo baseado em Rede Neural Profunda (MLP - Multi-Layer Perceptron)
Implementado em PyTorch com arquitetura feedforward totalmente conectada,
camadas ocultas customizáveis, regularização por Dropout e desnormalização precisa.
Todos os nomes de classes, métodos, variáveis e comentários estão em português.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class _RedeNeuralProfundaMLP(nn.Module):
    """
    Arquitetura de rede neural densa feedforward (MLP) com múltiplas camadas ocultas.
    Aprende mapeamentos não-lineares complexos entre indicadores técnicos/defasagens
    e a variação futura do preço do ativo.
    """

    def __init__(
        self,
        dimensao_entrada: int,
        dimensao_camada_1: int = 64,
        dimensao_camada_2: int = 32,
        taxa_dropout: float = 0.15,
        nome_ativacao: str = "relu"
    ):
        """
        Construtor da arquitetura neural.

        Parâmetros:
            dimensao_entrada: Número de features de entrada (lags + indicadores).
            dimensao_camada_1: Quantidade de neurônios da primeira camada oculta.
            dimensao_camada_2: Quantidade de neurônios da segunda camada oculta.
            taxa_dropout: Probabilidade de desligamento de neurônios para evitar overfitting.
            nome_ativacao: Função de ativação ('relu', 'gelu', 'tanh', 'leaky_relu').
        """
        super().__init__()

        # Mapeamento da função de ativação não-linear
        if nome_ativacao.lower() == "gelu":
            funcao_ativacao = nn.GELU()
        elif nome_ativacao.lower() == "tanh":
            funcao_ativacao = nn.Tanh()
        elif nome_ativacao.lower() == "leaky_relu":
            funcao_ativacao = nn.LeakyReLU(negative_slope=0.01)
        else:
            funcao_ativacao = nn.ReLU()

        # Definição das camadas lineares e de regularização
        self.camada_entrada = nn.Linear(dimensao_entrada, dimensao_camada_1)
        self.ativacao_1 = funcao_ativacao
        self.dropout_1 = nn.Dropout(taxa_dropout)

        self.camada_intermediaria = nn.Linear(dimensao_camada_1, dimensao_camada_2)
        self.ativacao_2 = funcao_ativacao
        self.dropout_2 = nn.Dropout(taxa_dropout)

        self.camada_saida = nn.Linear(dimensao_camada_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta (forward pass) dos dados através dos blocos de neurônios.

        Parâmetros:
            x: Tensor com as características de entrada [amostras, features].

        Retorno:
            Tensor com as estimativas preditas [amostras].
        """
        h1 = self.dropout_1(self.ativacao_1(self.camada_entrada(x)))
        h2 = self.dropout_2(self.ativacao_2(self.camada_intermediaria(h1)))
        saida = self.camada_saida(h2)
        return saida.squeeze(-1)


class ModeloRedeNeural(BaseAlgoritmo):
    """
    Estimador de Inteligência Artificial baseado em Rede Neural Profunda (MLP).
    Utiliza otimizador Adam com decaimento de peso (L2) e retropropagação do erro quadrático médio.
    """

    def __init__(
        self,
        numero_epocas: int = 45,
        taxa_aprendizado: float = 0.005,
        dimensao_camada_1: int = 64,
        dimensao_camada_2: int = 32,
        funcao_ativacao: str = "relu",
        janela_temporal: int = 10
    ):
        """
        Inicializa o algoritmo de Rede Neural.

        Parâmetros:
            numero_epocas: Ciclos completos de treinamento sobre o dataset histórico.
            taxa_aprendizado: Tamanho do passo de atualização dos gradientes.
            dimensao_camada_1: Número de neurônios na 1ª camada escondida.
            dimensao_camada_2: Número de neurônios na 2ª camada escondida.
            funcao_ativacao: Função de ativação dos neurônios ('relu', 'gelu', 'tanh').
            janela_temporal: Tamanho da janela de lags observados pelo modelo.
        """
        super().__init__(nome_identificador="Rede Neural (MLP)", janela_temporal=janela_temporal)
        self.numero_epocas = numero_epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.dimensao_camada_1 = dimensao_camada_1
        self.dimensao_camada_2 = dimensao_camada_2
        self.funcao_ativacao = funcao_ativacao
        self.rede: Optional[_RedeNeuralProfundaMLP] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo de treinamento supervisionado da Rede Neural e projeção futura.

        Parâmetros:
            dados_completos: DataFrame histórico com cotações [Open, High, Low, Close, Volume].
            horizonte_projecao: Quantidade de dias futuros para estimar.
            eh_criptomoeda: Se True, considera negociação contínua aos fins de semana.
            funcao_progresso: Callback de monitoramento (passo, total, mensagem, métrica).
            hiperparametros: Dicionário opcional para ajuste fino em tempo de execução.

        Retorno:
            Dicionário com 'metrics', 'history_df', 'forecast_df' e 'ga_history'.
        """
        configuracoes = hiperparametros or {}
        epocas = int(configuracoes.get("numero_epocas", self.numero_epocas))
        taxa_lr = float(configuracoes.get("taxa_aprendizado", self.taxa_aprendizado))
        camada_1 = int(configuracoes.get("dimensao_camada_1", self.dimensao_camada_1))
        camada_2 = int(configuracoes.get("dimensao_camada_2", self.dimensao_camada_2))
        ativacao = str(configuracoes.get("funcao_ativacao", self.funcao_ativacao))

        if funcao_progresso:
            funcao_progresso(5, 100, "Rede Neural: Preparando matriz de defasagens e indicadores...", 0.0)

        # 1. Extração de atributos e defasagens
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        dimensao_atributos = matriz_atributos.shape[1]
        tensores_x = torch.tensor(matriz_atributos, dtype=torch.float32)
        tensores_y = torch.tensor(vetor_alvo, dtype=torch.float32)

        # 2. Inicialização da arquitetura neural
        self.rede = _RedeNeuralProfundaMLP(
            dimensao_entrada=dimensao_atributos,
            dimensao_camada_1=camada_1,
            dimensao_camada_2=camada_2,
            taxa_dropout=0.15,
            nome_ativacao=ativacao
        )
        self.rede.train()

        funcao_perda = nn.MSELoss()
        otimizador = optim.Adam(self.rede.parameters(), lr=taxa_lr, weight_decay=1e-5)

        historico_perdas: List[float] = []

        # 3. Treinamento da Rede Neural com Retropropagação
        for epoca in range(1, epocas + 1):
            otimizador.zero_grad()
            saidas = self.rede(tensores_x)
            perda = funcao_perda(saidas, tensores_y)
            perda.backward()

            # Evita explosão de gradientes através de clipping
            nn.utils.clip_grad_norm_(self.rede.parameters(), max_norm=1.0)
            otimizador.step()

            valor_perda = float(perda.item())
            historico_perdas.append(valor_perda)

            if funcao_progresso and (epoca % max(1, epocas // 10) == 0 or epoca == epocas):
                progresso = 10 + int((epoca / epocas) * 65)
                funcao_progresso(
                    progresso,
                    100,
                    f"Rede Neural: Época {epoca}/{epocas} | Perda MSE: {valor_perda:.5f}",
                    valor_perda
                )

        # 4. Modo de Inferência e Avaliação Histórica In-Sample
        self.rede.eval()
        with torch.no_grad():
            predicoes_normalizadas = self.rede(tensores_x).numpy()

        precos_reais, precos_previstos_reais = self.desnormalizar_historico(
            dados_completos, len(matriz_atributos), predicoes_normalizadas
        )

        residuos = precos_reais - precos_previstos_reais
        desvio_padrao_erros = float(np.std(residuos))

        # 5. Projeção Recursiva Futura com a Rede Neural
        def prever_vetor_rede(v: np.ndarray) -> float:
            with torch.no_grad():
                tensor_entrada = torch.tensor(v, dtype=torch.float32)
                return float(self.rede(tensor_entrada).item())

        datas_futuras, previsoes_futuras_reais, limites_inf, limites_sup = self.projetar_futuro_recursivo(
            dados_completos=dados_completos,
            funcao_predicao_vetor=prever_vetor_rede,
            horizonte_projecao=horizonte_projecao,
            desvio_padrao_erros=desvio_padrao_erros,
            eh_criptomoeda=eh_criptomoeda
        )

        # 6. Estruturação dos DataFrames
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

        # 7. Apuração das Métricas Estatísticas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=precos_reais,
            precos_previstos=precos_previstos_reais,
            preco_real_final=float(precos_reais[-1]),
            preco_projetado_final=float(previsoes_futuras_reais[-1]),
            horizonte_dias=horizonte_projecao
        )

        # Histórico de aprendizado para gráficos
        self.historico_aprendizado = {
            "generations": list(range(1, len(historico_perdas) + 1)),
            "best_fitness": [round(1.0 / (p + 1e-4), 2) for p in historico_perdas],
            "avg_fitness": [round(1.0 / (np.mean(historico_perdas[:i+1]) + 1e-4), 2) for i in range(len(historico_perdas))],
            "perdas": historico_perdas
        }

        if funcao_progresso:
            funcao_progresso(
                100,
                100,
                f"Rede Neural: Concluído! Tendência: {metricas['tendencia_esperada']}",
                metricas["rmse"]
            )

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
