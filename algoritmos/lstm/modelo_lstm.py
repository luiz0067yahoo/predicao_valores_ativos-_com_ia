"""
Modelo LSTM (Long Short-Term Memory) para Previsão de Séries Temporais Financeiras
Implementado em PyTorch, utiliza células com portas de entrada, esquecimento e saída
para reter dependências temporais de longo prazo com amortecimento do gradiente.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class _RedeNeuralLSTM(nn.Module):
    """
    Arquitetura da rede neural profunda com camadas LSTM empilhadas e camada densa linear.
    """

    def __init__(self, dimensao_entrada: int, dimensao_oculta: int = 48, numero_camadas: int = 2, taxa_dropout: float = 0.15):
        """
        Inicializa as camadas da rede neural LSTM.

        Parâmetros:
            dimensao_entrada: Quantidade de variáveis de entrada por passo de tempo.
            dimensao_oculta: Número de unidades/neurônios ocultos em cada célula LSTM.
            numero_camadas: Quantidade de camadas LSTM empilhadas.
            taxa_dropout: Probabilidade de desativação aleatória de neurônios para regularização.
        """
        super().__init__()
        self.dimensao_oculta = dimensao_oculta
        self.numero_camadas = numero_camadas

        # Camada LSTM do PyTorch
        self.camada_lstm = nn.LSTM(
            input_size=dimensao_entrada,
            hidden_size=dimensao_oculta,
            num_layers=numero_camadas,
            batch_first=True,
            dropout=taxa_dropout if numero_camadas > 1 else 0.0
        )

        # Camada linear de saída para projeção escalar contínua
        self.camada_saida = nn.Linear(dimensao_oculta, 1)

    def forward(self, tensores_entrada: torch.Tensor) -> torch.Tensor:
        """
        Propagação direta (forward pass) pelos tensores da rede.

        Parâmetros:
            tensores_entrada: Tensor tridimensional [batch_size, sequence_length, num_features].

        Retorno:
            Tensor predito [batch_size, 1].
        """
        # Saída do LSTM: saida_lstm possui dimensões [batch, seq_len, hidden_dim]
        saida_lstm, _ = self.camada_lstm(tensores_entrada)

        # Seleciona o estado oculto correspondente ao último passo temporal da sequência
        ultimo_passo = saida_lstm[:, -1, :]

        # Passa pelo neurônio linear de regressão
        return self.camada_saida(ultimo_passo).squeeze(-1)


class ModeloLSTM(BaseAlgoritmo):
    """
    Controlador do modelo LSTM para treino, inferência recursiva e projeção de cotações.
    """

    def __init__(
        self,
        janela_temporal: int = 10,
        numero_epocas: int = 45,
        taxa_aprendizado: float = 0.008,
        dimensao_oculta: int = 48
    ):
        """
        Inicializa o modelo LSTM.

        Parâmetros:
            janela_temporal: Tamanho da sequência de dias passados.
            numero_epocas: Quantidade de ciclos completos de treinamento.
            taxa_aprendizado: Taxa de aprendizado do otimizador Adam.
            dimensao_oculta: Tamanho do vetor de estado oculto.
        """
        super().__init__(nome_identificador="LSTM", janela_temporal=janela_temporal)
        self.numero_epocas = numero_epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.dimensao_oculta = dimensao_oculta
        self.rede: Optional[_RedeNeuralLSTM] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o treinamento da rede LSTM em PyTorch e projeta o horizonte futuro.

        Parâmetros:
            dados_completos: DataFrame histórico com cotações.
            horizonte_projecao: Dias futuros a projetar.
            eh_criptomoeda: Se True, considera finais de semana.
            funcao_progresso: Callback para atualização periódica.
            hiperparametros: Dicionário opcional com sobrescrita de parâmetros.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        epocas = int(config.get("numero_epocas", self.numero_epocas))
        taxa_lr = float(config.get("taxa_aprendizado", self.taxa_aprendizado))
        tamanho_oculto = int(config.get("dimensao_oculta", self.dimensao_oculta))

        if funcao_progresso:
            funcao_progresso(5, 100, "LSTM: Construindo tensores sequenciais...", 0.0)

        # 1. Preparação dos dados tabulares
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        # Converte a matriz de atributos para formato sequencial 3D [amostras, seq_len, 1]
        quantidade_amostras, tamanho_janela = matriz_atributos.shape
        tensores_x = torch.tensor(matriz_atributos, dtype=torch.float32).unsqueeze(-1)
        tensores_y = torch.tensor(vetor_alvo, dtype=torch.float32)

        # 2. Inicialização da rede neural, função de custo e otimizador
        self.rede = _RedeNeuralLSTM(
            dimensao_entrada=1,
            dimensao_oculta=tamanho_oculto,
            numero_camadas=2,
            taxa_dropout=0.15
        )
        self.rede.train()

        funcao_custo = nn.MSELoss()
        otimizador = optim.Adam(self.rede.parameters(), lr=taxa_lr, weight_decay=1e-5)

        historico_perdas: List[float] = []

        # 3. Loop de treinamento com retropropagação (backpropagation)
        for epoca in range(1, epocas + 1):
            otimizador.zero_grad()
            saidas_rede = self.rede(tensores_x)
            perda = funcao_custo(saidas_rede, tensores_y)
            perda.backward()

            # Corte de gradiente para evitar explosão (gradient clipping)
            nn.utils.clip_grad_norm_(self.rede.parameters(), max_norm=1.0)
            otimizador.step()

            valor_perda = float(perda.item())
            historico_perdas.append(valor_perda)

            if funcao_progresso and (epoca % max(1, epocas // 10) == 0 or epoca == epocas):
                progresso_pct = 10 + int((epoca / epocas) * 65)
                msg_status = f"LSTM: Época {epoca}/{epocas} | Perda MSE: {valor_perda:.5f}"
                funcao_progresso(progresso_pct, 100, msg_status, valor_perda)

        # 4. Modo de inferência
        self.rede.eval()
        with torch.no_grad():
            predicoes_norm = self.rede(tensores_x).numpy()

        precos_reais, precos_previstos_reais = self.desnormalizar_historico(
            dados_completos, len(matriz_atributos), predicoes_norm
        )

        residuos = precos_reais - precos_previstos_reais
        desvio_padrao_erros = float(np.std(residuos))

        # 5. Projeção recursiva futura
        def prever_vetor_lstm(v: np.ndarray) -> float:
            with torch.no_grad():
                t = torch.tensor(v.reshape(1, -1, 1), dtype=torch.float32)
                return float(self.rede(t).item())

        datas_futuras, previsoes_futuras_reais, limites_inf, limites_sup = self.projetar_futuro_recursivo(
            dados_completos=dados_completos,
            funcao_predicao_vetor=prever_vetor_lstm,
            horizonte_projecao=horizonte_projecao,
            desvio_padrao_erros=desvio_padrao_erros,
            eh_criptomoeda=eh_criptomoeda
        )

        # 8. DataFrames de saída
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

        # 9. Métricas
        metricas = self.calcular_metricas_estatisticas(
            precos_reais=precos_reais,
            precos_previstos=precos_previstos_reais,
            preco_real_final=float(precos_reais[-1]),
            preco_projetado_final=float(previsoes_futuras_reais[-1]),
            horizonte_dias=horizonte_projecao
        )

        self.historico_aprendizado = {
            "generations": list(range(1, len(historico_perdas) + 1)),
            "best_fitness": [round(1.0 / (p + 1e-4), 2) for p in historico_perdas],
            "avg_fitness": [round(1.0 / (np.mean(historico_perdas[:i+1]) + 1e-4), 2) for i in range(len(historico_perdas))],
            "perdas": historico_perdas
        }

        if funcao_progresso:
            funcao_progresso(100, 100, f"LSTM: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
