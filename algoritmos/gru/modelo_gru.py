"""
Modelo GRU (Gated Recurrent Unit) para Previsão de Séries Temporais Financeiras
Implementado em PyTorch, utiliza arquitetura com portas de atualização e reset,
oferecendo convergência mais rápida que o LSTM com menor carga computacional.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class _RedeNeuralGRU(nn.Module):
    """
    Arquitetura de rede neural recorrente GRU com camadas empilhadas e projeção linear.
    """

    def __init__(self, dimensao_entrada: int, dimensao_oculta: int = 40, numero_camadas: int = 2, taxa_dropout: float = 0.1):
        """
        Inicializa a estrutura interna do módulo GRU.

        Parâmetros:
            dimensao_entrada: Número de variáveis preditoras por passo temporal.
            dimensao_oculta: Número de unidades ocultas por célula GRU.
            numero_camadas: Quantidade de blocos GRU empilhados verticalmente.
            taxa_dropout: Taxa de descarte estocástico para prevenção de sobreajuste.
        """
        super().__init__()
        self.camada_gru = nn.GRU(
            input_size=dimensao_entrada,
            hidden_size=dimensao_oculta,
            num_layers=numero_camadas,
            batch_first=True,
            dropout=taxa_dropout if numero_camadas > 1 else 0.0
        )
        self.camada_linear = nn.Linear(dimensao_oculta, 1)

    def forward(self, tensores_entrada: torch.Tensor) -> torch.Tensor:
        """
        Executa a passagem para frente (forward pass).

        Parâmetros:
            tensores_entrada: Tensor 3D no formato [batch, seq_len, features].

        Retorno:
            Tensor unidimensional com a estimativa do valor seguinte.
        """
        saida_gru, _ = self.camada_gru(tensores_entrada)
        ultimo_estado = saida_gru[:, -1, :]
        return self.camada_linear(ultimo_estado).squeeze(-1)


class ModeloGRU(BaseAlgoritmo):
    """
    Controlador do modelo GRU para projeção autorregressiva de ativos.
    """

    def __init__(
        self,
        janela_temporal: int = 10,
        numero_epocas: int = 40,
        taxa_aprendizado: float = 0.008,
        dimensao_oculta: int = 40
    ):
        """
        Inicializa os parâmetros do algoritmo GRU.

        Parâmetros:
            janela_temporal: Quantidade de dias anteriores analisados.
            numero_epocas: Ciclos de treinamento da rede.
            taxa_aprendizado: Taxa de passo do gradiente Adam.
            dimensao_oculta: Largura das camadas ocultas da célula GRU.
        """
        super().__init__(nome_identificador="GRU", janela_temporal=janela_temporal)
        self.numero_epocas = numero_epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.dimensao_oculta = dimensao_oculta
        self.rede: Optional[_RedeNeuralGRU] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Treina o estimador GRU em PyTorch e realiza projeções recursivas.

        Parâmetros:
            dados_completos: DataFrame histórico.
            horizonte_projecao: Quantidade de passos à frente.
            eh_criptomoeda: Flag para considerar fins de semana.
            funcao_progresso: Callback de atualização da interface.
            hiperparametros: Parâmetros opcionais de ajuste fino.

        Retorno:
            Dicionário com métricas e DataFrames de projeção.
        """
        config = hiperparametros or {}
        epocas = int(config.get("numero_epocas", self.numero_epocas))
        taxa_lr = float(config.get("taxa_aprendizado", self.taxa_aprendizado))
        tamanho_oculto = int(config.get("dimensao_oculta", self.dimensao_oculta))

        if funcao_progresso:
            funcao_progresso(5, 100, "GRU: Estruturando tensores temporais...", 0.0)

        # 1. Obtenção de matriz de defasagens
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        # Converte para formato 3D
        tensores_x = torch.tensor(matriz_atributos, dtype=torch.float32).unsqueeze(-1)
        tensores_y = torch.tensor(vetor_alvo, dtype=torch.float32)

        # 2. Inicialização da rede neural GRU
        self.rede = _RedeNeuralGRU(
            dimensao_entrada=1,
            dimensao_oculta=tamanho_oculto,
            numero_camadas=2,
            taxa_dropout=0.1
        )
        self.rede.train()

        funcao_custo = nn.MSELoss()
        otimizador = optim.Adam(self.rede.parameters(), lr=taxa_lr, weight_decay=1e-5)

        historico_perdas: List[float] = []

        # 3. Ciclo de treinamento
        for epoca in range(1, epocas + 1):
            otimizador.zero_grad()
            predicoes = self.rede(tensores_x)
            perda = funcao_custo(predicoes, tensores_y)
            perda.backward()

            nn.utils.clip_grad_norm_(self.rede.parameters(), max_norm=1.0)
            otimizador.step()

            val_perda = float(perda.item())
            historico_perdas.append(val_perda)

            if funcao_progresso and (epoca % max(1, epocas // 10) == 0 or epoca == epocas):
                pct = 10 + int((epoca / epocas) * 65)
                msg = f"GRU: Época {epoca}/{epocas} | Perda MSE: {val_perda:.5f}"
                funcao_progresso(pct, 100, msg, val_perda)

        # 4. Avaliação in-sample
        self.rede.eval()
        with torch.no_grad():
            predicoes_norm = self.rede(tensores_x).numpy()

        precos_reais, precos_previstos_reais = self.desnormalizar_historico(
            dados_completos, len(matriz_atributos), predicoes_norm
        )

        residuos = precos_reais - precos_previstos_reais
        desvio_padrao_erros = float(np.std(residuos))

        # 5. Projeção recursiva autorregressiva
        def prever_vetor_gru(v: np.ndarray) -> float:
            with torch.no_grad():
                t = torch.tensor(v.reshape(1, -1, 1), dtype=torch.float32)
                return float(self.rede(t).item())

        datas_futuras, previsoes_futuras_reais, limites_inf, limites_sup = self.projetar_futuro_recursivo(
            dados_completos=dados_completos,
            funcao_predicao_vetor=prever_vetor_gru,
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

        # 9. Métricas estatísticas
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
            funcao_progresso(100, 100, f"GRU: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
