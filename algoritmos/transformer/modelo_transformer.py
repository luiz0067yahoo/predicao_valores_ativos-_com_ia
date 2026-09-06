"""
Modelo Transformer com Autoatenção Temporal (Self-Attention) para Séries Temporais Financeiras
Implementado em PyTorch, utiliza mecanismos de atenção multi-cabeça (Multi-Head Attention)
para capturar correlações de longo alcance e dinâmicas complexas de mercado.
"""

from typing import Any, Callable, Dict, List, Optional
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from algoritmos.base_algoritmo import BaseAlgoritmo
from data_fetcher import DataFetcher


class _CodificadorPosicional(nn.Module):
    """
    Adiciona informação de ordem temporal (posição) aos vetores de características.
    """

    def __init__(self, dimensao_modelo: int, tamanho_maximo: int = 200):
        super().__init__()
        # Cria matriz de codificação posicional com senos e cossenos
        pe = torch.zeros(tamanho_maximo, dimensao_modelo)
        posicao = torch.arange(0, tamanho_maximo, dtype=torch.float).unsqueeze(1)
        termo_divisao = torch.exp(torch.arange(0, dimensao_modelo, 2).float() * (-math.log(10000.0) / dimensao_modelo))

        pe[:, 0::2] = torch.sin(posicao * termo_divisao)
        pe[:, 1::2] = torch.cos(posicao * termo_divisao)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soma a codificação posicional aos embeddings da entrada
        return x + self.pe[:, :x.size(1)]


class _RedeNeuralTransformer(nn.Module):
    """
    Arquitetura completa do Transformer Encoder para regressão de séries temporais.
    """

    def __init__(
        self,
        dimensao_entrada: int = 1,
        dimensao_modelo: int = 32,
        numero_cabecas: int = 4,
        dimensao_feedforward: int = 64,
        numero_camadas: int = 2,
        taxa_dropout: float = 0.1
    ):
        super().__init__()
        # Projeção linear inicial para elevar a dimensão de entrada ao espaço do modelo
        self.projecao_entrada = nn.Linear(dimensao_entrada, dimensao_modelo)
        self.posicional = _CodificadorPosicional(dimensao_modelo=dimensao_modelo)

        # Bloco de camadas Transformer Encoder do PyTorch
        camada_encoder = nn.TransformerEncoderLayer(
            d_model=dimensao_modelo,
            nhead=numero_cabecas,
            dim_feedforward=dimensao_feedforward,
            dropout=taxa_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(camada_encoder, num_layers=numero_camadas)

        # Camadas finais de regressão
        self.camada_densa = nn.Linear(dimensao_modelo, 16)
        self.ativacao = nn.ReLU()
        self.camada_saida = nn.Linear(16, 1)

    def forward(self, tensores_entrada: torch.Tensor) -> torch.Tensor:
        """
        Executa o fluxo de atenção temporal e projeta o preço futuro.
        """
        x = self.projecao_entrada(tensores_entrada)
        x = self.posicional(x)
        saida_atencao = self.transformer_encoder(x)

        # Média ponderada temporal (global average pooling no eixo da sequência)
        representacao_temporal = saida_atencao.mean(dim=1)

        oculto = self.ativacao(self.camada_densa(representacao_temporal))
        return self.camada_saida(oculto).squeeze(-1)


class ModeloTransformer(BaseAlgoritmo):
    """
    Controlador do modelo Transformer para previsão financeira com autoatenção.
    """

    def __init__(
        self,
        janela_temporal: int = 10,
        numero_epocas: int = 40,
        taxa_aprendizado: float = 0.005,
        dimensao_modelo: int = 32
    ):
        """
        Inicializa o algoritmo Transformer.

        Parâmetros:
            janela_temporal: Tamanho da janela temporal passada.
            numero_epocas: Épocas de treinamento da atenção.
            taxa_aprendizado: Taxa de aprendizado do Adam.
            dimensao_modelo: Dimensionalidade dos vetores de atenção.
        """
        super().__init__(nome_identificador="Transformer", janela_temporal=janela_temporal)
        self.numero_epocas = numero_epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.dimensao_modelo = dimensao_modelo
        self.rede: Optional[_RedeNeuralTransformer] = None

    def treinar_e_projetar(
        self,
        dados_completos: pd.DataFrame,
        horizonte_projecao: int = 5,
        eh_criptomoeda: bool = False,
        funcao_progresso: Optional[Callable[[int, int, str, float], None]] = None,
        hiperparametros: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Treina o modelo Transformer em PyTorch e realiza projeção futura recursiva.

        Parâmetros:
            dados_completos: DataFrame histórico com cotações.
            horizonte_projecao: Quantidade de dias para projeção.
            eh_criptomoeda: Flag de negociação contínua aos sábados/domingos.
            funcao_progresso: Callback de monitoramento.
            hiperparametros: Parâmetros opcionais de configuração.

        Retorno:
            Dicionário com métricas estatísticas e DataFrames.
        """
        config = hiperparametros or {}
        epocas = int(config.get("numero_epocas", self.numero_epocas))
        taxa_lr = float(config.get("taxa_aprendizado", self.taxa_aprendizado))
        dim_modelo = int(config.get("dimensao_modelo", self.dimensao_modelo))

        if funcao_progresso:
            funcao_progresso(5, 100, "Transformer: Preparando tensores com autoatenção temporal...", 0.0)

        # 1. Preparação de matriz de atributos defasados
        coletor = DataFetcher()
        matriz_atributos, vetor_alvo, datas_alvo, parametros_escala = coletor.prepare_lagged_features(
            dados_completos, lookback=self.janela_temporal
        )

        # Converte para tensores 3D [amostras, seq_len, 1]
        tensores_x = torch.tensor(matriz_atributos, dtype=torch.float32).unsqueeze(-1)
        tensores_y = torch.tensor(vetor_alvo, dtype=torch.float32)

        # 2. Inicialização do Transformer
        self.rede = _RedeNeuralTransformer(
            dimensao_entrada=1,
            dimensao_modelo=dim_modelo,
            numero_cabecas=4,
            dimensao_feedforward=64,
            numero_camadas=2,
            taxa_dropout=0.1
        )
        self.rede.train()

        funcao_custo = nn.MSELoss()
        otimizador = optim.Adam(self.rede.parameters(), lr=taxa_lr, weight_decay=1e-5)

        historico_perdas: List[float] = []

        # 3. Loop de treinamento
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
                msg = f"Transformer: Época {epoca}/{epocas} | Perda MSE: {val_perda:.5f}"
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
        def prever_vetor_transformer(v: np.ndarray) -> float:
            with torch.no_grad():
                t = torch.tensor(v.reshape(1, -1, 1), dtype=torch.float32)
                return float(self.rede(t).item())

        datas_futuras, previsoes_futuras_reais, limites_inf, limites_sup = self.projetar_futuro_recursivo(
            dados_completos=dados_completos,
            funcao_predicao_vetor=prever_vetor_transformer,
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
            funcao_progresso(100, 100, f"Transformer: Concluído! Tendência: {metricas['tendencia_esperada']}", metricas["rmse"])

        return {
            "metrics": metricas,
            "history_df": df_historico_saida,
            "forecast_df": df_projecao_saida,
            "ga_history": self.historico_aprendizado
        }
