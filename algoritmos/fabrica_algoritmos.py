"""
Fábrica e Gerenciador Central de Algoritmos Preditivos (fabrica_algoritmos.py)
Registra todos os algoritmos disponíveis, fornece metadados de desempenho,
e instancia dinamicamente o modelo selecionado pelo usuário.
Todos os nomes de arquivos, variáveis e comentários estão em português.
"""

from typing import Any, Dict, List, Type
from algoritmos.base_algoritmo import BaseAlgoritmo
from algoritmos.xgboost.modelo_xgboost import ModeloXGBoost
from algoritmos.lightgbm.modelo_lightgbm import ModeloLightGBM
from algoritmos.random_forest.modelo_random_forest import ModeloRandomForest
from algoritmos.lstm.modelo_lstm import ModeloLSTM
from algoritmos.gru.modelo_gru import ModeloGRU
from algoritmos.transformer.modelo_transformer import ModeloTransformer
from algoritmos.arima_sarima.modelo_arima_sarima import ModeloARIMA
from algoritmos.prophet.modelo_prophet import ModeloProphet
from algoritmos.regressao_linear.modelo_regressao_linear import ModeloRegressaoLinear
from algoritmos.algoritmo_genetico.modelo_algoritmo_genetico import ModeloAlgoritmoGenetico
from algoritmos.mirofish.simulador_mirofish import SimuladorMiroFish
from algoritmos.rede_neural.modelo_rede_neural import ModeloRedeNeural
from algoritmos.logica_fuzzy.modelo_logica_fuzzy import ModeloLogicaFuzzy


# Catálogo com todos os algoritmos suportados, seus metadados e parâmetros padrão
CATALOGO_ALGORITMOS: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "identificador": "xgboost",
        "nome": "XGBoost",
        "classe": ModeloXGBoost,
        "desempenho": "⭐⭐⭐⭐⭐",
        "uso": "Previsão de retorno/preço com múltiplos indicadores e defasagens não lineares",
        "icone": "⚡",
        "categoria": "Gradient Boosting",
        "parametros_padrao": {
            "numero_estimadores": 120,
            "taxa_aprendizado": 0.05,
            "profundidade_maxima": 4
        }
    },
    "lightgbm": {
        "identificador": "lightgbm",
        "nome": "LightGBM",
        "classe": ModeloLightGBM,
        "desempenho": "⭐⭐⭐⭐⭐",
        "uso": "Similar ao XGBoost, ultrarrápido com otimização baseada em histogramas",
        "icone": "🚀",
        "categoria": "Gradient Boosting",
        "parametros_padrao": {
            "numero_estimadores": 100,
            "taxa_aprendizado": 0.05,
            "numero_folhas": 31
        }
    },
    "random_forest": {
        "identificador": "random_forest",
        "nome": "Random Forest",
        "classe": ModeloRandomForest,
        "desempenho": "⭐⭐⭐⭐",
        "uso": "Ensemble de árvores com alta resiliência a ruído e dados não lineares",
        "icone": "🌲",
        "categoria": "Ensemble Learning",
        "parametros_padrao": {
            "numero_arvores": 150,
            "profundidade_maxima": 8
        }
    },
    "lstm": {
        "identificador": "lstm",
        "nome": "LSTM (Long Short-Term Memory)",
        "classe": ModeloLSTM,
        "desempenho": "⭐⭐⭐⭐",
        "uso": "Séries temporais com dependências sequenciais e memória de longo prazo",
        "icone": "🧠",
        "categoria": "Rede Neural Recorrente",
        "parametros_padrao": {
            "numero_epocas": 40,
            "taxa_aprendizado": 0.008,
            "dimensao_oculta": 48
        }
    },
    "gru": {
        "identificador": "gru",
        "nome": "GRU (Gated Recurrent Unit)",
        "classe": ModeloGRU,
        "desempenho": "⭐⭐⭐⭐",
        "uso": "Alternativa mais leve à LSTM, menor custo computacional e convergência rápida",
        "icone": "⚡",
        "categoria": "Rede Neural Recorrente",
        "parametros_padrao": {
            "numero_epocas": 35,
            "taxa_aprendizado": 0.008,
            "dimensao_oculta": 40
        }
    },
    "transformer": {
        "identificador": "transformer",
        "nome": "Transformer (Autoatenção Temporal)",
        "classe": ModeloTransformer,
        "desempenho": "⭐⭐⭐⭐⭐",
        "uso": "Séries temporais complexas com mecanismo de Multi-Head Self-Attention",
        "icone": "🔮",
        "categoria": "Deep Learning / Atenção",
        "parametros_padrao": {
            "numero_epocas": 35,
            "taxa_aprendizado": 0.005,
            "dimensao_modelo": 32
        }
    },
    "arima_sarima": {
        "identificador": "arima_sarima",
        "nome": "ARIMA / SARIMA",
        "classe": ModeloARIMA,
        "desempenho": "⭐⭐⭐",
        "uso": "Modelagem estatística clássica autoregressiva integrada de médias móveis",
        "icone": "📐",
        "categoria": "Estatística Clássica",
        "parametros_padrao": {
            "ordem_p": 2,
            "ordem_d": 1,
            "ordem_q": 2
        }
    },
    "prophet": {
        "identificador": "prophet",
        "nome": "Prophet (Meta/Facebook)",
        "classe": ModeloProphet,
        "desempenho": "⭐⭐",
        "uso": "Decomposição aditiva de tendência não-linear, sazonalidade e feriados",
        "icone": "📈",
        "categoria": "Decomposição Aditiva",
        "parametros_padrao": {
            "flexibilidade_tendencia": 0.05,
            "flexibilidade_sazonal": 10.0
        }
    },
    "regressao_linear": {
        "identificador": "regressao_linear",
        "nome": "Regressão Linear (Ridge)",
        "classe": ModeloRegressaoLinear,
        "desempenho": "⭐⭐",
        "uso": "Baseline paramétrico simples, interpretação direta e execução instantânea",
        "icone": "📏",
        "categoria": "Linear / Baseline",
        "parametros_padrao": {
            "forca_regularizacao": 1.0
        }
    },
    "algoritmo_genetico": {
        "identificador": "algoritmo_genetico",
        "nome": "Algoritmo Genético Evolutivo",
        "classe": ModeloAlgoritmoGenetico,
        "desempenho": "⭐⭐⭐⭐",
        "uso": "Otimização evolutiva global de pesos e osciladores técnicos com seleção e mutação",
        "icone": "🧬",
        "categoria": "Computação Evolutiva",
        "parametros_padrao": {
            "tamanho_populacao": 60,
            "numero_geracoes": 35,
            "taxa_mutacao": 0.15,
            "taxa_crossover": 0.85
        }
    },
    "mirofish": {
        "identificador": "mirofish",
        "nome": "MiroFish (Enxame de Agentes)",
        "classe": SimuladorMiroFish,
        "desempenho": "⭐⭐⭐⭐⭐",
        "uso": "Simulação de enxame de agentes autônomos com debate e consenso emergente",
        "icone": "🐟",
        "categoria": "Inteligência de Enxame",
        "parametros_padrao": {
            "numero_rodadas_debate": 15
        }
    },
    "rede_neural": {
        "identificador": "rede_neural",
        "nome": "Rede Neural (MLP)",
        "classe": ModeloRedeNeural,
        "desempenho": "⭐⭐⭐⭐⭐",
        "uso": "Rede Neural Profunda feedforward com camadas densas e Dropout",
        "icone": "🧠",
        "categoria": "Deep Learning",
        "parametros_padrao": {
            "numero_epocas": 45,
            "taxa_aprendizado": 0.005,
            "dimensao_camada_1": 64,
            "dimensao_camada_2": 32,
            "funcao_ativacao": "relu"
        }
    },
    "logica_fuzzy": {
        "identificador": "logica_fuzzy",
        "nome": "Lógica Fuzzy (TSK)",
        "classe": ModeloLogicaFuzzy,
        "desempenho": "⭐⭐⭐⭐",
        "uso": "Sistema de Inferência Difusa Takagi-Sugeno com funções gaussianas e centroide",
        "icone": "🌊",
        "categoria": "Lógica Difusa",
        "parametros_padrao": {
            "numero_conjuntos_fuzzy": 5,
            "largura_pertinencia": 1.0,
            "forca_regularizacao": 0.5
        }
    }
}


class FabricaAlgoritmos:
    """
    Fábrica estática para consulta de metadados e criação de instâncias de algoritmos.
    """

    @staticmethod
    def listar_algoritmos() -> List[Dict[str, Any]]:
        """
        Retorna uma lista resumida com metadados para preenchimento de menus na interface web.
        """
        lista: List[Dict[str, Any]] = []
        for ident, info in CATALOGO_ALGORITMOS.items():
            lista.append({
                "identificador": ident,
                "nome": info["nome"],
                "desempenho": info["desempenho"],
                "uso": info["uso"],
                "icone": info["icone"],
                "categoria": info["categoria"],
                "parametros_padrao": info["parametros_padrao"]
            })
        return lista

    @staticmethod
    def obter_instancia(identificador: str, janela_temporal: int = 10) -> BaseAlgoritmo:
        """
        Instancia a classe do algoritmo solicitado a partir do identificador.

        Parâmetros:
            identificador: Nome chave do algoritmo (ex: 'xgboost', 'lstm', 'mirofish').
            janela_temporal: Tamanho da janela de lags passados.

        Retorno:
            Instância que herda de BaseAlgoritmo.
        """
        chave = identificador.lower().strip()
        if chave not in CATALOGO_ALGORITMOS:
            # Fallback seguro para o algoritmo genético se chave não for encontrada
            chave = "algoritmo_genetico"

        classe_algoritmo: Type[BaseAlgoritmo] = CATALOGO_ALGORITMOS[chave]["classe"]
        return classe_algoritmo(janela_temporal=janela_temporal)
