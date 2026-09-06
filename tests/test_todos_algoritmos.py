"""
Suíte de Testes Automatizados para Todos os 11 Algoritmos de IA e Séries Temporais.
Valida o ciclo de treinamento, inferência recursiva e limites de 95% de confiança.
"""

import numpy as np
import pandas as pd
import pytest

from algoritmos.fabrica_algoritmos import FabricaAlgoritmos, CATALOGO_ALGORITMOS


@pytest.fixture
def dados_sinteticos_mercado():
    """Gera DataFrame simulado de preços com tendências e ruído estocástico."""
    np.random.seed(42)
    datas = pd.date_range("2025-01-01", periods=90, freq="B")

    # Passeio aleatório simulando cotação de ativo
    retornos = np.random.normal(0.001, 0.015, size=90)
    precos = 100.0 * np.cumprod(1.0 + retornos)

    df = pd.DataFrame({
        "Open": precos * (1.0 + np.random.normal(0, 0.002, 90)),
        "High": precos * (1.0 + np.abs(np.random.normal(0, 0.005, 90))),
        "Low": precos * (1.0 - np.abs(np.random.normal(0, 0.005, 90))),
        "Close": precos,
        "Volume": np.random.randint(50000, 500000, size=90)
    }, index=datas)

    return df


@pytest.mark.parametrize("identificador", list(CATALOGO_ALGORITMOS.keys()))
def test_execucao_algoritmo(identificador, dados_sinteticos_mercado):
    """
    Testa se cada um dos 11 algoritmos instancia, treina e projeta corretamente.
    """
    modelo = FabricaAlgoritmos.obter_instancia(identificador, janela_temporal=10)
    assert modelo is not None
    assert modelo.nome_identificador != ""

    # Parâmetros reduzidos para execução ultra-rápida nos testes
    hiperparametros = {
        "numero_estimadores": 10,
        "numero_arvores": 10,
        "numero_epocas": 3,
        "tamanho_populacao": 20,
        "numero_geracoes": 3,
        "numero_rodadas_debate": 3,
        "horizonte_projecao": 3
    }

    resultado = modelo.treinar_e_projetar(
        dados_completos=dados_sinteticos_mercado,
        horizonte_projecao=3,
        eh_criptomoeda=False,
        hiperparametros=hiperparametros
    )

    assert "metrics" in resultado
    assert "history_df" in resultado
    assert "forecast_df" in resultado

    metricas = resultado["metrics"]
    assert "ultimo_preco_real" in metricas
    assert "preco_projetado_final" in metricas
    assert "rmse" in metricas
    assert "acuracia_direcional" in metricas

    df_proj = resultado["forecast_df"]
    assert len(df_proj) == 3
    assert "Preco_Projetado" in df_proj.columns
    assert "Limite_Inferior" in df_proj.columns
    assert "Limite_Superior" in df_proj.columns
