"""
Testes unitários para os novos algoritmos (MAPP, Ensemble, Pattern Matching)
e verificação do catálogo de 16 algoritmos na fábrica.
"""

import numpy as np
import pandas as pd
import pytest
from algoritmos.fabrica_algoritmos import FabricaAlgoritmos
from algoritmos.mapp.modelo_mapp import ModeloMAPP
from algoritmos.ensemble.modelo_ensemble import ModeloEnsemble
from algoritmos.pattern_matching.modelo_pattern_matching import ModeloPatternMatching


@pytest.fixture
def sample_dataset():
    """Gera dataset para treino e teste de algoritmos."""
    np.random.seed(42)
    n = 180
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, n))
    high = close + np.random.uniform(0.5, 1.5, n)
    low = close - np.random.uniform(0.5, 1.5, n)
    open_ = close + np.random.uniform(-0.5, 0.5, n)
    volume = np.random.uniform(1e5, 5e5, n)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    return df


def test_catalog_16_algorithms():
    """Garante que todos os 16 algoritmos estão devidamente registrados na fábrica."""
    catalog = FabricaAlgoritmos.listar_algoritmos_disponiveis()
    assert len(catalog) == 16

    ids = [item['identificador'] for item in catalog]
    assert 'mapp' in ids
    assert 'ensemble' in ids
    assert 'pattern_matching' in ids
    assert 'xgboost' in ids
    assert 'rede_neural' in ids
    assert 'logica_fuzzy' in ids


def test_modelo_mapp(sample_dataset):
    """Testa treino e projeção do modelo MAPP com saída probabilística."""
    modelo = ModeloMAPP(janela_temporal=10)
    resultado = modelo.treinar_e_projetar(
        dados_completos=sample_dataset,
        horizonte_projecao=5,
        hiperparametros={"n_features": 10, "confidence_level": 0.95}
    )

    assert "forecast_df" in resultado
    assert len(resultado["forecast_df"]) == 5
    assert "metricas" in resultado
    assert "probabilidade_alta" in resultado
    assert "direcao_predita" in resultado
    assert "regime_detectado" in resultado


def test_modelo_ensemble(sample_dataset):
    """Testa treino e projeção do modelo Ensemble."""
    modelo = ModeloEnsemble(janela_temporal=10)
    resultado = modelo.treinar_e_projetar(
        dados_completos=sample_dataset,
        horizonte_projecao=5,
        hiperparametros={"mode": "regime_adaptive", "top_models": 3}
    )

    assert "forecast_df" in resultado
    assert len(resultado["forecast_df"]) == 5
    assert "metricas" in resultado


def test_modelo_pattern_matching(sample_dataset):
    """Testa treino e projeção do modelo Pattern Matching."""
    modelo = ModeloPatternMatching(janela_temporal=10)
    resultado = modelo.treinar_e_projetar(
        dados_completos=sample_dataset,
        horizonte_projecao=5,
        hiperparametros={"window_size": 15, "top_k_matches": 3}
    )

    assert "forecast_df" in resultado
    assert len(resultado["forecast_df"]) == 5
    assert "metricas" in resultado
