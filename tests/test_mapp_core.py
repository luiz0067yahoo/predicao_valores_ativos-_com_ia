"""
Testes unitários para os componentes centrais do M.A.P.P.:
- ForecastHorizon & NormalizadorHorizonte
- ProgressTracker (HH:MM:SS)
- MarketRegimeDetector (8 regimes)
- ValidadorAntiVazamento & SeletorCaracteristicas
"""

import time
import pytest
import numpy as np
import pandas as pd
from mapp.horizon import ForecastHorizon, NormalizadorHorizonte
from mapp.tracker import ProgressTracker
from mapp.market_regime import MarketRegimeDetector, TipoRegimeMercado, MarketRegime
from mapp.selection import SeletorCaracteristicas, ValidadorAntiVazamento


def test_horizon_normalization_equities():
    """Testa a normalização de horizontes para ações de bolsa tradicional (dias úteis)."""
    h_dias = NormalizadorHorizonte.normalizar(10, 'dias', eh_criptomoeda=False)
    assert h_dias.periodos_normalizados == 10
    assert h_dias.eh_criptomoeda is False

    h_sem = NormalizadorHorizonte.normalizar(2, 'semanas', eh_criptomoeda=False)
    assert h_sem.periodos_normalizados == 10  # 2 semanas * 5 dias úteis

    h_mes = NormalizadorHorizonte.normalizar(1, 'meses', eh_criptomoeda=False)
    assert h_mes.periodos_normalizados == 21  # 21 dias úteis em média por mês

    h_ano = NormalizadorHorizonte.normalizar(1, 'anos', eh_criptomoeda=False)
    assert h_ano.periodos_normalizados == 252  # 252 dias úteis


def test_horizon_normalization_crypto():
    """Testa a normalização de horizontes para criptoativos (24/7 - dias corridos)."""
    h_sem = NormalizadorHorizonte.normalizar(2, 'semanas', eh_criptomoeda=True)
    assert h_sem.periodos_normalizados == 14  # 2 * 7 dias corridos
    assert h_sem.eh_criptomoeda is True

    h_mes = NormalizadorHorizonte.normalizar(1, 'meses', eh_criptomoeda=True)
    assert h_mes.periodos_normalizados == 30  # 30 dias corridos

    h_ano = NormalizadorHorizonte.normalizar(1, 'anos', eh_criptomoeda=True)
    assert h_ano.periodos_normalizados == 365


def test_progress_tracker_formatting():
    """Testa o monitor de tempo real e a formatação HH:MM:SS."""
    tracker = ProgressTracker(total_passos=10, nome_tarefa="Teste")
    tracker.iniciar()

    time.sleep(0.05)
    snap = tracker.atualizar(passo_atual=2, mensagem="Processando", algoritmo_atual="ModeloTeste")

    assert snap.passo_atual == 2
    assert snap.total_passos == 10
    assert snap.porcentagem == 20.0
    assert snap.algoritmo_atual == "ModeloTeste"
    # Formato HH:MM:SS tem dois ':'
    assert snap.tempo_decorrido_str.count(":") == 2
    assert snap.tempo_restante_str.count(":") == 2
    assert snap.tempo_total_estimado_str.count(":") == 2

    # Finalização
    final_snap = tracker.concluir(mensagem="Finalizado com sucesso")
    assert final_snap.porcentagem == 100.0
    assert final_snap.tempo_restante_seg == 0.0


def test_market_regime_detector():
    """Testa a classificação dos 8 regimes de mercado e a extração de features."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    
    # Série sintética com tendência altista
    trend = np.linspace(100, 180, n)
    noise = np.random.normal(0, 1.5, n)
    close = trend + noise
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    open_ = close + np.random.uniform(-0.5, 0.5, n)
    volume = np.random.uniform(1e5, 5e5, n)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    # Teste de detecção individual na barra mais recente
    resultado = MarketRegimeDetector.detectar(df)
    assert isinstance(resultado.regime_primario, TipoRegimeMercado)
    assert 0.0 <= resultado.confianca <= 1.0
    assert len(resultado.vetor_features_regime) == 8

    # Teste de geração da série histórica de regimes
    serie_regimes = MarketRegimeDetector.gerar_serie_regimes(df, janela=30)
    assert len(serie_regimes) == n
    assert serie_regimes.name == "Market_Regime"


def test_anti_leakage_and_feature_selector():
    """Testa a validação contra vazamento de dados e o seletor de features."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        'f1': np.random.normal(0, 1, n),
        'f2': np.random.normal(0, 1, n),
        'f3_colinear': np.zeros(n),
        'target': np.random.choice([0, 1], n)
    })
    # f3 quase idêntica a f1 (alta colinearidade)
    df['f3_colinear'] = df['f1'] + np.random.normal(0, 0.001, n)

    selector = SeletorCaracteristicas(n_features_selecionar=2, limite_correlacao=0.85)
    X = df[['f1', 'f2', 'f3_colinear']]
    y = df['target']
    selected = selector.selecionar_features(X, y)

    # f3_colinear deve ter sido filtrada pela correlação com f1
    assert len(selected) <= 2
    assert not ('f1' in selected and 'f3_colinear' in selected)

    # Teste de perturbação anti-vazamento
    def dummy_feature_gen(data):
        return pd.DataFrame({'lag_1': data['target'].shift(1).fillna(0)})

    res_val = ValidadorAntiVazamento.verificar_ausencia_vazamento(df, dummy_feature_gen, coluna_alvo='target')
    assert res_val['sem_vazamento'] is True
