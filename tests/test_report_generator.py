"""
Testes Unitários para Geração de Relatórios DOCX e Planilhamento de Ensaios
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from report_generator import ReportGenerator
from config import BASE_DIR, REPORTS_DIR, LOGS_DIR


def test_log_trial(tmp_path):
    trial_data = {
        "Data_Hora": "2024-01-01 10:00:00",
        "Ativo": "Dólar (USD/BRL)",
        "Ticker": "USDBRL=X",
        "Moeda": "BRL",
        "Periodo": "1y",
        "Populacao": 50,
        "Geracoes": 20,
        "Taxa_Mutacao": 0.15,
        "Taxa_Crossover": 0.85,
        "Lookback": 10,
        "Horizonte_Dias": 5,
        "Ultimo_Preco_Real": 5.42,
        "Preco_Projetado_Final": 5.48,
        "Variacao_Esperada_Pct": 1.10,
        "Tendencia": "Alta (Subida)",
        "RMSE": 0.045,
        "MAE": 0.035,
        "MAPE_Pct": 0.65,
        "R2": 0.92,
        "Acuracia_Direcional_Pct": 68.5
    }

    log_file = ReportGenerator.log_trial(trial_data)
    assert os.path.exists(log_file)

    # Verifica se os dados foram gravados lendo com pandas
    df = pd.read_csv(log_file)
    assert len(df) >= 1
    assert "USDBRL=X" in df["Ticker"].values


def test_generate_docx_report_with_charts(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    future_dates = pd.date_range("2024-02-15", periods=5, freq="B")

    df_hist = pd.DataFrame({
        "Preco_Real": 100.0 + np.linspace(0, 10, len(dates)),
        "Preco_Previsto_IA": 100.0 + np.linspace(0, 9.8, len(dates)),
        "Erro_Absoluto": np.abs(np.random.normal(0, 0.5, len(dates))),
        "Erro_Percentual": np.abs(np.random.normal(0, 0.5, len(dates)))
    }, index=dates)

    df_fore = pd.DataFrame({
        "Preco_Projetado": [110.5, 111.0, 111.4, 111.8, 112.3],
        "Limite_Inferior": [109.0, 109.2, 109.5, 109.8, 110.0],
        "Limite_Superior": [112.0, 112.8, 113.3, 113.8, 114.5]
    }, index=future_dates)

    prediction_results = {
        "metrics": {
            "ultimo_preco_real": 110.0,
            "preco_projetado_final": 112.3,
            "variacao_esperada_pct": 2.09,
            "tendencia_esperada": "Alta (Subida)",
            "horizonte_dias": 5,
            "rmse": 0.45,
            "mae": 0.38,
            "mape": 0.35,
            "r2": 0.94,
            "acuracia_direcional": 72.0
        },
        "history_df": df_hist,
        "forecast_df": df_fore
    }

    ga_history = {
        "generation": list(range(1, 11)),
        "best_fitness": [200.0 + i * 15 for i in range(10)],
        "avg_fitness": [150.0 + i * 10 for i in range(10)]
    }

    # Exporta gráficos
    charts = ReportGenerator.export_charts_for_report(
        prediction_results=prediction_results,
        ga_history=ga_history,
        asset_name="Dólar Teste",
        currency="BRL",
        temp_dir=tmp_path
    )

    assert os.path.exists(charts["prediction_chart"])
    assert os.path.exists(charts["convergence_chart"])

    # Gera DOCX
    out_docx = str(tmp_path / "test_relatorio.docx")
    ga_config = {
        "population_size": 40,
        "generations": 10,
        "crossover_rate": 0.85,
        "mutation_rate": 0.15,
        "elitism_ratio": 0.08,
        "lookback_window": 5
    }

    docx_path = ReportGenerator.generate_docx_report(
        asset_name="Dólar Teste",
        ticker="USDBRL=X",
        currency="BRL",
        period_str="1y",
        ga_config=ga_config,
        prediction_results=prediction_results,
        ga_history=ga_history,
        chart_paths=charts,
        output_filepath=out_docx
    )

    assert os.path.exists(docx_path)
    assert os.path.getsize(docx_path) > 10000  # O documento com imagens deve ter tamanho razoável (> 10KB)
