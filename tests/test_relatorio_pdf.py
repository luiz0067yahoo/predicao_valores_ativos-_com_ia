"""
Testes unitários para o módulo de geração de relatórios corporativos em PDF (relatorio_pdf.py).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from relatorio_pdf import GeradorRelatorioPDF


@pytest.fixture
def dados_teste_relatorio(tmp_path):
    """Cria dados sintéticos de teste para geração do PDF."""
    datas_hist = pd.date_range("2025-01-01", periods=30, freq="B")
    datas_proj = pd.date_range("2025-02-12", periods=5, freq="B")

    df_hist = pd.DataFrame({
        "Preco_Real": np.linspace(100.0, 110.0, 30),
        "Preco_Previsto_IA": np.linspace(99.5, 109.8, 30)
    }, index=datas_hist)

    df_proj = pd.DataFrame({
        "Preco_Projetado": [111.0, 112.5, 113.0, 114.2, 115.0],
        "Limite_Inferior": [108.0, 109.0, 109.5, 110.0, 110.5],
        "Limite_Superior": [114.0, 116.0, 117.0, 118.5, 120.0]
    }, index=datas_proj)

    metricas = {
        "ultimo_preco_real": 110.0,
        "preco_projetado_final": 115.0,
        "variacao_esperada_pct": 4.55,
        "tendencia_esperada": "Alta Estimada (Bullish)",
        "rmse": 0.45,
        "mae": 0.38,
        "mape": 0.35,
        "r2": 0.98,
        "acuracia_direcional": 88.5
    }

    resultados_predicao = {
        "metrics": metricas,
        "history_df": df_hist,
        "forecast_df": df_proj
    }

    historico_treino = {
        "generations": list(range(1, 11)),
        "best_fitness": [800.0 + i * 15 for i in range(10)],
        "avg_fitness": [400.0 + i * 10 for i in range(10)]
    }

    return resultados_predicao, historico_treino, tmp_path


def test_geracao_relatorio_pdf(dados_teste_relatorio):
    """Valida a criação do documento PDF com gráficos e tabelas embutidos."""
    resultados_predicao, historico_treino, pasta_temp = dados_teste_relatorio

    # 1. Exporta gráficos
    caminhos_graficos = GeradorRelatorioPDF.exportar_graficos_analiticos(
        resultados_predicao=resultados_predicao,
        historico_treinamento=historico_treino,
        nome_ativo="Dólar (USD/BRL)",
        moeda="BRL",
        diretorio_saida=pasta_temp
    )

    assert "grafico_predicao" in caminhos_graficos
    assert "grafico_convergencia" in caminhos_graficos
    assert os.path.exists(caminhos_graficos["grafico_predicao"])
    assert os.path.exists(caminhos_graficos["grafico_convergencia"])

    # 2. Gera PDF
    caminho_pdf_destino = str(pasta_temp / "teste_relatorio.pdf")
    caminho_gerado = GeradorRelatorioPDF.gerar_relatorio_pdf(
        nome_ativo="Dólar (USD/BRL)",
        ticker="USDBRL=X",
        moeda="BRL",
        nome_algoritmo="XGBoost",
        periodo_escolhido="1 Ano",
        parametros_algoritmo={"numero_estimadores": 100},
        resultados_predicao=resultados_predicao,
        historico_treinamento=historico_treino,
        caminhos_graficos=caminhos_graficos,
        caminho_arquivo_saida=caminho_pdf_destino
    )

    assert os.path.exists(caminho_gerado)
    tamanho_bytes = os.path.getsize(caminho_gerado)
    assert tamanho_bytes > 20000  # PDF completo com imagens deve ter tamanho razoável (>20KB)
