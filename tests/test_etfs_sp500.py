"""
Testes Unitários para os 5 ETFs do S&P 500 (IVVB11, SPXI11, SPXB11, SPXR11, SPBZ11)
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

from config import ASSETS, DB_DIR
from data_fetcher import DataFetcher
from web_app import app, _parse_train_params


ETFS_ESPERADOS = [
    {
        "nome": "IVVB11 (iShares S&P 500)",
        "ticker": "IVVB11.SA",
        "codigo": "IVVB11",
        "excel": "etf_ivvb11.xls",
        "cambio": "Exposto ao dólar",
        "sp500": "Sim"
    },
    {
        "nome": "SPXI11 (IT NOW S&P500 TRN)",
        "ticker": "SPXI11.SA",
        "codigo": "SPXI11",
        "excel": "etf_spxi11.xls",
        "cambio": "Exposto ao dólar",
        "sp500": "Sim"
    },
    {
        "nome": "SPXB11 (BTG Pactual S&P 500)",
        "ticker": "SPXB11.SA",
        "codigo": "SPXB11",
        "excel": "etf_spxb11.xls",
        "cambio": "Exposto ao dólar",
        "sp500": "Sim"
    },
    {
        "nome": "SPXR11 (IT NOW S&P 500 Futures Quanto)",
        "ticker": "SPXR11.SA",
        "codigo": "SPXR11",
        "excel": "etf_spxr11.xls",
        "cambio": "Protegido do dólar",
        "sp500": "S&P 500 Futures"
    },
    {
        "nome": "SPBZ11 (BTG Pactual S&P 500 Futures Quanto)",
        "ticker": "SPBZ11.SA",
        "codigo": "SPBZ11",
        "excel": "etf_spbz11.xls",
        "cambio": "Protegido do dólar",
        "sp500": "S&P 500 Futures"
    }
]


def test_etfs_cadastrados_em_assets():
    """Valida se todos os 5 ETFs estão corretamente configurados no dicionário ASSETS."""
    for etf in ETFS_ESPERADOS:
        nome = etf["nome"]
        assert nome in ASSETS, f"ETF {nome} não encontrado em ASSETS"
        info = ASSETS[nome]
        assert info["ticker"] == etf["ticker"]
        assert info["category"] == "ETFs (S&P 500)"
        assert info["currency"] == "BRL"
        assert info["cambio"] == etf["cambio"]
        assert info["sp500"] == etf["sp500"]


def test_obter_caminho_arquivo_excel_etfs():
    """Valida se a rota canônica para arquivos Excel dos ETFs está correta."""
    for etf in ETFS_ESPERADOS:
        # Com sufixo .SA
        caminho_sa = DataFetcher.obter_caminho_arquivo_excel(etf["ticker"])
        assert caminho_sa.name == etf["excel"]
        assert caminho_sa.parent == DB_DIR

        # Apenas código puro
        caminho_puro = DataFetcher.obter_caminho_arquivo_excel(etf["codigo"])
        assert caminho_puro.name == etf["excel"]


def test_parse_train_params_resolucao_flexivel():
    """Valida se _parse_train_params resolve os ETFs tanto pelo nome completo, ticker ou código."""
    for etf in ETFS_ESPERADOS:
        # 1. Pelo nome completo
        res1 = _parse_train_params({"asset": etf["nome"]})
        assert res1["ticker"] == etf["ticker"]
        assert res1["moeda"] == "BRL"

        # 2. Pelo ticker com .SA
        res2 = _parse_train_params({"asset": etf["ticker"]})
        assert res2["ticker"] == etf["ticker"]
        assert res2["moeda"] == "BRL"

        # 3. Pelo código curto (ex: IVVB11)
        res3 = _parse_train_params({"asset": etf["codigo"]})
        assert res3["ticker"] == etf["ticker"]
        assert res3["moeda"] == "BRL"


def test_api_config_retorna_etfs():
    """Valida se o endpoint Flask /api/config expõe os ETFs adicionados."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/api/config")
        assert resp.status_code == 200
        dados = resp.get_json()
        assert "assets" in dados
        for etf in ETFS_ESPERADOS:
            assert etf["nome"] in dados["assets"]
            info = dados["assets"][etf["nome"]]
            assert info["category"] == "ETFs (S&P 500)"
            assert info["currency"] == "BRL"
            assert info["cambio"] == etf["cambio"]


def test_simulador_executar_resolucao_etf():
    """Valida se a rota do simulador aceita os ETFs e inicia a simulação com sucesso."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        with patch("threading.Thread.start"):
            resp = client.post("/api/simulator/run", json={
                "asset": "IVVB11 (iShares S&P 500)",
                "capital": 5000,
                "period": "1y",
                "algorithms": ["mapp"]
            })
            assert resp.status_code == 200
            json_resp = resp.get_json()
            assert json_resp["success"] is True
            assert "sim_id" in json_resp
