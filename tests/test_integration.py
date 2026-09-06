"""
Script de Integração Ponta a Ponta:
Baixa dados reais do Yahoo Finance, roda o GA, gera predições e o relatório DOCX.
"""

import os
from data_fetcher import DataFetcher
from genetic_engine import GeneticEngine
from model_predictor import ModelPredictor
from report_generator import ReportGenerator

def run_integration():
    print("=== 1. Coleta do Yahoo Finance para USDBRL=X ===")
    fetcher = DataFetcher()
    df = fetcher.fetch_asset_data("USDBRL=X", period="6mo")
    print(f"Total de registros baixados: {len(df)}")
    last_val = df["Close"].iloc[-1]
    print(f"Último fechamento: R$ {last_val:.4f}")

    print("=== 2. Preparação de features e lags ===")
    X, y, target_dates, scaler_params = fetcher.prepare_lagged_features(df, lookback=10)
    print(f"Formato de X: {X.shape}, y: {y.shape}")

    print("=== 3. Execução do Algoritmo Genético ===")
    engine = GeneticEngine(population_size=30, generations=15, crossover_rate=0.85, mutation_rate=0.15)
    best_ind = engine.evolve(X, y)
    print(f"Melhor Fitness: {best_ind.fitness:.2f} | RMSE: {best_ind.rmse:.4f}")

    print("=== 4. Avaliação e Projeção Futura ===")
    predictor = ModelPredictor(best_ind, lookback_window=10)
    results = predictor.evaluate_and_predict(df, X, target_dates, forecast_horizon=5)
    metrics = results["metrics"]
    print(f"RMSE Real: R$ {metrics['rmse']:.4f} | MAE: R$ {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")
    print(f"Variação Projetada: {metrics['variacao_esperada_pct']:+.2f}% ({metrics['tendencia_esperada']})")

    print("=== 5. Planilhamento do Ensaio ===")
    trial_data = {
        "Data_Hora": "2026-09-05 23:48:00",
        "Ativo": "Dólar (USD/BRL)",
        "Ticker": "USDBRL=X",
        "Moeda": "BRL",
        "Periodo": "6 Meses",
        "Populacao": 30,
        "Geracoes": 15,
        "Taxa_Mutacao": 0.15,
        "Taxa_Crossover": 0.85,
        "Lookback": 10,
        "Horizonte_Dias": 5,
        "Ultimo_Preco_Real": metrics["ultimo_preco_real"],
        "Preco_Projetado_Final": metrics["preco_projetado_final"],
        "Variacao_Esperada_Pct": metrics["variacao_esperada_pct"],
        "Tendencia": metrics["tendencia_esperada"],
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "MAPE_Pct": metrics["mape"],
        "R2": metrics["r2"],
        "Acuracia_Direcional_Pct": metrics["acuracia_direcional"]
    }
    log_file = ReportGenerator.log_trial(trial_data)
    print(f"Ensaio registrado em: {log_file}")

    print("=== 6. Geração de Relatório Executivo DOCX com Gráficos ===")
    charts = ReportGenerator.export_charts_for_report(results, engine.history, "Dólar (USD/BRL)", "BRL")
    report_path = ReportGenerator.generate_docx_report(
        asset_name="Dólar (USD/BRL)",
        ticker="USDBRL=X",
        currency="BRL",
        period_str="6 Meses",
        ga_config={"population_size": 30, "generations": 15, "crossover_rate": 0.85, "mutation_rate": 0.15, "elitism_ratio": 0.08, "lookback_window": 10, "forecast_horizon": 5},
        prediction_results=results,
        ga_history=engine.history,
        chart_paths=charts
    )
    print(f"Relatório gerado com sucesso em: {report_path}")
    assert os.path.exists(report_path)
    print("\n>>> TESTE DE INTEGRACAO COMPLETO COM 100% DE SUCESSO! <<<")

if __name__ == "__main__":
    run_integration()
