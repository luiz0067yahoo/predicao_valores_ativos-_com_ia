import json
import time
import pytest
import web_app

def test_simulator_e2e():
    client = web_app.app.test_client()

    payload = {
        'asset': 'petr4',
        'ticker': 'PETR4.SA',
        'period': '2020_2025',
        'start_date': '2020-01-01',
        'end_date': '2025-12-31',
        'forecast_target_date': '2026-09-16',
        'horizon_value': 30,
        'horizon_unit': 'days',
        'capital': 10000.0,
        'aporte_periodico': 500.0,
        'taxa_corretagem_pct': 0.05,
        'slippage_pct': 0.02,
        'algorithms': ['arima_sarima', 'regressao_linear'],
        'algorithm_params': {}
    }

    res = client.post('/api/simulator/run', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.json
    assert data.get('success') is True
    sim_id = data.get('sim_id')
    assert sim_id

    completed = False
    for _ in range(60):
        time.sleep(1)
        p_res = client.get(f'/api/simulator/progress?sim_id={sim_id}')
        assert p_res.status_code == 200
        p_data = p_res.json
        if p_data.get('concluido') is True:
            completed = True
            result = p_data.get('resultado')
            assert result is not None
            assert 'deviation_comparison' in result
            assert 'individual_results' in result
            assert 'comparison_table' in result
            assert len(result['deviation_comparison']['itens']) == 2
            
            # Test export PDF for regressao_linear
            pdf_res = client.get(f'/api/simulator/export-pdf?sim_id={sim_id}&algorithm=regressao_linear')
            assert pdf_res.status_code == 200
            assert pdf_res.mimetype == 'application/pdf'
            assert len(pdf_res.data) > 1000
            
            # Test export DOCX for regressao_linear
            docx_res = client.get(f'/api/simulator/export-docx?sim_id={sim_id}&algorithm=regressao_linear')
            assert docx_res.status_code == 200
            assert len(docx_res.data) > 1000
            break
        elif p_data.get('erro'):
            pytest.fail(f"Simulation failed with error: {p_data.get('erro')}")

    assert completed, "Simulation did not complete within timeout"
