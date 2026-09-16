/**
 * Controlador Integrado do Simulador de Investimentos Multi-Algoritmo
 * (static/js/simulator_integrated.js)
 * Funciona diretamente na página principal (/), compartilhando ativo, datas e configurações.
 */

(function () {
  let activeSimulationId = null;
  let pollInterval = null;
  let lastSimulationData = null;
  let currentSelectedIndividualAlgo = null;

  // Instâncias dos gráficos Chart.js
  let deviationChart = null;
  let indivForecastChart = null;
  let indivEquityChart = null;
  let multiEquityChart = null;

  // Paleta de cores para os 16 algoritmos
  const colorPalette = {
    mapp: { border: '#10B981', fill: 'rgba(16, 185, 129, 0.15)', name: 'M.A.P.P. Quantitativo' },
    ensemble: { border: '#8B5CF6', fill: 'rgba(139, 92, 246, 0.15)', name: 'Prediction Ensemble' },
    pattern_matching: { border: '#F59E0B', fill: 'rgba(245, 158, 11, 0.15)', name: 'Pattern Matching' },
    xgboost: { border: '#3B82F6', fill: 'rgba(59, 130, 246, 0.15)', name: 'XGBoost' },
    lightgbm: { border: '#14B8A6', fill: 'rgba(20, 184, 166, 0.15)', name: 'LightGBM' },
    random_forest: { border: '#06B6D4', fill: 'rgba(6, 182, 212, 0.15)', name: 'Random Forest' },
    lstm: { border: '#EC4899', fill: 'rgba(236, 72, 153, 0.15)', name: 'LSTM Recorrente' },
    gru: { border: '#F43F5E', fill: 'rgba(244, 63, 94, 0.15)', name: 'GRU Recorrente' },
    transformer: { border: '#A855F7', fill: 'rgba(168, 85, 247, 0.15)', name: 'Transformer' },
    arima_sarima: { border: '#6366F1', fill: 'rgba(99, 102, 241, 0.15)', name: 'ARIMA / SARIMA' },
    prophet: { border: '#EAB308', fill: 'rgba(234, 179, 8, 0.15)', name: 'Prophet' },
    regressao_linear: { border: '#94A3B8', fill: 'rgba(148, 163, 184, 0.15)', name: 'Regressão Linear' },
    algoritmo_genetico: { border: '#10B981', fill: 'rgba(16, 185, 129, 0.15)', name: 'Algoritmo Genético' },
    mirofish: { border: '#0284C7', fill: 'rgba(2, 132, 199, 0.15)', name: 'MiroFish Multiagente' },
    rede_neural: { border: '#8B5CF6', fill: 'rgba(139, 92, 246, 0.15)', name: 'Rede Neural MLP' },
    logica_fuzzy: { border: '#06B6D4', fill: 'rgba(6, 182, 212, 0.15)', name: 'Lógica Fuzzy' }
  };

  document.addEventListener('DOMContentLoaded', () => {
    initElementsAndEvents();
    checkUrlTabParameter();
  });

  function checkUrlTabParameter() {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('tab') === 'simulator') {
      activateSimulatorTab();
    }
  }

  function activateSimulatorTab() {
    const tabBtn = document.getElementById('tab-btn-simulator');
    if (tabBtn) tabBtn.click();
  }

  function initElementsAndEvents() {
    const btnNavSim = document.getElementById('btn-nav-simulator');
    if (btnNavSim) {
      btnNavSim.addEventListener('click', activateSimulatorTab);
    }

    // Botões da Toolbar de Algoritmos
    const btnSelectAll = document.getElementById('sim-int-select-all');
    const btnSelectTop = document.getElementById('sim-int-select-top');
    const btnDeselectAll = document.getElementById('sim-int-deselect-all');
    const btnRunSim = document.getElementById('btn-run-integrated-sim');

    if (btnSelectAll) {
      btnSelectAll.addEventListener('click', () => {
        document.querySelectorAll('input[name="sim_int_algorithms"]').forEach(cb => cb.checked = true);
        updateAlgoCardsVisual();
      });
    }

    if (btnSelectTop) {
      btnSelectTop.addEventListener('click', () => {
        const top5 = ['mapp', 'ensemble', 'xgboost', 'lightgbm', 'transformer'];
        document.querySelectorAll('input[name="sim_int_algorithms"]').forEach(cb => {
          cb.checked = top5.includes(cb.value);
        });
        updateAlgoCardsVisual();
      });
    }

    if (btnDeselectAll) {
      btnDeselectAll.addEventListener('click', () => {
        document.querySelectorAll('input[name="sim_int_algorithms"]').forEach(cb => cb.checked = false);
        updateAlgoCardsVisual();
      });
    }

    document.querySelectorAll('input[name="sim_int_algorithms"]').forEach(cb => {
      cb.addEventListener('change', updateAlgoCardsVisual);
    });

    if (btnRunSim) {
      btnRunSim.addEventListener('click', executeIntegratedSimulation);
    }

    // Botões de Exportação Individual
    const btnExportIndivPdf = document.getElementById('btn-export-sim-pdf');
    const btnExportIndivDocx = document.getElementById('btn-export-sim-docx');

    if (btnExportIndivPdf) {
      btnExportIndivPdf.addEventListener('click', () => {
        if (!activeSimulationId || !currentSelectedIndividualAlgo) {
          showNotification('Selecione um algoritmo simulado para exportar.', 'warning');
          return;
        }
        window.open(`/api/simulator/export-pdf?sim_id=${encodeURIComponent(activeSimulationId)}&algorithm=${encodeURIComponent(currentSelectedIndividualAlgo)}`, '_blank');
      });
    }

    if (btnExportIndivDocx) {
      btnExportIndivDocx.addEventListener('click', () => {
        if (!activeSimulationId || !currentSelectedIndividualAlgo) {
          showNotification('Selecione um algoritmo simulado para exportar.', 'warning');
          return;
        }
        window.location.href = `/api/simulator/export-docx?sim_id=${encodeURIComponent(activeSimulationId)}&algorithm=${encodeURIComponent(currentSelectedIndividualAlgo)}`;
      });
    }
  }

  function updateAlgoCardsVisual() {
    document.querySelectorAll('input[name="sim_int_algorithms"]').forEach(cb => {
      const card = cb.closest('.algo-checkbox-card');
      if (card) {
        if (cb.checked) {
          card.classList.add('selected');
        } else {
          card.classList.remove('selected');
        }
      }
    });
  }

  function getSelectedAlgorithms() {
    const checked = document.querySelectorAll('input[name="sim_int_algorithms"]:checked');
    return Array.from(checked).map(c => c.value);
  }

  async function executeIntegratedSimulation() {
    const selectedAlgos = getSelectedAlgorithms();
    if (selectedAlgos.length === 0) {
      showNotification('Selecione pelo menos um algoritmo para iniciar a simulação.', 'warning');
      return;
    }

    const selectAsset = document.getElementById('select-asset');
    const inputCustomTicker = document.getElementById('input-custom-ticker');
    const selectPeriod = document.getElementById('select-period');
    const inputStartDate = document.getElementById('input-start-date');
    const inputEndDate = document.getElementById('input-end-date');
    const inputTargetDate = document.getElementById('param-forecast-target-date') || document.getElementById('input-end-date');

    const capital = parseFloat(document.getElementById('sim-int-capital')?.value) || 10000;
    const aporte = parseFloat(document.getElementById('sim-int-aporte')?.value) || 0;
    const taxa = parseFloat(document.getElementById('sim-int-taxa')?.value) || 0.05;
    const slippage = parseFloat(document.getElementById('sim-int-slippage')?.value) || 0.02;

    const payload = {
      asset: selectAsset ? selectAsset.value : 'petr4',
      custom_ticker: inputCustomTicker ? inputCustomTicker.value.trim() : '',
      period: selectPeriod ? selectPeriod.value : '2020_2025',
      start_date: inputStartDate ? inputStartDate.value : '2020-01-01',
      end_date: inputEndDate ? inputEndDate.value : '2025-12-31',
      forecast_target_date: inputTargetDate ? inputTargetDate.value : '',
      capital: capital,
      aporte_periodico: aporte,
      taxa_corretagem_pct: taxa,
      slippage_pct: slippage,
      horizon_value: 30,
      horizon_unit: 'dias',
      algorithms: selectedAlgos,
      algorithm_params: {}
    };

    const progressBox = document.getElementById('sim-int-progress-box');
    const resultsBox = document.getElementById('sim-int-results-box');

    if (progressBox) progressBox.classList.remove('hidden');
    if (resultsBox) resultsBox.classList.add('hidden');

    resetProgressUI();
    showNotification(`Iniciando simulação de ${selectedAlgos.length} algoritmos de IA...`, 'info');

    try {
      const res = await fetch('/api/simulator/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (!res.ok || !data.success) {
        showNotification(data.error || 'Falha ao iniciar simulação.', 'error');
        if (progressBox) progressBox.classList.add('hidden');
        return;
      }

      activeSimulationId = data.sim_id;
      startProgressPolling();
    } catch (e) {
      showNotification('Erro de comunicação: ' + e.message, 'error');
      if (progressBox) progressBox.classList.add('hidden');
    }
  }

  function resetProgressUI() {
    const fill = document.getElementById('sim-int-progress-fill');
    const pct = document.getElementById('sim-int-prog-pct');
    const elapsed = document.getElementById('sim-int-elapsed');
    const remaining = document.getElementById('sim-int-remaining');
    const status = document.getElementById('sim-int-status-msg');

    if (fill) fill.style.width = '0%';
    if (pct) pct.textContent = '0.0%';
    if (elapsed) elapsed.textContent = '00:00:00';
    if (remaining) remaining.textContent = 'Calculando...';
    if (status) status.textContent = 'Carregando série histórica de dados...';
  }

  function startProgressPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      if (!activeSimulationId) return;

      try {
        const res = await fetch(`/api/simulator/progress?sim_id=${encodeURIComponent(activeSimulationId)}`);
        const state = await res.json();

        if (state.erro) {
          clearInterval(pollInterval);
          showNotification('Erro na simulação: ' + state.erro, 'error');
          return;
        }

        const pct = state.progresso || 0;
        const fill = document.getElementById('sim-int-progress-fill');
        const pctEl = document.getElementById('sim-int-prog-pct');
        const elapsed = document.getElementById('sim-int-elapsed');
        const remaining = document.getElementById('sim-int-remaining');
        const status = document.getElementById('sim-int-status-msg');

        if (fill) fill.style.width = `${pct}%`;
        if (pctEl) pctEl.textContent = `${pct.toFixed(1)}%`;
        if (elapsed) elapsed.textContent = state.tempo_decorrido || '00:00:00';
        if (remaining) remaining.textContent = state.tempo_restante || '00:00:00';
        if (status) status.textContent = state.status_message || 'Processando modelos...';

        if (state.concluido && state.resultado) {
          clearInterval(pollInterval);
          showNotification('Simulação comparativa concluída com sucesso!', 'success');
          lastSimulationData = state.resultado;

          const progressBox = document.getElementById('sim-int-progress-box');
          const resultsBox = document.getElementById('sim-int-results-box');
          if (progressBox) progressBox.classList.add('hidden');
          if (resultsBox) resultsBox.classList.remove('hidden');

          renderIntegratedResults(state.resultado);
        }
      } catch (e) {
        console.warn('Falha no polling da simulação:', e);
      }
    }, 800);
  }

  function renderIntegratedResults(data) {
    const moeda = data.currency || 'BRL';
    const simbMoeda = moeda === 'USD' ? '$' : 'R$';

    // 1. Destaques
    const highlights = data.highlights || {};
    const hlBest = document.getElementById('sim-hl-best');
    const hlDev = document.getElementById('sim-hl-deviation');
    const hlRet = document.getElementById('sim-hl-return');
    const hlSharpe = document.getElementById('sim-hl-sharpe');
    const hlDd = document.getElementById('sim-hl-drawdown');

    if (hlBest) hlBest.textContent = highlights.best_overall || '—';
    if (hlDev) hlDev.textContent = highlights.best_deviation_today || '—';
    if (hlRet) hlRet.textContent = highlights.best_return || '—';
    if (hlSharpe) hlSharpe.textContent = highlights.best_sharpe || '—';
    if (hlDd) hlDd.textContent = highlights.lowest_drawdown || '—';

    // 2. Gráfico Geral Detalhado de Desvio na Data Atual (Hoje)
    renderDeviationSection(data.deviation_comparison, moeda);

    // 3. Análise, Gráficos & Documentos Individuais
    setupIndividualSection(data);

    // 4. Curvas de Capital Combinadas
    renderMultiEquityChart(data.equity_curves, moeda);

    // 5. Tabela Comparativa Completa
    renderComparativeTable(data.comparison_table, moeda);
  }

  // =========================================================================
  // GRÁFICO GERAL DETALHADO DE DESVIO HOJE
  // =========================================================================
  function renderDeviationSection(devData, moeda) {
    if (!devData) return;
    const simbMoeda = moeda === 'USD' ? '$' : 'R$';

    const quickStatsDiv = document.getElementById('sim-deviation-quick-stats');
    if (quickStatsDiv) {
      quickStatsDiv.innerHTML = `
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
          <span style="font-size: 0.75rem; color: #94A3B8; display: block;">COTAÇÃO REAL HOJE (${devData.data_referencia || 'Hoje'})</span>
          <strong style="font-size: 1.1rem; color: #10B981;">${simbMoeda} ${(devData.preco_real_hoje || 0).toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</strong>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
          <span style="font-size: 0.75rem; color: #94A3B8; display: block;">MODELO MAIS ASSERTIVO HOJE</span>
          <strong style="font-size: 1.1rem; color: #38BDF8;">${devData.campeao_hoje_nome || '—'} (${devData.campeao_hoje_desvio > 0 ? '+' : ''}${devData.campeao_hoje_desvio.toFixed(2)}%)</strong>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
          <span style="font-size: 0.75rem; color: #94A3B8; display: block;">DESVIO MÉDIO GERAL</span>
          <strong style="font-size: 1.1rem; color: #F59E0B;">±${(devData.desvio_medio_geral_pct || 0).toFixed(2)}%</strong>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
          <span style="font-size: 0.75rem; color: #94A3B8; display: block;">TAXA ACERTO DIREÇÃO</span>
          <strong style="font-size: 1.1rem; color: #EC4899;">${(devData.taxa_acerto_direcao_pct || 0).toFixed(1)}%</strong>
        </div>
      `;
    }

    const tbody = document.getElementById('sim-tbody-deviation');
    if (tbody) {
      tbody.innerHTML = '';
      const items = devData.itens || [];
      items.forEach((it, idx) => {
        const tr = document.createElement('tr');
        const isAssertive = it.status === 'ASSERTIVO';
        const diffSign = it.diferenca_pct > 0 ? '+' : '';
        const diffColor = Math.abs(it.diferenca_pct) <= 5 ? '#10B981' : (Math.abs(it.diferenca_pct) <= 15 ? '#F59E0B' : '#EF4444');

        tr.innerHTML = `
          <td><span class="rank-badge ${idx === 0 ? 'rank-1' : ''}">#${idx + 1}</span></td>
          <td><strong>${it.nome_exibicao}</strong></td>
          <td>${it.preco_projetado_hoje !== null ? `${simbMoeda} ${it.preco_projetado_hoje.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}` : '—'}</td>
          <td><strong style="color: #10B981;">${it.preco_real_hoje !== null ? `${simbMoeda} ${it.preco_real_hoje.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}` : '—'}</strong></td>
          <td>${it.diferenca_absoluta !== null ? `${it.diferenca_absoluta > 0 ? '+' : ''}${simbMoeda} ${it.diferenca_absoluta.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}` : '—'}</td>
          <td><strong style="color: ${diffColor};">${diffSign}${it.diferenca_pct.toFixed(2)}%</strong></td>
          <td>${it.acuracia_pct.toFixed(2)}%</td>
          <td>${it.acertou_direcao ? '<span class="badge-tag" style="background: rgba(16,185,129,0.2); color: #10B981;">✓ Correta</span>' : '<span class="badge-tag" style="background: rgba(239,68,68,0.2); color: #EF4444;">✗ Divergente</span>'}</td>
          <td><span class="score-badge" style="background: ${isAssertive ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.08)'}; color: ${isAssertive ? '#10B981' : '#94A3B8'};">${it.status}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    const canvas = document.getElementById('sim-chart-deviation-bars');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (deviationChart) deviationChart.destroy();

    const items = devData.itens || [];
    const labels = items.map(i => i.nome_exibicao);
    const dataValues = items.map(i => i.diferenca_pct);
    const bgColors = dataValues.map(v => Math.abs(v) <= 5 ? 'rgba(16, 185, 129, 0.75)' : (Math.abs(v) <= 15 ? 'rgba(245, 158, 11, 0.75)' : 'rgba(239, 68, 68, 0.75)'));
    const borderColors = dataValues.map(v => Math.abs(v) <= 5 ? '#10B981' : (Math.abs(v) <= 15 ? '#F59E0B' : '#EF4444'));

    deviationChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Desvio Hoje (%) vs. Cotação Real de Mercado',
          data: dataValues,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const it = items[context.dataIndex];
                return [
                  `Desvio Hoje: ${it.diferenca_pct > 0 ? '+' : ''}${it.diferenca_pct.toFixed(2)}%`,
                  `Previsão IA: ${simbMoeda} ${it.preco_projetado_hoje !== null ? it.preco_projetado_hoje.toFixed(2) : '—'}`,
                  `Cotação Real: ${simbMoeda} ${it.preco_real_hoje !== null ? it.preco_real_hoje.toFixed(2) : '—'}`,
                  `Acurácia: ${it.acuracia_pct.toFixed(1)}% | Direção: ${it.acertou_direcao ? 'Acertou' : 'Errou'}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            ticks: {
              color: '#94A3B8',
              callback: (val) => `${val > 0 ? '+' : ''}${val}%`
            }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#F8FAFC', font: { weight: 600 } }
          }
        }
      }
    });
  }

  // =========================================================================
  // ANÁLISE, GRÁFICOS E RELATÓRIOS INDIVIDUAIS POR ALGORITMO
  // =========================================================================
  function setupIndividualSection(data) {
    const tabsContainer = document.getElementById('sim-indiv-algo-tabs');
    if (!tabsContainer) return;

    tabsContainer.innerHTML = '';
    const indivResults = data.individual_results || {};
    const algoKeys = Object.keys(indivResults);

    if (algoKeys.length === 0) return;

    algoKeys.forEach((algoId, idx) => {
      const info = indivResults[algoId];
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `tab-btn ${idx === 0 ? 'active' : ''}`;
      btn.innerHTML = `${info.icone || '⚡'} ${info.nome || algoId}`;
      btn.dataset.algo = algoId;

      btn.addEventListener('click', () => {
        tabsContainer.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectIndividualAlgorithm(algoId, data);
      });

      tabsContainer.appendChild(btn);
    });

    selectIndividualAlgorithm(algoKeys[0], data);
  }

  function selectIndividualAlgorithm(algoId, data) {
    currentSelectedIndividualAlgo = algoId;
    const indiv = (data.individual_results || {})[algoId];
    if (!indiv) return;

    const moeda = data.currency || 'BRL';
    const simbMoeda = moeda === 'USD' ? '$' : 'R$';

    const lblPdf = document.getElementById('sim-label-btn-pdf');
    const lblDocx = document.getElementById('sim-label-btn-docx');
    if (lblPdf) lblPdf.textContent = `PDF (${indiv.nome})`;
    if (lblDocx) lblDocx.textContent = `Word (${indiv.nome})`;

    const kpiGrid = document.getElementById('sim-indiv-kpi-grid');
    if (kpiGrid) {
      const m = indiv.metrics || {};
      const t = indiv.target_comparison || {};
      kpiGrid.innerHTML = `
        <div class="kpi-card" style="background: rgba(15, 23, 42, 0.6);">
          <div class="kpi-header"><span class="kpi-title">RETORNO LÍQUIDO</span></div>
          <div class="kpi-body"><span class="kpi-value font-mono" style="color: ${m.total_return_pct >= 0 ? '#10B981' : '#EF4444'};">${m.total_return_pct > 0 ? '+' : ''}${(m.total_return_pct || 0).toFixed(2)}%</span></div>
          <div class="kpi-footer"><span class="kpi-sub">Lucro: ${simbMoeda} ${(m.total_profit || 0).toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</span></div>
        </div>
        <div class="kpi-card" style="background: rgba(15, 23, 42, 0.6);">
          <div class="kpi-header"><span class="kpi-title">DESVIO HOJE</span></div>
          <div class="kpi-body"><span class="kpi-value font-mono" style="color: ${Math.abs(t.diferenca_pct || 0) <= 5 ? '#10B981' : '#F59E0B'};">${t.diferenca_pct !== null ? `${t.diferenca_pct > 0 ? '+' : ''}${t.diferenca_pct.toFixed(2)}%` : '—'}</span></div>
          <div class="kpi-footer"><span class="kpi-sub">Real: ${simbMoeda} ${(t.preco_real_hoje || 0).toFixed(2)} | IA: ${(t.preco_projetado_hoje || 0).toFixed(2)}</span></div>
        </div>
        <div class="kpi-card" style="background: rgba(15, 23, 42, 0.6);">
          <div class="kpi-header"><span class="kpi-title">ÍNDICE DE SHARPE</span></div>
          <div class="kpi-body"><span class="kpi-value font-mono" style="color: #38BDF8;">${(m.sharpe_ratio || 0).toFixed(2)}</span></div>
          <div class="kpi-footer"><span class="kpi-sub">Sortino: ${(m.sortino_ratio || 0).toFixed(2)}</span></div>
        </div>
        <div class="kpi-card" style="background: rgba(15, 23, 42, 0.6);">
          <div class="kpi-header"><span class="kpi-title">MAX DRAWDOWN</span></div>
          <div class="kpi-body"><span class="kpi-value font-mono" style="color: #F43F5E;">${(m.max_drawdown_pct || 0).toFixed(2)}%</span></div>
          <div class="kpi-footer"><span class="kpi-sub">Calmar: ${(m.calmar_ratio || 0).toFixed(2)}</span></div>
        </div>
        <div class="kpi-card" style="background: rgba(15, 23, 42, 0.6);">
          <div class="kpi-header"><span class="kpi-title">TAXA DE ACERTO</span></div>
          <div class="kpi-body"><span class="kpi-value font-mono" style="color: #A855F7;">${(m.win_rate_pct || 0).toFixed(1)}%</span></div>
          <div class="kpi-footer"><span class="kpi-sub">Trades: ${m.total_trades || 0}</span></div>
        </div>
      `;
    }

    renderIndividualForecastChart(indiv, simbMoeda);
    renderIndividualEquityChart(indiv, simbMoeda);
  }

  function renderIndividualForecastChart(indiv, simbMoeda) {
    const canvas = document.getElementById('sim-chart-indiv-forecast');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (indivForecastChart) indivForecastChart.destroy();

    const chartData = indiv.chart_data || {};
    const histDates = chartData.dates_history || [];
    const histReal = chartData.prices_real || [];
    const histModel = chartData.prices_model || [];
    const futDates = chartData.dates_forecast || [];
    const futProj = chartData.prices_forecast || [];
    const futLower = chartData.prices_lower || [];
    const futUpper = chartData.prices_upper || [];

    const allDates = [...histDates, ...futDates];
    const nHist = histDates.length;

    const paddedReal = [...histReal, ...new Array(futDates.length).fill(null)];
    const paddedModel = [...histModel, ...new Array(futDates.length).fill(null)];
    const paddedProj = [...new Array(nHist).fill(null), ...futProj];
    const paddedLower = [...new Array(nHist).fill(null), ...futLower];
    const paddedUpper = [...new Array(nHist).fill(null), ...futUpper];

    if (histReal.length > 0 && futProj.length > 0) {
      paddedProj[nHist - 1] = histReal[nHist - 1];
      paddedLower[nHist - 1] = histReal[nHist - 1];
      paddedUpper[nHist - 1] = histReal[nHist - 1];
    }

    indivForecastChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allDates,
        datasets: [
          {
            label: 'Cotação Real Histórica',
            data: paddedReal,
            borderColor: '#38BDF8',
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Ajuste In-Sample do Modelo',
            data: paddedModel,
            borderColor: '#F59E0B',
            borderWidth: 1.5,
            borderDash: [3, 3],
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Projeção Futura IA',
            data: paddedProj,
            borderColor: '#10B981',
            borderWidth: 2.5,
            pointRadius: 1,
            fill: false
          },
          {
            label: 'Limite Superior (95%)',
            data: paddedUpper,
            borderColor: 'rgba(16, 185, 129, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Intervalo de Confiança 95%',
            data: paddedLower,
            borderColor: 'rgba(16, 185, 129, 0.25)',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '-1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: '#94A3B8', boxWidth: 12 } }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748B', maxTicksLimit: 12 } },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94A3B8', callback: (val) => `${simbMoeda} ${val.toFixed(2)}` }
          }
        }
      }
    });
  }

  function renderIndividualEquityChart(indiv, simbMoeda) {
    const canvas = document.getElementById('sim-chart-indiv-equity');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (indivEquityChart) indivEquityChart.destroy();

    const eqCurve = indiv.equity_curve || [];
    const labels = eqCurve.map(p => p.date);
    const capitalData = eqCurve.map(p => p.capital);
    const ddData = eqCurve.map(p => -(p.drawdown_pct || 0));

    indivEquityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Evolução do Capital',
            data: capitalData,
            borderColor: '#10B981',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            yAxisID: 'y'
          },
          {
            label: 'Drawdown (%)',
            data: ddData,
            borderColor: '#F43F5E',
            backgroundColor: 'rgba(244, 63, 94, 0.12)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: '#94A3B8', boxWidth: 12 } }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748B', maxTicksLimit: 10 } },
          y: {
            position: 'left',
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#10B981', callback: (val) => `${simbMoeda} ${val.toLocaleString('pt-BR')}` }
          },
          y1: {
            position: 'right',
            grid: { display: false },
            ticks: { color: '#F43F5E', callback: (val) => `${val.toFixed(1)}%` }
          }
        }
      }
    });
  }

  function renderMultiEquityChart(equityCurves, moeda) {
    const canvas = document.getElementById('sim-chart-multi-equity');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (multiEquityChart) multiEquityChart.destroy();

    const simbMoeda = moeda === 'USD' ? '$' : 'R$';
    const datasets = [];
    let commonLabels = [];

    Object.keys(equityCurves || {}).forEach(algoId => {
      const curve = equityCurves[algoId] || [];
      if (commonLabels.length === 0 && curve.length > 0) {
        commonLabels = curve.map(pt => pt.date);
      }
      const p = colorPalette[algoId] || { border: '#94A3B8', name: algoId };
      datasets.push({
        label: p.name || algoId,
        data: curve.map(pt => pt.capital),
        borderColor: p.border,
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0.1
      });
    });

    multiEquityChart = new Chart(ctx, {
      type: 'line',
      data: { labels: commonLabels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: '#94A3B8', boxWidth: 12 } }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748B', maxTicksLimit: 12 } },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94A3B8', callback: (val) => `${simbMoeda} ${val.toLocaleString('pt-BR')}` }
          }
        }
      }
    });
  }

  function renderComparativeTable(tableData, moeda) {
    const tbody = document.getElementById('sim-tbody-comparative');
    if (!tbody) return;
    tbody.innerHTML = '';
    const simbMoeda = moeda === 'USD' ? '$' : 'R$';

    (tableData || []).forEach((row, idx) => {
      const tr = document.createElement('tr');
      const retColor = row.retorno_pct >= 0 ? '#10B981' : '#EF4444';
      const devColor = row.desvio_hoje_pct !== null ? (Math.abs(row.desvio_hoje_pct) <= 5 ? '#10B981' : '#F59E0B') : '#94A3B8';

      tr.innerHTML = `
        <td><span class="rank-badge ${idx === 0 ? 'rank-1' : (idx === 1 ? 'rank-2' : (idx === 2 ? 'rank-3' : ''))}">#${idx + 1}</span></td>
        <td><strong>${row.nome_exibicao}</strong></td>
        <td>${simbMoeda} ${row.capital_final.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</td>
        <td style="color: ${retColor}; font-weight: 600;">${row.lucro_total > 0 ? '+' : ''}${simbMoeda} ${row.lucro_total.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</td>
        <td style="color: ${retColor}; font-weight: 700;">${row.retorno_pct > 0 ? '+' : ''}${row.retorno_pct.toFixed(2)}%</td>
        <td style="color: ${devColor}; font-weight: 600;">${row.desvio_hoje_pct !== null ? `${row.desvio_hoje_pct > 0 ? '+' : ''}${row.desvio_hoje_pct.toFixed(2)}%` : '—'}</td>
        <td>${row.preco_projetado_hoje !== null ? `${simbMoeda} ${row.preco_projetado_hoje.toFixed(2)}` : '—'}</td>
        <td style="color: #38BDF8; font-weight: 600;">${row.sharpe.toFixed(2)}</td>
        <td style="color: #F43F5E;">${row.max_drawdown.toFixed(2)}%</td>
        <td>${row.win_rate.toFixed(1)}%</td>
        <td>${row.rmse.toFixed(4)}</td>
        <td><span class="score-badge">${row.score_multicriterio.toFixed(1)}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function showNotification(msg, type) {
    if (typeof window.showToast === 'function') {
      window.showToast(msg, type);
    } else {
      console.log(`[Simulator ${type}]: ${msg}`);
    }
  }
})();
