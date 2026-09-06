/**
 * Controlador Front-End do Simulador de Investimentos (static/js/simulator.js)
 * Gerencia o Wizard em 5 etapas, a consulta em tempo real do ProgressTracker,
 * a renderização de múltiplos gráficos com Chart.js e a tabela de ranqueamento.
 */

document.addEventListener('DOMContentLoaded', () => {
  let currentStep = 1;
  const totalSteps = 5;
  let activeSimulationId = null;
  let pollInterval = null;

  let equityChart = null;
  let drawdownChart = null;
  let riskReturnChart = null;

  // Paleta elegante de cores para gráficos multi-algoritmo
  const colorPalette = {
    mapp: { border: '#10B981', fill: 'rgba(16, 185, 129, 0.1)' },
    ensemble: { border: '#8B5CF6', fill: 'rgba(139, 92, 246, 0.1)' },
    xgboost: { border: '#3B82F6', fill: 'rgba(59, 130, 246, 0.1)' },
    random_forest: { border: '#06B6D4', fill: 'rgba(6, 182, 212, 0.1)' },
    lstm: { border: '#EC4899', fill: 'rgba(236, 72, 153, 0.1)' },
    pattern_matching: { border: '#F59E0B', fill: 'rgba(245, 158, 11, 0.1)' },
    lightgbm: { border: '#14B8A6', fill: 'rgba(20, 184, 166, 0.1)' },
    regressao_linear: { border: '#94A3B8', fill: 'rgba(148, 163, 184, 0.1)' }
  };

  // Elementos do DOM
  const wizardSection = document.getElementById('wizard-section');
  const progressSection = document.getElementById('sim-progress-section');
  const resultsSection = document.getElementById('sim-results-section');

  const btnPrev = document.getElementById('wizard-btn-prev');
  const btnNext = document.getElementById('wizard-btn-next');
  const btnExecute = document.getElementById('wizard-btn-execute');
  const btnNewSim = document.getElementById('btn-new-simulation');

  const selectAsset = document.getElementById('sim-select-asset');
  const groupCustomTicker = document.getElementById('sim-group-custom-ticker');
  const inputCustomTicker = document.getElementById('sim-input-custom-ticker');

  const inputHorizonVal = document.getElementById('sim-input-horizon-val');
  const selectHorizonUnit = document.getElementById('sim-select-horizon-unit');
  const horizonSummary = document.getElementById('sim-horizon-summary');

  const paramsTabsContainer = document.getElementById('sim-params-tabs');

  // =========================================================================
  // 1. NAVEGAÇÃO DO WIZARD
  // =========================================================================
  function goToStep(step) {
    if (step < 1 || step > totalSteps) return;
    currentStep = step;

    // Atualiza indicadores visuais
    document.querySelectorAll('.wizard-steps-indicator .step-badge').forEach(badge => {
      const bStep = parseInt(badge.dataset.step);
      badge.classList.remove('active', 'completed');
      if (bStep === currentStep) {
        badge.classList.add('active');
      } else if (bStep < currentStep) {
        badge.classList.add('completed');
      }
    });

    // Alterna visibilidade dos painéis
    document.querySelectorAll('.wizard-step-pane').forEach((pane, idx) => {
      if (idx + 1 === currentStep) {
        pane.classList.add('active');
      } else {
        pane.classList.remove('active');
      }
    });

    // Atualiza botões
    if (currentStep === 1) {
      btnPrev.classList.add('hidden');
    } else {
      btnPrev.classList.remove('hidden');
    }

    if (currentStep === totalSteps) {
      btnNext.classList.add('hidden');
      btnExecute.classList.remove('hidden');
      renderParamsTabs();
    } else {
      btnNext.classList.remove('hidden');
      btnExecute.classList.add('hidden');

      const nextLabels = {
        1: 'Próximo: Capital',
        2: 'Próximo: Horizonte',
        3: 'Próximo: Algoritmos',
        4: 'Próximo: Parâmetros'
      };
      btnNext.textContent = nextLabels[currentStep] || 'Próximo';
    }
  }

  btnNext.addEventListener('click', () => {
    if (currentStep === 1 && selectAsset.value === 'custom' && !inputCustomTicker.value.trim()) {
      showToast('Por favor, informe o Ticker personalizado.', 'warning');
      inputCustomTicker.focus();
      return;
    }
    if (currentStep === 4) {
      const selected = getSelectedAlgorithms();
      if (selected.length === 0) {
        showToast('Selecione pelo menos um algoritmo para a simulação.', 'warning');
        return;
      }
    }
    goToStep(currentStep + 1);
  });

  btnPrev.addEventListener('click', () => {
    goToStep(currentStep - 1);
  });

  // Alternância do ticker personalizado
  selectAsset.addEventListener('change', () => {
    if (selectAsset.value === 'custom') {
      groupCustomTicker.classList.remove('hidden');
    } else {
      groupCustomTicker.classList.add('hidden');
    }
    updateHorizonPreview();
  });

  // Atualização do resumo do horizonte
  function updateHorizonPreview() {
    const val = parseInt(inputHorizonVal.value) || 30;
    const unit = selectHorizonUnit.value;
    const isCrypto = selectAsset.value.includes('BTC') || selectAsset.value.includes('ETH');

    let periodsEst = val;
    if (unit === 'dias') {
      periodsEst = isCrypto ? val : (val <= 14 ? val : Math.round(val * 21 / 30.4));
    } else if (unit === 'semanas') {
      periodsEst = isCrypto ? val * 7 : val * 5;
    } else if (unit === 'meses') {
      periodsEst = isCrypto ? Math.round(val * 30.4) : val * 21;
    } else if (unit === 'anos') {
      periodsEst = isCrypto ? Math.round(val * 365) : val * 252;
    }

    const mercTexto = isCrypto ? 'mercado contínuo 24/7' : 'pregões úteis comerciais';
    horizonSummary.textContent = `${val} ${unit} normalizados em ~${periodsEst} passos de rebalanceamento (${mercTexto}).`;
  }

  inputHorizonVal.addEventListener('input', updateHorizonPreview);
  selectHorizonUnit.addEventListener('change', updateHorizonPreview);
  updateHorizonPreview();

  // Seleção de cards de algoritmo
  document.querySelectorAll('.algo-checkbox-card input').forEach(checkbox => {
    checkbox.addEventListener('change', (e) => {
      const card = e.target.closest('.algo-checkbox-card');
      if (e.target.checked) {
        card.classList.add('selected');
      } else {
        card.classList.remove('selected');
      }
    });
  });

  function getSelectedAlgorithms() {
    const checked = document.querySelectorAll('input[name="sim_algorithms"]:checked');
    return Array.from(checked).map(c => c.value);
  }

  // =========================================================================
  // 2. ETAPA 5: PARÂMETROS ESPECÍFICOS POR ALGORITMO
  // =========================================================================
  function renderParamsTabs() {
    const selected = getSelectedAlgorithms();
    if (selected.length === 0) {
      paramsTabsContainer.innerHTML = '<p>Nenhum algoritmo selecionado.</p>';
      return;
    }

    const algoNames = {
      mapp: 'M.A.P.P.',
      ensemble: 'Prediction Ensemble',
      xgboost: 'XGBoost',
      random_forest: 'Random Forest',
      lstm: 'LSTM',
      pattern_matching: 'Pattern Matching',
      lightgbm: 'LightGBM',
      regressao_linear: 'Regressão Linear'
    };

    let html = '<div class="tabs-nav">';
    selected.forEach((algo, idx) => {
      html += `<button type="button" class="tab-btn ${idx === 0 ? 'active' : ''}" data-algo="${algo}">${algoNames[algo] || algo}</button>`;
    });
    html += '</div><div class="tabs-content">';

    selected.forEach((algo, idx) => {
      html += `<div class="tab-pane ${idx === 0 ? 'active' : ''}" id="tab-pane-${algo}">`;

      if (algo === 'mapp') {
        html += `
          <div class="form-grid">
            <div class="form-group">
              <label>Máximo de Atributos (Feature Selection)</label>
              <input type="number" id="p_mapp_features" class="form-control" value="20" min="5" max="50">
            </div>
            <div class="form-group">
              <label>Peso do Regime de Mercado</label>
              <input type="number" id="p_mapp_peso_regime" class="form-control" value="0.30" min="0.0" max="1.0" step="0.05">
            </div>
          </div>
        `;
      } else if (algo === 'ensemble') {
        html += `
          <div class="form-grid">
            <div class="form-group">
              <label>Estratégia de Pesagem</label>
              <select id="p_ensemble_tipo" class="form-control">
                <option value="regime" selected>Dependente do Regime de Mercado (Recomendado)</option>
                <option value="estatico">Pesos Estáticos Balanceados</option>
              </select>
            </div>
          </div>
        `;
      } else if (algo === 'xgboost') {
        html += `
          <div class="form-grid">
            <div class="form-group">
              <label>Número de Estimadores (Árvores)</label>
              <input type="number" id="p_xgb_n_estimators" class="form-control" value="100" min="20" max="500" step="10">
            </div>
            <div class="form-group">
              <label>Taxa de Aprendizado (Learning Rate)</label>
              <input type="number" id="p_xgb_lr" class="form-control" value="0.05" min="0.005" max="0.3" step="0.01">
            </div>
          </div>
        `;
      } else if (algo === 'random_forest') {
        html += `
          <div class="form-grid">
            <div class="form-group">
              <label>Número de Árvores</label>
              <input type="number" id="p_rf_trees" class="form-control" value="120" min="20" max="500" step="10">
            </div>
            <div class="form-group">
              <label>Profundidade Máxima</label>
              <input type="number" id="p_rf_depth" class="form-control" value="6" min="2" max="20">
            </div>
          </div>
        `;
      } else if (algo === 'pattern_matching') {
        html += `
          <div class="form-grid">
            <div class="form-group">
              <label>K Vizinhos Análogos</label>
              <input type="number" id="p_pm_k" class="form-control" value="5" min="2" max="20">
            </div>
          </div>
        `;
      } else {
        html += `<p class="text-muted">Parâmetros ótimos pré-configurados para ${algoNames[algo] || algo}.</p>`;
      }

      html += '</div>';
    });

    html += '</div>';
    paramsTabsContainer.innerHTML = html;

    // Conecta clique nas tabs
    paramsTabsContainer.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        paramsTabsContainer.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        paramsTabsContainer.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const target = document.getElementById(`tab-pane-${btn.dataset.algo}`);
        if (target) target.classList.add('active');
      });
    });
  }

  function extractAlgorithmParams() {
    const params = {};
    const selected = getSelectedAlgorithms();

    selected.forEach(algo => {
      params[algo] = {};
      if (algo === 'mapp') {
        const feat = document.getElementById('p_mapp_features');
        const reg = document.getElementById('p_mapp_peso_regime');
        if (feat) params[algo]['max_features'] = parseInt(feat.value);
        if (reg) params[algo]['peso_regime'] = parseFloat(reg.value);
      } else if (algo === 'ensemble') {
        const t = document.getElementById('p_ensemble_tipo');
        if (t) params[algo]['tipo_pesagem'] = t.value;
      } else if (algo === 'xgboost') {
        const est = document.getElementById('p_xgb_n_estimators');
        const lr = document.getElementById('p_xgb_lr');
        if (est) params[algo]['numero_estimadores'] = parseInt(est.value);
        if (lr) params[algo]['taxa_aprendizado'] = parseFloat(lr.value);
      } else if (algo === 'random_forest') {
        const tr = document.getElementById('p_rf_trees');
        const dp = document.getElementById('p_rf_depth');
        if (tr) params[algo]['numero_arvores'] = parseInt(tr.value);
        if (dp) params[algo]['profundidade_maxima'] = parseInt(dp.value);
      } else if (algo === 'pattern_matching') {
        const k = document.getElementById('p_pm_k');
        if (k) params[algo]['k_vizinhos'] = parseInt(k.value);
      }
    });

    return params;
  }

  // =========================================================================
  // 3. EXECUÇÃO DA SIMULAÇÃO & PROGRESS TRACKER
  // =========================================================================
  btnExecute.addEventListener('click', async () => {
    const selectedAlgos = getSelectedAlgorithms();
    if (selectedAlgos.length === 0) {
      showToast('Selecione pelo menos um algoritmo.', 'warning');
      return;
    }

    const payload = {
      asset: selectAsset.value,
      custom_ticker: inputCustomTicker.value.trim(),
      period: document.getElementById('sim-select-period').value,
      capital: parseFloat(document.getElementById('sim-input-capital').value) || 10000,
      aporte_periodico: parseFloat(document.getElementById('sim-input-aporte').value) || 0,
      taxa_corretagem_pct: parseFloat(document.getElementById('sim-input-taxa').value) || 0.05,
      slippage_pct: parseFloat(document.getElementById('sim-input-slippage').value) || 0.02,
      horizon_value: parseInt(inputHorizonVal.value) || 30,
      horizon_unit: selectHorizonUnit.value,
      algorithms: selectedAlgos,
      algorithm_params: extractAlgorithmParams()
    };

    // Alterna visualização para o monitor de progresso
    wizardSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    progressSection.classList.remove('hidden');

    resetProgressUI();
    showToast('Iniciando simulação multi-algoritmo...', 'info');

    try {
      const res = await fetch('/api/simulator/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (!res.ok || !data.success) {
        showToast(data.error || 'Falha ao iniciar simulação.', 'error');
        wizardSection.classList.remove('hidden');
        progressSection.classList.add('hidden');
        return;
      }

      activeSimulationId = data.sim_id;
      startProgressPolling();
    } catch (e) {
      showToast('Erro de comunicação: ' + e.message, 'error');
      wizardSection.classList.remove('hidden');
      progressSection.classList.add('hidden');
    }
  });

  function resetProgressUI() {
    document.getElementById('sim-progress-fill').style.width = '0%';
    document.getElementById('sim-prog-pct').textContent = '0.0%';
    document.getElementById('sim-elapsed-time').textContent = '00:00:00';
    document.getElementById('sim-remaining-time').textContent = 'Calculando...';
    document.getElementById('sim-total-time').textContent = 'Calculando...';
    document.getElementById('sim-status-text').textContent = 'Iniciando simulação de carteira...';
  }

  function startProgressPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      if (!activeSimulationId) return;

      try {
        const res = await fetch(`/api/simulator/progress?sim_id=${activeSimulationId}`);
        const state = await res.json();

        if (state.erro) {
          clearInterval(pollInterval);
          showToast('Erro na simulação: ' + state.erro, 'error');
          wizardSection.classList.remove('hidden');
          progressSection.classList.add('hidden');
          return;
        }

        // Atualiza UI do ProgressTracker
        const pct = state.progresso || 0;
        document.getElementById('sim-progress-fill').style.width = `${pct}%`;
        document.getElementById('sim-prog-pct').textContent = `${pct.toFixed(1)}%`;
        document.getElementById('sim-elapsed-time').textContent = state.tempo_decorrido || '00:00:00';
        document.getElementById('sim-remaining-time').textContent = state.tempo_restante || '00:00:00';
        document.getElementById('sim-total-time').textContent = state.tempo_estimado_total || '00:00:00';
        document.getElementById('sim-status-text').textContent = state.status_message || 'Executando...';

        if (state.concluido && state.resultado) {
          clearInterval(pollInterval);
          showToast('Simulação concluída com sucesso!', 'success');
          progressSection.classList.add('hidden');
          resultsSection.classList.remove('hidden');
          renderSimulationResults(state.resultado);
        }
      } catch (e) {
        console.warn('Falha no polling de simulação:', e);
      }
    }, 1000);
  }

  // =========================================================================
  // 4. RENDERIZAÇÃO DOS RESULTADOS COMPARATIVOS E GRÁFICOS
  // =========================================================================
  function renderSimulationResults(data) {
    const highlights = data.highlights || {};
    document.getElementById('hl-best-overall').textContent = highlights.best_overall || '—';
    document.getElementById('hl-best-return').textContent = highlights.best_return || '—';
    document.getElementById('hl-best-sharpe').textContent = highlights.best_sharpe || '—';
    document.getElementById('hl-lowest-dd').textContent = highlights.lowest_drawdown || '—';

    const capIni = data.comparison_table?.[0]?.capital_inicial || 10000;
    document.getElementById('label-capital-ini').textContent = `R$ ${capIni.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}`;

    // 1. Tabela Comparativa
    const tbody = document.getElementById('sim-table-body');
    tbody.innerHTML = '';

    const tableRows = data.comparison_table || [];
    tableRows.forEach((row, idx) => {
      const tr = document.createElement('tr');
      const isWinner = idx === 0;
      const profitClass = row.lucro_total >= 0 ? 'text-profit' : 'text-loss';
      const returnClass = row.retorno_pct >= 0 ? 'text-profit' : 'text-loss';

      tr.innerHTML = `
        <td><span class="rank-badge ${isWinner ? 'rank-1' : ''}">#${idx + 1}</span></td>
        <td><strong>${row.nome_exibicao}</strong></td>
        <td>R$ ${row.capital_final.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</td>
        <td class="${profitClass}">R$ ${row.lucro_total.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</td>
        <td class="${returnClass}">${row.retorno_pct >= 0 ? '+' : ''}${row.retorno_pct.toFixed(2)}%</td>
        <td>${row.sharpe.toFixed(2)}</td>
        <td>${row.sortino.toFixed(2)}</td>
        <td class="text-loss">-${Math.abs(row.max_drawdown).toFixed(2)}%</td>
        <td>${row.win_rate.toFixed(1)}%</td>
        <td>${row.profit_factor.toFixed(2)}</td>
        <td>${row.rmse.toFixed(4)}</td>
        <td><span class="score-badge">${row.score_multicriterio.toFixed(1)}</span></td>
      `;
      tbody.appendChild(tr);
    });

    // 2. Gráfico de Evolução de Capital (Equity Curves)
    renderEquityChart(data.equity_curves);

    // 3. Gráficos de Drawdown e Risco x Retorno
    renderSecondaryCharts(data);
  }

  function renderEquityChart(curvesMap) {
    const ctx = document.getElementById('chart-equity-comparison').getContext('2d');
    if (equityChart) equityChart.destroy();

    const datasets = [];
    let labels = [];

    Object.entries(curvesMap).forEach(([algoId, points]) => {
      if (!labels.length && points.length) {
        labels = points.map(p => p.date);
      }
      const style = colorPalette[algoId] || { border: '#CBD5E1', fill: 'transparent' };
      datasets.push({
        label: algoId.replace('_', ' ').toUpperCase(),
        data: points.map(p => p.capital),
        borderColor: style.border,
        backgroundColor: style.fill,
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0
      });
    });

    equityChart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: {
              color: '#94A3B8',
              callback: (val) => `R$ ${val.toLocaleString('pt-BR')}`
            }
          }
        },
        plugins: {
          legend: { labels: { color: '#E2E8F0', font: { family: 'Outfit', size: 12 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: R$ ${ctx.raw.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}`
            }
          }
        }
      }
    });
  }

  function renderSecondaryCharts(data) {
    // Drawdown chart
    const ctxDd = document.getElementById('chart-drawdown-comparison').getContext('2d');
    if (drawdownChart) drawdownChart.destroy();

    const ddDatasets = [];
    let ddLabels = [];

    const detailed = data.detailed_results || {};
    Object.entries(detailed).forEach(([algoId, res]) => {
      const points = res.drawdown_curve || [];
      if (!ddLabels.length && points.length) {
        ddLabels = points.map(p => p.date);
      }
      const style = colorPalette[algoId] || { border: '#EF4444' };
      ddDatasets.push({
        label: algoId.replace('_', ' ').toUpperCase(),
        data: points.map(p => -Math.abs(p.drawdown_pct)),
        borderColor: style.border,
        borderWidth: 1.5,
        tension: 0.1,
        pointRadius: 0
      });
    });

    drawdownChart = new Chart(ctxDd, {
      type: 'line',
      data: { labels: ddLabels, datasets: ddDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { display: false },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94A3B8', callback: v => `${v}%` }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });

    // Scatter Risk x Return
    const ctxScatter = document.getElementById('chart-risk-return-scatter').getContext('2d');
    if (riskReturnChart) riskReturnChart.destroy();

    const scatterData = (data.comparison_table || []).map(row => ({
      x: row.max_drawdown,
      y: row.retorno_pct,
      label: row.nome_exibicao
    }));

    riskReturnChart = new Chart(ctxScatter, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Algoritmos',
          data: scatterData,
          backgroundColor: '#38BDF8',
          borderColor: '#0284C7',
          pointRadius: 6,
          pointHoverRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Risco Máximo: Drawdown (%)', color: '#94A3B8' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94A3B8' }
          },
          y: {
            title: { display: true, text: 'Retorno Acumulado (%)', color: '#94A3B8' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94A3B8' }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const item = ctx.raw;
                return `${item.label}: Retorno +${item.y.toFixed(1)}%, Max DD -${item.x.toFixed(1)}%`;
              }
            }
          },
          legend: { display: false }
        }
      }
    });
  }

  btnNewSim.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    wizardSection.classList.remove('hidden');
    goToStep(1);
  });

  // Notificações Toast
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
});
