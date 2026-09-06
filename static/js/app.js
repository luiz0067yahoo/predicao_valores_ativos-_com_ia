/**
 * AI ASSET PREDICTOR - CLIENT CONTROLLER & VISUALIZATION ENGINE
 * Integrates Chart.js, AJAX with Flask Session, Multi-Model Dynamic Forms,
 * and PDF / DOCX Executive Reporting.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Estado Global da Aplicação
  const state = {
    assetsConfig: {},
    periodsConfig: {},
    defaultGa: {},
    algorithmsCatalog: [],
    selectedAlgorithm: 'mapp',
    currentCurrency: 'USD',
    forecastChart: null,
    convergenceChart: null,
    pollingInterval: null,
    currentTaskId: null,
    isRunning: false
  };

  // Referências aos Elementos do DOM
  const elements = {
    selectAlgorithm: document.getElementById('select-algorithm'),
    algorithmDesc: document.getElementById('algorithm-desc'),
    algoBadgeHeader: document.getElementById('algo-badge-header'),
    selectAsset: document.getElementById('select-asset'),
    customTickerGroup: document.getElementById('custom-ticker-group'),
    inputCustomTicker: document.getElementById('input-custom-ticker'),
    selectPeriod: document.getElementById('select-period'),
    customDatesGroup: document.getElementById('custom-dates-group'),
    inputStartDate: document.getElementById('input-start-date'),
    inputEndDate: document.getElementById('input-end-date'),
    dynamicParamsContainer: document.getElementById('dynamic-params-container'),
    paramsSectionTitle: document.getElementById('params-section-title'),
    btnRun: document.getElementById('btn-run'),
    btnExportPdf: document.getElementById('btn-export-pdf'),
    btnExportDocx: document.getElementById('btn-export-docx'),

    // Barra de Progresso e Status
    progressContainer: document.getElementById('progress-container'),
    progressStatusText: document.getElementById('progress-status-text'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    progGen: document.getElementById('prog-gen'),
    progFit: document.getElementById('prog-fit'),
    progRmse: document.getElementById('prog-rmse'),
    progExtra1: document.getElementById('prog-extra-1'),
    progExtra2: document.getElementById('prog-extra-2'),
    progExtra3: document.getElementById('prog-extra-3'),
    connectionBadge: document.getElementById('connection-badge'),
    connectionText: document.getElementById('connection-text'),

    // Cards de KPIs
    kpiActual: document.getElementById('kpi-actual'),
    kpiPredicted: document.getElementById('kpi-predicted'),
    kpiVariation: document.getElementById('kpi-variation'),
    kpiTrendTag: document.getElementById('kpi-trend-tag'),
    kpiRmse: document.getElementById('kpi-rmse'),
    kpiDirectional: document.getElementById('kpi-directional'),
    kpiTickerLabel: document.getElementById('kpi-ticker-label'),
    kpiHorizonLabel: document.getElementById('kpi-horizon-label'),
    kpiAlgoBadge: document.getElementById('kpi-algo-badge'),

    // Canvas e Títulos dos Gráficos
    forecastCanvas: document.getElementById('chart-forecast-canvas'),
    convergenceCanvas: document.getElementById('chart-convergence-canvas'),
    chartForecastTitle: document.getElementById('chart-forecast-title'),
    chartConvTitle: document.getElementById('chart-conv-title'),
    chartConvSubtitle: document.getElementById('chart-conv-subtitle'),

    // Tabela Passo a Passo
    forecastTableBody: document.getElementById('forecast-table-body'),
    tableInfoCounter: document.getElementById('table-info-counter'),

    // Abas de Navegação
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    // Modais
    btnViewHistory: document.getElementById('btn-view-history'),
    modalHistory: document.getElementById('modal-history'),
    btnCloseHistory: document.getElementById('btn-close-history'),
    btnCloseHistoryFooter: document.getElementById('btn-close-history-footer'),
    btnRefreshHistory: document.getElementById('btn-refresh-history'),
    historyTableBody: document.getElementById('history-table-body'),

    btnViewReports: document.getElementById('btn-view-reports'),
    modalReports: document.getElementById('modal-reports'),
    btnCloseReports: document.getElementById('btn-close-reports'),
    btnCloseReportsFooter: document.getElementById('btn-close-reports-footer'),
    btnRefreshReports: document.getElementById('btn-refresh-reports'),
    reportsListContainer: document.getElementById('reports-list-container'),

    toastContainer: document.getElementById('toast-container')
  };

  // =========================================================================
  // 1. INICIALIZAÇÃO E CARREGAMENTO DE CONFIGURAÇÕES
  // =========================================================================
  async function initApp() {
    try {
      setupDatesDefault();
      initCharts();
      setupEventListeners();
      await loadConfig();
    } catch (err) {
      console.error('Falha na inicialização:', err);
      showToast('Erro ao carregar configurações do servidor', 'error');
    }
  }

  function setupDatesDefault() {
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);

    elements.inputStartDate.value = oneYearAgo.toISOString().split('T')[0];
    elements.inputEndDate.value = today.toISOString().split('T')[0];
  }

  async function loadConfig() {
    const res = await fetch('/api/config');
    if (!res.ok) throw new Error('Não foi possível obter /api/config');
    const data = await res.json();

    state.assetsConfig = data.assets || {};
    state.periodsConfig = data.periods || {};
    state.defaultGa = data.default_ga || {};
    state.algorithmsCatalog = data.algoritmos || [];

    populateAssetSelect();
    populateAlgorithmSelect();
  }

  function populateAlgorithmSelect() {
    if (!state.algorithmsCatalog || state.algorithmsCatalog.length === 0) return;

    elements.selectAlgorithm.innerHTML = '';
    state.algorithmsCatalog.forEach(algo => {
      const opt = document.createElement('option');
      opt.value = algo.identificador;
      opt.textContent = `${algo.icone} ${algo.nome} (${algo.desempenho})`;
      elements.selectAlgorithm.appendChild(opt);
    });

    elements.selectAlgorithm.value = 'mapp';
    onAlgorithmChange();
  }

  function populateAssetSelect() {
    elements.selectAsset.innerHTML = '';

    const categories = {};
    for (const [name, info] of Object.entries(state.assetsConfig)) {
      const cat = info.category || 'Outros';
      if (!categories[cat]) categories[cat] = [];
      categories[cat].push({ name, ...info });
    }

    for (const [catName, assets] of Object.entries(categories)) {
      const optGroup = document.createElement('optgroup');
      optGroup.label = catName;
      assets.forEach(asset => {
        const opt = document.createElement('option');
        opt.value = asset.name;
        opt.textContent = `${asset.name} (${asset.ticker})`;
        optGroup.appendChild(opt);
      });
      elements.selectAsset.appendChild(optGroup);
    }

    const customGroup = document.createElement('optgroup');
    customGroup.label = 'Personalizado';
    const optCustom = document.createElement('option');
    optCustom.value = 'Personalizado';
    optCustom.textContent = 'Custom Ticker (Yahoo Finance)';
    customGroup.appendChild(optCustom);
    elements.selectAsset.appendChild(customGroup);

    if (elements.selectAsset.options.length > 0) {
      elements.selectAsset.selectedIndex = 0;
      onAssetChange();
    }
  }

  // =========================================================================
  // 2. FORMULÁRIO DINÂMICO DE HIPERPARÂMETROS POR ALGORITMO
  // =========================================================================
  function onAlgorithmChange() {
    const selectedId = elements.selectAlgorithm.value;
    state.selectedAlgorithm = selectedId;

    const algoInfo = state.algorithmsCatalog.find(a => a.identificador === selectedId);
    if (!algoInfo) return;

    elements.algorithmDesc.textContent = `${algoInfo.uso} | Categoria: ${algoInfo.categoria}`;
    elements.algoBadgeHeader.textContent = algoInfo.categoria;
    elements.paramsSectionTitle.textContent = `4. Parâmetros do ${algoInfo.nome}`;

    renderDynamicParameters(selectedId, algoInfo.parametros_padrao);
  }

  function renderDynamicParameters(algoId, defaultParams = {}) {
    elements.dynamicParamsContainer.innerHTML = '';

    const fields = [];

    // Adiciona campo padrão de Horizonte de Projeção com suporte a Unidades
    fields.push({
      id: 'param-horizon',
      name: 'horizon_value',
      label: 'Horizonte de Projeção',
      type: 'number',
      val: 2,
      min: 1,
      max: 365,
      step: 1,
      hint: 'Valor do horizonte futuro'
    });
    fields.push({
      id: 'param-horizon-unit',
      name: 'horizon_unit',
      label: 'Unidade do Horizonte',
      type: 'select',
      val: 'anos',
      options: [
        { value: 'dias', text: 'Dias' },
        { value: 'semanas', text: 'Semanas' },
        { value: 'meses', text: 'Meses' },
        { value: 'anos', text: 'Anos' },
      ],
      hint: 'Escala temporal calibrada'
    });

    if (algoId === 'xgboost' || algoId === 'lightgbm') {
      fields.push({
        id: 'param-estimators',
        name: 'numero_estimadores',
        label: 'Nº Estimadores',
        type: 'number',
        val: defaultParams.numero_estimadores || 100,
        min: 20,
        max: 300,
        step: 10,
        hint: 'Árvores boosting'
      });
      fields.push({
        id: 'param-lr',
        name: 'taxa_aprendizado',
        label: 'Taxa de Aprendizado',
        type: 'number',
        val: defaultParams.taxa_aprendizado || 0.05,
        min: 0.001,
        max: 0.5,
        step: 0.01,
        hint: 'Learning rate'
      });
      if (algoId === 'xgboost') {
        fields.push({
          id: 'param-depth',
          name: 'profundidade_maxima',
          label: 'Profundidade',
          type: 'number',
          val: defaultParams.profundidade_maxima || 4,
          min: 2,
          max: 12,
          step: 1,
          hint: 'Max depth'
        });
      } else {
        fields.push({
          id: 'param-leaves',
          name: 'numero_folhas',
          label: 'Nº de Folhas',
          type: 'number',
          val: defaultParams.numero_folhas || 31,
          min: 10,
          max: 128,
          step: 1,
          hint: 'Num leaves'
        });
      }
    } else if (algoId === 'random_forest') {
      fields.push({
        id: 'param-trees',
        name: 'numero_arvores',
        label: 'Nº de Árvores',
        type: 'number',
        val: defaultParams.numero_arvores || 150,
        min: 20,
        max: 300,
        step: 10,
        hint: 'Ensemble bagging'
      });
      fields.push({
        id: 'param-depth',
        name: 'profundidade_maxima',
        label: 'Profundidade Máx.',
        type: 'number',
        val: defaultParams.profundidade_maxima || 8,
        min: 2,
        max: 20,
        step: 1,
        hint: 'Max depth'
      });
    } else if (algoId === 'lstm' || algoId === 'gru' || algoId === 'transformer') {
      fields.push({
        id: 'param-epochs',
        name: 'numero_epocas',
        label: 'Épocas Treino',
        type: 'number',
        val: defaultParams.numero_epocas || 35,
        min: 10,
        max: 120,
        step: 5,
        hint: 'Ciclos de ajuste'
      });
      fields.push({
        id: 'param-lr',
        name: 'taxa_aprendizado',
        label: 'Taxa Aprendizado',
        type: 'number',
        val: defaultParams.taxa_aprendizado || 0.008,
        min: 0.001,
        max: 0.05,
        step: 0.001,
        hint: 'Adam LR'
      });
      fields.push({
        id: 'param-hidden',
        name: algoId === 'transformer' ? 'dimensao_modelo' : 'dimensao_oculta',
        label: algoId === 'transformer' ? 'Dimensão Modelo' : 'Neurônios Ocultos',
        type: 'number',
        val: defaultParams.dimensao_modelo || defaultParams.dimensao_oculta || 40,
        min: 16,
        max: 128,
        step: 8,
        hint: 'Capacidade rede'
      });
    } else if (algoId === 'arima_sarima') {
      fields.push({
        id: 'param-p',
        name: 'ordem_p',
        label: 'Ordem AR (p)',
        type: 'number',
        val: defaultParams.ordem_p || 2,
        min: 0,
        max: 5,
        step: 1,
        hint: 'Defasagem AR'
      });
      fields.push({
        id: 'param-d',
        name: 'ordem_d',
        label: 'Diferenciação (d)',
        type: 'number',
        val: defaultParams.ordem_d || 1,
        min: 0,
        max: 2,
        step: 1,
        hint: 'Estacionariedade'
      });
      fields.push({
        id: 'param-q',
        name: 'ordem_q',
        label: 'Ordem MA (q)',
        type: 'number',
        val: defaultParams.ordem_q || 2,
        min: 0,
        max: 5,
        step: 1,
        hint: 'Média móvel'
      });
    } else if (algoId === 'prophet') {
      fields.push({
        id: 'param-trend',
        name: 'flexibilidade_tendencia',
        label: 'Flex. Tendência',
        type: 'number',
        val: defaultParams.flexibilidade_tendencia || 0.05,
        min: 0.001,
        max: 0.5,
        step: 0.01,
        hint: 'Changepoints'
      });
      fields.push({
        id: 'param-seasonal',
        name: 'flexibilidade_sazonal',
        label: 'Flex. Sazonal',
        type: 'number',
        val: defaultParams.flexibilidade_sazonal || 10.0,
        min: 0.1,
        max: 25.0,
        step: 1.0,
        hint: 'Seasonality prior'
      });
    } else if (algoId === 'regressao_linear') {
      fields.push({
        id: 'param-reg',
        name: 'forca_regularizacao',
        label: 'Regularização (Alfa)',
        type: 'number',
        val: defaultParams.forca_regularizacao || 1.0,
        min: 0.01,
        max: 10.0,
        step: 0.5,
        hint: 'Penalidade Ridge L2'
      });
    } else if (algoId === 'algoritmo_genetico') {
      fields.push({
        id: 'param-pop',
        name: 'tamanho_populacao',
        label: 'População',
        type: 'number',
        val: defaultParams.tamanho_populacao || 60,
        min: 20,
        max: 200,
        step: 10,
        hint: 'Indivíduos'
      });
      fields.push({
        id: 'param-gen',
        name: 'numero_geracoes',
        label: 'Gerações',
        type: 'number',
        val: defaultParams.numero_geracoes || 35,
        min: 10,
        max: 100,
        step: 5,
        hint: 'Épocas evolutivas'
      });
      fields.push({
        id: 'param-mut',
        name: 'taxa_mutacao',
        label: 'Taxa Mutação (%)',
        type: 'number',
        val: Math.round((defaultParams.taxa_mutacao || 0.15) * 100),
        min: 1,
        max: 50,
        step: 1,
        hint: 'Perturbação genética'
      });
    } else if (algoId === 'mirofish') {
      fields.push({
        id: 'param-rounds',
        name: 'numero_rodadas_debate',
        label: 'Rodadas de Debate',
        type: 'number',
        val: defaultParams.numero_rodadas_debate || 15,
        min: 5,
        max: 50,
        step: 5,
        hint: 'Interação dos agentes'
      });
    } else if (algoId === 'rede_neural') {
      fields.push({
        id: 'param-nn-epochs',
        name: 'numero_epocas',
        label: 'Épocas de Treino',
        type: 'number',
        val: defaultParams.numero_epocas || 45,
        min: 10,
        max: 150,
        step: 5,
        hint: 'Ciclos de retropropagação'
      });
      fields.push({
        id: 'param-nn-lr',
        name: 'taxa_aprendizado',
        label: 'Taxa Aprendizado',
        type: 'number',
        val: defaultParams.taxa_aprendizado || 0.005,
        min: 0.0005,
        max: 0.05,
        step: 0.001,
        hint: 'Passo do otimizador Adam'
      });
      fields.push({
        id: 'param-nn-l1',
        name: 'dimensao_camada_1',
        label: 'Camada Oculta 1',
        type: 'number',
        val: defaultParams.dimensao_camada_1 || 64,
        min: 16,
        max: 128,
        step: 16,
        hint: 'Neurônios densos'
      });
      fields.push({
        id: 'param-nn-l2',
        name: 'dimensao_camada_2',
        label: 'Camada Oculta 2',
        type: 'number',
        val: defaultParams.dimensao_camada_2 || 32,
        min: 8,
        max: 64,
        step: 8,
        hint: 'Neurônios secundários'
      });
    } else if (algoId === 'logica_fuzzy') {
      fields.push({
        id: 'param-fuzzy-sets',
        name: 'numero_conjuntos_fuzzy',
        label: 'Conjuntos Fuzzy',
        type: 'number',
        val: defaultParams.numero_conjuntos_fuzzy || 5,
        min: 3,
        max: 9,
        step: 2,
        hint: 'Regimes linguísticos'
      });
      fields.push({
        id: 'param-fuzzy-width',
        name: 'largura_pertinencia',
        label: 'Largura Gaussiana',
        type: 'number',
        val: defaultParams.largura_pertinencia || 1.0,
        min: 0.2,
        max: 3.0,
        step: 0.1,
        hint: 'Dispersão pertinência'
      });
      fields.push({
        id: 'param-fuzzy-reg',
        name: 'forca_regularizacao',
        label: 'Suavização Regras',
        type: 'number',
        val: defaultParams.forca_regularizacao || 0.5,
        min: 0.05,
        max: 5.0,
        step: 0.1,
        hint: 'Penalidade TSK'
      });
    } else if (algoId === 'mapp') {
      fields.push({
        id: 'param-mapp-features',
        name: 'n_features',
        label: 'Qtd. Características',
        type: 'number',
        val: defaultParams.n_features || 15,
        min: 5,
        max: 40,
        step: 1,
        hint: 'Features selecionadas (anti-leak)'
      });
      fields.push({
        id: 'param-mapp-conf',
        name: 'confidence_level',
        label: 'Confiança (%)',
        type: 'number',
        val: Math.round((defaultParams.confidence_level || 0.95) * 100),
        min: 80,
        max: 99,
        step: 1,
        hint: 'Intervalo de incerteza'
      });
    } else if (algoId === 'ensemble') {
      fields.push({
        id: 'param-ens-mode',
        name: 'mode',
        label: 'Modo de Ponderação',
        type: 'select',
        val: defaultParams.mode || 'regime_adaptive',
        options: [
          { value: 'regime_adaptive', text: 'Adaptativo por Regime' },
          { value: 'equal', text: 'Equiponderado (Média Simples)' }
        ],
        hint: 'Ponderação dos modelos'
      });
      fields.push({
        id: 'param-ens-k',
        name: 'top_models',
        label: 'Top Modelos',
        type: 'number',
        val: defaultParams.top_models || 3,
        min: 2,
        max: 8,
        step: 1,
        hint: 'Qtd. modelos no ensemble'
      });
    } else if (algoId === 'pattern_matching') {
      fields.push({
        id: 'param-pm-window',
        name: 'window_size',
        label: 'Janela do Padrão',
        type: 'number',
        val: defaultParams.window_size || 20,
        min: 5,
        max: 60,
        step: 5,
        hint: 'Barras do padrão atual'
      });
      fields.push({
        id: 'param-pm-matches',
        name: 'top_k_matches',
        label: 'Análogos Históricos',
        type: 'number',
        val: defaultParams.top_k_matches || 5,
        min: 2,
        max: 20,
        step: 1,
        hint: 'Top padrões similares'
      });
    }

    fields.forEach(f => {
      const group = document.createElement('div');
      group.className = 'form-group';
      if (f.type === 'select') {
        const optionsHtml = (f.options || []).map(opt =>
          `<option value="${opt.value}" ${opt.value === f.val ? 'selected' : ''}>${opt.text}</option>`
        ).join('');
        group.innerHTML = `
          <label for="${f.id}">${f.label}</label>
          <select id="${f.id}" name="${f.name}" class="form-control dynamic-param">
            ${optionsHtml}
          </select>
          <span class="field-hint">${f.hint}</span>
        `;
      } else {
        group.innerHTML = `
          <label for="${f.id}">${f.label}</label>
          <input type="${f.type}" id="${f.id}" name="${f.name}" class="form-control dynamic-param"
                 value="${f.val}" min="${f.min || ''}" max="${f.max || ''}" step="${f.step || '1'}">
          <span class="field-hint">${f.hint}</span>
        `;
      }
      elements.dynamicParamsContainer.appendChild(group);
    });
  }

  // =========================================================================
  // 3. CONFIGURAÇÃO DOS GRÁFICOS (CHART.JS)
  // =========================================================================
  function initCharts() {
    const ctxForecast = elements.forecastCanvas.getContext('2d');
    state.forecastChart = new Chart(ctxForecast, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Preço Real (Mercado)',
            data: [],
            borderColor: '#38bdf8',
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.1
          },
          {
            label: 'Ajuste do Modelo IA',
            data: [],
            borderColor: '#fb923c',
            backgroundColor: 'transparent',
            borderWidth: 1.6,
            borderDash: [5, 4],
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.1
          },
          {
            label: 'Projeção Futura (IA)',
            data: [],
            borderColor: '#10b981',
            backgroundColor: 'transparent',
            borderWidth: 2.2,
            pointRadius: 3.5,
            pointBackgroundColor: '#10b981',
            pointHoverRadius: 5,
            tension: 0.15
          },
          {
            label: 'Limite Superior (95%)',
            data: [],
            borderColor: 'rgba(16, 185, 129, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            tension: 0.15
          },
          {
            label: 'Limite Inferior (95%)',
            data: [],
            borderColor: 'rgba(16, 185, 129, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            tension: 0.15
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.92)',
            titleColor: '#f8fafc',
            bodyColor: '#cbd5e1',
            borderColor: 'rgba(255, 255, 255, 0.12)',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: function (ctx) {
                const val = ctx.parsed.y;
                if (val === null || val === undefined || isNaN(val)) return null;
                return `${ctx.dataset.label}: ${formatCurrency(val, state.currentCurrency)}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748b', maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#64748b',
              callback: function (val) { return formatNumber(val); }
            }
          }
        }
      }
    });

    const ctxConv = elements.convergenceCanvas.getContext('2d');
    state.convergenceChart = new Chart(ctxConv, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Melhor Desempenho / 1/Loss',
            data: [],
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.08)',
            borderWidth: 2.2,
            fill: true,
            pointRadius: 1.5,
            tension: 0.1
          },
          {
            label: 'Média de Aprendizado',
            data: [],
            borderColor: '#3b82f6',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.92)',
            titleColor: '#f8fafc',
            bodyColor: '#cbd5e1',
            borderColor: 'rgba(255, 255, 255, 0.12)',
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Ciclo / Passo Temporal', color: '#64748b', font: { size: 11 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748b' }
          },
          y: {
            title: { display: true, text: 'Métrica de Ajuste', color: '#64748b', font: { size: 11 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748b' }
          }
        }
      }
    });
  }

  // =========================================================================
  // 4. LISTENERS DE EVENTOS
  // =========================================================================
  function setupEventListeners() {
    elements.selectAlgorithm.addEventListener('change', onAlgorithmChange);
    elements.selectAsset.addEventListener('change', onAssetChange);
    elements.selectPeriod.addEventListener('change', onPeriodChange);

    elements.btnRun.addEventListener('click', startOptimizationAjax);
    elements.btnExportPdf.addEventListener('click', exportPdfReport);
    elements.btnExportDocx.addEventListener('click', exportDocxReport);

    // Abas
    elements.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        elements.tabButtons.forEach(b => b.classList.remove('active'));
        elements.tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        const pane = document.getElementById(targetId);
        if (pane) pane.classList.add('active');

        if (state.forecastChart) state.forecastChart.resize();
        if (state.convergenceChart) state.convergenceChart.resize();
      });
    });

    // Modais
    elements.btnViewHistory.addEventListener('click', openHistoryModal);
    elements.btnCloseHistory.addEventListener('click', () => elements.modalHistory.classList.add('hidden'));
    elements.btnCloseHistoryFooter.addEventListener('click', () => elements.modalHistory.classList.add('hidden'));
    elements.btnRefreshHistory.addEventListener('click', loadHistoryData);

    elements.btnViewReports.addEventListener('click', openReportsModal);
    elements.btnCloseReports.addEventListener('click', () => elements.modalReports.classList.add('hidden'));
    elements.btnCloseReportsFooter.addEventListener('click', () => elements.modalReports.classList.add('hidden'));
    elements.btnRefreshReports.addEventListener('click', loadReportsData);
  }

  function onAssetChange() {
    const val = elements.selectAsset.value;
    if (val === 'Personalizado') {
      elements.customTickerGroup.classList.remove('hidden');
      state.currentCurrency = 'Unidade';
    } else {
      elements.customTickerGroup.classList.add('hidden');
      const info = state.assetsConfig[val];
      state.currentCurrency = info ? info.currency : 'USD';
    }
  }

  function onPeriodChange() {
    const val = elements.selectPeriod.value;
    if (val === 'custom') {
      elements.customDatesGroup.classList.remove('hidden');
    } else {
      elements.customDatesGroup.classList.add('hidden');
    }
  }

  // =========================================================================
  // 5. PIPELINE VIA SESSÃO E AJAX (POLLING ASSÍNCRONO)
  // =========================================================================
  async function startOptimizationAjax() {
    if (state.isRunning) return;

    const asset = elements.selectAsset.value;
    const customTicker = elements.inputCustomTicker.value.trim();
    const period = elements.selectPeriod.value;
    const startDate = elements.inputStartDate.value;
    const endDate = elements.inputEndDate.value;
    const algo = elements.selectAlgorithm.value;

    if (asset === 'Personalizado' && !customTicker) {
      showToast('Por favor, informe o Ticker Yahoo Finance.', 'error');
      elements.inputCustomTicker.focus();
      return;
    }

    // Coleta hiperparâmetros dinâmicos do formulário
    const payload = {
      asset: asset,
      custom_ticker: customTicker,
      period: period,
      start_date: startDate,
      end_date: endDate,
      algoritmo: algo
    };

    const dynamicInputs = elements.dynamicParamsContainer.querySelectorAll('.dynamic-param');
    dynamicInputs.forEach(inp => {
      payload[inp.name] = inp.value;
    });

    setRunningState(true);

    try {
      // 1. Inicia tarefa no servidor via POST AJAX
      const res = await fetch('/api/iniciar-treinamento', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.erro || 'Falha ao iniciar processamento.');
      }

      const initData = await res.json();
      state.currentTaskId = initData.id_tarefa;

      elements.progressStatusText.textContent = 'Tarefa iniciada. Monitorando progresso...';
      elements.progressBarFill.style.width = '8%';
      elements.progressPercentage.textContent = '8%';

      // 2. Loop de polling AJAX consultando a Sessão do servidor a cada 350ms
      if (state.pollingInterval) clearInterval(state.pollingInterval);

      state.pollingInterval = setInterval(async () => {
        try {
          const pollRes = await fetch(`/api/progresso-sessao?id_tarefa=${encodeURIComponent(state.currentTaskId)}`);
          if (!pollRes.ok) return;

          const pollData = await pollRes.json();

          // Atualiza barra de progresso e textos
          const prog = pollData.progresso || 0;
          elements.progressBarFill.style.width = `${prog}%`;
          elements.progressPercentage.textContent = `${prog}%`;
          elements.progressStatusText.textContent = pollData.mensagem_status || 'Processando...';

          if (pollData.concluido) {
            clearInterval(state.pollingInterval);
            state.pollingInterval = null;

            if (pollData.erro) {
              showToast(`Erro na execução: ${pollData.erro}`, 'error');
              elements.progressStatusText.textContent = 'Falha no processamento.';
              setRunningState(false);
            } else if (pollData.resultado) {
              handleOptimizationSuccess(pollData.resultado);
              setRunningState(false);
            }
          }
        } catch (errPoll) {
          console.warn('Falha na consulta de polling AJAX:', errPoll);
        }
      }, 350);

    } catch (err) {
      console.error('Erro ao disparar treinamento:', err);
      showToast(`Erro: ${err.message}`, 'error');
      setRunningState(false);
    }
  }

  function setRunningState(running) {
    state.isRunning = running;
    elements.btnRun.disabled = running;
    elements.btnExportPdf.disabled = running;
    elements.btnExportDocx.disabled = running;
    elements.selectAlgorithm.disabled = running;
    elements.selectAsset.disabled = running;
    elements.selectPeriod.disabled = running;

    if (running) {
      elements.progressContainer.classList.remove('hidden');
      elements.connectionBadge.className = 'badge-status badge-busy';
      elements.connectionText.textContent = 'Processando com IA...';
      elements.progressBarFill.style.width = '5%';
      elements.progressPercentage.textContent = '5%';
      elements.progressStatusText.textContent = 'Iniciando modelo...';
    } else {
      elements.connectionBadge.className = 'badge-status badge-ready';
      elements.connectionText.textContent = 'Pronto para Otimização';
    }
  }

  function handleOptimizationSuccess(data) {
    elements.progressBarFill.style.width = '100%';
    elements.progressPercentage.textContent = '100%';
    elements.progressStatusText.textContent = 'Otimização concluída!';
    elements.btnExportPdf.disabled = false;
    elements.btnExportDocx.disabled = false;

    state.currentCurrency = data.currency || 'USD';
    const metrics = data.metrics;

    // Atualiza KPIs
    elements.kpiActual.textContent = formatCurrency(metrics.ultimo_preco_real, state.currentCurrency);
    elements.kpiPredicted.textContent = formatCurrency(metrics.preco_projetado_final, state.currentCurrency);
    elements.kpiTickerLabel.textContent = `${data.asset_name} (${data.ticker})`;
    elements.kpiHorizonLabel.textContent = `${data.forecast.length} dias projetados`;
    elements.kpiAlgoBadge.textContent = data.nome_algoritmo || 'MODELO IA';

    const varPct = metrics.variacao_esperada_pct;
    elements.kpiVariation.textContent = `${varPct > 0 ? '+' : ''}${varPct.toFixed(2)}%`;

    if (varPct > 0.05) {
      elements.kpiTrendTag.className = 'kpi-badge badge-success';
      elements.kpiTrendTag.textContent = 'ALTA (BULL)';
      elements.kpiVariation.style.color = '#10b981';
    } else if (varPct < -0.05) {
      elements.kpiTrendTag.className = 'kpi-badge badge-danger';
      elements.kpiTrendTag.textContent = 'BAIXA (BEAR)';
      elements.kpiVariation.style.color = '#f43f5e';
    } else {
      elements.kpiTrendTag.className = 'kpi-badge badge-neutral';
      elements.kpiTrendTag.textContent = 'NEUTRO';
      elements.kpiVariation.style.color = '#f8fafc';
    }

    elements.kpiRmse.textContent = metrics.rmse.toFixed(4);
    elements.kpiDirectional.textContent = `${metrics.acuracia_direcional.toFixed(1)}%`;

    // Gráficos
    renderForecastChart(data);
    renderConvergenceChart(data.ga_history, data.nome_algoritmo);

    // Tabela
    renderForecastTable(data, metrics.ultimo_preco_real);

    showToast(`Previsão com ${data.nome_algoritmo} concluída! Tendência: ${metrics.tendencia_esperada}`, 'success');
  }

  function renderForecastChart(data) {
    const history = data.history || [];
    const forecast = data.forecast || [];

    const histLabels = history.map(h => formatDateLabel(h.date));
    const histReal = history.map(h => h.real);
    const histPred = history.map(h => h.pred);

    const lastHistDate = histLabels[histLabels.length - 1];
    const lastHistPrice = histReal[histReal.length - 1];

    const foreLabels = [lastHistDate, ...forecast.map(f => formatDateLabel(f.date))];
    const foreProj = [lastHistPrice, ...forecast.map(f => f.projected)];
    const foreUpper = [lastHistPrice, ...forecast.map(f => f.upper)];
    const foreLower = [lastHistPrice, ...forecast.map(f => f.lower)];

    const totalLabels = [...histLabels, ...foreLabels.slice(1)];
    const paddedHistReal = [...histReal, ...new Array(forecast.length).fill(null)];
    const paddedHistPred = [...histPred, ...new Array(forecast.length).fill(null)];

    const paddedForeProj = [...new Array(histReal.length - 1).fill(null), ...foreProj];
    const paddedForeUpper = [...new Array(histReal.length - 1).fill(null), ...foreUpper];
    const paddedForeLower = [...new Array(histReal.length - 1).fill(null), ...foreLower];

    elements.chartForecastTitle.textContent = `Série Histórica e Projeção IA [${data.nome_algoritmo}] - ${data.asset_name}`;

    state.forecastChart.data.labels = totalLabels;
    state.forecastChart.data.datasets[0].data = paddedHistReal;
    state.forecastChart.data.datasets[1].data = paddedHistPred;
    state.forecastChart.data.datasets[2].data = paddedForeProj;
    state.forecastChart.data.datasets[3].data = paddedForeUpper;
    state.forecastChart.data.datasets[4].data = paddedForeLower;

    state.forecastChart.update();
  }

  function renderConvergenceChart(gaHistory, nomeAlgo) {
    if (!gaHistory || !gaHistory.generations) return;

    elements.chartConvTitle.textContent = `Curva de Aprendizado / Convergência [${nomeAlgo || 'Modelo'}]`;
    state.convergenceChart.data.labels = gaHistory.generations;
    state.convergenceChart.data.datasets[0].data = gaHistory.best_fitness;
    state.convergenceChart.data.datasets[1].data = gaHistory.avg_fitness;
    state.convergenceChart.update();
  }

  function renderForecastTable(data, lastRefPrice) {
    elements.forecastTableBody.innerHTML = '';
    const forecast = data.forecast || [];
    elements.tableInfoCounter.textContent = `${forecast.length} dias projetados`;

    if (forecast.length === 0) {
      elements.forecastTableBody.innerHTML = '<tr><td colspan="6" class="empty-state-cell">Nenhum dado projetado.</td></tr>';
      return;
    }

    forecast.forEach(item => {
      const stepVar = ((item.projected - lastRefPrice) / (lastRefPrice + 1e-9)) * 100;
      const isUp = stepVar >= 0;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="font-mono">${formatDateDisplay(item.date)}</td>
        <td class="font-mono font-bold" style="color: #38bdf8;">${formatCurrency(item.projected, data.currency)}</td>
        <td class="font-mono text-muted">${formatCurrency(item.lower, data.currency)}</td>
        <td class="font-mono text-muted">${formatCurrency(item.upper, data.currency)}</td>
        <td class="font-mono" style="color: ${isUp ? '#10b981' : '#f43f5e'}; font-weight: 600;">
          ${isUp ? '+' : ''}${stepVar.toFixed(2)}%
        </td>
        <td>
          <span class="badge-mini" style="background: ${isUp ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)'}; color: ${isUp ? '#6ee7b7' : '#fda4af'};">
            ${isUp ? '▲ Alta' : '▼ Baixa'}
          </span>
        </td>
      `;
      elements.forecastTableBody.appendChild(tr);
    });
  }

  // =========================================================================
  // 6. EXPORTAÇÃO DE RELATÓRIOS (PDF E DOCX)
  // =========================================================================
  async function exportPdfReport() {
    showToast('Gerando gráficos de alta resolução e compilando PDF...', 'info');
    try {
      const response = await fetch('/api/export-pdf');
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        showToast(errData.error || 'Erro ao gerar PDF.', 'error');
        return;
      }

      let filename = 'relatorio.pdf';
      const disposition = response.headers.get('Content-Disposition');
      if (disposition && disposition.indexOf('filename=') !== -1) {
        const match = disposition.match(/filename=["']?([^"';]+)["']?/);
        if (match && match[1]) filename = match[1];
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      showToast(`Download de "${filename}" concluído!`, 'success');
    } catch (e) {
      showToast('Falha na comunicação ao exportar PDF: ' + e.message, 'error');
    }
  }

  async function exportDocxReport() {
    showToast('Gerando gráficos e relatório Word (.docx)...', 'info');
    try {
      const response = await fetch('/api/export-docx');
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        showToast(errData.error || 'Erro ao gerar Word (.docx).', 'error');
        return;
      }

      let filename = 'relatorio.docx';
      const disposition = response.headers.get('Content-Disposition');
      if (disposition && disposition.indexOf('filename=') !== -1) {
        const match = disposition.match(/filename=["']?([^"';]+)["']?/);
        if (match && match[1]) filename = match[1];
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      showToast(`Download de "${filename}" concluído!`, 'success');
    } catch (e) {
      showToast('Falha na comunicação ao exportar Word: ' + e.message, 'error');
    }
  }

  // =========================================================================
  // 7. HISTÓRICO E ARQUIVO DE RELATÓRIOS
  // =========================================================================
  async function openHistoryModal() {
    elements.modalHistory.classList.remove('hidden');
    await loadHistoryData();
  }

  async function loadHistoryData() {
    elements.historyTableBody.innerHTML = '<tr><td colspan="10" class="empty-state-cell">Carregando registros...</td></tr>';
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      const records = data.records || [];

      if (records.length === 0) {
        elements.historyTableBody.innerHTML = '<tr><td colspan="10" class="empty-state-cell">Nenhum ensaio registrado ainda.</td></tr>';
        return;
      }

      elements.historyTableBody.innerHTML = '';
      records.forEach(rec => {
        const tr = document.createElement('tr');
        const varVal = parseFloat(rec.Variacao_Esperada_Pct) || 0;
        const isUp = varVal >= 0;

        tr.innerHTML = `
          <td class="font-mono text-muted">${rec.Data_Hora || '-'}</td>
          <td><strong>${rec.Ativo || '-'}</strong></td>
          <td class="font-mono">${rec.Ticker || '-'}</td>
          <td>${rec.Periodo || '-'}</td>
          <td class="font-mono">${formatCurrency(parseFloat(rec.Ultimo_Preco_Real), rec.Moeda)}</td>
          <td class="font-mono">${formatCurrency(parseFloat(rec.Preco_Projetado_Final), rec.Moeda)}</td>
          <td class="font-mono" style="color: ${isUp ? '#10b981' : '#f43f5e'}; font-weight: 600;">
            ${isUp ? '+' : ''}${varVal.toFixed(2)}%
          </td>
          <td class="font-mono">${parseFloat(rec.RMSE || 0).toFixed(4)}</td>
          <td class="font-mono">${parseFloat(rec.Acuracia_Direcional_Pct || 0).toFixed(1)}%</td>
          <td><span class="badge-mini">${rec.Tendencia || '-'}</span></td>
        `;
        elements.historyTableBody.appendChild(tr);
      });
    } catch (err) {
      elements.historyTableBody.innerHTML = '<tr><td colspan="10" class="empty-state-cell">Erro ao carregar histórico.</td></tr>';
    }
  }

  async function openReportsModal() {
    elements.modalReports.classList.remove('hidden');
    await loadReportsData();
  }

  async function loadReportsData() {
    elements.reportsListContainer.innerHTML = '<p class="empty-state-cell">Carregando lista de relatórios...</p>';
    try {
      const res = await fetch('/api/reports');
      const data = await res.json();
      const reports = data.reports || [];

      if (reports.length === 0) {
        elements.reportsListContainer.innerHTML = '<p class="empty-state-cell">Nenhum relatório emitido ainda.</p>';
        return;
      }

      elements.reportsListContainer.innerHTML = '';
      reports.forEach(r => {
        const item = document.createElement('div');
        item.className = 'report-item';
        const isPdf = r.ext === 'pdf';
        const btnClass = isPdf ? 'btn-pdf' : 'btn-docx';
        const badgeLabel = isPdf ? 'PDF' : 'DOCX';

        item.innerHTML = `
          <div class="report-info">
            <span class="report-filename">
              <span class="badge-mini" style="${isPdf ? 'background: rgba(225,29,72,0.2); color: #fda4af;' : 'background: rgba(16,185,129,0.2); color: #6ee7b7;'}">${badgeLabel}</span>
              ${r.filename}
            </span>
            <span class="report-meta">Tamanho: ${r.size_kb} KB | Criado em: ${r.modified}</span>
          </div>
          <a href="/api/reports/${encodeURIComponent(r.filename)}" class="btn ${btnClass}" style="width: auto; padding: 6px 14px; font-size: 0.78rem;">
            Baixar ${badgeLabel}
          </a>
        `;
        elements.reportsListContainer.appendChild(item);
      });
    } catch (err) {
      elements.reportsListContainer.innerHTML = '<p class="empty-state-cell">Erro ao carregar relatórios.</p>';
    }
  }

  // =========================================================================
  // 8. FORMATADORES & TOASTS
  // =========================================================================
  function formatCurrency(val, currency = 'USD') {
    if (val === null || val === undefined || isNaN(val)) return '--';
    const num = Number(val);
    const curr = currency === 'BRL' ? 'R$' : currency === 'USD' ? '$' : '';
    return `${curr} ${num.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`;
  }

  function formatNumber(val) {
    if (val === null || val === undefined || isNaN(val)) return '';
    return Number(val).toLocaleString('pt-BR', { maximumFractionDigits: 2 });
  }

  function formatDateLabel(dateStr) {
    if (!dateStr) return '';
    const parts = dateStr.split('-');
    if (parts.length === 3) return `${parts[2]}/${parts[1]}`;
    return dateStr;
  }

  function formatDateDisplay(dateStr) {
    if (!dateStr) return '';
    const parts = dateStr.split('-');
    if (parts.length === 3) return `${parts[2]}/${parts[1]}/${parts[0]}`;
    return dateStr;
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    toast.innerHTML = `<span><strong>${icon}</strong></span> <span>${message}</span>`;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  initApp();
});
