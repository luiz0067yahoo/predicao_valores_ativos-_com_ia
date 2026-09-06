# AI Asset Predictor: Previsão de Ativos Financeiros e Commodities com Algoritmo Genético e Yahoo Finance

Sistema em Python com Interface Gráfica Interativa (GUI) que integra inteligência artificial evolutiva (**Algoritmo Genético**) e modelos de séries temporais para modelagem, otimização de pesos e projeção de cotações futuras de ativos financeiros, moedas e commodities agrícolas/pecuárias em tempo real via API do Yahoo Finance (`yfinance`).

---

## 🎯 Ativos Pré-configurados e Suportados

| Categoria | Ativo | Ticker Yahoo Finance | Descrição |
| :--- | :--- | :--- | :--- |
| **Câmbio** | Dólar Americano | `USDBRL=X` | Cotação USD/BRL em Reais |
| **Câmbio** | Euro | `EURBRL=X` | Cotação EUR/BRL em Reais |
| **Criptomoedas** | Bitcoin (USD) | `BTC-USD` | Cotação do BTC em Dólares |
| **Criptomoedas** | Bitcoin (BRL) | `BTC-BRL` | Cotação do BTC em Reais |
| **Índices** | Ibovespa | `^BVSP` | Índice de Ações da B3 (Brasil) |
| **Commodities** | Soja | `ZS=F` | Contratos Futuros de Soja (CBOT) |
| **Commodities** | Milho | `ZC=F` | Contratos Futuros de Milho (CBOT) |
| **Commodities** | Café Arábica | `KC=F` | Contratos Futuros de Café (ICE) |
| **Pecuária** | Boi Gordo | `LE=F` | Live Cattle Futures (CME) |
| **Pecuária** | Gado de Engorda | `GF=F` | Feeder Cattle Futures |
| **Personalizado** | Custom Ticker | *Livre escolha* | Suporta qualquer ação, índice ou ETF global (ex: `PETR4.SA`, `VALE3.SA`, `AAPL`, `SPY`) |

---

## 🧠 Arquitetura do Algoritmo Genético

O motor evolutivo foi projetado especificamente para séries temporais financeiras:

- **Representação Cromossômica:** Vetor contínuo de genes em ponto flutuante que pondera defasagens temporais autorregressivas (*lags* de lookback $t-1, \dots, t-k$), indicadores de momentum e osciladores técnicos (*RSI*, *MACD*, *Volatilidade*, *Médias Móveis* e *Bandas de Bollinger*) com termo de viés (*bias*).
- **Função de Aptidão (Fitness):** Minimiza o erro quadrático e absoluto ponderado ($\text{RMSE}$ e $\text{MAE}$), penaliza dispersões extremas através de regularização L2 implícita e bonifica assertividade no sentido direcional da variação de preços:
  $$\text{Fitness} = \frac{1000}{1.0 + 10 \cdot \text{RMSE} + 5 \cdot \text{MAE}} \times (1.0 + 0.4 \cdot \text{Acurácia Direcional}) - 0.01 \sum w_i^2$$
- **Seleção:** Seleção por Torneio estocástico configurável ($k=4$).
- **Recombinação (Crossover):** Cruzamento aritmético ponderado combinado com dispersão espacial exploratória (BLX-$\alpha$).
- **Mutação:** Mutação gaussiana com perturbação adaptativa decrescente ao longo das gerações $\sigma(g) = \sigma_0 \cdot (1 - g/G)$ com taxa de reset estocástica para evasão de mínimos locais.
- **Elitismo:** Preservação estrita dos melhores indivíduos no topo de cada geração.

---

## 📊 Relatórios Executivos em DOCX e Planilhamento

1. **Planilhamento Automático de Ensaios (`output/logs/historico_ensaios.csv`):**
   - Registra cada experimento com timestamp, ativo, hiperparâmetros do GA, métricas estatísticas ($\text{RMSE}$, $\text{MAE}$, $\text{MAPE}$, $R^2$, Acurácia Direcional) e preços futuros.
2. **Relatório Executivo em Word (`.docx`):**
   - Geração com layout corporativo profissional.
   - Tabelas estilizadas com formatação de cores e bordas elegantes.
   - Gráficos analíticos de alta resolução (300 DPI) embutidos diretamente no documento:
     - Gráfico da Série Histórica, Ajuste do GA e Projeção Futura com intervalo de confiança empírico (95%).
     - Gráfico da Curva de Aprendizado e Convergência Evolutiva (Fitness Máximo vs. Médio).

---

## 🚀 Instalação e Execução

### 1. Pré-requisitos
Certifique-se de possuir o Python 3.10+ instalado em seu sistema.

### 2. Instalação das Dependências
Clone ou acesse o repositório e instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 3. Executando a Interface Gráfica (GUI)
Para iniciar o aplicativo com dashboard interativo:
```bash
python main.py
```

### 4. Executando os Testes Automatizados
Para rodar a suíte completa de testes unitários:
```bash
python -m pytest -v
```

---

## 📂 Estrutura Modular do Projeto

```
predicao_valores_ativos-_com_ia/
├── config.py                 # Mapeamento de tickers, hiperparâmetros padrão e diretórios
├── data_fetcher.py           # Conexão com Yahoo Finance, cache e indicadores técnicos
├── genetic_engine.py         # Motor do Algoritmo Genético (Cromossomos, Seleção, Crossover, Mutação)
├── model_predictor.py        # Reconstrução de escalas, métricas (RMSE, MAE, MAPE, R²) e projeção recursiva
├── report_generator.py       # Planilhamento de ensaios e emissão do relatório executivo em Word (.docx)
├── gui.py                    # Interface Gráfica moderna com Tkinter e Matplotlib embutido
├── main.py                   # Ponto de entrada da aplicação
├── requirements.txt          # Dependências do projeto
├── output/
│   ├── reports/              # Relatórios DOCX e gráficos gerados
│   └── logs/                 # Planilha CSV de histórico de ensaios
└── tests/                    # Suíte de testes unitários e de integração
    ├── test_data_fetcher.py
    ├── test_genetic_engine.py
    ├── test_model_predictor.py
    ├── test_report_generator.py
    └── test_integration.py
```
