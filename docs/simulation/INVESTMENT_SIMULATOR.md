# Simulador de Investimentos Multi-Algoritmo

## 1. Visão Geral
O **Simulador de Investimentos M.A.P.P.** (`/portfolio-simulator`) permite comparar em paralelo o desempenho de múltiplos algoritmos de Inteligência Artificial e modelos preditivos em um mesmo ativo, simulando curvas de patrimônio (Equity Curve), métricas de drawdown e taxas de retorno ajustadas ao risco.

---

## 2. Wizard em 5 Etapas
A interface do simulador guia o usuário através de um fluxo estruturado:
1. **Passo 1: Seleção do Ativo**: Escolha entre ativos pré-configurados (Ibovespa, Criptomoedas, Ações BR, Ações Globais, Forex) ou Ticker personalizado do Yahoo Finance.
2. **Passo 2: Capital Inicial & Custos**: Configuração do saldo inicial (Ex: R$ 10.000,00), moeda e custos operacionais (corretagem e slippage).
3. **Passo 3: Horizonte Temporal**: Definição do período de backtesting e normalização do horizonte (dias, semanas, meses, anos), com diferenciação automática entre dias úteis de bolsa e mercados 24/7 de criptoativos.
4. **Passo 4: Seleção de Algoritmos**: Escolha de 2 ou mais algoritmos para confronto simultâneo.
5. **Passo 5: Parâmetros & Execução**: Ajuste fino de hiperparâmetros de cada modelo selecionado e disparo da simulação.

---

## 3. Monitoramento em Tempo Real (`ProgressTracker`)
Durante o processamento da simulação, o sistema não utiliza timers fictícios ou barras estáticas. O `ProgressTracker` monitora a execução real em segundo plano através de:
- **Tempo Decorrido Real**: Medido via `time.perf_counter()`;
- **Tempo Restante Estimado**: Calculado a partir da taxa de processamento por passo ($T_{\text{médio}} \times N_{\text{restante}}$);
- **Tempo Total Estimado**: Soma dinâmica formatada rigorosamente no padrão `HH:MM:SS`.

---

## 4. Sistema de Ranqueamento Multicritério
Ao término da simulação, o sistema classifica os modelos através de uma pontuação multicritério ponderada:

$$\text{Score} = w_{\text{ret}} \cdot S(\text{Retorno}) + w_{\text{sharpe}} \cdot S(\text{Sharpe}) + w_{\text{sortino}} \cdot S(\text{Sortino}) - w_{\text{dd}} \cdot S(\text{MaxDD}) - w_{\text{rmse}} \cdot S(\text{RMSE})$$

Onde cada indicador $S(\cdot)$ é normalizado entre 0 e 100 em relação aos pares do teste.
O algoritmo com maior pontuação é premiado com a medalha de ouro 🥇 e destacado no relatório executivo.
