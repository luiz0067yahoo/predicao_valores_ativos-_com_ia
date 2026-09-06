# Motor de Backtesting Quantitativo e Validação Walk-Forward

## 1. Visão Geral
O módulo `mapp.backtesting` implementa um motor de simulação histórica realista (`BacktestEngine`) desenhado para avaliar estratégias de trading e algoritmos preditivos sem qualquer viés de antecipação (lookahead bias) ou sobreajuste (overfitting).

---

## 2. Validação Walk-Forward
Diferente da validação cruzada K-Fold padrão (que quebra a estrutura causal temporal dos mercados), o motor utiliza a metodologia **Walk-Forward** com janelas expansivas ou deslizantes:
1. O modelo é treinado até a data $t$;
2. A previsão é gerada para o período de teste $t \to t+h$;
3. A janela de treino avança para $t+h$;
4. O processo é repetido até cobrir todo o período histórico disponível.

---

## 3. Custos Operacionais e Modelagem de Execução
Para garantir fidelidade com o ambiente real de negociação, o simulador incorpora:
- **Taxas de Corretagem e Emolumentos**: Percentual parametrizável por ordem executada (padrão: 0.05% para ações B3 / 0.10% para cripto);
- **Slippage (Deslizamento de Preço)**: Penalidade percentual sobre o preço de execução simulando atrasos e impacto no livro de ofertas (padrão: 0.02%);
- **Dimensionamento de Posição (Position Sizing)**: Alocação proporcional ao capital líquido disponível ou fração de volatilidade;
- **Stop Loss e Take Profit**: Saídas automáticas baseadas em múltiplos de ATR ou percentual de perda máxima.

---

## 4. Métricas Financeiras e Estatísticas de Risco
O módulo `CalculadorMetricas` calcula um conjunto completo de métricas institucionais:
- **Retorno Acumulado Total (%)**: $\frac{V_{\text{final}} - V_{\text{inicial}}}{V_{\text{inicial}}} \times 100$
- **Taxa de Acerto (Win Rate %)**: $\frac{\text{Trades Positivos}}{\text{Total de Trades}} \times 100$
- **Fator de Lucro (Profit Factor)**: $\frac{\sum \text{Lucros}}{\sum |\text{Prejuízos}|}$
- **Máximo Rebaixamento (Max Drawdown %)**: $\max_t \left(\frac{\text{Pico}_t - \text{Vale}_t}{\text{Pico}_t}\right) \times 100$
- **Índice de Sharpe**: Retorno excedente anualizado dividido pelo desvio padrão dos retornos.
- **Índice de Sortino**: Retorno excedente anualizado dividido pelo desvio padrão negativo (Downside Deviation).
- **Índice de Calmar**: Retorno anualizado composto dividido pelo Máximo Drawdown.
- **Métricas de Erro Preditivo**: RMSE, MAE, MAPE e Coeficiente de Determinação $R^2$.
