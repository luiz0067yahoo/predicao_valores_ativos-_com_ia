# Modelo M.A.P.P. (Market Analysis Pattern Prediction)

## 1. Visão Geral
O **M.A.P.P.** é o modelo quantitativo proprietário da plataforma que sintetiza quatro pilares essenciais da análise de mercado em uma previsão probabilística calibrada:
1. **Regime de Mercado**: Identificação do estado macro/estrutural do ativo através do `MarketRegimeDetector` (Tendência de Alta, Tendência de Baixa, Lateralização, Alta/Baixa Volatilidade, Rompimento, Crash, Recuperação).
2. **Estrutura e Acumulação**: Análise de suportes, resistências, acumulação de volume (VWAP, OBV) e compressão de volatilidade.
3. **Padrões Técnicos**: Reconhecimento de velas japonesas clássicas, padrões gráficos (topos/fundos duplos, triângulos, OCO) e retrações de Fibonacci.
4. **Momento & Divergências**: RSI, MACD, Estocástico e divergências altistas/baixistas de preço vs osciladores.

---

## 2. Hipótese Teórica e Formulação Matemática
O modelo assume que o retorno esperado $R_{t+h}$ em um horizonte $h$ é condicionado ao regime de mercado $S_t \in \{1, \dots, K\}$ e a um vetor de características selecionadas $X_t$:

$$P(Y_{t+h} = 1 \mid X_t, S_t) = \sigma\left( \mathbf{w}_{S_t}^T X_t + b_{S_t} \right)$$

Onde:
- $Y_{t+h} = 1$ indica movimento altista ($R_{t+h} > 0$);
- $\mathbf{w}_{S_t}$ são os coeficientes calibrados especificamente para o regime $S_t$;
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ é a função sigmoide logística.

A magnitude esperada do movimento é modelada por:

$$\hat{\Delta} = \text{E}[|R_{t+h}|] \cdot (2 \cdot P(\text{Alta}) - 1)$$

O intervalo de confiança de 95% é calculado considerando a volatilidade realizada recente $\sigma_{\text{realizada}}$ e a raiz quadrada do horizonte ajustado:

$$\text{IC}_{95\%} = \left[ P_t \cdot \left(1 + \hat{\Delta} - 1.96 \cdot \sigma_{\text{realizada}} \sqrt{\frac{h}{252}}\right), \; P_t \cdot \left(1 + \hat{\Delta} + 1.96 \cdot \sigma_{\text{realizada}} \sqrt{\frac{h}{252}}\right) \right]$$

---

## 3. Seleção de Características e Prevenção de Vazamento
- **Seleção**: Utiliza o `SeletorCaracteristicas` com eliminação de colinearidade (Pearson $|\rho| > 0.85$) e ranqueamento por Informação Mútua (Mutual Information).
- **Anti-Vazamento**: Verificado pelo `ValidadorAntiVazamento`, garantindo que $X_t$ contenha estritamente informações disponíveis no momento $t$ (sem olhar para frente).

---

## 4. Parâmetros de Configuração
| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `n_features` | int | 15 | Quantidade de características mais preditivas a reter |
| `confidence_level` | float | 0.95 | Nível de confiança para o cone de incerteza |
| `horizonte_projecao` | int | 5 | Número de barras de previsão futura |

---

## 5. Casos de Uso Recomendados
- Ativos com forte alternância de regimes de volatilidade e tendência (ações de alta liquidez, índices, criptomoedas e commodities).
- Operações posicionais (swing trade e position) onde o controle de risco e intervalos de confiança são críticos.
