# Modelo Ensemble Multi-Regime

## 1. Visão Geral
O **Modelo Ensemble** combina as previsões de múltiplos algoritmos da plataforma através de um mecanismo de consenso ponderado. A ponderação pode ser estática (equiponderada) ou dinâmica, adaptando-se em tempo real ao **Regime de Mercado** ativo identificado pelo `MarketRegimeDetector`.

---

## 2. Ponderação Adaptativa por Regime
Diferentes famílias de algoritmos apresentam desempenhos assimétricos sob diferentes condições de mercado:
- **Tendência Forte (`BULL_TREND` / `BEAR_TREND`)**: Modelos baseados em momentum e árvores de decisão (`xgboost`, `lightgbm`, `mapp`) recebem maior peso.
- **Lateralização (`SIDEWAYS` / `LOW_VOLATILITY`)**: Modelos autoregressivos e estocásticos (`arima_sarima`, `pattern_matching`, `logica_fuzzy`) recebem maior peso.
- **Alta Volatilidade / Rompimento (`BREAKOUT` / `HIGH_VOLATILITY`)**: Redes neurais recorrentes (`lstm`, `gru`, `transformer`) e o modelo `mapp` são priorizados.

A previsão combinada $\hat{y}_t^{\text{ensemble}}$ é obtida por:

$$\hat{y}_t^{\text{ensemble}} = \sum_{m=1}^{M} w_m(S_t) \cdot \hat{y}_{t}^{(m)}$$

Onde $\sum_{m=1}^{M} w_m(S_t) = 1$ e $w_m(S_t) \ge 0$.

---

## 3. Parâmetros de Configuração
| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `mode` | str | `"regime_adaptive"` | Estratégia de ponderação (`regime_adaptive` ou `equal`) |
| `top_models` | int | 3 | Quantidade máxima de modelos combinados |
| `horizonte_projecao` | int | 5 | Número de barras de previsão futura |

---

## 4. Vantagens Teóricas
- **Redução de Variância**: A média ponderada reduz o erro de generalização assintótico (princípio de Condorcet e teorema de Bias-Variance).
- **Robustez Estrutural**: Reduz a sensibilidade do sistema a falhas isoladas de um modelo individual durante transições abruptas de regime.
