w# Classificador de Regimes de Mercado (`MarketRegimeDetector`)

## 1. Visão Geral
O classificador de regimes de mercado categoriza dinamicamente cada barra histórica em um dos 8 regimes fundamentais, utilizando métricas simultâneas de tendência, volatilidade, momentum e volume.

---

## 2. Os 8 Regimes de Mercado
| Código do Regime | Nome em Português | Condição Técnica Principal | Comportamento Típico de Mercado |
| :--- | :--- | :--- | :--- |
| `BULL_TREND` | Tendência de Alta | $P > SMA_{50} > SMA_{200}$, $ADX > 22$, inclinação positiva | Movimentos direcionais ascendentes consistentes |
| `BEAR_TREND` | Tendência de Baixa | $P < SMA_{50} < SMA_{200}$, $ADX > 22$, inclinação negativa | Movimentos direcionais descendentes consistentes |
| `SIDEWAYS` | Lateralização | $ADX < 20$, médias móveis entrelaçadas | Faixa de preços estreita, sem direção clara |
| `HIGH_VOLATILITY` | Alta Volatilidade | $ATR > 1.8 \times \overline{ATR}_{50}$ ou $BB_{width} > 2.0 \sigma$ | Grandes oscilações intradiárias, risco elevado |
| `LOW_VOLATILITY` | Baixa Volatilidade | $ATR < 0.6 \times \overline{ATR}_{50}$, compressão de Bollinger | Acumulação prévia a rompimentos |
| `BREAKOUT` | Rompimento | Fechamento rompendo máxima de 20 barras com volume $> 1.8 \times$ média | Início explosivo de nova tendência |
| `CRASH` | Queda Acentuada | Queda acumulada de 3 dias $> 2.5 \times ATR$ com pico de volume | Venda em pânico, liquidações forçadas |
| `RECOVERY` | Recuperação | Reversão após queda acentuada, RSI saindo de sobrevenda com divergência | Repique técnico ou formação de fundo |

---

## 3. Características Numéricas Extraídas
O `MarketRegimeDetector` enriquece o dataset com as seguintes features normalizadas:
- `regime_bull_trend_flag`: Variável binária de alta persistente;
- `regime_bear_trend_flag`: Variável binária de baixa persistente;
- `regime_volatility_score`: Nível relativo de volatilidade recente [0, 1];
- `regime_trend_strength`: Intensidade da tendência medida por $ADX / 100$;
- `regime_code`: Identificador ordinal categórico [0 a 7].
