# Pipeline de Engenharia de Características M.A.P.P.

O módulo `mapp.features` implementa um pipeline unificado e sem vazamento de dados (`extrair_todas_caracteristicas_mapp`) que transforma dados brutos de OHLCV em mais de 70 indicadores quantitativos, técnicos e estatísticos.

---

## Módulos de Características

| Módulo | Arquivo Fonte | Principais Indicadores | Documentação |
| :--- | :--- | :--- | :--- |
| **Regimes de Mercado** | `market_regime.py` | 8 Regimes estruturais, força da tendência, score de volatilidade | [MARKET_REGIME.md](./MARKET_REGIME.md) |
| **Tendência & Médias** | `features/trend.py` | SMAs (20, 50, 100, 200), EMAs (9, 21, 50), ADX, Inclinações, Aceleração | [TREND.md](./TREND.md) |
| **Volume & Fluxo** | `features/volume.py` | Volume Relativo, Z-Score de Volume, OBV, VWAP, Volume de Rompimento | [VOLUME.md](./VOLUME.md) |
| **Volatilidade & Faixas** | `features/volatility.py` | ATR, ATR%, Volatilidade Histórica/Realizada, Bandas de Bollinger, Compressão | [VOLATILITY.md](./VOLATILITY.md) |
| **Momento & Osciladores** | `features/momentum.py` | Retornos Multiperíodo (1, 3, 5, 10, 20), RSI, Estocástico, Williams %R, ROC, MACD | [MOMENTUM.md](./MOMENTUM.md) |
| **Velas Japonesas** | `features/candlesticks.py` | Martelo, Estrela Cadente, Engolfo de Alta/Baixa, Doji, Nuvem Negra, Estrela da Manhã | [CANDLESTICKS.md](./CANDLESTICKS.md) |
| **Suportes & Resistências** | `features/support_resistance.py` | Pivôs locais, Contagem de toques, Distância ao suporte/resistência mais próximo | [SUPPORT_RESISTANCE.md](./SUPPORT_RESISTANCE.md) |
| **Retrações de Fibonacci** | `features/fibonacci.py` | Níveis 23.6%, 38.2%, 50.0%, 61.8%, 78.6%, Rejeições, Confluências | [FIBONACCI.md](./FIBONACCI.md) |
| **Divergências** | `features/divergences.py` | Divergências altistas e baixistas entre preço e RSI / MACD / OBV | [DIVERGENCES.md](./DIVERGENCES.md) |

---

## Garantias Anti-Vazamento (Anti-Data-Leakage)
1. **Sem Olhar para Frente**: Todas as médias móveis, pivôs e osciladores utilizam estritamente o operador `.shift(1)` ou janelas de cálculo terminadas na barra corrente $t$.
2. **Normalização Local**: Escalas e Z-scores são computados através de janelas móveis de histórico passado, nunca utilizando estatísticas de todo o dataset (Global Mean/Std).
3. **Validação Automática**: O módulo `ValidadorAntiVazamento` realiza testes de perturbação no futuro para atestar a independência temporal dos sinais gerados.
