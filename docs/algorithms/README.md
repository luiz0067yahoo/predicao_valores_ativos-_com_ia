# Catálogo de Algoritmos Quantitativos e IA

A plataforma conta com **16 algoritmos preditivos** cobrindo Aprendizado de Máquina Supervisionado, Deep Learning Recorrente e com Atenção, Séries Temporais Clássicas, Computação Evolucionária, Simulação Multiagente, Lógica Fuzzy e Metodologias Proprietárias de Engenharia de Características.

---

## Tabela Comparativa de Algoritmos

| Algoritmo | Família / Categoria | Desempenho Típico | Horizonte Ideal | Complexidade Computacional | Documentação |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M.A.P.P.** | Quantitativo / Multi-Pilar | ⭐⭐⭐⭐⭐ | Curto a Médio (1 a 30 dias) | Média | [MAPP.md](./MAPP.md) |
| **Ensemble** | Meta-Modelo Adaptativo | ⭐⭐⭐⭐⭐ | Curto a Longo (1 a 90 dias) | Média-Alta | [ENSEMBLE.md](./ENSEMBLE.md) |
| **Pattern Matching** | Análogos Históricos / KNN | ⭐⭐⭐⭐ | Curto a Médio (5 a 20 dias) | Baixa-Média | [PATTERN_MATCHING.md](./PATTERN_MATCHING.md) |
| **XGBoost** | Gradient Tree Boosting | ⭐⭐⭐⭐⭐ | Curto a Médio (1 a 15 dias) | Baixa-Média | [XGBOOST.md](./XGBOOST.md) |
| **LightGBM** | Fast Leaf-wise Boosting | ⭐⭐⭐⭐⭐ | Curto a Médio (1 a 15 dias) | Muito Baixa | [LIGHTGBM.md](./LIGHTGBM.md) |
| **Random Forest** | Bagging Ensemble | ⭐⭐⭐⭐ | Médio (5 a 30 dias) | Baixa | [RANDOM_FOREST.md](./RANDOM_FOREST.md) |
| **LSTM** | Deep Learning Recorrente | ⭐⭐⭐⭐ | Médio a Longo (10 a 60 dias) | Média | [LSTM.md](./LSTM.md) |
| **GRU** | Deep Learning Recorrente Leve | ⭐⭐⭐⭐ | Médio (5 a 45 dias) | Baixa-Média | [GRU.md](./GRU.md) |
| **Transformer** | Mecanismo de Auto-Atenção | ⭐⭐⭐⭐⭐ | Médio a Longo (15 a 90 dias) | Alta | [TRANSFORMER.md](./TRANSFORMER.md) |
| **ARIMA / SARIMA** | Modelagem Estocástica Clássica | ⭐⭐⭐ | Curto (1 a 7 dias) | Baixa | [ARIMA.md](./ARIMA.md) |
| **Prophet** | Decomposição Aditiva / Sazonal | ⭐⭐ | Médio a Longo (30 a 180 dias) | Baixa | [PROPHET.md](./PROPHET.md) |
| **Regressão Linear** | Regularização L2 (Ridge) | ⭐⭐ | Curto (1 a 5 dias) | Extremamente Baixa | [REGRESSAO_LINEAR.md](./REGRESSAO_LINEAR.md) |
| **Algoritmo Genético** | Otimização Evolutiva (GA) | ⭐⭐⭐⭐ | Curto a Médio (1 a 20 dias) | Média | [ALGORITMO_GENETICO.md](./ALGORITMO_GENETICO.md) |
| **Rede Neural MLP** | Perceptron Multicamadas Profundo | ⭐⭐⭐⭐⭐ | Curto a Médio (5 a 30 dias) | Baixa-Média | [REDE_NEURAL.md](./REDE_NEURAL.md) |
| **Lógica Fuzzy** | Inferência Takagi-Sugeno-Kang | ⭐⭐⭐⭐ | Curto a Médio (3 a 20 dias) | Baixa | [LOGICA_FUZZY.md](./LOGICA_FUZZY.md) |
| **MiroFish** | Simulação Multiagente Social | ⭐⭐⭐⭐⭐ | Médio (5 a 30 dias) | Média | [MIROFISH.md](./MIROFISH.md) |

---

## Diretrizes de Escolha de Modelo
1. **Ambiente com Alta Mudança de Regime**: Utilize o `M.A.P.P.` ou `Ensemble` com modo `regime_adaptive`.
2. **Previsão Rápida e Eficiente de Curto Prazo**: `LightGBM` ou `XGBoost`.
3. **Padrões de Longa Dependência Sequencial**: `Transformer` ou `LSTM`.
4. **Sazonalidade Evidente e Horizontes Maiores**: `Prophet` ou `ARIMA/SARIMA`.
5. **Auditoria e Explicabilidade Direta**: `Pattern Matching` e `Lógica Fuzzy`.
