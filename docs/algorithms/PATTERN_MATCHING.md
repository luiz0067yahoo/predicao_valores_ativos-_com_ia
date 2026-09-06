# Modelo de Pattern Matching (Análogos Históricos)

## 1. Visão Geral
O modelo de **Pattern Matching** baseia-se no princípio clássico da análise técnica e da teoria fractal dos mercados: *"a história tende a se repetir, ou pelo menos a rimar"*. Em vez de estimar parâmetros paramétricos em todo o histórico, o modelo busca nos dados passados as janelas temporais com trajetórias normalizadas mais semelhantes à janela recente, projetando a continuação média observada nesses análogos.

---

## 2. Metodologia Matemática
1. **Normalização Min-Max / Z-Score Local**:
   Para cada subjanela de comprimento $W$, os preços de fechamento são normalizados para remover escala absoluta:
   $$z_\tau = \frac{P_\tau - \min(P_{[t-W, t]})}{\max(P_{[t-W, t]}) - \min(P_{[t-W, t]})}$$

2. **Cálculo de Distância / Similaridade**:
   A distância euclidiana normalizada entre o padrão atual $Q$ e cada padrão histórico candidato $C_i$ é calculada por:
   $$d(Q, C_i) = \sqrt{\frac{1}{W} \sum_{k=1}^W (Q_k - C_{i,k})^2}$$

3. **Seleção e Ponderação dos Top-K Análogos**:
   Os $K$ padrões com menor distância são selecionados. O peso de cada análogo decai inversamente com sua distância:
   $$w_i = \frac{1 / (d(Q, C_i) + \epsilon)}{\sum_{j=1}^K 1 / (d(Q, C_j) + \epsilon)}$$

4. **Projeção Futura**:
   A trajetória futura de $h$ passos observada após cada análogo é escalonada e combinada linearmente:
   $$\hat{P}_{t+h} = P_t \cdot \left( 1 + \sum_{i=1}^K w_i \cdot R_{i, +h} \right)$$

---

## 3. Parâmetros de Configuração
| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `window_size` | int | 20 | Comprimento da janela do padrão de busca (em barras) |
| `top_k_matches` | int | 5 | Quantidade de episódios históricos análogos a combinar |
| `horizonte_projecao` | int | 5 | Quantidade de barras a projetar no futuro |

---

## 4. Casos de Uso e Vantagens
- Altamente intuitivo e interpretável, permitindo auditoria visual direta dos momentos históricos análogos.
- Funciona eficazmente em ativos que exibem padrões gráficos claros (consolidações pré-rompimento, fundos arredondados, repiques em suportes).
