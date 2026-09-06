"""
Módulo do Motor de Backtesting Walk-Forward (mapp/backtesting/engine.py)
Executa simulação quantitativa estritamente temporal (sem olhar para o futuro).
Incorpora taxas de corretagem, slippage, dimensionamento de posição, stop loss
e take profit, gerando curva de capital diária e métricas consolidadas.
Todos os nomes em português.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from mapp.backtesting.metrics import CalculadorMetricas, MetricasBacktesting


class BacktestEngine:
    """
    Motor de simulação e backtesting com janela expansiva / deslizante (Walk-Forward).
    """

    def __init__(
        self,
        capital_inicial: float = 10000.0,
        taxa_corretagem_pct: float = 0.05,  # 0.05% por operação (compra/venda)
        slippage_pct: float = 0.02,          # 0.02% de derrapagem média
        stop_loss_pct: Optional[float] = 0.05,  # 5% de stop loss
        take_profit_pct: Optional[float] = 0.10  # 10% de alvo de lucro
    ):
        self.capital_inicial = capital_inicial
        self.taxa_corretagem = taxa_corretagem_pct / 100.0
        self.slippage = slippage_pct / 100.0
        self.stop_loss = stop_loss_pct
        self.take_profit = take_profit_pct

    def executar_simulacao_walk_forward(
        self,
        df: pd.DataFrame,
        funcao_predicao: Callable[[pd.DataFrame, int], float],
        horizonte_dias: int = 5,
        janela_treino_minima: int = 40,
        passo_rebalanceamento: int = 5,
        funcao_progresso: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executa backtest walk-forward ao longo da série temporal.

        Parâmetros:
            df: DataFrame histórico com cotações [Open, High, Low, Close, Volume].
            funcao_predicao: Callback f(sub_df_ate_t, horizonte) -> preco_projetado.
            horizonte_dias: Horizonte de projeção da estratégia.
            janela_treino_minima: Barras iniciais necessárias antes da primeira operação.
            passo_rebalanceamento: Frequência de avaliação de posições em barras.
        """
        close = df["Close"].values
        datas = df.index
        n = len(df)

        if n <= janela_treino_minima + horizonte_dias:
            raise ValueError(f"Série temporal muito curta ({n} barras) para backtest com janela {janela_treino_minima}.")

        capital_atual = self.capital_inicial
        curva_capital = [capital_atual] * janela_treino_minima
        datas_capital = list(datas[:janela_treino_minima])

        posicao = 0.0  # 1.0 (Comprado) ou 0.0 (Líquido)
        preco_entrada = 0.0
        qtd_ativos = 0.0
        operacoes_retornos: List[float] = []
        historico_trades: List[Dict[str, Any]] = []

        precos_reais_avaliados: List[float] = []
        precos_previstos_avaliados: List[float] = []

        # Pontos de rebalanceamento
        pontos_decisao = list(range(janela_treino_minima, n - 1, passo_rebalanceamento))
        total_decisoes = len(pontos_decisao)

        for idx_step, t in enumerate(pontos_decisao):
            if funcao_progresso:
                pct = (idx_step / max(1, total_decisoes)) * 100.0
                funcao_progresso(pct, f"Executando Walk-Forward na barra {t}/{n}...")

            sub_df = df.iloc[:t + 1]
            preco_atual = close[t]

            # 1. Obter sinal/previsão estritamente com dados passados até t
            try:
                preco_previsto = funcao_predicao(sub_df, horizonte_dias)
            except Exception:
                preco_previsto = preco_atual

            # Registra previsão e preço real do horizonte
            idx_futuro = min(n - 1, t + horizonte_dias)
            precos_reais_avaliados.append(close[idx_futuro])
            precos_previstos_avaliados.append(preco_previsto)

            retorno_esperado = (preco_previsto - preco_atual) / (preco_atual + 1e-9)

            # 2. Lógica de Execução com Stop Loss, Take Profit e Custos
            # Se previsão for altista (> +0.5%) e estivermos líquidos: COMPRA
            custo_total_fator = self.taxa_corretagem + self.slippage

            fim_janela = min(n, t + passo_rebalanceamento)

            for d in range(t + 1, fim_janela):
                p_dia = close[d]

                if posicao == 1.0:
                    # Verifica stop loss ou take profit intra-período
                    var_desde_entrada = (p_dia - preco_entrada) / preco_entrada

                    saiu = False
                    if self.stop_loss and var_desde_entrada <= -self.stop_loss:
                        # Stop Loss atingido
                        preco_saida = p_dia * (1.0 - self.slippage)
                        capital_atual = qtd_ativos * preco_saida * (1.0 - self.taxa_corretagem)
                        ret_trade = (preco_saida / preco_entrada) - 1.0
                        operacoes_retornos.append(ret_trade)
                        historico_trades.append({
                            "data_saida": str(datas[d].date()),
                            "tipo": "STOP_LOSS",
                            "retorno_pct": round(ret_trade * 100.0, 2),
                            "capital": round(capital_atual, 2)
                        })
                        posicao = 0.0
                        qtd_ativos = 0.0
                        saiu = True

                    elif self.take_profit and var_desde_entrada >= self.take_profit:
                        # Take Profit atingido
                        preco_saida = p_dia * (1.0 - self.slippage)
                        capital_atual = qtd_ativos * preco_saida * (1.0 - self.taxa_corretagem)
                        ret_trade = (preco_saida / preco_entrada) - 1.0
                        operacoes_retornos.append(ret_trade)
                        historico_trades.append({
                            "data_saida": str(datas[d].date()),
                            "tipo": "TAKE_PROFIT",
                            "retorno_pct": round(ret_trade * 100.0, 2),
                            "capital": round(capital_atual, 2)
                        })
                        posicao = 0.0
                        qtd_ativos = 0.0
                        saiu = True

                    if not saiu:
                        # Valor de mercado marcado a mercado (MtM)
                        valor_mtm = qtd_ativos * p_dia
                        curva_capital.append(valor_mtm)
                    else:
                        curva_capital.append(capital_atual)
                else:
                    curva_capital.append(capital_atual)

                datas_capital.append(datas[d])

            # Decisão de início/manutenção de posição no próximo passo
            if retorno_esperado > 0.005 and posicao == 0.0:
                # Entra comprado com 95% do capital disponível
                preco_compra = close[fim_janela - 1] * (1.0 + self.slippage)
                capital_investido = capital_atual * 0.95
                qtd_ativos = capital_investido / preco_compra
                custo_compra = capital_investido * self.taxa_corretagem
                capital_atual = capital_atual - capital_investido - custo_compra
                preco_entrada = preco_compra
                posicao = 1.0
                historico_trades.append({
                    "data_entrada": str(datas[fim_janela - 1].date()),
                    "tipo": "COMPRA",
                    "preco": round(preco_compra, 2)
                })

            elif retorno_esperado < -0.002 and posicao == 1.0:
                # Zera posição se o modelo indicar fraqueza ou queda
                preco_saida = close[fim_janela - 1] * (1.0 - self.slippage)
                capital_resgatado = qtd_ativos * preco_saida * (1.0 - self.taxa_corretagem)
                ret_trade = (preco_saida / preco_entrada) - 1.0
                capital_atual = capital_atual + capital_resgatado
                operacoes_retornos.append(ret_trade)
                historico_trades.append({
                    "data_saida": str(datas[fim_janela - 1].date()),
                    "tipo": "VENDA_SINAL",
                    "retorno_pct": round(ret_trade * 100.0, 2),
                    "capital": round(capital_atual, 2)
                })
                posicao = 0.0
                qtd_ativos = 0.0

        # Se terminou comprado, liquida na última cotação para apuração final
        if posicao == 1.0:
            preco_final = close[-1] * (1.0 - self.slippage)
            capital_atual = capital_atual + (qtd_ativos * preco_final * (1.0 - self.taxa_corretagem))

        # Alinha tamanhos
        serie_curva = pd.Series(curva_capital[:len(datas)], index=datas[:len(curva_capital)])

        # Calcula métricas consolidadas
        metricas = CalculadorMetricas.calcular_metricas_completas(
            curva_capital=serie_curva,
            retornos_operacoes=operacoes_retornos,
            precos_reais=np.array(precos_reais_avaliados),
            precos_previstos=np.array(precos_previstos_avaliados)
        )

        # Gráfico de drawdown
        picos = serie_curva.cummax()
        serie_drawdown = ((serie_curva - picos) / (picos + 1e-9)) * 100.0

        return {
            "metrics": metricas.para_dicionario(),
            "equity_curve": [
                {"date": d.strftime("%Y-%m-%d"), "capital": round(float(c), 2)}
                for d, c in serie_curva.items()
            ],
            "drawdown_curve": [
                {"date": d.strftime("%Y-%m-%d"), "drawdown_pct": round(float(dd), 2)}
                for d, dd in serie_drawdown.items()
            ],
            "trades": historico_trades
        }
