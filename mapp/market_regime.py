"""
Módulo de Detecção de Regimes de Mercado (MarketRegimeDetector)
Classifica o estado estrutural e comportamental do ativo em 8 regimes:
BULL_TREND, BEAR_TREND, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY,
BREAKOUT, CRASH e RECOVERY.
Fornece contexto para que osciladores não produzam sinais espúrios.
Todos os nomes de arquivos, funções, variáveis e comentários em português.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TipoRegimeMercado(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"


# Alias em inglês para compatibilidade
MarketRegime = TipoRegimeMercado


@dataclass
class ResultadoRegimeMercado:
    """Resultado detalhado da classificação do regime de mercado."""
    regime_primario: TipoRegimeMercado
    regime_secundario: Optional[TipoRegimeMercado]
    confianca: float  # [0.0, 1.0]
    descricao: str
    metricas: Dict[str, float]
    vetor_features_regime: Dict[str, float]

    def para_dicionario(self) -> Dict[str, Any]:
        return {
            "primary_regime": self.regime_primario.value,
            "secondary_regime": self.regime_secundario.value if self.regime_secundario else None,
            "confidence": round(self.confianca, 3),
            "description": self.descricao,
            "metrics": {k: round(v, 4) for k, v in self.metricas.items()},
            "regime_features": {k: round(v, 4) for k, v in self.vetor_features_regime.items()}
        }


class MarketRegimeDetector:
    """
    Detector de regimes de mercado baseado em tendência, volatilidade,
    momentum e fluxo de volume sem vazamento de dados futuros.
    """

    @classmethod
    def detectar(cls, df: pd.DataFrame) -> ResultadoRegimeMercado:
        """
        Analisa a série histórica e classifica o regime na barra mais recente.
        """
        if len(df) < 30:
            return cls._regime_padrao_insuficiente()

        close = df["Close"].values
        high = df["High"].values if "High" in df.columns else close
        low = df["Low"].values if "Low" in df.columns else close
        volume = df["Volume"].values if "Volume" in df.columns else np.ones_like(close)

        n = len(close)

        # 1. Indicadores de Tendência e Médias
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-min(50, n):])
        sma_200 = np.mean(close[-min(200, n):]) if n >= 60 else sma_50
        preco_atual = close[-1]

        # Inclinação de 20 períodos normalizada
        retornos_20 = (close[-1] - close[-20]) / (close[-20] + 1e-9)
        retornos_5 = (close[-1] - close[-5]) / (close[-5] + 1e-9)

        # 2. Volatilidade Realizada e Bandas de Bollinger
        janela_vol = close[-20:]
        std_20 = float(np.std(janela_vol))
        media_20 = float(np.mean(janela_vol))
        bb_width = (4.0 * std_20) / (media_20 + 1e-9)

        # Volatilidade média histórica de longo prazo para comparação
        if n >= 60:
            std_longo = float(np.std(close[-60:]))
            razao_vol = std_20 / (std_longo + 1e-9)
        else:
            razao_vol = 1.0

        # ATR simplificado de 14 períodos
        tr = np.maximum(
            high[-14:] - low[-14:],
            np.maximum(
                np.abs(high[-14:] - close[-15:-1]),
                np.abs(low[-14:] - close[-15:-1])
            )
        )
        atr_14 = float(np.mean(tr))
        atr_pct = atr_14 / (preco_atual + 1e-9)

        # 3. Volume Relativo
        vol_recente = np.mean(volume[-5:])
        vol_historico = np.mean(volume[-min(30, n):]) + 1e-9
        razao_volume = float(vol_recente / vol_historico)

        # 4. Detecção de Rompimentos e Extremos
        max_20 = float(np.max(high[-21:-1]))
        min_20 = float(np.min(low[-21:-1]))
        rompeu_maxima = preco_atual > max_20
        rompeu_minima = preco_atual < min_20

        # 5. Algoritmo de Regras Ponderadas para Classificação do Regime
        pontuacoes: Dict[TipoRegimeMercado, float] = {
            TipoRegimeMercado.BULL_TREND: 0.0,
            TipoRegimeMercado.BEAR_TREND: 0.0,
            TipoRegimeMercado.SIDEWAYS: 0.0,
            TipoRegimeMercado.HIGH_VOLATILITY: 0.0,
            TipoRegimeMercado.LOW_VOLATILITY: 0.0,
            TipoRegimeMercado.BREAKOUT: 0.0,
            TipoRegimeMercado.CRASH: 0.0,
            TipoRegimeMercado.RECOVERY: 0.0
        }

        # Avaliação de Bull Trend
        if preco_atual > sma_20 > sma_50 and retornos_20 > 0.03:
            pontuacoes[TipoRegimeMercado.BULL_TREND] += 0.5 + min(0.4, retornos_20 * 5)
            if preco_atual > sma_200:
                pontuacoes[TipoRegimeMercado.BULL_TREND] += 0.2

        # Avaliação de Bear Trend
        if preco_atual < sma_20 < sma_50 and retornos_20 < -0.03:
            pontuacoes[TipoRegimeMercado.BEAR_TREND] += 0.5 + min(0.4, abs(retornos_20) * 5)
            if preco_atual < sma_200:
                pontuacoes[TipoRegimeMercado.BEAR_TREND] += 0.2

        # Avaliação de Crash (queda abrupta com alta volatilidade e volume)
        if retornos_5 < -0.08 or (retornos_20 < -0.15 and razao_volume > 1.3):
            pontuacoes[TipoRegimeMercado.CRASH] += 0.85
            pontuacoes[TipoRegimeMercado.HIGH_VOLATILITY] += 0.4

        # Avaliação de Recovery (repique forte após fundo recente)
        min_recente = float(np.min(low[-15:]))
        if retornos_5 > 0.06 and (preco_atual - min_recente) / (min_recente + 1e-9) > 0.08 and retornos_20 < 0:
            pontuacoes[TipoRegimeMercado.RECOVERY] += 0.75

        # Avaliação de Breakout
        if (rompeu_maxima or rompeu_minima) and razao_volume > 1.4:
            pontuacoes[TipoRegimeMercado.BREAKOUT] += 0.8
            if rompeu_maxima:
                pontuacoes[TipoRegimeMercado.BULL_TREND] += 0.3
            else:
                pontuacoes[TipoRegimeMercado.BEAR_TREND] += 0.3

        # Avaliação de Volatilidade Alta / Baixa
        if razao_vol > 1.5 or bb_width > 0.12 or atr_pct > 0.045:
            pontuacoes[TipoRegimeMercado.HIGH_VOLATILITY] += 0.65
        elif razao_vol < 0.7 and bb_width < 0.04:
            pontuacoes[TipoRegimeMercado.LOW_VOLATILITY] += 0.75
            pontuacoes[TipoRegimeMercado.SIDEWAYS] += 0.4

        # Avaliação de Sideways (Lateralização)
        if abs(retornos_20) < 0.03 and abs(preco_atual - sma_20) / (sma_20 + 1e-9) < 0.02:
            pontuacoes[TipoRegimeMercado.SIDEWAYS] += 0.6
            if bb_width < 0.06:
                pontuacoes[TipoRegimeMercado.SIDEWAYS] += 0.25

        # Ordena regimes por pontuação decrescente
        regimes_ordenados = sorted(pontuacoes.items(), key=lambda x: x[1], reverse=True)
        primario, score_primario = regimes_ordenados[0]
        secundario, score_secundario = regimes_ordenados[1]

        # Se pontuação primária for muito baixa, classifica como SIDEWAYS por segurança
        if score_primario < 0.25:
            primario = TipoRegimeMercado.SIDEWAYS
            confianca = 0.55
        else:
            confianca = min(0.95, max(0.45, score_primario))

        # Gera vetor de características normalizadas (one-hot ponderado)
        vetor_regime = {
            f"regime_{r.value.lower()}": float(np.clip(pontuacoes[r], 0.0, 1.0))
            for r in TipoRegimeMercado
        }

        metricas = {
            "retorno_20d_pct": float(retornos_20 * 100.0),
            "retorno_5d_pct": float(retornos_5 * 100.0),
            "bb_width": float(bb_width),
            "atr_pct": float(atr_pct * 100.0),
            "razao_volume": float(razao_volume),
            "razao_volatilidade": float(razao_vol),
            "distancia_sma20_pct": float(((preco_atual / (sma_20 + 1e-9)) - 1.0) * 100.0),
            "distancia_sma50_pct": float(((preco_atual / (sma_50 + 1e-9)) - 1.0) * 100.0)
        }

        descricao = cls._gerar_descricao_regime(primario, confianca, metricas)

        return ResultadoRegimeMercado(
            regime_primario=primario,
            regime_secundario=secundario if score_secundario > 0.3 else None,
            confianca=confianca,
            descricao=descricao,
            metricas=metricas,
            vetor_features_regime=vetor_regime
        )

    @classmethod
    def gerar_serie_regimes(cls, df: pd.DataFrame, janela: int = 30) -> pd.Series:
        """
        Gera uma série temporal contendo o regime histórico para cada barra,
        garantindo estritamente ausência de vazamento temporal (lookahead bias).
        """
        regimes = []
        indices = df.index

        for i in range(len(df)):
            if i < janela:
                regimes.append(TipoRegimeMercado.SIDEWAYS.value)
            else:
                sub_df = df.iloc[:i + 1]
                resultado = cls.detectar(sub_df)
                regimes.append(resultado.regime_primario.value)

        return pd.Series(regimes, index=indices, name="Market_Regime")

    @staticmethod
    def _gerar_descricao_regime(regime: TipoRegimeMercado, conf: float, m: Dict[str, float]) -> str:
        conf_pct = int(conf * 100)
        if regime == TipoRegimeMercado.BULL_TREND:
            return f"Tendência de Alta estabelecida ({conf_pct}% conf.), preços acima das médias com retorno 20d de +{m['retorno_20d_pct']:.1f}%."
        elif regime == TipoRegimeMercado.BEAR_TREND:
            return f"Tendência de Baixa ativa ({conf_pct}% conf.), pressão vendedora com retorno 20d de {m['retorno_20d_pct']:.1f}%."
        elif regime == TipoRegimeMercado.CRASH:
            return f"Regime de Queda Severa / Pânico ({conf_pct}% conf.), alta volatilidade ({m['atr_pct']:.1f}% ATR) e forte vazão."
        elif regime == TipoRegimeMercado.RECOVERY:
            return f"Regime de Repique e Recuperação ({conf_pct}% conf.) após teste de suporte relevante."
        elif regime == TipoRegimeMercado.BREAKOUT:
            return f"Rompimento Estrutural em Andamento ({conf_pct}% conf.), volume {m['razao_volume']:.1f}x acima da média."
        elif regime == TipoRegimeMercado.HIGH_VOLATILITY:
            return f"Volatilidade Elevada ({conf_pct}% conf.), expansão de bandas e oscilações atípicas."
        elif regime == TipoRegimeMercado.LOW_VOLATILITY:
            return f"Compressão de Volatilidade ({conf_pct}% conf.), bandas estreitas indicando potencial breakout iminente."
        return f"Mercado em Consolidação / Lateralização ({conf_pct}% conf.), sem direção direcional dominante."

    @staticmethod
    def _regime_padrao_insuficiente() -> ResultadoRegimeMercado:
        vetor = {f"regime_{r.value.lower()}": 0.125 for r in TipoRegimeMercado}
        vetor["regime_sideways"] = 0.5
        return ResultadoRegimeMercado(
            regime_primario=TipoRegimeMercado.SIDEWAYS,
            regime_secundario=None,
            confianca=0.50,
            descricao="Histórico curto para confirmação estatística de regime. Classificado preliminarmente como Lateral.",
            metricas={"bb_width": 0.05, "retorno_20d_pct": 0.0},
            vetor_features_regime=vetor
        )
