"""
Módulo de Normalização de Horizonte de Previsão (ForecastHorizon)
Converte parâmetros genéricos de tempo (dias, semanas, meses, anos) para
o número adequado de passos/períodos de negociação, respeitando o calendário
de mercado (dias úteis vs cripto 24/7).
Todos os nomes de arquivos, funções, variáveis e comentários em português.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ForecastHorizon:
    """
    Representação estruturada e normalizada de um horizonte de previsão.
    """
    valor: int
    unidade: str  # 'dias', 'semanas', 'meses', 'anos'
    periodos_normalizados: int  # quantidade real de passos/barras no timeframe
    timeframe: str  # ex: '1D'
    data_alvo: pd.Timestamp
    eh_criptomoeda: bool = False

    def em_dias_corridos(self) -> int:
        """Retorna o horizonte aproximado em dias corridos."""
        unidade_limpa = self.unidade.lower().strip()
        if "dia" in unidade_limpa:
            return self.valor
        elif "semana" in unidade_limpa:
            return self.valor * 7
        elif "mês" in unidade_limpa or "mes" in unidade_limpa:
            return int(self.valor * 30.4375)
        elif "ano" in unidade_limpa:
            return int(self.valor * 365.25)
        return self.valor

    def formatar_legenda(self) -> str:
        """Formata uma string legível para gráficos e relatórios."""
        unidade_singular_plural = self.unidade
        return f"{self.valor} {unidade_singular_plural} ({self.periodos_normalizados} períodos)"


class NormalizadorHorizonte:
    """
    Utilitário para normalizar qualquer entrada de horizonte temporal em períodos efetivos.
    """

    # Média histórica de pregões na B3 / NYSE:
    # 1 mês = ~21 a 22 pregões úteis (NUNCA 30 pregões exatos)
    # 1 ano = ~252 pregões úteis
    PREGOES_POR_MES = 21
    PREGOES_POR_ANO = 252

    @classmethod
    def normalizar(
        cls,
        valor: int,
        unidade: str = "dias",
        data_base: Optional[pd.Timestamp] = None,
        eh_criptomoeda: bool = False,
        timeframe: str = "1D"
    ) -> ForecastHorizon:
        """
        Calcula o número preciso de passos de projeção e a data alvo final.

        Parâmetros:
            valor: Número inteiro positivo (ex: 30, 6, 2).
            unidade: Unidade temporal ('dias', 'semanas', 'meses', 'anos').
            data_base: Timestamp da última barra disponível (padrão: agora).
            eh_criptomoeda: Se True, considera negociação contínua 24/7 (sábados e domingos contam).
            timeframe: Periodicidade da barra (padrão '1D').
        """
        if valor <= 0:
            valor = 1

        unidade_limpa = unidade.lower().strip()
        dt_base = pd.to_datetime(data_base) if data_base is not None else pd.Timestamp.now().normalize()

        # Determina a quantidade de passos (barras diárias) conforme mercado
        if "dia" in unidade_limpa:
            if eh_criptomoeda:
                periodos = valor
                delta_dias = valor
            else:
                if valor <= 14:
                    periodos = valor
                    delta_dias = cls._dias_corridos_para_pregoes(valor)
                else:
                    periodos = max(1, int(round(valor * (cls.PREGOES_POR_MES / 30.4375))))
                    delta_dias = valor

        elif "semana" in unidade_limpa:
            if eh_criptomoeda:
                periodos = valor * 7
            else:
                periodos = valor * 5  # 5 pregões por semana comercial
            delta_dias = valor * 7

        elif "mês" in unidade_limpa or "mes" in unidade_limpa:
            if eh_criptomoeda:
                periodos = int(round(valor * 30.4375))
            else:
                periodos = valor * cls.PREGOES_POR_MES
            delta_dias = int(round(valor * 30.4375))

        elif "ano" in unidade_limpa:
            if eh_criptomoeda:
                periodos = int(round(valor * 365.25))
            else:
                periodos = valor * cls.PREGOES_POR_ANO
            delta_dias = int(round(valor * 365.25))

        else:
            periodos = max(1, valor)
            delta_dias = valor

        periodos = max(1, min(periodos, 1260))

        if eh_criptomoeda:
            data_alvo = dt_base + timedelta(days=delta_dias)
        else:
            data_corrente = dt_base
            pregoes_contados = 0
            while pregoes_contados < periodos:
                data_corrente = data_corrente + timedelta(days=1)
                if data_corrente.weekday() < 5:
                    pregoes_contados += 1
            data_alvo = data_corrente

        return ForecastHorizon(
            valor=valor,
            unidade=unidade,
            periodos_normalizados=periodos,
            timeframe=timeframe,
            data_alvo=data_alvo,
            eh_criptomoeda=eh_criptomoeda
        )

    @classmethod
    def parse_string_horizonte(cls, texto: str, eh_criptomoeda: bool = False) -> ForecastHorizon:
        """
        Converte strings comuns como '30 dias', '6 meses', '1 ano', '14d', '2y' em ForecastHorizon.
        """
        texto_limpo = str(texto).strip().lower()

        mapa_rapido = {
            "1d": (1, "dias"),
            "1 dia": (1, "dias"),
            "3d": (3, "dias"),
            "5d": (5, "dias"),
            "7d": (7, "dias"),
            "7 dias": (7, "dias"),
            "14d": (14, "dias"),
            "14 dias": (14, "dias"),
            "30d": (30, "dias"),
            "30 dias": (30, "dias"),
            "1m": (1, "meses"),
            "1 mês": (1, "meses"),
            "1 mes": (1, "meses"),
            "3m": (3, "meses"),
            "3 meses": (3, "meses"),
            "6m": (6, "meses"),
            "6 meses": (6, "meses"),
            "1y": (1, "anos"),
            "1 ano": (1, "anos"),
            "2y": (2, "anos"),
            "2 anos": (2, "anos"),
            "5y": (5, "anos"),
            "5 anos": (5, "anos"),
        }

        if texto_limpo in mapa_rapido:
            val, un = mapa_rapido[texto_limpo]
            return cls.normalizar(val, un, eh_criptomoeda=eh_criptomoeda)

        partes = texto_limpo.split()
        if len(partes) >= 2 and partes[0].isdigit():
            val = int(partes[0])
            un = partes[1]
            return cls.normalizar(val, un, eh_criptomoeda=eh_criptomoeda)

        if texto_limpo.isdigit():
            return cls.normalizar(int(texto_limpo), "dias", eh_criptomoeda=eh_criptomoeda)

        return cls.normalizar(5, "dias", eh_criptomoeda=eh_criptomoeda)

    @staticmethod
    def _dias_corridos_para_pregoes(pregoes: int) -> int:
        semanas = pregoes // 5
        resto = pregoes % 5
        return semanas * 7 + resto + (2 if resto >= 4 else 0)
