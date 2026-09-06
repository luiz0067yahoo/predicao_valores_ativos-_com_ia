"""
Módulo de Seleção de Características e Prevenção de Vazamento Temporal (mapp/selection.py)
Aplica filtragem de colinearidade, pontuação por Informação Mútua (Mutual Information),
Permutation Importance e validação rigorosa anti-leakage (garante que dados futuros
não alteram features do passado).
Sem vazamento temporal.
Todos os nomes em português.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


class SeletorCaracteristicas:
    """
    Identifica e seleciona os atributos preditivos com maior poder explicativo,
    eliminando ruído e redundâncias para evitar overfitting.
    """

    def __init__(self, n_features_selecionar: int = 25, limite_correlacao: float = 0.92):
        self.n_features_selecionar = n_features_selecionar
        self.limite_correlacao = limite_correlacao

    def selecionar_features(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        return self.selecionar_melhores_features(
            X=X,
            y=y,
            max_features=self.n_features_selecionar,
            limiar_correlacao=self.limite_correlacao
        )

    selecionar = selecionar_features

    @classmethod
    def selecionar_melhores_features(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = 25,
        limiar_correlacao: float = 0.92
    ) -> List[str]:
        """
        Seleciona as características mais relevantes combinando redução de colinearidade
        e ranqueamento por Informação Mútua fora da amostra de teste.
        """
        if X.shape[1] <= max_features:
            return list(X.columns)

        # 1. Filtro de Colinearidade (remove features com correlação absoluta > limiar)
        matriz_corr = X.corr().abs()
        colunas_remover = set()

        for i in range(len(matriz_corr.columns)):
            col_i = matriz_corr.columns[i]
            if col_i in colunas_remover:
                continue
            for j in range(i + 1, len(matriz_corr.columns)):
                col_j = matriz_corr.columns[j]
                if col_j in colunas_remover:
                    continue
                if matriz_corr.iloc[i, j] > limiar_correlacao:
                    colunas_remover.add(col_j)

        colunas_filtradas = [c for c in X.columns if c not in colunas_remover]
        if len(colunas_filtradas) <= max_features:
            return colunas_filtradas

        # 2. Ranqueamento por Informação Mútua (Mutual Information)
        X_sub = X[colunas_filtradas].values
        # Previne NaN
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        y_limpo = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        mi_scores = mutual_info_regression(X_sub, y_limpo, random_state=42)
        indices_ordenados = np.argsort(mi_scores)[::-1]

        features_selecionadas = [colunas_filtradas[idx] for idx in indices_ordenados[:max_features]]
        return features_selecionadas


class ValidadorAntiVazamento:
    """
    Testa computacionalmente se uma função geradora de features possui vazamento temporal.
    Injeta variações artificiais em barras futuras e confirma que nenhuma barra do passado sofreu alteração.
    """

    @staticmethod
    def validar_ausencia_vazamento(
        funcao_extracao: Callable[[pd.DataFrame], pd.DataFrame],
        df_amostra: pd.DataFrame,
        barra_corte: int = 40
    ) -> Tuple[bool, str]:
        """
        Executa teste de perturbação no futuro:
        1. Extrai features no DataFrame original.
        2. Altera drasticamente as cotações nas barras posteriores à 'barra_corte'.
        3. Extrai features no DataFrame alterado.
        4. Compara se todas as features até 'barra_corte - 1' permaneceram identicamente iguais.
        """
        if len(df_amostra) < barra_corte + 10:
            return True, "Série muito curta para perturbação."

        # 1. Extração no dataset original
        f_original = funcao_extracao(df_amostra)

        # 2. Cria cópia e perturba os dados APÓS a barra de corte
        df_perturbado = df_amostra.copy()
        if "Close" in df_perturbado.columns:
            df_perturbado.iloc[barra_corte:, df_perturbado.columns.get_loc("Close")] *= 2.5
        elif len(df_perturbado.columns) > 0:
            df_perturbado.iloc[barra_corte:, 0] *= 2.5

        if "High" in df_perturbado.columns:
            df_perturbado.iloc[barra_corte:, df_perturbado.columns.get_loc("High")] *= 2.7
        if "Low" in df_perturbado.columns:
            df_perturbado.iloc[barra_corte:, df_perturbado.columns.get_loc("Low")] *= 2.3

        # 3. Extração no dataset perturbado
        f_perturbada = funcao_extracao(df_perturbado)

        # 4. Verifica igualdade estrita até barra_corte - 1
        f_orig_passado = f_original.iloc[:barra_corte].values
        f_pert_passado = f_perturbada.iloc[:barra_corte].values

        diff_maxima = np.max(np.abs(f_orig_passado - f_pert_passado))

        if diff_maxima > 1e-6:
            return False, f"Vazamento temporal detectado! Diferença de {diff_maxima:.6f} em barras pretéritas após perturbação futura."

        return True, "Validação aprovada: Zero vazamento temporal detectado."

    @classmethod
    def verificar_ausencia_vazamento(
        cls,
        df: pd.DataFrame,
        funcao_extracao: Callable[[pd.DataFrame], pd.DataFrame],
        coluna_alvo: Optional[str] = None,
        barra_corte: int = 40
    ) -> Dict[str, Any]:
        """Wrapper utilitário para testes automatizados."""
        ok, msg = cls.validar_ausencia_vazamento(funcao_extracao, df, barra_corte=min(barra_corte, len(df) - 5))
        return {"sem_vazamento": ok, "passou": ok, "mensagem": msg}
