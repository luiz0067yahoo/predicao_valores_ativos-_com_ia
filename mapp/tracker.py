"""
Módulo de Rastreamento de Progresso e Estimativa de Tempo Real (ProgressTracker)
Calcula com rigor cronométrico tempo decorrido, restante e total estimado.
Não utiliza números fictícios.
Todos os nomes de arquivos, classes, variáveis e comentários em português.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EstadoProgresso:
    """Snapshot do estado de execução do tracker."""
    porcentagem: float
    tempo_decorrido_seg: float
    tempo_decorrido_str: str
    tempo_restante_seg: float
    tempo_restante_str: str
    tempo_total_estimado_seg: float
    tempo_total_estimado_str: str
    passo_atual: int
    total_passos: int
    mensagem_status: str
    algoritmo_atual: str = ""
    detalhes: Dict[str, Any] = field(default_factory=dict)

    def para_dicionario(self) -> Dict[str, Any]:
        """Serializa em formato JSON amigável."""
        return {
            "progress_percentage": round(self.porcentagem, 1),
            "elapsed_time": self.tempo_decorrido_str,
            "elapsed_seconds": round(self.tempo_decorrido_seg, 1),
            "remaining_time": self.tempo_restante_str,
            "remaining_seconds": round(self.tempo_restante_seg, 1),
            "estimated_total_time": self.tempo_total_estimado_str,
            "estimated_total_seconds": round(self.tempo_total_estimado_seg, 1),
            "current_step": self.passo_atual,
            "total_steps": self.total_passos,
            "status_message": self.mensagem_status,
            "current_algorithm": self.algoritmo_atual,
            "details": self.detalhes
        }


class ProgressTracker:
    """
    Rastreador de progresso multi-etapa com estimativa adaptativa de tempo restante.
    """

    def __init__(self, total_passos: int = 100, nome_tarefa: str = "Otimização / Predição"):
        self.total_passos = max(1, total_passos)
        self.nome_tarefa = nome_tarefa
        self.tempo_inicio: float = time.perf_counter()
        self.passo_atual: int = 0
        self.mensagem_atual: str = "Iniciando processamento..."
        self.algoritmo_atual: str = ""
        self.historico_etapas: List[float] = []
        self.ultimo_tempo_etapa: float = self.tempo_inicio
        self.detalhes_extras: Dict[str, Any] = {}

    def iniciar(self):
        """Reinicia o cronômetro do rastreador."""
        self.tempo_inicio = time.perf_counter()
        self.ultimo_tempo_etapa = self.tempo_inicio
        self.passo_atual = 0
        self.historico_etapas.clear()

    def atualizar(
        self,
        passo_atual: int,
        total_passos: Optional[int] = None,
        mensagem: Optional[str] = None,
        algoritmo_atual: Optional[str] = None,
        detalhes: Optional[Dict[str, Any]] = None
    ) -> EstadoProgresso:
        """
        Registra o avanço de um ou mais passos e recalcula dinamicamente os tempos.
        """
        agora = time.perf_counter()
        if total_passos is not None and total_passos > 0:
            self.total_passos = total_passos

        self.passo_atual = min(self.total_passos, max(0, passo_atual))

        if mensagem:
            self.mensagem_atual = mensagem
        if algoritmo_atual:
            self.algoritmo_atual = algoritmo_atual
        if detalhes:
            self.detalhes_extras.update(detalhes)

        tempo_delta = agora - self.ultimo_tempo_etapa
        self.ultimo_tempo_etapa = agora
        if tempo_delta > 0:
            self.historico_etapas.append(tempo_delta)
            # Mantém histórico deslizante das últimas 50 etapas
            if len(self.historico_etapas) > 50:
                self.historico_etapas.pop(0)

        return self.obter_estado()

    def obter_estado(self) -> EstadoProgresso:
        """Calcula o snapshot atual com formatação rigorosa em HH:MM:SS."""
        agora = time.perf_counter()
        decorrido = max(0.0, agora - self.tempo_inicio)
        fracao_concluida = self.passo_atual / self.total_passos
        porcentagem = fracao_concluida * 100.0

        if self.passo_atual >= self.total_passos:
            restante = 0.0
            total_estimado = decorrido
        elif fracao_concluida > 0.01:
            # Estimativa ponderada: média global vs média recente das etapas
            estimativa_global = decorrido / fracao_concluida
            if self.historico_etapas:
                media_passo = sum(self.historico_etapas) / len(self.historico_etapas)
                passos_faltantes = self.total_passos - self.passo_atual
                estimativa_local = decorrido + (media_passo * passos_faltantes)
                # Combinação convexa (70% global, 30% local recente para suavidade)
                total_estimado = (0.7 * estimativa_global) + (0.3 * estimativa_local)
            else:
                total_estimado = estimativa_global
            restante = max(0.0, total_estimado - decorrido)
        else:
            # Início imediato: estimativa preliminar conservadora
            restante = 0.0
            total_estimado = decorrido

        return EstadoProgresso(
            porcentagem=porcentagem,
            tempo_decorrido_seg=decorrido,
            tempo_decorrido_str=self.formatar_tempo_hhmmss(decorrido),
            tempo_restante_seg=restante,
            tempo_restante_str=self.formatar_tempo_hhmmss(restante),
            tempo_total_estimado_seg=total_estimado,
            tempo_total_estimado_str=self.formatar_tempo_hhmmss(total_estimado),
            passo_atual=self.passo_atual,
            total_passos=self.total_passos,
            mensagem_status=self.mensagem_atual,
            algoritmo_atual=self.algoritmo_atual,
            detalhes=self.detalhes_extras
        )

    def concluir(self, mensagem: str = "Processamento concluído.") -> EstadoProgresso:
        """Marca como concluído com 100% de progresso e tempo restante zerado."""
        self.passo_atual = self.total_passos
        self.mensagem_atual = mensagem
        return self.obter_estado()

    # Alias em inglês
    finish = concluir

    @staticmethod
    def formatar_tempo_hhmmss(segundos: float) -> str:
        """Converte valor em segundos para o formato '00:04:37'."""
        if segundos is None or segundos < 0:
            segundos = 0.0
        segundos_inteiros = int(round(segundos))
        horas = segundos_inteiros // 3600
        minutos = (segundos_inteiros % 3600) // 60
        segs = segundos_inteiros % 60
        return f"{horas:02d}:{minutos:02d}:{segs:02d}"
