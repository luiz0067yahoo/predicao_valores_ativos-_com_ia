"""
Servidor Web Flask e API REST/AJAX com Sessão para o AI Asset Predictor.
Permite executar múltiplos algoritmos de Inteligência Artificial e Séries Temporais,
acompanhar o progresso em tempo real via Sessão Flask e AJAX, e exportar relatórios
executivos em formatos PDF e DOCX pelo navegador.
"""

import csv
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Optional
import pandas as pd

from flask import Flask, Response, jsonify, render_template, request, send_file, session
from flask_cors import CORS

from config import (
    ASSETS,
    DEFAULT_GA_CONFIG,
    PERIOD_CHOICES,
    REPORTS_DIR,
    LOGS_DIR,
    TRIALS_LOG_FILE
)
from data_fetcher import DataFetcher
from report_generator import ReportGenerator
from relatorio_pdf import GeradorRelatorioPDF
from genetic_engine import GeneticEngine as _OriginalGeneticEngine
from model_predictor import ModelPredictor as _OriginalModelPredictor
GeneticEngine = _OriginalGeneticEngine
ModelPredictor = _OriginalModelPredictor
from algoritmos.fabrica_algoritmos import FabricaAlgoritmos, CATALOGO_ALGORITMOS
from mapp.horizon import ForecastHorizon, NormalizadorHorizonte
from mapp.tracker import ProgressTracker
from mapp.simulator import InvestmentSimulator, HyperparameterOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "chave_secreta_ai_asset_predictor_2026_financeiro")
CORS(app, supports_credentials=True)

# Registro em memória de tarefas ativas para monitoramento por Sessão / AJAX
TAREFAS_PROGRESSO: Dict[str, Dict[str, Any]] = {}
TRAVA_TAREFAS = threading.Lock()

# Registro de simulações ativas do simulador de investimentos
SIMULACOES_PROGRESSO: Dict[str, Dict[str, Any]] = {}
TRAVA_SIMULACOES = threading.Lock()

# Armazenamento em memória do último resultado executado para exportação instantânea
ULTIMO_RESULTADO: Dict[str, Any] = {
    "prediction_results": None,
    "ga_history": None,
    "asset_name": None,
    "ticker": None,
    "currency": "USD",
    "period_str": "1y",
    "algoritmo": "algoritmo_genetico",
    "nome_algoritmo": "Algoritmo Genético",
    "hiperparametros": {}
}
TRAVA_ESTADO = threading.Lock()


@app.route("/")
def index():
    """Renderiza a página principal do dashboard moderno."""
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def obter_configuracoes():
    """Retorna ativos disponíveis, períodos, configurações padrão e catálogo de algoritmos."""
    return jsonify({
        "assets": ASSETS,
        "periods": PERIOD_CHOICES,
        "default_ga": DEFAULT_GA_CONFIG,
        "algoritmos": FabricaAlgoritmos.listar_algoritmos()
    })


@app.route("/api/algoritmos", methods=["GET"])
def listar_algoritmos():
    """Retorna o catálogo completo de todos os 11 algoritmos implementados."""
    return jsonify({
        "algoritmos": FabricaAlgoritmos.listar_algoritmos()
    })


def _extrair_parametros_requisicao(dados: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida e normaliza os parâmetros recebidos via requisição HTTP (JSON ou query params).
    Todas as variáveis tratadas em português.
    """
    ativo_selecionado = dados.get("asset", dados.get("ativo", "Dólar (USD/BRL)"))
    ticker_customizado = dados.get("custom_ticker", dados.get("ticker_personalizado", "")).strip().upper()

    if "Personalizado" in ativo_selecionado or ativo_selecionado == "custom":
        if not ticker_customizado:
            raise ValueError("Por favor, informe o código do Ticker personalizado (ex: PETR4.SA, AAPL, BTC-USD).")
        ticker = ticker_customizado
        nome_ativo = f"Custom ({ticker})"
        moeda = "Unidade"
    else:
        if ativo_selecionado not in ASSETS:
            # Tenta resolver por código ou ticker (ex: IVVB11, IVVB11.SA, SPXI11, etc.)
            encontrado = None
            ativo_norm = ativo_selecionado.strip().upper()
            for chave, info in ASSETS.items():
                tickers_candidatos = [
                    chave.upper(),
                    info["ticker"].upper(),
                    info["ticker"].upper().replace(".SA", "")
                ]
                if ativo_norm in tickers_candidatos or chave.upper().startswith(ativo_norm):
                    encontrado = chave
                    break
            if encontrado:
                ativo_selecionado = encontrado
            else:
                raise ValueError(f"Ativo '{ativo_selecionado}' não reconhecido no mapeamento.")
        informacoes_ativo = ASSETS[ativo_selecionado]
        ticker = informacoes_ativo["ticker"]
        nome_ativo = ativo_selecionado
        moeda = informacoes_ativo.get("currency", "USD")

    escolha_periodo = dados.get("period", dados.get("periodo", "1 Ano"))
    str_periodo = PERIOD_CHOICES.get(escolha_periodo, escolha_periodo if escolha_periodo in ["1mo", "3mo", "6mo", "1y", "2y", "5y"] else "1y")
    data_inicio = dados.get("start_date", dados.get("data_inicio", "")).strip() or None
    data_fim = dados.get("end_date", dados.get("data_fim", "")).strip() or None

    identificador_algoritmo = dados.get("algoritmo", dados.get("algorithm", "algoritmo_genetico")).lower().strip()
    if identificador_algoritmo not in CATALOGO_ALGORITMOS:
        identificador_algoritmo = "algoritmo_genetico"

    eh_cripto = ("BTC" in ticker) or ("ETH" in ticker) or ("-USD" in ticker and "USDBRL" not in ticker)
    val_horiz = dados.get("horizon_value", dados.get("horizonte_valor"))
    unit_horiz = dados.get("horizon_unit", dados.get("horizonte_unidade", "dias"))
    if val_horiz is not None:
        try:
            h_obj = NormalizadorHorizonte.normalizar(int(val_horiz), str(unit_horiz), eh_criptomoeda=eh_cripto)
            horizonte = h_obj.periodos_normalizados
        except Exception:
            horizonte = int(dados.get("forecast_horizon", dados.get("horizonte_projecao", 5)))
    else:
        horizonte = int(dados.get("forecast_horizon", dados.get("horizonte_projecao", 5)))

    # Extrai hiperparâmetros específicos
    hiperparametros: Dict[str, Any] = {}
    for chave, valor in dados.items():
        if chave in [
            "numero_estimadores", "taxa_aprendizado", "profundidade_maxima",
            "numero_folhas", "numero_arvores", "numero_epocas", "dimensao_oculta",
            "dimensao_modelo", "dimensao_camada_1", "dimensao_camada_2", "funcao_ativacao",
            "ordem_p", "ordem_d", "ordem_q", "flexibilidade_tendencia",
            "flexibilidade_sazonal", "forca_regularizacao", "tamanho_populacao",
            "numero_geracoes", "taxa_mutacao", "taxa_crossover", "numero_rodadas_debate",
            "numero_conjuntos_fuzzy", "largura_pertinencia",
            "max_features", "peso_regime", "peso_padroes", "tipo_pesagem", "k_vizinhos",
            "population_size", "generations", "mutation_rate"
        ]:
            hiperparametros[chave] = valor

    # Mapeamento de retrocompatibilidade de nomes
    if "population_size" in dados and "tamanho_populacao" not in hiperparametros:
        hiperparametros["tamanho_populacao"] = int(dados["population_size"])
    if "generations" in dados and "numero_geracoes" not in hiperparametros:
        hiperparametros["numero_geracoes"] = int(dados["generations"])
    if "mutation_rate" in dados and "taxa_mutacao" not in hiperparametros:
        mut_val = float(dados["mutation_rate"])
        hiperparametros["taxa_mutacao"] = mut_val / 100.0 if mut_val > 1.0 else mut_val

    data_alvo_projecao = dados.get("data_alvo_projecao", dados.get("data_fim_projecao", dados.get("target_date", ""))).strip() or None

    taxa_mut_val = float(hiperparametros.get("taxa_mutacao", 0.1))
    pop_size_val = int(hiperparametros.get("tamanho_populacao", 50))
    gen_val = int(hiperparametros.get("numero_geracoes", 20))

    return {
        "ticker": ticker,
        "nome_ativo": nome_ativo,
        "asset": nome_ativo,
        "moeda": moeda,
        "currency": moeda,
        "escolha_periodo": escolha_periodo,
        "period": escolha_periodo,
        "str_periodo": str_periodo,
        "data_inicio": data_inicio,
        "start_date": data_inicio,
        "data_fim": data_fim,
        "end_date": data_fim,
        "data_alvo_projecao": data_alvo_projecao,
        "algoritmo": identificador_algoritmo,
        "algorithm": identificador_algoritmo,
        "horizonte": horizonte,
        "horizon": horizonte,
        "forecast_horizon": horizonte,
        "pop_size": pop_size_val,
        "generations": gen_val,
        "mutation_rate": taxa_mut_val,
        "hiperparametros": hiperparametros
    }


# Aliases para retrocompatibilidade com testes existentes
_parse_train_params = _extrair_parametros_requisicao
LAST_RESULT = ULTIMO_RESULTADO


def _executar_pipeline_modelo(parametros: Dict[str, Any], callback_progresso=None) -> Dict[str, Any]:
    """
    Executa o download de cotações, instancia o algoritmo selecionado e executa a projeção.
    """
    ticker = parametros["ticker"]
    nome_ativo = parametros["nome_ativo"]
    moeda = parametros["moeda"]
    str_periodo = parametros["str_periodo"]
    data_inicio = parametros["data_inicio"]
    data_fim = parametros["data_fim"]
    data_alvo_projecao = parametros.get("data_alvo_projecao")
    algoritmo_id = parametros["algoritmo"]
    horizonte = parametros["horizonte"]
    hiperparametros = parametros["hiperparametros"]

    if callback_progresso:
        callback_progresso(5, 100, f"Baixando cotações de {ticker} no Yahoo Finance...", 0.0)

    coletor = DataFetcher()
    dados_brutos = coletor.fetch_asset_data(
        ticker=ticker,
        period=str_periodo if str_periodo != "custom" else None,
        start_date=data_inicio,
        end_date=data_fim
    )

    if len(dados_brutos) < 25:
        raise ValueError(f"Série temporal insuficiente ({len(dados_brutos)} registros). Escolha um período maior.")

    if callback_progresso:
        callback_progresso(15, 100, f"{len(dados_brutos)} registros carregados. Inicializando {algoritmo_id}...", 0.0)

    # Cria instância do algoritmo selecionado através da fábrica
    instancia_algoritmo = FabricaAlgoritmos.obter_instancia(algoritmo_id, janela_temporal=10)
    nome_exibicao = CATALOGO_ALGORITMOS.get(algoritmo_id, {}).get("nome", algoritmo_id)

    eh_criptomoeda = ("BTC" in ticker) or ("ETH" in ticker) or ("-USD" in ticker and "USDBRL" not in ticker)

    # Executa o modelo (com suporte a mocks em testes legados)
    if GeneticEngine is not _OriginalGeneticEngine or ModelPredictor is not _OriginalModelPredictor:
        matriz_x, vetor_y, datas_alvo, params_esc = coletor.prepare_lagged_features(dados_brutos, lookback=10)
        motor_ga = GeneticEngine(
            population_size=hiperparametros.get("tamanho_populacao", 20),
            generations=hiperparametros.get("numero_geracoes", 5),
            crossover_rate=0.85,
            mutation_rate=hiperparametros.get("taxa_mutacao", 0.1),
            elitism_ratio=0.08
        )
        best_chromo = motor_ga.evolve(matriz_x, vetor_y)
        preditor = ModelPredictor(best_individual=best_chromo, lookback_window=10)
        resultados = preditor.evaluate_and_predict(
            df_raw=dados_brutos,
            X=matriz_x,
            dates=datas_alvo,
            forecast_horizon=horizonte,
            is_crypto=eh_criptomoeda
        )
    else:
        resultados = instancia_algoritmo.treinar_e_projetar(
            dados_completos=dados_brutos,
            horizonte_projecao=horizonte,
            eh_criptomoeda=eh_criptomoeda,
            funcao_progresso=callback_progresso,
            hiperparametros=hiperparametros
        )

    metricas = resultados["metrics"]

    # Registra no arquivo histórico de ensaios CSV
    registro_ensaio = {
        "Data_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ativo": nome_ativo,
        "Ticker": ticker,
        "Moeda": moeda,
        "Periodo": parametros["escolha_periodo"],
        "Populacao": hiperparametros.get("tamanho_populacao", hiperparametros.get("numero_estimadores", 0)),
        "Geracoes": hiperparametros.get("numero_geracoes", hiperparametros.get("numero_epocas", 0)),
        "Taxa_Mutacao": hiperparametros.get("taxa_mutacao", hiperparametros.get("taxa_aprendizado", 0.0)),
        "Taxa_Crossover": 0.85,
        "Lookback": 10,
        "Horizonte_Dias": horizonte,
        "Ultimo_Preco_Real": metricas["ultimo_preco_real"],
        "Preco_Projetado_Final": metricas["preco_projetado_final"],
        "Variacao_Esperada_Pct": metricas["variacao_esperada_pct"],
        "Tendencia": f"{metricas['tendencia_esperada']} [{nome_exibicao}]",
        "RMSE": metricas["rmse"],
        "MAE": metricas["mae"],
        "MAPE_Pct": metricas["mape"],
        "R2": metricas["r2"],
        "Acuracia_Direcional_Pct": metricas["acuracia_direcional"]
    }
    ReportGenerator.log_trial(registro_ensaio)

    # Serialização dos DataFrames para retorno JSON
    df_hist = resultados["history_df"].tail(90)
    pontos_historico = [
        {
            "date": dt.strftime("%Y-%m-%d"),
            "real": float(linha["Preco_Real"]),
            "pred": float(linha["Preco_Previsto_IA"]) if "Preco_Previsto_IA" in linha else float(linha["Preco_Real"])
        }
        for dt, linha in df_hist.iterrows()
    ]

    df_proj = resultados["forecast_df"]
    comparacao_data_final = {
        "tem_valor_real": False,
        "data_alvo": None,
        "preco_projetado": None,
        "preco_real": None,
        "diferenca_absoluta": None,
        "diferenca_pct": None,
        "acuracia_pct": None,
        "acertou_direcao": None,
        "mensagem": "Projeção futura (cotação de mercado em aberto)."
    }

    df_validacao = None
    try:
        dt_inicio_proj = pd.to_datetime(df_proj.index.min()).strftime("%Y-%m-%d")
        dt_fim_proj = pd.to_datetime(df_proj.index.max()).strftime("%Y-%m-%d")
        if pd.to_datetime(dt_inicio_proj) <= pd.Timestamp.now().normalize():
            df_validacao = coletor.fetch_asset_data(
                ticker=ticker,
                start_date=dt_inicio_proj,
                end_date=dt_fim_proj
            )
    except Exception as e_val:
        logger.info(f"Cotações de validação para o horizonte de projeção não disponíveis: {e_val}")
        df_validacao = None

    pontos_projecao = []
    for dt, linha in df_proj.iterrows():
        dt_str = dt.strftime("%Y-%m-%d")
        preco_proj = float(linha["Preco_Projetado"])
        preco_real_val = None
        erro_pct = None

        if df_validacao is not None and not df_validacao.empty:
            dt_norm = pd.to_datetime(dt).normalize()
            if dt_norm in df_validacao.index:
                col_c = "Close" if "Close" in df_validacao.columns else df_validacao.columns[0]
                preco_real_val = float(df_validacao.loc[dt_norm, col_c])
                erro_pct = ((preco_proj - preco_real_val) / preco_real_val) * 100.0

        pontos_projecao.append({
            "date": dt_str,
            "projected": preco_proj,
            "lower": float(linha["Limite_Inferior"]),
            "upper": float(linha["Limite_Superior"]),
            "real": preco_real_val,
            "error_pct": erro_pct
        })

    # Análise de comparação na Data Final (Date Picker)
    if pontos_projecao:
        ultimo_ponto = pontos_projecao[-1]
        data_final_str = data_alvo_projecao or ultimo_ponto["date"]
        ponto_alvo = next((p for p in pontos_projecao if p["date"] == data_final_str), ultimo_ponto)

        if ponto_alvo.get("real") is not None:
            p_proj = ponto_alvo["projected"]
            p_real = ponto_alvo["real"]
            diff_abs = p_proj - p_real
            diff_pct = ((p_proj - p_real) / p_real) * 100.0
            ultimo_real = float(metricas["ultimo_preco_real"])
            direcao_real = p_real >= ultimo_real
            direcao_proj = p_proj >= ultimo_real

            comparacao_data_final = {
                "tem_valor_real": True,
                "data_alvo": ponto_alvo["date"],
                "preco_projetado": p_proj,
                "preco_real": p_real,
                "diferenca_absoluta": diff_abs,
                "diferenca_pct": diff_pct,
                "acuracia_pct": max(0.0, 100.0 - abs(diff_pct)),
                "acertou_direcao": (direcao_real == direcao_proj),
                "mensagem": f"Cotação real em {ponto_alvo['date']}: {moeda} {p_real:.2f} (Desvio IA: {diff_pct:+.2f}%)"
            }
        else:
            comparacao_data_final = {
                "tem_valor_real": False,
                "data_alvo": ponto_alvo["date"],
                "preco_projetado": ponto_alvo["projected"],
                "preco_real": None,
                "diferenca_absoluta": None,
                "diferenca_pct": None,
                "acuracia_pct": None,
                "acertou_direcao": None,
                "mensagem": f"Data futura em aberto: Projeção estimada em {moeda} {ponto_alvo['projected']:.2f} para {ponto_alvo['date']}."
            }

    historico_aprendizado = resultados.get("ga_history", {})

    # Formata as datas inicial e final para nomeação amigável de relatórios (ex: 'bitcoin 30-12-2024 a 30-12-2025')
    try:
        if parametros.get("data_inicio"):
            dt_ini_obj = pd.to_datetime(parametros["data_inicio"])
        else:
            dt_ini_obj = pd.to_datetime(dados_brutos.index.min())

        if parametros.get("data_fim"):
            dt_fim_obj = pd.to_datetime(parametros["data_fim"])
        else:
            dt_fim_obj = pd.to_datetime(dados_brutos.index.max())

        data_inicio_formatada = dt_ini_obj.strftime("%d-%m-%Y")
        data_fim_formatada = dt_fim_obj.strftime("%d-%m-%Y")
    except Exception:
        data_inicio_formatada = pd.to_datetime(dados_brutos.index.min()).strftime("%d-%m-%Y")
        data_fim_formatada = pd.to_datetime(dados_brutos.index.max()).strftime("%d-%m-%Y")

    try:
        caminho_excel = DataFetcher.obter_caminho_arquivo_excel(ticker)
        if isinstance(caminho_excel, Path):
            nome_base_arquivo = caminho_excel.stem
        elif isinstance(caminho_excel, str):
            nome_base_arquivo = Path(caminho_excel).stem
        else:
            nome_base_arquivo = ticker.lower().replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")
    except Exception:
        nome_base_arquivo = ticker.lower().replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")

    # Armazena na memória global para exportação rápida de PDF ou DOCX
    with TRAVA_ESTADO:
        ULTIMO_RESULTADO["prediction_results"] = resultados
        ULTIMO_RESULTADO["ga_history"] = historico_aprendizado
        ULTIMO_RESULTADO["asset_name"] = nome_ativo
        ULTIMO_RESULTADO["ticker"] = ticker
        ULTIMO_RESULTADO["currency"] = moeda
        ULTIMO_RESULTADO["period_str"] = parametros["escolha_periodo"]
        ULTIMO_RESULTADO["algoritmo"] = algoritmo_id
        ULTIMO_RESULTADO["nome_algoritmo"] = nome_exibicao
        ULTIMO_RESULTADO["hiperparametros"] = hiperparametros
        ULTIMO_RESULTADO["data_inicio_formatada"] = data_inicio_formatada
        ULTIMO_RESULTADO["data_fim_formatada"] = data_fim_formatada
        ULTIMO_RESULTADO["nome_base_arquivo"] = nome_base_arquivo
        ULTIMO_RESULTADO["target_comparison"] = comparacao_data_final

    return {
        "success": True,
        "asset_name": nome_ativo,
        "ticker": ticker,
        "currency": moeda,
        "algoritmo": algoritmo_id,
        "nome_algoritmo": nome_exibicao,
        "metrics": metricas,
        "history": pontos_historico,
        "forecast": pontos_projecao,
        "target_comparison": comparacao_data_final,
        "ga_history": historico_aprendizado,
        "data_inicio_formatada": data_inicio_formatada,
        "data_fim_formatada": data_fim_formatada,
        "nome_base_arquivo": nome_base_arquivo,
        "nome_arquivo_pdf": f"{nome_base_arquivo} {data_inicio_formatada} a {data_fim_formatada}.pdf",
        "nome_arquivo_docx": f"{nome_base_arquivo} {data_inicio_formatada} a {data_fim_formatada}.docx"
    }


# =========================================================================
# ENDPOINTS DE CONTROLE VIA SESSÃO E AJAX
# =========================================================================

@app.route("/api/iniciar-treinamento", methods=["POST"])
def iniciar_treinamento_ajax():
    """
    Endpoint AJAX para iniciar o treinamento em thread de fundo.
    Salva o identificador único da tarefa na Sessão do usuário.
    """
    try:
        dados_corpo = request.get_json() or {}
        parametros = _extrair_parametros_requisicao(dados_corpo)

        # Gera ID único para a tarefa
        id_tarefa = str(uuid.uuid4())
        session["id_tarefa_atual"] = id_tarefa

        with TRAVA_TAREFAS:
            TAREFAS_PROGRESSO[id_tarefa] = {
                "progresso": 5,
                "mensagem_status": "Iniciando processamento...",
                "concluido": False,
                "resultado": None,
                "erro": None,
                "timestamp_inicio": time.time()
            }

        def tarefa_trabalhador():
            def atualizar_progresso(passo, total, mensagem, metrica):
                with TRAVA_TAREFAS:
                    if id_tarefa in TAREFAS_PROGRESSO:
                        TAREFAS_PROGRESSO[id_tarefa]["progresso"] = passo
                        TAREFAS_PROGRESSO[id_tarefa]["mensagem_status"] = mensagem

            try:
                res = _executar_pipeline_modelo(parametros, callback_progresso=atualizar_progresso)
                with TRAVA_TAREFAS:
                    if id_tarefa in TAREFAS_PROGRESSO:
                        TAREFAS_PROGRESSO[id_tarefa]["progresso"] = 100
                        TAREFAS_PROGRESSO[id_tarefa]["mensagem_status"] = "Concluído com sucesso!"
                        TAREFAS_PROGRESSO[id_tarefa]["concluido"] = True
                        TAREFAS_PROGRESSO[id_tarefa]["resultado"] = res
            except Exception as e:
                logger.error(f"Erro no trabalhador da tarefa {id_tarefa}: {e}", exc_info=True)
                with TRAVA_TAREFAS:
                    if id_tarefa in TAREFAS_PROGRESSO:
                        TAREFAS_PROGRESSO[id_tarefa]["concluido"] = True
                        TAREFAS_PROGRESSO[id_tarefa]["erro"] = str(e)
                        TAREFAS_PROGRESSO[id_tarefa]["mensagem_status"] = f"Erro: {e}"

        thread = threading.Thread(target=tarefa_trabalhador, daemon=True)
        thread.start()

        return jsonify({
            "sucesso": True,
            "id_tarefa": id_tarefa,
            "mensagem": "Tarefa de treinamento iniciada com sucesso."
        })

    except Exception as ex:
        logger.error(f"Falha ao disparar tarefa AJAX: {ex}", exc_info=True)
        return jsonify({"sucesso": False, "erro": str(ex)}), 400


@app.route("/api/progresso-sessao", methods=["GET"])
def consultar_progresso_sessao():
    """
    Endpoint AJAX consultado periodicamente (polling) pelo cliente.
    Recupera o id_tarefa armazenado na sessão ou recebido como parâmetro de consulta.
    """
    id_tarefa = request.args.get("id_tarefa") or session.get("id_tarefa_atual")

    if not id_tarefa:
        return jsonify({
            "progresso": 0,
            "mensagem_status": "Nenhuma tarefa em andamento nesta sessão.",
            "concluido": True,
            "resultado": None,
            "erro": None
        })

    with TRAVA_TAREFAS:
        info_tarefa = TAREFAS_PROGRESSO.get(id_tarefa)

    if not info_tarefa:
        return jsonify({
            "progresso": 0,
            "mensagem_status": "Tarefa não encontrada ou expirada.",
            "concluido": True,
            "resultado": None,
            "erro": "Identificador de tarefa não localizado."
        })

    return jsonify({
        "id_tarefa": id_tarefa,
        "progresso": info_tarefa["progresso"],
        "mensagem_status": info_tarefa["mensagem_status"],
        "concluido": info_tarefa["concluido"],
        "resultado": info_tarefa["resultado"],
        "erro": info_tarefa["erro"]
    })


# =========================================================================
# ENDPOINTS REST E SSE CONVENCIONAIS (FALLBACK)
# =========================================================================

@app.route("/api/train-stream")
def train_stream():
    """Endpoint SSE mantido como canal secundário de streaming."""
    try:
        parametros = _extrair_parametros_requisicao(request.args)
    except ValueError as e:
        def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return Response(err_gen(), mimetype="text/event-stream")

    import queue
    fila_progresso = queue.Queue()

    def trabalhador():
        def callback_sse(passo, total, mensagem, metrica):
            fila_progresso.put({
                "type": "progress",
                "gen": passo,
                "max_gen": total,
                "best_fit": round(metrica, 2) if metrica > 0 else passo,
                "avg_fit": round(metrica * 0.8, 2) if metrica > 0 else passo,
                "best_rmse": round(metrica, 4) if metrica > 0 else 0.0,
                "progress": passo
            })
        try:
            res = _executar_pipeline_modelo(parametros, callback_progresso=callback_sse)
            fila_progresso.put({"type": "complete", "data": res})
        except Exception as ex:
            fila_progresso.put({"type": "error", "message": str(ex)})

    threading.Thread(target=trabalhador, daemon=True).start()

    def stream_eventos():
        yield f"data: {json.dumps({'type': 'started', 'message': 'Pipeline iniciado.'})}\n\n"
        while True:
            try:
                msg = fila_progresso.get(timeout=45.0)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("complete", "error"):
                    break
            except Exception:
                yield f"data: {json.dumps({'type': 'heartbeat', 'message': 'Processando...'})}\n\n"

    return Response(stream_eventos(), mimetype="text/event-stream")


@app.route("/api/predict", methods=["POST"])
def predict_sync():
    """Endpoint síncrono para testes ou chamadas de API."""
    try:
        dados = request.get_json() or {}
        parametros = _extrair_parametros_requisicao(dados)
        res = _executar_pipeline_modelo(parametros)
        res["success"] = True
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def _obter_nome_arquivo_exportacao(extensao: str) -> str:
    """
    Retorna o nome do arquivo para exportação conforme padrão solicitado:
    <nome_arquivo> <data_inicio> a <data_fim>.<extensão>
    Exemplo: 'bitcoin 30-12-2024 a 30-12-2025.pdf' ou 'bitcoin 30-12-2024 a 30-12-2025.docx'
    """
    with TRAVA_ESTADO:
        ticker = ULTIMO_RESULTADO.get("ticker") or "BTC-USD"
        nome_base = ULTIMO_RESULTADO.get("nome_base_arquivo")
        if not nome_base:
            try:
                caminho_excel = DataFetcher.obter_caminho_arquivo_excel(ticker)
                if isinstance(caminho_excel, Path):
                    nome_base = caminho_excel.stem
                elif isinstance(caminho_excel, str):
                    nome_base = Path(caminho_excel).stem
                else:
                    nome_base = ticker.lower().replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")
            except Exception:
                nome_base = ticker.lower().replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")

        dt_inicio_str = ULTIMO_RESULTADO.get("data_inicio_formatada")
        dt_fim_str = ULTIMO_RESULTADO.get("data_fim_formatada")

        if not dt_inicio_str or not dt_fim_str:
            pred_res = ULTIMO_RESULTADO.get("prediction_results")
            if pred_res and "history_df" in pred_res and not pred_res["history_df"].empty:
                idx = pred_res["history_df"].index
                dt_inicio_str = pd.to_datetime(idx.min()).strftime("%d-%m-%Y")
                dt_fim_str = pd.to_datetime(idx.max()).strftime("%d-%m-%Y")
            else:
                agora = datetime.now()
                dt_fim_str = agora.strftime("%d-%m-%Y")
                dt_inicio_str = (agora - timedelta(days=365)).strftime("%d-%m-%Y")

    ext_limpa = extensao.lstrip(".")
    return f"{nome_base} {dt_inicio_str} a {dt_fim_str}.{ext_limpa}"


# =========================================================================
# EXPORTAÇÃO DE RELATÓRIOS: PDF E DOCX
# =========================================================================

@app.route("/api/export-pdf", methods=["GET", "POST"])
def exportar_relatorio_pdf():
    """Gera e retorna o relatório executivo corporativo em formato PDF com nomenclatura amigável."""
    with TRAVA_ESTADO:
        if not ULTIMO_RESULTADO["prediction_results"]:
            return jsonify({"success": False, "error": "Nenhuma previsão foi executada recentemente para gerar o PDF."}), 400

        pred_res = ULTIMO_RESULTADO["prediction_results"]
        hist_treino = ULTIMO_RESULTADO["ga_history"]
        nome_ativo = ULTIMO_RESULTADO["asset_name"]
        ticker = ULTIMO_RESULTADO["ticker"]
        moeda = ULTIMO_RESULTADO["currency"]
        periodo = ULTIMO_RESULTADO["period_str"]
        nome_algoritmo = ULTIMO_RESULTADO["nome_algoritmo"]
        hiperparametros = ULTIMO_RESULTADO["hiperparametros"]

    try:
        nome_arquivo_download = _obter_nome_arquivo_exportacao("pdf")
        caminho_pdf_customizado = str(REPORTS_DIR / nome_arquivo_download)

        # Exporta imagens dos gráficos analíticos
        caminhos_graficos = GeradorRelatorioPDF.exportar_graficos_analiticos(
            resultados_predicao=pred_res,
            historico_treinamento=hist_treino,
            nome_ativo=nome_ativo,
            moeda=moeda
        )

        # Compila o PDF com o nome padronizado
        caminho_pdf = GeradorRelatorioPDF.gerar_relatorio_pdf(
            nome_ativo=nome_ativo,
            ticker=ticker,
            moeda=moeda,
            nome_algoritmo=nome_algoritmo,
            periodo_escolhido=periodo,
            parametros_algoritmo=hiperparametros,
            resultados_predicao=pred_res,
            historico_treinamento=hist_treino,
            caminhos_graficos=caminhos_graficos,
            caminho_arquivo_saida=caminho_pdf_customizado
        )

        resposta = send_file(
            caminho_pdf,
            as_attachment=True,
            download_name=nome_arquivo_download,
            mimetype="application/pdf"
        )
        resposta.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return resposta
    except Exception as e:
        logger.error(f"Erro ao gerar relatório PDF: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export-docx", methods=["GET", "POST"])
def exportar_relatorio_docx():
    """Gera e retorna o relatório executivo em formato Word (.docx) com nomenclatura amigável."""
    with TRAVA_ESTADO:
        if not ULTIMO_RESULTADO["prediction_results"]:
            return jsonify({"success": False, "error": "Nenhuma previsão executada para exportar DOCX."}), 400

        pred_res = ULTIMO_RESULTADO["prediction_results"]
        hist_treino = ULTIMO_RESULTADO["ga_history"]
        nome_ativo = ULTIMO_RESULTADO["asset_name"]
        ticker = ULTIMO_RESULTADO["ticker"]
        moeda = ULTIMO_RESULTADO["currency"]
        periodo = ULTIMO_RESULTADO["period_str"]
        hiperparametros = ULTIMO_RESULTADO["hiperparametros"]

    try:
        nome_arquivo_download = _obter_nome_arquivo_exportacao("docx")
        caminho_docx_customizado = str(REPORTS_DIR / nome_arquivo_download)

        charts = ReportGenerator.export_charts_for_report(
            prediction_results=pred_res,
            ga_history=hist_treino if "generation" in hist_treino else {"generation": [1, 2], "best_fitness": [1, 2], "avg_fitness": [1, 2]},
            asset_name=nome_ativo,
            currency=moeda
        )

        doc_path = ReportGenerator.generate_docx_report(
            asset_name=nome_ativo,
            ticker=ticker,
            currency=moeda,
            period_str=periodo,
            ga_config=hiperparametros or DEFAULT_GA_CONFIG,
            prediction_results=pred_res,
            ga_history=hist_treino if "generation" in hist_treino else {"generation": [1, 2], "best_fitness": [1, 2], "avg_fitness": [1, 2]},
            chart_paths=charts,
            output_filepath=caminho_docx_customizado
        )

        resposta = send_file(
            doc_path,
            as_attachment=True,
            download_name=nome_arquivo_download,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        resposta.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return resposta
    except Exception as e:
        logger.error(f"Erro ao exportar DOCX: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def obter_historico_ensaios():
    """Retorna registros de ensaios salvos no CSV."""
    if not TRIALS_LOG_FILE.exists():
        return jsonify({"records": []})

    registros = []
    try:
        with open(TRIALS_LOG_FILE, mode="r", encoding="utf-8") as f:
            leitor = csv.DictReader(f)
            for linha in leitor:
                registros.append(linha)
        registros.reverse()
        return jsonify({"records": registros})
    except Exception as e:
        logger.error(f"Erro ao ler histórico: {e}", exc_info=True)
        return jsonify({"records": [], "error": str(e)}), 500


@app.route("/api/reports", methods=["GET"])
def listar_arquivos_relatorios():
    """Lista todos os arquivos de relatórios PDF e DOCX arquivados."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for f in REPORTS_DIR.glob("*.*"):
        if f.suffix.lower() in [".pdf", ".docx"]:
            st = f.stat()
            arquivos.append({
                "filename": f.name,
                "ext": f.suffix.lower().replace(".", ""),
                "size_kb": round(st.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
    arquivos.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify({"reports": arquivos})


@app.route("/api/reports/<path:filename>", methods=["GET"])
def baixar_arquivo_relatorio(filename: str):
    """Permite download direto de qualquer relatório PDF ou DOCX arquivado."""
    caminho = REPORTS_DIR / filename
    if not caminho.exists() or not caminho.is_file():
        return jsonify({"error": "Arquivo não encontrado."}), 404

    mimetype = "application/pdf" if filename.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return send_file(
        caminho,
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype
    )


# =========================================================================
# NOVA PÁGINA E APIS: SIMULADOR DE INVESTIMENTO M.A.P.P.
# =========================================================================

@app.route("/portfolio-simulator")
def pagina_simulador_investimento():
    """Renderiza a página corporativa do Simulador de Investimentos."""
    return render_template("simulator.html")


@app.route("/api/simulator/run", methods=["POST"])
def iniciar_simulacao_carteira():
    """
    Inicia simulação comparativa multi-algoritmo em thread de fundo
    e acompanha o tempo real via ProgressTracker.
    """
    try:
        dados = request.get_json() or {}
        if not dados:
            return jsonify({"error": "Parâmetros não fornecidos."}), 400
        ativo = dados.get("asset", dados.get("ativo", "Petrobras (PETR4.SA)"))
        custom_ticker = dados.get("custom_ticker", "").strip().upper()

        if "Personalizado" in ativo or ativo == "custom":
            ticker = custom_ticker or "PETR4.SA"
            nome_ativo = f"Custom ({ticker})"
            moeda = "BRL"
        elif ativo in ASSETS:
            ticker = ASSETS[ativo]["ticker"]
            nome_ativo = ativo
            moeda = ASSETS[ativo].get("currency", "BRL")
        else:
            encontrado = None
            ativo_norm = ativo.strip().upper()
            for chave, info in ASSETS.items():
                tickers_candidatos = [
                    chave.upper(),
                    info["ticker"].upper(),
                    info["ticker"].upper().replace(".SA", "")
                ]
                if ativo_norm in tickers_candidatos or chave.upper().startswith(ativo_norm):
                    encontrado = chave
                    break
            if encontrado:
                ticker = ASSETS[encontrado]["ticker"]
                nome_ativo = encontrado
                moeda = ASSETS[encontrado].get("currency", "BRL")
            else:
                ticker = custom_ticker or "PETR4.SA"
                nome_ativo = ativo
                moeda = "BRL"

        capital = float(dados.get("capital", dados.get("capital_inicial", 10000.0)))
        aporte = float(dados.get("aporte_periodico", 0.0))
        periodo = dados.get("period", dados.get("periodo", "1y"))

        # Normaliza horizonte
        h_val = int(dados.get("horizon_value", dados.get("horizonte_valor", 30)))
        h_unit = str(dados.get("horizon_unit", dados.get("horizonte_unidade", "dias")))
        eh_cripto = ("BTC" in ticker) or ("ETH" in ticker) or ("-USD" in ticker and "USDBRL" not in ticker)

        horizonte_obj = NormalizadorHorizonte.normalizar(
            valor=h_val,
            unidade=h_unit,
            eh_criptomoeda=eh_cripto
        )

        algoritmos_escolhidos = dados.get("algorithms", dados.get("algoritmos", ["mapp", "xgboost", "random_forest"]))
        params_algos = dados.get("algorithm_params", {})

        id_simulacao = str(uuid.uuid4())
        session["id_simulacao_atual"] = id_simulacao

        with TRAVA_SIMULACOES:
            SIMULACOES_PROGRESSO[id_simulacao] = {
                "progresso": 0.0,
                "tempo_decorrido": "00:00:00",
                "tempo_restante": "Calculando...",
                "tempo_estimado_total": "Calculando...",
                "algoritmo_atual": "",
                "status_message": "Carregando série histórica de dados...",
                "concluido": False,
                "resultado": None,
                "erro": None
            }

        def trabalhador_simulacao():
            try:
                coletor = DataFetcher()
                str_p = PERIOD_CHOICES.get(periodo, periodo if periodo in ["1mo", "3mo", "6mo", "1y", "2y", "5y"] else "1y")
                df_dados = coletor.fetch_asset_data(ticker=ticker, period=str_p)

                if len(df_dados) < 30:
                    raise ValueError(f"Série temporal insuficiente ({len(df_dados)} registros) para simulação de investimentos.")

                simulador = InvestmentSimulator(
                    capital_inicial=capital,
                    aporte_periodico=aporte,
                    moeda=moeda
                )

                def callback_progresso_sim(snap: Dict[str, Any]):
                    with TRAVA_SIMULACOES:
                        if id_simulacao in SIMULACOES_PROGRESSO:
                            SIMULACOES_PROGRESSO[id_simulacao].update({
                                "progresso": snap.get("progress_percentage", 0.0),
                                "tempo_decorrido": snap.get("elapsed_time", "00:00:00"),
                                "tempo_restante": snap.get("remaining_time", "Calculando..."),
                                "tempo_estimado_total": snap.get("estimated_total_time", "Calculando..."),
                                "algoritmo_atual": snap.get("current_algorithm", ""),
                                "status_message": snap.get("status_message", "Executando...")
                            })

                resultado = simulador.simular_multiplos_algoritmos(
                    df=df_dados,
                    algoritmos_selecionados=algoritmos_escolhidos,
                    horizonte=horizonte_obj,
                    parametros_por_algoritmo=params_algos,
                    callback_progresso=callback_progresso_sim
                )

                resultado["asset_name"] = nome_ativo
                resultado["ticker"] = ticker
                resultado["currency"] = moeda

                with TRAVA_SIMULACOES:
                    if id_simulacao in SIMULACOES_PROGRESSO:
                        SIMULACOES_PROGRESSO[id_simulacao].update({
                            "progresso": 100.0,
                            "tempo_restante": "00:00:00",
                            "status_message": "Simulação de investimentos concluída com sucesso!",
                            "concluido": True,
                            "resultado": resultado
                        })

            except Exception as e_sim:
                logger.error(f"Erro no trabalhador de simulação {id_simulacao}: {e_sim}", exc_info=True)
                with TRAVA_SIMULACOES:
                    if id_simulacao in SIMULACOES_PROGRESSO:
                        SIMULACOES_PROGRESSO[id_simulacao].update({
                            "concluido": True,
                            "erro": str(e_sim),
                            "status_message": f"Erro: {str(e_sim)}"
                        })

        thread = threading.Thread(target=trabalhador_simulacao, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "sim_id": id_simulacao,
            "message": "Simulação de carteira iniciada."
        })

    except Exception as e:
        logger.error(f"Erro ao iniciar simulação: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/simulator/progress", methods=["GET"])
def obter_progresso_simulacao():
    """Retorna o estado em tempo real com cronômetro decorrido e estimado."""
    id_sim = request.args.get("sim_id") or session.get("id_simulacao_atual")
    if not id_sim:
        return jsonify({"concluido": False, "erro": "Nenhuma simulação ativa informada."}), 404

    with TRAVA_SIMULACOES:
        estado = SIMULACOES_PROGRESSO.get(id_sim)
        if not estado:
            return jsonify({"concluido": False, "erro": "Identificador de simulação expirado ou inexistente."}), 404
        return jsonify(estado)


@app.route("/api/optimize", methods=["POST"])
def otimizar_hiperparametros_api():
    """Executa busca e otimização multiobjetivo de hiperparâmetros fora da amostra."""
    try:
        dados = request.get_json() or {}
        algoritmo_id = dados.get("algorithm", "xgboost").lower()
        ticker = dados.get("ticker", "USDBRL=X")
        grade = dados.get("param_grid") or {
            "numero_estimadores": [50, 100, 150],
            "taxa_aprendizado": [0.03, 0.05, 0.10],
            "profundidade_maxima": [3, 5]
        }

        coletor = DataFetcher()
        df = coletor.fetch_asset_data(ticker=ticker, period="1y")

        resultado_otim = HyperparameterOptimizer.otimizar_algoritmo(
            algoritmo_id=algoritmo_id,
            df=df,
            grade_parametros=grade,
            horizonte_passos=5,
            max_iteracoes=10
        )

        return jsonify({"success": True, "optimization": resultado_otim})
    except Exception as e:
        logger.error(f"Erro na otimização de hiperparâmetros: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 400


def run_web_server(host="127.0.0.1", port=5000, debug=False):
    """Inicia o servidor Flask."""
    logger.info(f"Iniciando AI Asset Predictor Web em http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web_server()
