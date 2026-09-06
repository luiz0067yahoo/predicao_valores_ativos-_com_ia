"""
Interface Gráfica do Usuário (GUI) Moderna para Previsão de Séries Temporais com IA
Desenvolvido com Tkinter, ttk e integração nativa com Matplotlib
"""

import logging
import os
import subprocess
import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config import ASSETS, DEFAULT_GA_CONFIG, PERIOD_CHOICES, REPORTS_DIR, LOGS_DIR
from data_fetcher import DataFetcher
from genetic_engine import GeneticEngine
from model_predictor import ModelPredictor
from report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ModernFinancialGUI(tk.Tk):
    """
    Janela Principal da Aplicação de Previsão de Ativos com Algoritmo Genético.
    """

    def __init__(self):
        super().__init__()

        self.title("AI Asset Predictor - Algoritmo Genético & Yahoo Finance")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        # Variáveis de estado
        self.data_fetcher = DataFetcher()
        self.raw_df = None
        self.prediction_results = None
        self.ga_history = None
        self.current_ticker = None
        self.current_asset_name = None
        self.current_currency = "USD"
        self.current_period_str = "1y"

        # Configuração de Estilo ttk
        self._setup_styles()

        # Criação dos componentes de layout
        self._create_layout()

        # Carrega dados padrão iniciais na interface
        self._set_default_values()

    def _setup_styles(self):
        """Define estilos visuais modernos para a interface."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Paleta de Cores
        self.bg_color = "#F5F7FA"
        self.sidebar_bg = "#FFFFFF"
        self.primary_color = "#102C57"
        self.accent_color = "#1679AB"
        self.accent_hover = "#0D5B84"
        self.text_color = "#1E293B"

        self.configure(bg=self.bg_color)

        self.style.configure(".", font=("Segoe UI", 9), background=self.bg_color, foreground=self.text_color)
        self.style.configure("Sidebar.TFrame", background=self.sidebar_bg)
        self.style.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=self.primary_color, background=self.sidebar_bg)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9, "bold"), foreground="#475569", background=self.sidebar_bg)
        self.style.configure("Value.TLabel", font=("Segoe UI", 13, "bold"), foreground=self.primary_color, background="#FFFFFF")
        self.style.configure("MetricTitle.TLabel", font=("Segoe UI", 8), foreground="#64748B", background="#FFFFFF")

        self.style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.accent_color,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=8
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", self.accent_hover), ("disabled", "#94A3B8")]
        )

        self.style.configure(
            "Action.TButton",
            font=("Segoe UI", 9, "bold"),
            background="#059669",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6
        )
        self.style.map(
            "Action.TButton",
            background=[("active", "#047857"), ("disabled", "#94A3B8")]
        )

    def _create_layout(self):
        """Monta a estrutura principal com barra lateral e painel de exibição."""
        main_container = ttk.Frame(self, style="Sidebar.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Barra Lateral de Controles (Sidebar)
        self.sidebar = ttk.Frame(main_container, width=340, style="Sidebar.TFrame", padding=14)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        self._build_sidebar_controls()

        # Separador vertical
        sep = ttk.Separator(main_container, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y)

        # 2. Área Central / Dashboard de Visualização
        self.content_area = ttk.Frame(main_container, style="Sidebar.TFrame", padding=12)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_content_dashboard()

    def _build_sidebar_controls(self):
        """Monta os controles e inputs na barra lateral."""
        # Cabeçalho da Sidebar
        lbl_brand = ttk.Label(self.sidebar, text="⚡ AI ASSET PREDICTOR", style="Header.TLabel")
        lbl_brand.pack(anchor=tk.W, pady=(0, 2))
        lbl_sub = ttk.Label(self.sidebar, text="Algoritmo Genético & Yahoo Finance", style="SubHeader.TLabel")
        lbl_sub.pack(anchor=tk.W, pady=(0, 12))

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Seleção de Ativo
        lbl_asset = ttk.Label(self.sidebar, text="1. Seleção do Ativo", style="SubHeader.TLabel")
        lbl_asset.pack(anchor=tk.W, pady=(6, 2))

        self.combo_asset = ttk.Combobox(self.sidebar, values=list(ASSETS.keys()) + ["Personalizado (Custom Ticker)"], state="readonly")
        self.combo_asset.pack(fill=tk.X, pady=(0, 4))
        self.combo_asset.bind("<<ComboboxSelected>>", self._on_asset_changed)

        # Campo para Custom Ticker (inicialmente oculto ou desabilitado)
        self.frame_custom = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.frame_custom.pack(fill=tk.X, pady=(0, 6))
        lbl_custom = ttk.Label(self.frame_custom, text="Ticker Yahoo:", style="SubHeader.TLabel")
        lbl_custom.pack(side=tk.LEFT, padx=(0, 4))
        self.entry_custom_ticker = ttk.Entry(self.frame_custom, width=15)
        self.entry_custom_ticker.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_custom_ticker.insert(0, "PETR4.SA")

        # Seleção de Período Histórico
        lbl_period = ttk.Label(self.sidebar, text="2. Intervalo de Leitura", style="SubHeader.TLabel")
        lbl_period.pack(anchor=tk.W, pady=(8, 2))

        self.combo_period = ttk.Combobox(self.sidebar, values=list(PERIOD_CHOICES.keys()), state="readonly")
        self.combo_period.pack(fill=tk.X, pady=(0, 4))
        self.combo_period.bind("<<ComboboxSelected>>", self._on_period_changed)

        # Datas personalizadas
        self.frame_dates = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.frame_dates.pack(fill=tk.X, pady=(0, 8))

        lbl_start = ttk.Label(self.frame_dates, text="Início (AAAA-MM-DD):", style="SubHeader.TLabel")
        lbl_start.pack(anchor=tk.W)
        self.entry_start = ttk.Entry(self.frame_dates)
        self.entry_start.pack(fill=tk.X, pady=(0, 4))

        lbl_end = ttk.Label(self.frame_dates, text="Fim (AAAA-MM-DD):", style="SubHeader.TLabel")
        lbl_end.pack(anchor=tk.W)
        self.entry_end = ttk.Entry(self.frame_dates)
        self.entry_end.pack(fill=tk.X, pady=(0, 4))

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Parâmetros do Algoritmo Genético
        lbl_ga = ttk.Label(self.sidebar, text="3. Hiperparâmetros do GA", style="SubHeader.TLabel")
        lbl_ga.pack(anchor=tk.W, pady=(4, 6))

        # Grid de parâmetros
        param_grid = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        param_grid.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(param_grid, text="População:", style="SubHeader.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.spin_pop = ttk.Spinbox(param_grid, from_=20, to=300, increment=10, width=8)
        self.spin_pop.grid(row=0, column=1, sticky=tk.E, pady=2)

        ttk.Label(param_grid, text="Gerações:", style="SubHeader.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.spin_gen = ttk.Spinbox(param_grid, from_=10, to=150, increment=5, width=8)
        self.spin_gen.grid(row=1, column=1, sticky=tk.E, pady=2)

        ttk.Label(param_grid, text="Taxa Mutação (%):", style="SubHeader.TLabel").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.spin_mut = ttk.Spinbox(param_grid, from_=1, to=50, increment=1, width=8)
        self.spin_mut.grid(row=2, column=1, sticky=tk.E, pady=2)

        ttk.Label(param_grid, text="Projeção (Dias):", style="SubHeader.TLabel").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.spin_horizon = ttk.Spinbox(param_grid, from_=1, to=30, increment=1, width=8)
        self.spin_horizon.grid(row=3, column=1, sticky=tk.E, pady=2)

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Botão Principal de Execução
        self.btn_run = ttk.Button(
            self.sidebar,
            text="🚀 Otimizar & Prever com IA",
            style="Accent.TButton",
            command=self._start_training_thread
        )
        self.btn_run.pack(fill=tk.X, pady=(4, 8))

        # Barra de Progresso e Status
        self.lbl_status = ttk.Label(self.sidebar, text="Pronto para iniciar.", style="SubHeader.TLabel")
        self.lbl_status.pack(anchor=tk.W, pady=(2, 2))

        self.progress_bar = ttk.Progressbar(self.sidebar, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Ações de Relatório
        self.btn_report = ttk.Button(
            self.sidebar,
            text="📄 Gerar Relatório Executivo (.docx)",
            style="Action.TButton",
            command=self._export_docx_report,
            state="disabled"
        )
        self.btn_report.pack(fill=tk.X, pady=(2, 4))

        self.btn_open_folder = ttk.Button(
            self.sidebar,
            text="📂 Abrir Pasta de Relatórios",
            command=self._open_reports_folder
        )
        self.btn_open_folder.pack(fill=tk.X, pady=(2, 2))

    def _build_content_dashboard(self):
        """Monta o painel com KPIs e abas de visualização gráfica."""
        # 1. Cards de Métricas Principais (KPIs)
        self.cards_frame = ttk.Frame(self.content_area, style="Sidebar.TFrame")
        self.cards_frame.pack(fill=tk.X, pady=(0, 10))

        self.card_actual = self._create_kpi_card(self.cards_frame, "ÚLTIMO PREÇO REAL", "-", 0)
        self.card_predicted = self._create_kpi_card(self.cards_frame, "PREVISÃO IA (FUTURO)", "-", 1)
        self.card_variation = self._create_kpi_card(self.cards_frame, "VARIAÇÃO ESTIMADA", "-", 2)
        self.card_rmse = self._create_kpi_card(self.cards_frame, "RMSE (ERRO)", "-", 3)
        self.card_directional = self._create_kpi_card(self.cards_frame, "ACURÁCIA DIRECIONAL", "-", 4)

        # 2. Caderno de Abas (Notebook) para Gráficos
        self.notebook = ttk.Notebook(self.content_area)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Aba 1: Gráfico Principal de Preços e Projeção
        self.tab_prediction = ttk.Frame(self.notebook, style="Sidebar.TFrame")
        self.notebook.add(self.tab_prediction, text="  📈 Série Histórica & Projeção Futura  ")

        self.fig_pred, self.ax_pred = plt.subplots(figsize=(8, 4.5), dpi=100)
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=self.tab_prediction)
        self.canvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_pred = NavigationToolbar2Tk(self.canvas_pred, self.tab_prediction)
        self.toolbar_pred.update()

        # Aba 2: Gráfico da Curva de Convergência do GA
        self.tab_convergence = ttk.Frame(self.notebook, style="Sidebar.TFrame")
        self.notebook.add(self.tab_convergence, text="  🧬 Curva de Aprendizado (GA)  ")

        self.fig_conv, self.ax_conv = plt.subplots(figsize=(8, 4.5), dpi=100)
        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=self.tab_convergence)
        self.canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_conv = NavigationToolbar2Tk(self.canvas_conv, self.tab_convergence)
        self.toolbar_conv.update()

        # Aba 3: Tabela de Previsão Detalhada
        self.tab_table = ttk.Frame(self.notebook, style="Sidebar.TFrame")
        self.notebook.add(self.tab_table, text="  📋 Tabela de Projeção Passo a Passo  ")

        self._build_projection_table()

    def _create_kpi_card(self, parent: ttk.Frame, title: str, initial_value: str, col: int) -> ttk.Label:
        """Cria um card visual de KPI com borda sutil e fundo branco."""
        card = tk.Frame(parent, bg="#FFFFFF", highlightbackground="#E2E8F0", highlightthickness=1, padx=12, pady=8)
        card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        lbl_t = tk.Label(card, text=title, font=("Segoe UI", 7, "bold"), fg="#64748B", bg="#FFFFFF")
        lbl_t.pack(anchor=tk.W)

        lbl_v = tk.Label(card, text=initial_value, font=("Segoe UI", 12, "bold"), fg=self.primary_color, bg="#FFFFFF")
        lbl_v.pack(anchor=tk.W, pady=(2, 0))
        return lbl_v

    def _build_projection_table(self):
        """Cria a tabela Treeview para exibição dos valores futuros projetados."""
        columns = ("Data", "Preco_Projetado", "Limite_Inferior", "Limite_Superior", "Variacao")
        self.tree_forecast = ttk.Treeview(self.tab_table, columns=columns, show="headings", height=10)

        self.tree_forecast.heading("Data", text="Data Projetada")
        self.tree_forecast.heading("Preco_Projetado", text="Preço Estimado")
        self.tree_forecast.heading("Limite_Inferior", text="Limite Inferior (95%)")
        self.tree_forecast.heading("Limite_Superior", text="Limite Superior (95%)")
        self.tree_forecast.heading("Variacao", text="Variação vs. Atual (%)")

        for col in columns:
            self.tree_forecast.column(col, anchor=tk.CENTER, width=140)

        scroll_y = ttk.Scrollbar(self.tab_table, orient=tk.VERTICAL, command=self.tree_forecast.yview)
        self.tree_forecast.configure(yscrollcommand=scroll_y.set)

        self.tree_forecast.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y, pady=8)

    def _set_default_values(self):
        """Preenche valores padrão na interface gráfica."""
        self.combo_asset.current(0)  # Dólar USD/BRL
        self.combo_period.current(3)  # 1 Ano

        today = datetime.today()
        one_year_ago = today - timedelta(days=365)
        self.entry_start.insert(0, one_year_ago.strftime("%Y-%m-%d"))
        self.entry_end.insert(0, today.strftime("%Y-%m-%d"))

        self.spin_pop.set(DEFAULT_GA_CONFIG["population_size"])
        self.spin_gen.set(DEFAULT_GA_CONFIG["generations"])
        self.spin_mut.set(int(DEFAULT_GA_CONFIG["mutation_rate"] * 100))
        self.spin_horizon.set(DEFAULT_GA_CONFIG["forecast_horizon"])

        self._on_asset_changed()
        self._on_period_changed()

    def _on_asset_changed(self, event=None):
        selected = self.combo_asset.get()
        if "Personalizado" in selected:
            self.frame_custom.pack(fill=tk.X, pady=(0, 6))
        else:
            self.frame_custom.pack_forget()

    def _on_period_changed(self, event=None):
        selected = self.combo_period.get()
        if selected == "Personalizado":
            self.frame_dates.pack(fill=tk.X, pady=(0, 8))
        else:
            self.frame_dates.pack_forget()

    def _start_training_thread(self):
        """Dispara a execução do treinamento em uma thread separada para não travar a GUI."""
        self.btn_run.config(state="disabled")
        self.btn_report.config(state="disabled")
        self.lbl_status.config(text="Baixando cotações do Yahoo Finance...")
        self.progress_bar["value"] = 5

        thread = threading.Thread(target=self._run_optimization_pipeline, daemon=True)
        thread.start()

    def _run_optimization_pipeline(self):
        """Pipeline completo executado em background."""
        try:
            # 1. Determina Ticker e Metadados
            selected_asset = self.combo_asset.get()
            if "Personalizado" in selected_asset:
                ticker = self.entry_custom_ticker.get().strip().upper()
                asset_name = f"Custom ({ticker})"
                currency = "Unidade"
            else:
                info = ASSETS[selected_asset]
                ticker = info["ticker"]
                asset_name = selected_asset
                currency = info.get("currency", "USD")

            self.current_ticker = ticker
            self.current_asset_name = asset_name
            self.current_currency = currency

            period_choice = self.combo_period.get()
            period_str = PERIOD_CHOICES.get(period_choice, "1y")
            self.current_period_str = period_str

            start_date = None
            end_date = None
            if period_str == "custom":
                start_date = self.entry_start.get().strip()
                end_date = self.entry_end.get().strip()

            # 2. Download dos dados
            self.raw_df = self.data_fetcher.fetch_asset_data(
                ticker=ticker,
                period=period_str if period_str != "custom" else None,
                start_date=start_date,
                end_date=end_date
            )

            # 3. Engenharia de Atributos e Lags
            lookback = DEFAULT_GA_CONFIG["lookback_window"]
            X, y, target_dates, scaler_params = self.data_fetcher.prepare_lagged_features(
                self.raw_df, lookback=lookback
            )

            if len(X) < 20:
                raise ValueError(f"Série histórica insuficiente ({len(X)} pontos). Escolha um período maior.")

            # 4. Configuração do GA a partir dos inputs
            pop_size = int(self.spin_pop.get())
            generations = int(self.spin_gen.get())
            mut_rate = float(self.spin_mut.get()) / 100.0
            horizon = int(self.spin_horizon.get())

            self.ga_config = {
                "population_size": pop_size,
                "generations": generations,
                "crossover_rate": DEFAULT_GA_CONFIG["crossover_rate"],
                "mutation_rate": mut_rate,
                "elitism_ratio": DEFAULT_GA_CONFIG["elitism_ratio"],
                "lookback_window": lookback,
                "forecast_horizon": horizon
            }

            ga_engine = GeneticEngine(
                population_size=pop_size,
                generations=generations,
                crossover_rate=self.ga_config["crossover_rate"],
                mutation_rate=mut_rate,
                elitism_ratio=self.ga_config["elitism_ratio"]
            )

            # Callback para atualizar a barra de progresso da GUI
            def ga_callback(gen: int, max_gen: int, best_fit: float, avg_fit: float, best_rmse: float):
                progress = 10 + int((gen / max_gen) * 85)
                self.after(0, lambda: self._update_progress_ui(gen, max_gen, best_fit, best_rmse, progress))

            # Execução da evolução
            best_individual = ga_engine.evolve(X, y, callback=ga_callback)
            self.ga_history = ga_engine.history

            # 5. Predição, Métricas e Projeção Futura
            is_crypto = "BTC" in ticker or "ETH" in ticker
            predictor = ModelPredictor(best_individual=best_individual, lookback_window=lookback)
            self.prediction_results = predictor.evaluate_and_predict(
                df_raw=self.raw_df,
                X=X,
                dates=target_dates,
                forecast_horizon=horizon,
                is_crypto=is_crypto
            )

            # 6. Planilhamento automático do ensaio em CSV
            metrics = self.prediction_results["metrics"]
            trial_record = {
                "Data_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ativo": asset_name,
                "Ticker": ticker,
                "Moeda": currency,
                "Periodo": period_choice,
                "Populacao": pop_size,
                "Geracoes": generations,
                "Taxa_Mutacao": mut_rate,
                "Taxa_Crossover": self.ga_config["crossover_rate"],
                "Lookback": lookback,
                "Horizonte_Dias": horizon,
                "Ultimo_Preco_Real": metrics["ultimo_preco_real"],
                "Preco_Projetado_Final": metrics["preco_projetado_final"],
                "Variacao_Esperada_Pct": metrics["variacao_esperada_pct"],
                "Tendencia": metrics["tendencia_esperada"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "MAPE_Pct": metrics["mape"],
                "R2": metrics["r2"],
                "Acuracia_Direcional_Pct": metrics["acuracia_direcional"]
            }
            ReportGenerator.log_trial(trial_record)

            # Atualização da interface na thread principal
            self.after(0, self._render_results_ui)

        except Exception as e:
            logger.error(f"Erro no pipeline de otimização: {e}", exc_info=True)
            self.after(0, lambda: self._show_error_message(str(e)))
        finally:
            self.after(0, lambda: self.btn_run.config(state="normal"))

    def _update_progress_ui(self, gen: int, max_gen: int, best_fit: float, best_rmse: float, progress: int):
        self.progress_bar["value"] = progress
        self.lbl_status.config(text=f"Geração {gen}/{max_gen} | Fitness: {best_fit:.1f} | RMSE: {best_rmse:.4f}")

    def _render_results_ui(self):
        """Renderiza os KPIs, gráficos e tabelas após conclusão da otimização."""
        metrics = self.prediction_results["metrics"]
        curr = self.current_currency

        # Atualiza KPIs
        self.card_actual.config(text=f"{curr} {metrics['ultimo_preco_real']:,.2f}")
        self.card_predicted.config(text=f"{curr} {metrics['preco_projetado_final']:,.2f}")

        var_pct = metrics["variacao_esperada_pct"]
        var_color = "#16A34A" if var_pct >= 0 else "#DC2626"
        self.card_variation.config(text=f"{var_pct:+.2f}%", fg=var_color)

        self.card_rmse.config(text=f"{metrics['rmse']:,.4f}")
        self.card_directional.config(text=f"{metrics['acuracia_direcional']:.1f}%")

        # 1. Atualiza Gráfico de Predição
        self.ax_pred.clear()
        df_hist = self.prediction_results["history_df"].tail(90)
        df_fore = self.prediction_results["forecast_df"]

        self.ax_pred.plot(df_hist.index, df_hist["Preco_Real"], label="Preço Real (Mercado)", color="#1F77B4", lw=2.0)
        self.ax_pred.plot(df_hist.index, df_hist["Preco_Previsto_IA"], label="Ajuste Algoritmo Genético", color="#FF7F0E", lw=1.6, linestyle="--")

        # Projeção futura
        last_date = df_hist.index[-1]
        last_price = df_hist["Preco_Real"].iloc[-1]
        proj_dates = [last_date] + list(df_fore.index)
        proj_prices = [last_price] + list(df_fore["Preco_Projetado"])
        proj_lower = [last_price] + list(df_fore["Limite_Inferior"])
        proj_upper = [last_price] + list(df_fore["Limite_Superior"])

        self.ax_pred.plot(proj_dates, proj_prices, label="Projeção Futura (IA)", color="#2CA02C", lw=2.2, marker="o", markersize=4)
        self.ax_pred.fill_between(proj_dates, proj_lower, proj_upper, color="#2CA02C", alpha=0.18, label="Intervalo de Confiança (95%)")

        self.ax_pred.set_title(f"Série Histórica e Projeção IA - {self.current_asset_name}", fontsize=11, fontweight="bold")
        self.ax_pred.set_ylabel(f"Cotação ({self.current_currency})", fontsize=9)
        self.ax_pred.legend(loc="best", fontsize=8)
        self.ax_pred.grid(True, linestyle=":", alpha=0.6)
        self.fig_pred.tight_layout()
        self.canvas_pred.draw()

        # 2. Atualiza Gráfico de Convergência do GA
        self.ax_conv.clear()
        gens = self.ga_history["generation"]
        best_fit = self.ga_history["best_fitness"]
        avg_fit = self.ga_history["avg_fitness"]

        self.ax_conv.plot(gens, best_fit, label="Melhor Fitness da População", color="#DC2626", lw=2.2)
        self.ax_conv.plot(gens, avg_fit, label="Fitness Médio da População", color="#2563EB", lw=1.6, linestyle=":")

        self.ax_conv.set_title("Curva de Aprendizado e Convergência Evolutiva", fontsize=11, fontweight="bold")
        self.ax_conv.set_xlabel("Geração", fontsize=9)
        self.ax_conv.set_ylabel("Fitness", fontsize=9)
        self.ax_conv.legend(loc="best", fontsize=8)
        self.ax_conv.grid(True, linestyle=":", alpha=0.6)
        self.fig_conv.tight_layout()
        self.canvas_conv.draw()

        # 3. Atualiza Tabela Passo a Passo
        for item in self.tree_forecast.get_children():
            self.tree_forecast.delete(item)

        initial_ref_price = metrics["ultimo_preco_real"]
        for date_val, row in df_fore.iterrows():
            step_var = ((row["Preco_Projetado"] - initial_ref_price) / (initial_ref_price + 1e-9)) * 100.0
            self.tree_forecast.insert(
                "",
                tk.END,
                values=(
                    date_val.strftime("%d/%m/%Y"),
                    f"{row['Preco_Projetado']:,.4f}",
                    f"{row['Limite_Inferior']:,.4f}",
                    f"{row['Limite_Superior']:,.4f}",
                    f"{step_var:+.2f}%"
                )
            )

        self.progress_bar["value"] = 100
        self.lbl_status.config(text=f"Predição concluída com sucesso! ({metrics['tendencia_esperada']})")
        self.btn_report.config(state="normal")

    def _export_docx_report(self):
        """Gera o relatório em formato DOCX com os gráficos de alta resolução embutidos."""
        if not self.prediction_results or not self.ga_history:
            messagebox.showwarning("Aviso", "Execute primeiro a otimização antes de gerar o relatório.")
            return

        try:
            self.lbl_status.config(text="Gerando gráficos de alta resolução e relatório Word...")
            charts = ReportGenerator.export_charts_for_report(
                prediction_results=self.prediction_results,
                ga_history=self.ga_history,
                asset_name=self.current_asset_name,
                currency=self.current_currency
            )

            report_path = ReportGenerator.generate_docx_report(
                asset_name=self.current_asset_name,
                ticker=self.current_ticker,
                currency=self.current_currency,
                period_str=self.combo_period.get(),
                ga_config=self.ga_config,
                prediction_results=self.prediction_results,
                ga_history=self.ga_history,
                chart_paths=charts
            )

            self.lbl_status.config(text="Relatório DOCX gerado com sucesso!")
            messagebox.showinfo("Sucesso", f"Relatório executivo gerado com sucesso em:\n{report_path}")

        except Exception as e:
            logger.error(f"Erro ao gerar DOCX: {e}", exc_info=True)
            messagebox.showerror("Erro", f"Falha ao gerar relatório DOCX:\n{e}")

    def _open_reports_folder(self):
        """Abre a pasta de relatórios no Explorer do Windows."""
        try:
            folder = str(REPORTS_DIR.resolve())
            os.startfile(folder)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir o diretório: {e}")

    def _show_error_message(self, err_msg: str):
        self.lbl_status.config(text="Erro durante o processamento.")
        self.progress_bar["value"] = 0
        messagebox.showerror("Erro na Execução", f"Ocorreu um erro durante a execução:\n{err_msg}")


if __name__ == "__main__":
    app = ModernFinancialGUI()
    app.mainloop()
