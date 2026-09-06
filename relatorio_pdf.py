"""
Módulo de Geração de Relatórios Executivos em PDF
Gera relatórios corporativos completos utilizando ReportLab e Matplotlib.
Todos os nomes de funções, variáveis e comentários estão em português.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import HRFlowable, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import REPORTS_DIR


class GeradorRelatorioPDF:
    """
    Classe responsável por estruturar, formatar e compilar relatórios executivos
    em formato PDF para análise e projeção de ativos com Inteligência Artificial.
    """

    @staticmethod
    def exportar_graficos_analiticos(
        resultados_predicao: Dict[str, Any],
        historico_treinamento: Optional[Dict[str, Any]],
        nome_ativo: str,
        moeda: str,
        diretorio_saida: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        Exporta gráficos em formato PNG de alta resolução para inserção no PDF.

        Parâmetros:
            resultados_predicao: Dicionário contendo DataFrames de histórico e projeção.
            historico_treinamento: Métricas por época/geração do treinamento do modelo.
            nome_ativo: Nome amigável do ativo (ex: Dólar, Bitcoin, Soja).
            moeda: Símbolo ou sigla da moeda (BRL, USD).
            diretorio_saida: Pasta onde os arquivos de imagem temporários serão gravados.

        Retorno:
            Dicionário com os caminhos absolutos dos gráficos gerados.
        """
        # Define o diretório de destino dos gráficos
        pasta_destino = diretorio_saida or REPORTS_DIR
        pasta_destino.mkdir(parents=True, exist_ok=True)

        # Cria carimbo de data/hora único para evitar sobreposição de arquivos
        carimbo_tempo = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_grafico_predicao = str(pasta_destino / f"grafico_pred_pdf_{carimbo_tempo}.png")
        caminho_grafico_convergencia = str(pasta_destino / f"grafico_conv_pdf_{carimbo_tempo}.png")

        # -------------------------------------------------------------
        # 1. Gráfico de Série Histórica e Cone de Projeção (95% Confiança)
        # -------------------------------------------------------------
        figura_pred, eixo_pred = plt.subplots(figsize=(8.5, 4.0), dpi=300)
        eixo_pred.set_facecolor("#FAFAFA")

        df_historico = resultados_predicao.get("history_df", pd.DataFrame())
        df_projecao = resultados_predicao.get("forecast_df", pd.DataFrame())

        # Exibe os últimos 60 dias de histórico para clareza visual no documento impresso
        df_recorte_hist = df_historico.tail(60) if not df_historico.empty else pd.DataFrame()

        if not df_recorte_hist.empty:
            eixo_pred.plot(
                df_recorte_hist.index,
                df_recorte_hist["Preco_Real"],
                label="Preço Real de Mercado",
                color="#0284C7",
                linewidth=1.8
            )
            if "Preco_Previsto_IA" in df_recorte_hist.columns:
                eixo_pred.plot(
                    df_recorte_hist.index,
                    df_recorte_hist["Preco_Previsto_IA"],
                    label="Ajuste do Modelo IA",
                    color="#F97316",
                    linewidth=1.4,
                    linestyle="--"
                )

        if not df_projecao.empty and not df_recorte_hist.empty:
            # Ponto de ancoragem entre o último valor real e o primeiro valor projetado
            data_ancora = df_recorte_hist.index[-1]
            preco_ancora = df_recorte_hist["Preco_Real"].iloc[-1]

            datas_proj = [data_ancora] + list(df_projecao.index)
            precos_proj = [preco_ancora] + list(df_projecao["Preco_Projetado"])
            limites_inferiores = [preco_ancora] + list(df_projecao["Limite_Inferior"])
            limites_superiores = [preco_ancora] + list(df_projecao["Limite_Superior"])

            # Linha da previsão futura
            eixo_pred.plot(
                datas_proj,
                precos_proj,
                label="Projeção Futura (IA)",
                color="#10B981",
                linewidth=2.0,
                marker="o",
                markersize=3.5
            )

            # Preenchimento sombreado da área de confiança (95%)
            eixo_pred.fill_between(
                datas_proj,
                limites_inferiores,
                limites_superiores,
                color="#10B981",
                alpha=0.18,
                label="Intervalo de Confiança (95%)"
            )

        eixo_pred.set_title(f"Série Temporal e Projeção IA - {nome_ativo}", fontsize=11, fontweight="bold", pad=8)
        eixo_pred.set_xlabel("Data", fontsize=8)
        eixo_pred.set_ylabel(f"Cotação ({moeda})", fontsize=8)
        eixo_pred.tick_params(labelsize=7)
        eixo_pred.grid(True, linestyle=":", alpha=0.6)
        eixo_pred.legend(loc="best", fontsize=7, framealpha=0.85)

        figura_pred.tight_layout()
        figura_pred.savefig(caminho_grafico_predicao, dpi=300)
        plt.close(figura_pred)

        # -------------------------------------------------------------
        # 2. Gráfico de Aprendizado ou Convergência
        # -------------------------------------------------------------
        figura_conv, eixo_conv = plt.subplots(figsize=(8.5, 3.2), dpi=300)
        eixo_conv.set_facecolor("#FAFAFA")

        if historico_treinamento and "generations" in historico_treinamento:
            # Formato Algoritmo Genético
            epocas = historico_treinamento["generations"]
            melhores = historico_treinamento.get("best_fitness", [])
            medias = historico_treinamento.get("avg_fitness", [])

            eixo_conv.plot(epocas, melhores, label="Melhor Fitness da População", color="#E11D48", linewidth=1.8)
            if medias:
                eixo_conv.plot(epocas, medias, label="Fitness Médio", color="#3B82F6", linewidth=1.3, linestyle=":")
            eixo_conv.set_ylabel("Fitness (Aptidão)", fontsize=8)
            eixo_conv.set_xlabel("Geração Evolutiva", fontsize=8)
            eixo_conv.set_title("Curva de Convergência Evolutiva do Algoritmo", fontsize=10, fontweight="bold", pad=6)
        elif historico_treinamento and "perdas" in historico_treinamento:
            # Formato Redes Neurais (LSTM/GRU/Transformer)
            epocas = list(range(1, len(historico_treinamento["perdas"]) + 1))
            perdas = historico_treinamento["perdas"]
            eixo_conv.plot(epocas, perdas, label="Função de Custo (MSE)", color="#8B5CF6", linewidth=1.8)
            eixo_conv.set_ylabel("Perda (Loss)", fontsize=8)
            eixo_conv.set_xlabel("Época de Treinamento", fontsize=8)
            eixo_conv.set_title("Curva de Aprendizado e Minimização de Perda", fontsize=10, fontweight="bold", pad=6)
        else:
            # Fallback informativo para modelos que não usam épocas
            eixo_conv.text(
                0.5, 0.5,
                "Modelo ajustado analiticamente em etapa única.\nConvergência ótima obtida diretamente.",
                horizontalalignment="center", verticalalignment="center",
                fontsize=9, color="#64748B", style="italic"
            )
            eixo_conv.set_xticks([])
            eixo_conv.set_yticks([])
            eixo_conv.set_title("Otimização Analítica do Modelo", fontsize=10, fontweight="bold", pad=6)

        eixo_conv.tick_params(labelsize=7)
        eixo_conv.grid(True, linestyle=":", alpha=0.6)
        eixo_conv.legend(loc="best", fontsize=7, framealpha=0.85)

        figura_conv.tight_layout()
        figura_conv.savefig(caminho_grafico_convergencia, dpi=300)
        plt.close(figura_conv)

        return {
            "grafico_predicao": caminho_grafico_predicao,
            "grafico_convergencia": caminho_grafico_convergencia
        }

    @staticmethod
    def gerar_relatorio_pdf(
        nome_ativo: str,
        ticker: str,
        moeda: str,
        nome_algoritmo: str,
        periodo_escolhido: str,
        parametros_algoritmo: Dict[str, Any],
        resultados_predicao: Dict[str, Any],
        historico_treinamento: Optional[Dict[str, Any]],
        caminhos_graficos: Dict[str, str],
        caminho_arquivo_saida: Optional[str] = None
    ) -> str:
        """
        Compila o documento PDF com formatação profissional, tabelas de métricas,
        gráficos e detalhamento diário de projeções.

        Parâmetros:
            nome_ativo: Identificação do ativo.
            ticker: Símbolo Yahoo Finance.
            moeda: Moeda de negociação (USD, BRL).
            nome_algoritmo: Nome do algoritmo selecionado (ex: XGBoost, LSTM, MiroFish).
            periodo_escolhido: Janela de leitura histórica (1y, 6mo, etc.).
            parametros_algoritmo: Hiperparâmetros empregados no treino.
            resultados_predicao: Métricas e projeções calculadas.
            historico_treinamento: Log de perdas ou fitness.
            caminhos_graficos: Imagens dos gráficos gerados.
            caminho_arquivo_saida: Destino opcional do arquivo PDF gerado.

        Retorno:
            Caminho absoluto do arquivo PDF criado.
        """
        # Define caminho do PDF de saída
        if not caminho_arquivo_saida:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            from data_fetcher import DataFetcher
            nome_base = DataFetcher.obter_caminho_arquivo_excel(ticker).stem

            if resultados_predicao and "history_df" in resultados_predicao and not resultados_predicao["history_df"].empty:
                idx = resultados_predicao["history_df"].index
                dt_ini = pd.to_datetime(idx.min()).strftime("%d-%m-%Y")
                dt_fim = pd.to_datetime(idx.max()).strftime("%d-%m-%Y")
            else:
                agora = datetime.now()
                dt_fim = agora.strftime("%d-%m-%Y")
                dt_ini = (agora - timedelta(days=365)).strftime("%d-%m-%Y")

            caminho_final = str(REPORTS_DIR / f"{nome_base} {dt_ini} a {dt_fim}.pdf")
        else:
            caminho_final = caminho_arquivo_saida

        # Configura documento PDF (A4 com margens de 1.8 cm)
        doc = SimpleDocTemplate(
            caminho_final,
            pagesize=A4,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm
        )

        # Paleta de Cores Corporativas
        cor_primaria = colors.HexColor("#0F172A")    # Azul ardósia escuro
        cor_secundaria = colors.HexColor("#0284C7")  # Ciano escuro elegante
        cor_fundo_tabela = colors.HexColor("#F8FAFC")# Cinza muito claro
        cor_sucesso = colors.HexColor("#10B981")     # Verde esmeralda
        cor_perigo = colors.HexColor("#E11D48")      # Vermelho rubi
        cor_texto_escuro = colors.HexColor("#1E293B")
        cor_cinza_medio = colors.HexColor("#64748B")

        # Estilos de Parágrafos
        folha_estilos = getSampleStyleSheet()

        estilo_categoria = ParagraphStyle(
            "CategoriaDoc",
            parent=folha_estilos["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=cor_secundaria,
            spaceAfter=3,
            textTransform="uppercase"
        )

        estilo_titulo = ParagraphStyle(
            "TituloDoc",
            parent=folha_estilos["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=cor_primaria,
            alignment=TA_LEFT,
            spaceAfter=6
        )

        estilo_meta = ParagraphStyle(
            "MetaDoc",
            parent=folha_estilos["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            textColor=cor_cinza_medio,
            spaceAfter=12
        )

        estilo_secao = ParagraphStyle(
            "SecaoDoc",
            parent=folha_estilos["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=cor_primaria,
            spaceBefore=10,
            spaceAfter=6
        )

        estilo_corpo = ParagraphStyle(
            "CorpoDoc",
            parent=folha_estilos["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=cor_texto_escuro,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        )

        estilo_celula = ParagraphStyle(
            "CelulaDoc",
            parent=folha_estilos["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=cor_texto_escuro,
            alignment=TA_CENTER
        )

        estilo_celula_bold = ParagraphStyle(
            "CelulaBoldDoc",
            parent=folha_estilos["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=cor_primaria,
            alignment=TA_CENTER
        )

        elementos: List[Any] = []

        # -------------------------------------------------------------
        # 1. Cabeçalho Principal do Relatório
        # -------------------------------------------------------------
        elementos.append(Paragraph("RELATÓRIO EXECUTIVO DE INTELIGÊNCIA ARTIFICIAL", estilo_categoria))
        elementos.append(Paragraph("Projeção e Modelagem de Séries Temporais Financeiras", estilo_titulo))

        data_extenso = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        texto_metadados = (
            f"<b>Ativo:</b> {nome_ativo} ({ticker}) | "
            f"<b>Moeda:</b> {moeda} | "
            f"<b>Algoritmo:</b> {nome_algoritmo} | "
            f"<b>Emissão:</b> {data_extenso}"
        )
        elementos.append(Paragraph(texto_metadados, estilo_meta))
        elementos.append(HRFlowable(width="100%", thickness=1.5, color=cor_secundaria, spaceBefore=0, spaceAfter=10))

        # -------------------------------------------------------------
        # 2. Resumo Executivo e Métricas Principais (KPIs)
        # -------------------------------------------------------------
        metricas = resultados_predicao.get("metrics", {})
        preco_real_atual = metricas.get("ultimo_preco_real", 0.0)
        preco_proj_final = metricas.get("preco_projetado_final", 0.0)
        variacao_pct = metricas.get("variacao_esperada_pct", 0.0)
        tendencia = metricas.get("tendencia_esperada", "Indefinida")
        rmse = metricas.get("rmse", 0.0)
        mae = metricas.get("mae", 0.0)
        mape = metricas.get("mape", 0.0)
        r2 = metricas.get("r2", 0.0)
        acuracia_dir = metricas.get("acuracia_direcional", 0.0)

        sinal_cor = cor_sucesso if variacao_pct >= 0 else cor_perigo
        sinal_txt = f"+{variacao_pct:.2f}%" if variacao_pct >= 0 else f"{variacao_pct:.2f}%"

        elementos.append(Paragraph("1. Painel de Indicadores de Desempenho e Projeção (KPIs)", estilo_secao))

        # Tabela formatada em grade com os KPIs principais
        dados_tabela_kpis = [
            [
                Paragraph("<b>ÚLTIMO PREÇO REAL</b>", estilo_celula),
                Paragraph("<b>ALVO PROJETADO IA</b>", estilo_celula),
                Paragraph("<b>VARIAÇÃO ESPERADA</b>", estilo_celula),
                Paragraph("<b>RMSE (ERRO)</b>", estilo_celula),
                Paragraph("<b>ACURÁCIA DIRECIONAL</b>", estilo_celula),
            ],
            [
                Paragraph(f"<b>{moeda} {preco_real_atual:,.2f}</b>", estilo_celula_bold),
                Paragraph(f"<b>{moeda} {preco_proj_final:,.2f}</b>", estilo_celula_bold),
                Paragraph(f"<b><font color='{sinal_cor.hexval()}'>{sinal_txt}</font></b>", estilo_celula_bold),
                Paragraph(f"<b>{rmse:.4f}</b>", estilo_celula_bold),
                Paragraph(f"<b>{acuracia_dir:.1f}%</b>", estilo_celula_bold),
            ]
        ]

        tabela_kpis = Table(dados_tabela_kpis, colWidths=[3.4 * cm] * 5)
        tabela_kpis.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, 1), cor_fundo_tabela),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ]))
        elementos.append(tabela_kpis)
        elementos.append(Spacer(1, 8))

        # Texto explicativo corporativo
        texto_resumo = (
            f"O modelo <b>{nome_algoritmo}</b> foi treinado para projetar o comportamento de preços de <b>{nome_ativo}</b> "
            f"considerando o histórico de cotações ({periodo_escolhido}). A projeção aponta para uma tendência de <b>{tendencia}</b>, "
            f"com variação calculada em <b>{sinal_txt}</b> até o horizonte estipulado. "
            f"No conjunto de teste, o modelo obteve coeficiente de determinação <b>R² = {r2:.4f}</b>, "
            f"erro absoluto médio <b>MAE = {mae:,.4f}</b> e acurácia de direção de <b>{acuracia_dir:.1f}%</b>."
        )
        elementos.append(Paragraph(texto_resumo, estilo_corpo))
        elementos.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # 3. Gráficos Analíticos
        # -------------------------------------------------------------
        elementos.append(Paragraph("2. Visualizações Gráficas Analíticas", estilo_secao))

        caminho_grafico_pred = caminhos_graficos.get("grafico_predicao")
        if caminho_grafico_pred and os.path.exists(caminho_grafico_pred):
            img_pred = Image(caminho_grafico_pred, width=17.2 * cm, height=8.1 * cm)
            elementos.append(img_pred)
            elementos.append(Spacer(1, 6))

        caminho_grafico_conv = caminhos_graficos.get("grafico_convergencia")
        if caminho_grafico_conv and os.path.exists(caminho_grafico_conv):
            img_conv = Image(caminho_grafico_conv, width=17.2 * cm, height=6.4 * cm)
            elementos.append(img_conv)
            elementos.append(Spacer(1, 8))

        # -------------------------------------------------------------
        # 4. Tabela Passo a Passo com Projeção e Intervalos de Confiança (95%)
        # -------------------------------------------------------------
        elementos.append(Paragraph("3. Detalhamento Diário da Projeção e Limites Estatísticos (95% IC)", estilo_secao))

        df_proj = resultados_predicao.get("forecast_df", pd.DataFrame())
        linhas_tabela_proj = [
            [
                Paragraph("<b>Data</b>", estilo_celula),
                Paragraph(f"<b>Preço Projetado ({moeda})</b>", estilo_celula),
                Paragraph(f"<b>Limite Inferior (95%)</b>", estilo_celula),
                Paragraph(f"<b>Limite Superior (95%)</b>", estilo_celula),
                Paragraph("<b>Variação vs. Atual</b>", estilo_celula),
                Paragraph("<b>Direção</b>", estilo_celula),
            ]
        ]

        if not df_proj.empty:
            for data_item, linha in df_proj.iterrows():
                preco_passo = linha["Preco_Projetado"]
                lim_inf = linha["Limite_Inferior"]
                lim_sup = linha["Limite_Superior"]
                delta_passo = ((preco_passo - preco_real_atual) / (preco_real_atual + 1e-9)) * 100.0

                sinal_passo_cor = cor_sucesso if delta_passo >= 0 else cor_perigo
                sinal_passo_txt = f"+{delta_passo:.2f}%" if delta_passo >= 0 else f"{delta_passo:.2f}%"
                direcao_txt = "▲ Alta" if delta_passo >= 0 else "▼ Baixa"

                linhas_tabela_proj.append([
                    Paragraph(data_item.strftime("%d/%m/%Y"), estilo_celula),
                    Paragraph(f"<b>{preco_passo:,.4f}</b>", estilo_celula_bold),
                    Paragraph(f"{lim_inf:,.4f}", estilo_celula),
                    Paragraph(f"{lim_sup:,.4f}", estilo_celula),
                    Paragraph(f"<font color='{sinal_passo_cor.hexval()}'>{sinal_passo_txt}</font>", estilo_celula_bold),
                    Paragraph(f"<font color='{sinal_passo_cor.hexval()}'>{direcao_txt}</font>", estilo_celula),
                ])

        tabela_projecao = Table(linhas_tabela_proj, colWidths=[2.6 * cm, 3.2 * cm, 3.0 * cm, 3.0 * cm, 2.8 * cm, 2.6 * cm])
        tabela_projecao.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, cor_fundo_tabela]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        elementos.append(tabela_projecao)
        elementos.append(Spacer(1, 14))

        # Rodapé corporativo
        texto_disclaimer = (
            "<i>Aviso Legal: As projeções geradas por inteligência artificial e séries temporais possuem finalidade "
            "estritamente analítica e educacional. Cotações de ativos financeiros e commodities estão sujeitas à volatilidade "
            "e eventos estocásticos de mercado. Não constitui recomendação de investimento.</i>"
        )
        elementos.append(Paragraph(texto_disclaimer, estilo_meta))

        # Compila e gera o PDF final
        doc.build(elementos)
        return caminho_final
