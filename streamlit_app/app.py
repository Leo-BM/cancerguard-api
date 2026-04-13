"""
CancerGuard — Interface de Predição de Câncer de Mama
Fase 7: Streamlit UI
"""

import os
import requests
import streamlit as st

# ─── Configuração da página (deve ser o PRIMEIRO comando Streamlit) ────────────
st.set_page_config(
    page_title="CancerGuard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Variável de ambiente para URL da API ─────────────────────────────────────
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ─── CSS Global ───────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Header ── */
        .cg-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.5rem 0 0.5rem 0;
            border-bottom: 2px solid #2d2d2d;
            margin-bottom: 1.5rem;
        }
        .cg-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #f0f0f0;
            margin: 0;
            line-height: 1;
        }
        .cg-subtitle {
            font-size: 0.95rem;
            color: #888;
            margin: 0.25rem 0 0 0;
        }

        /* ── Status badges ── */
        .badge-online {
            background: #0d3321;
            color: #4ade80;
            border: 1px solid #166534;
            border-radius: 999px;
            padding: 0.3rem 0.9rem;
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .badge-offline {
            background: #3b0a0a;
            color: #f87171;
            border: 1px solid #7f1d1d;
            border-radius: 999px;
            padding: 0.3rem 0.9rem;
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
        }

        /* ── Result cards ── */
        .card-malignant {
            background: #1a0505;
            border: 2px solid #dc2626;
            border-radius: 12px;
            padding: 1.8rem 2rem;
            margin: 1.5rem 0;
        }
        .card-benign {
            background: #031a0a;
            border: 2px solid #16a34a;
            border-radius: 12px;
            padding: 1.8rem 2rem;
            margin: 1.5rem 0;
        }
        .card-title {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
        }
        .card-malignant .card-title { color: #f87171; }
        .card-benign .card-title { color: #4ade80; }
        .card-prob {
            font-size: 2.8rem;
            font-weight: 700;
            line-height: 1;
            margin: 0.5rem 0;
        }
        .card-malignant .card-prob { color: #f87171; }
        .card-benign .card-prob { color: #4ade80; }
        .card-label {
            font-size: 0.85rem;
            color: #aaa;
            margin: 0;
        }

        /* ── Risk badges ── */
        .badge-high {
            background: #450a0a;
            color: #f87171;
            border: 1px solid #dc2626;
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-medium {
            background: #431407;
            color: #fb923c;
            border: 1px solid #ea580c;
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-low {
            background: #052e16;
            color: #4ade80;
            border: 1px solid #16a34a;
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* ── SHAP bars ── */
        .shap-bar {
            font-family: monospace;
            font-size: 0.88rem;
            color: #d1d5db;
            line-height: 1.9;
        }
        .shap-feature {
            color: #9ca3af;
            font-size: 0.8rem;
        }
        .shap-positive { color: #f87171; }
        .shap-negative { color: #4ade80; }

        /* ── Info section table ── */
        .info-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }
        .info-table th {
            background: #1e1e2e;
            color: #c4b5fd;
            text-align: left;
            padding: 0.6rem 0.9rem;
            border-bottom: 1px solid #374151;
        }
        .info-table td {
            padding: 0.55rem 0.9rem;
            border-bottom: 1px solid #1f2937;
            color: #d1d5db;
            vertical-align: top;
        }
        .info-table tr:hover td { background: #111827; }

        /* ── Misc ── */
        .divider { border-color: #2d2d2d; margin: 1.5rem 0; }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #c4b5fd;
            margin: 1.2rem 0 0.6rem 0;
            letter-spacing: 0.03em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Verificar saúde da API ───────────────────────────────────────────────────
def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ─── Header ───────────────────────────────────────────────────────────────────
def render_header(api_online: bool) -> None:
    badge = (
        '<span class="badge-online">● API Online</span>'
        if api_online
        else '<span class="badge-offline">✕ API Offline</span>'
    )
    st.markdown(
        f"""
        <div class="cg-header">
            <div>
                <p class="cg-title">🔬 CancerGuard</p>
                <p class="cg-subtitle">Sistema de auxílio diagnóstico para câncer de mama · Wisconsin Breast Cancer Dataset</p>
            </div>
            <div style="margin-left:auto">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Seção informativa ────────────────────────────────────────────────────────
def render_about_section() -> None:
    with st.expander("ℹ️  O que é este sistema e como usar?", expanded=False):
        col1, col2 = st.columns([1.2, 1], gap="large")

        with col1:
            st.markdown(
                """
                **CancerGuard** é um sistema de apoio ao diagnóstico baseado em
                Machine Learning (SVM com kernel RBF) treinado no Wisconsin Breast
                Cancer Dataset (569 amostras, 30 features).

                > ⚠️ **Aviso importante:** Esta ferramenta é exclusivamente para fins
                > educacionais e de demonstração. Não substitui avaliação médica
                > profissional.

                **Como funciona:**
                1. Preencha os campos com os valores da biópsia da célula tumoral.
                2. Clique em **Analisar Tumor**.
                3. O sistema retorna a classificação (*Benigno* ou *Maligno*),
                   a probabilidade e os fatores que mais pesaram na decisão (SHAP).

                **Como obter os valores?**
                Os valores são gerados por software de análise de imagem de biópsia
                (ex.: ImageJ, CellProfiler) que mede propriedades geométricas e
                texturais dos núcleos celulares amostrados.
                """
            )

        with col2:
            st.markdown(
                """
                <table class="info-table">
                <thead>
                  <tr>
                    <th>Grupo</th>
                    <th>Descrição</th>
                    <th>Aba</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Médias (mean)</strong></td>
                    <td>Valor médio de cada característica calculada para todos os
                        núcleos celulares da amostra.</td>
                    <td>📊 Médias</td>
                  </tr>
                  <tr>
                    <td><strong>Erros Padrão (SE)</strong></td>
                    <td>Variação (incerteza) das medições ao redor da média.
                        Indica heterogeneidade entre os núcleos.</td>
                    <td>📉 Erros Padrão</td>
                  </tr>
                  <tr>
                    <td><strong>Piores Valores (worst)</strong></td>
                    <td>Média dos três maiores valores de cada característica.
                        Captura as células mais atípicas da amostra.</td>
                    <td>⚠️ Piores Valores</td>
                  </tr>
                  <tr>
                    <td><strong>Raio</strong></td>
                    <td>Distância média do centro ao perímetro do núcleo.</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td><strong>Textura</strong></td>
                    <td>Desvio padrão dos tons de cinza da imagem.</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td><strong>Perímetro / Área</strong></td>
                    <td>Tamanho e contorno do núcleo.</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td><strong>Suavidade</strong></td>
                    <td>Variação local nos comprimentos do raio.</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td><strong>Compacidade</strong></td>
                    <td>Perímetro² / Área − 1,0</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td><strong>Côncavidade / Pontos côncavos</strong></td>
                    <td>Gravidade e número de porções côncavas do contorno.</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td><strong>Simetria / Dimensão fractal</strong></td>
                    <td>Simetria do núcleo e aproximação da "coastline".</td>
                    <td>—</td>
                  </tr>
                </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )


# ─── Formulário ───────────────────────────────────────────────────────────────
def render_form() -> dict:
    """Renderiza as 3 abas e retorna o payload pronto para a API."""

    st.markdown('<p class="section-title">📋 Dados da biópsia</p>', unsafe_allow_html=True)

    # ── Valores padrão: caso benigno típico do dataset Wisconsin ──────────────
    defaults = {
        # mean
        "radius_mean": 12.0,
        "texture_mean": 17.0,
        "perimeter_mean": 78.0,
        "area_mean": 440.0,
        "smoothness_mean": 0.095,
        "compactness_mean": 0.080,
        "concavity_mean": 0.050,
        "concave_points_mean": 0.030,
        "symmetry_mean": 0.180,
        "fractal_dimension_mean": 0.060,
        # se
        "radius_se": 0.30,
        "texture_se": 1.20,
        "perimeter_se": 2.00,
        "area_se": 25.0,
        "smoothness_se": 0.006,
        "compactness_se": 0.018,
        "concavity_se": 0.020,
        "concave_points_se": 0.008,
        "symmetry_se": 0.020,
        "fractal_dimension_se": 0.003,
        # worst
        "radius_worst": 14.0,
        "texture_worst": 24.0,
        "perimeter_worst": 92.0,
        "area_worst": 600.0,
        "smoothness_worst": 0.130,
        "compactness_worst": 0.200,
        "concavity_worst": 0.200,
        "concave_points_worst": 0.100,
        "symmetry_worst": 0.280,
        "fractal_dimension_worst": 0.080,
    }

    # ── Labels e tooltips em português ────────────────────────────────────────
    labels = {
        "radius": "Raio",
        "texture": "Textura",
        "perimeter": "Perímetro",
        "area": "Área",
        "smoothness": "Suavidade",
        "compactness": "Compacidade",
        "concavity": "Côncavidade",
        "concave_points": "Pontos côncavos",
        "symmetry": "Simetria",
        "fractal_dimension": "Dimensão fractal",
    }
    tooltips = {
        "radius": "Distância média do centro ao perímetro do núcleo celular.",
        "texture": "Desvio padrão dos tons de cinza da imagem.",
        "perimeter": "Comprimento do contorno do núcleo.",
        "area": "Área total da secção transversal do núcleo.",
        "smoothness": "Variação local nos comprimentos do raio.",
        "compactness": "Calculado como: Perímetro² / Área − 1,0.",
        "concavity": "Gravidade das porções côncavas do contorno.",
        "concave_points": "Número de porções côncavas do contorno.",
        "symmetry": "Grau de simetria do núcleo celular.",
        "fractal_dimension": "Aproximação fractal do contorno ('coastline approximation').",
    }

    features = list(labels.keys())
    suffixes = ["mean", "se", "worst"]
    tab_labels = ["📊 Médias (mean)", "📉 Erros Padrão (SE)", "⚠️ Piores Valores (worst)"]

    payload: dict = {}

    tabs = st.tabs(tab_labels)
    for tab, suffix in zip(tabs, suffixes):
        with tab:
            col1, col2 = st.columns(2, gap="medium")
            for i, feat in enumerate(features):
                key = f"{feat}_{suffix}"
                label = f"{labels[feat]}"
                tip = tooltips[feat]
                col = col1 if i % 2 == 0 else col2
                val = col.number_input(
                    label=label,
                    min_value=0.0,
                    value=float(defaults[key]),
                    step=0.001 if defaults[key] < 1 else 0.1,
                    format="%.4f" if defaults[key] < 1 else "%.2f",
                    help=tip,
                    key=key,
                )
                payload[key] = val

    return payload


# ─── Exibir resultado ─────────────────────────────────────────────────────────
def render_result(response: dict) -> None:
    label = response.get("prediction", "UNKNOWN")
    prob = response.get("probability", 0.0)
    risk = response.get("risk_level", "N/A")
    shap_values: list[dict] = response.get("shap_values", [])

    is_malignant = label == "MALIGNANT"
    card_class = "card-malignant" if is_malignant else "card-benign"
    emoji = "🔴" if is_malignant else "🟢"
    label_pt = "Maligno" if is_malignant else "Benigno"
    prob_pct = f"{prob * 100:.1f}%"

    # ── Badge de risco ────────────────────────────────────────────────────────
    risk_map = {
        "Alto": ("badge-high", "Alto"),
        "Moderado": ("badge-medium", "Moderado"),
        "Baixo": ("badge-low", "Baixo"),
        "High": ("badge-high", "Alto"),
        "Moderate": ("badge-medium", "Moderado"),
        "Low": ("badge-low", "Baixo"),
    }
    badge_class, risk_pt = risk_map.get(risk, ("badge-low", risk))

    # ── Card principal ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="{card_class}">
            <p class="card-title">{emoji} {label_pt}</p>
            <p class="card-label">Probabilidade de malignidade</p>
            <p class="card-prob">{prob_pct}</p>
            <span class="{badge_class}">Risco: {risk_pt}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Gauge de probabilidade ────────────────────────────────────────────────
    st.markdown('<p class="section-title">📈 Probabilidade de malignidade</p>', unsafe_allow_html=True)
    st.progress(prob, text=f"{prob_pct} de probabilidade de tumor maligno")

    # ── SHAP — top features ───────────────────────────────────────────────────
    if shap_values:
        st.markdown('<p class="section-title">🔍 Fatores mais relevantes (SHAP)</p>', unsafe_allow_html=True)
        st.caption(
            "Valores positivos (vermelho) aumentam a probabilidade de malignidade. "
            "Valores negativos (verde) reduzem."
        )

        # Ordenar pelo valor absoluto — top 10
        sorted_shap = sorted(shap_values, key=lambda x: abs(x["value"]), reverse=True)[:10]
        max_abs = max(abs(x["value"]) for x in sorted_shap) or 1.0

        bars_html = '<div class="shap-bar">'
        for item in sorted_shap:
            feat = item["feature"].replace("_", " ")
            val = item["value"]
            bar_len = int(abs(val) / max_abs * 25)
            bar_char = "█" * bar_len
            sign = "+" if val >= 0 else "−"
            color_class = "shap-positive" if val >= 0 else "shap-negative"
            bars_html += (
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">'
                f'<span class="shap-feature" style="width:200px;text-align:right;">{feat}</span>'
                f'<span class="{color_class}">{bar_char}</span>'
                f'<span class="{color_class}" style="font-size:0.78rem;">{sign}{abs(val):.4f}</span>'
                f'</div>'
            )
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ Este resultado é gerado por um modelo de Machine Learning para fins educacionais. "
        "Não substitui diagnóstico médico profissional."
    )


# ─── App principal ────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()

    api_online = check_api_health()
    render_header(api_online)
    render_about_section()

    st.markdown("---")

    if not api_online:
        st.error(
            "⚠️ A API não está acessível em `"
            + API_URL
            + "`. Inicie o servidor com `uvicorn app.main:app --reload` e recarregue esta página."
        )
        return

    payload = render_form()

    st.markdown("---")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        submitted = st.button("🔬 Analisar Tumor", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Analisando amostra..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    render_result(resp.json())
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"Erro da API ({resp.status_code}): {detail}")
            except requests.Timeout:
                st.error("A API demorou mais de 30 s para responder. Verifique se o servidor está rodando.")
            except requests.ConnectionError:
                st.error("Não foi possível conectar à API. Verifique se o servidor está ativo.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Erro inesperado: {exc}")


if __name__ == "__main__":
    main()
