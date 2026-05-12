import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from scipy.stats import norm, chi2

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EcotoxLab",
    layout="wide",
    page_icon="⚗"
)

# ═══════════════════════════════════════════════════════════════
# ESTILO
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>

.stApp {
    background-color: #0a0e14;
    color: white;
}

h1, h2, h3, h4, h5, h6, p, div, label {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: #13191f;
}

.metric-card {
    background: #13191f;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #2d333b;
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FUNÇÕES MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════

def qnorm(p):
    return norm.ppf(p)

def pnorm(x):
    return norm.cdf(x)

def chiSqPval(chi2_val, df):
    if df <= 0:
        return np.nan
    return 1 - chi2.cdf(chi2_val, df)

def abbott(p, ctrl):
    if ctrl >= 1:
        return p
    return max(0, min(1, (p - ctrl) / (1 - ctrl)))

# ═══════════════════════════════════════════════════════════════
# SPEARMAN-KARBER
# ═══════════════════════════════════════════════════════════════

def calc_spearman_karber(doses, deaths, totals):

    pairs = []

    for d, dead, total in zip(doses, deaths, totals):

        if d > 0:

            pairs.append({
                "d": d,
                "p": dead / total,
                "n": total,
                "k": dead
            })

    if len(pairs) < 2:
        return None

    pairs = sorted(pairs, key=lambda x: x["d"])

    ld = [math.log10(p["d"]) for p in pairs]
    pr = [p["p"] for p in pairs]

    area = 0

    for i in range(1, len(ld)):

        area += (
            (ld[i] - ld[i - 1]) *
            (pr[i] + pr[i - 1]) / 2
        )

    logCL = ld[-1] - area

    v = 0

    for i in range(1, len(ld)):

        pi = (pr[i] + pr[i - 1]) / 2
        ni = pairs[i]["n"]

        v += (
            ((ld[i] - ld[i - 1]) ** 2) *
            pi *
            (1 - pi) /
            max(ni - 1, 1)
        )

    se = math.sqrt(v)

    f = 10 ** (1.96 * se)

    chi2_val = 0

    for p in pairs:

        expected = p["n"] * p["p"]

        chi2_val += (
            ((p["k"] - expected) ** 2) /
            max(expected * (1 - p["p"]), 1e-6)
        )

    cl = 10 ** logCL

    return {
        "cl": cl,
        "lcl": cl / f,
        "ucl": cl * f,
        "logCL": logCL,
        "slope": None,
        "intercept": None,
        "chi2": chi2_val,
        "pgof": chiSqPval(
            chi2_val,
            max(1, len(pairs) - 2)
        ),
        "variance": None,
        "zValue": None
    }

# ═══════════════════════════════════════════════════════════════
# GLM LOGIT / PROBIT
# ═══════════════════════════════════════════════════════════════

def calc_glm(doses, deaths, totals, link="probit"):

    pairs = []

    for d, dead, total in zip(doses, deaths, totals):

        x = math.log10(max(d, 1e-10))

        pairs.append({
            "x": x,
            "p": dead / total,
            "n": total,
            "k": dead
        })

    if len(pairs) < 2:
        return None

    pts = [p for p in pairs if 0 < p["p"] < 1]

    if len(pts) < 2:
        return None

    xs = [p["x"] for p in pts]

    if link == "probit":
        ys = [qnorm(p["p"]) for p in pts]
    else:
        ys = [
            math.log(p["p"] / (1 - p["p"]))
            for p in pts
        ]

    b1, b0 = np.polyfit(xs, ys, 1)

    logCL = -b0 / b1

    cl = 10 ** logCL

    residuals = []

    for p in pairs:

        eta = b0 + b1 * p["x"]

        if link == "probit":
            mu = pnorm(eta)
        else:
            mu = 1 / (1 + math.exp(-eta))

        residuals.append((p["p"] - mu) ** 2)

    variance = np.mean(residuals)

    se = math.sqrt(max(variance, 1e-10))

    f = 10 ** (1.96 * se)

    chi2_val = 0

    for p in pairs:

        eta = b0 + b1 * p["x"]

        if link == "probit":
            mu = pnorm(eta)
        else:
            mu = 1 / (1 + math.exp(-eta))

        chi2_val += (
            ((p["k"] - p["n"] * mu) ** 2) /
            max(p["n"] * mu * (1 - mu), 1e-6)
        )

    df = max(1, len(pairs) - 2)

    return {
        "cl": cl,
        "lcl": cl / f,
        "ucl": cl * f,
        "logCL": logCL,
        "slope": b1,
        "intercept": b0,
        "chi2": chi2_val,
        "pgof": chiSqPval(chi2_val, df),
        "variance": variance,
        "zValue": b1 / se,
        "_b0": b0,
        "_b1": b1,
        "_link": link
    }

# ═══════════════════════════════════════════════════════════════
# PREVISÃO DA CURVA
# ═══════════════════════════════════════════════════════════════

def predict_y(x, res, method="probit"):

    logX = math.log10(max(x, 1e-10))

    if "_b0" in res:

        eta = res["_b0"] + res["_b1"] * logX

        if method == "probit":
            return pnorm(eta) * 100

        return 100 / (1 + math.exp(-eta))

    return 100 / (1 + (x / res["cl"]) ** -4.5)

# ═══════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════

UNITS_CONC = [
    "µg/L",
    "µg/g",
    "µg/mg",
    "µg/kg",
    "mg/L",
    "mg/g",
    "mg/kg",
    "ng/L",
    "ng/g"
]

METHODS = {
    "LC Probit": "probit",
    "LC Logit": "logit",
    "Spearman-Karber": "spearman"
}

# ═══════════════════════════════════════════════════════════════
# TÍTULO
# ═══════════════════════════════════════════════════════════════

st.title("⚗ EcotoxLab")
st.subheader("Análise de Toxicidade Aquática")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

st.sidebar.title("Configurações")

substance = st.sidebar.text_input(
    "Substância / Espécie",
    value="Substância X"
)

reps = st.sidebar.number_input(
    "Repetições",
    min_value=1,
    value=3
)

indiv = st.sidebar.number_input(
    "Indivíduos",
    min_value=1,
    value=10
)

unit_conc = st.sidebar.selectbox(
    "Unidade",
    UNITS_CONC
)

method_name = st.sidebar.selectbox(
    "Método",
    list(METHODS.keys())
)

n_doses = st.sidebar.slider(
    "Número de doses",
    min_value=2,
    max_value=20,
    value=5
)

method = METHODS[method_name]

total = reps * indiv

st.sidebar.info(
    f"Total de organismos por grupo: {total}"
)

# ═══════════════════════════════════════════════════════════════
# ENTRADA DE DADOS
# ═══════════════════════════════════════════════════════════════

st.subheader("Dados Experimentais")

rows = []

header = st.columns(3)

header[0].markdown("### Dose")
header[1].markdown("### Mortos")
header[2].markdown("### Mortalidade")

for i in range(n_doses):

    c1, c2, c3 = st.columns(3)

    dose = c1.number_input(
        f"Dose {i+1}",
        min_value=0.0,
        value=0.0,
        key=f"dose_{i}"
    )

    dead = c2.number_input(
        f"Mortos {i+1}",
        min_value=0,
        max_value=total,
        value=0,
        key=f"dead_{i}"
    )

    pct = (dead / total) * 100

    c3.metric(
        "Mortalidade",
        f"{pct:.1f}%"
    )

    rows.append({
        "dose": dose,
        "dead": dead
    })

# ═══════════════════════════════════════════════════════════════
# BOTÃO DE CÁLCULO
# ═══════════════════════════════════════════════════════════════

if st.button("▶ Calcular CL50"):

    try:

        doses = [r["dose"] for r in rows]
        deaths = [r["dead"] for r in rows]
        totals = [total] * len(rows)

        if all(d == 0 for d in doses):

            st.error(
                "Adicione pelo menos uma dose maior que zero."
            )

            st.stop()

        ctrl_idx = doses.index(0) if 0 in doses else None

        ctrl_p = (
            deaths[ctrl_idx] / total
            if ctrl_idx is not None
            else 0
        )

        corr_p = [
            abbott(dead / total, ctrl_p)
            for dead in deaths
        ]

        corr_d = [
            round(p * total)
            for p in corr_p
        ]

        axs = [
            d for d in doses if d > 0
        ]

        ade = [
            corr_d[i]
            for i, d in enumerate(doses)
            if d > 0
        ]

        ato = [
            total
            for d in doses
            if d > 0
        ]

        if method == "spearman":

            res = calc_spearman_karber(
                axs,
                ade,
                ato
            )

        else:

            res = calc_glm(
                axs,
                ade,
                ato,
                method
            )

        if res is None:

            st.error(
                "Não foi possível calcular. "
                "Verifique os dados."
            )

            st.stop()

        # ═══════════════════════════════════════════════════════
        # RESULTADOS
        # ═══════════════════════════════════════════════════════

        st.success("Cálculo realizado com sucesso.")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "CL50",
            f"{res['cl']:.4f} {unit_conc}"
        )

        c2.metric(
            "LI 95%",
            f"{res['lcl']:.4f}"
        )

        c3.metric(
            "LS 95%",
            f"{res['ucl']:.4f}"
        )

        if res["slope"] is not None:

            c4, c5, c6, c7 = st.columns(4)

            c4.metric(
                "Slope",
                f"{res['slope']:.4f}"
            )

            c5.metric(
                "Intercept",
                f"{res['intercept']:.4f}"
            )

            c6.metric(
                "z-value",
                f"{res['zValue']:.4f}"
            )

            c7.metric(
                "Variância",
                f"{res['variance']:.6f}"
            )

        c8, c9 = st.columns(2)

        c8.metric(
            "χ² Pearson",
            f"{res['chi2']:.4f}"
        )

        c9.metric(
            "p-valor GOF",
            f"{res['pgof']:.4f}"
        )

        # ═══════════════════════════════════════════════════════
        # CURVA
        # ═══════════════════════════════════════════════════════

        minX = min(axs)
        maxX = max(axs)

        xvals = np.linspace(
            minX * 0.75,
            maxX * 1.25,
            200
        )

        yvals = [
            predict_y(x, res, method)
            for x in xvals
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=xvals,
            y=yvals,
            mode="lines",
            name="Curva"
        ))

        obs_y = [
            (a / total) * 100
            for a in ade
        ]

        fig.add_trace(go.Scatter(
            x=axs,
            y=obs_y,
            mode="markers",
            name="Observado"
        ))

        fig.add_hline(y=50)

        fig.add_vline(x=res["cl"])

        fig.update_layout(
            title=f"{substance} — Curva Dose-Resposta",
            xaxis_title=f"Concentração ({unit_conc})",
            yaxis_title="Mortalidade (%)",
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ═══════════════════════════════════════════════════════
        # TABELA
        # ═══════════════════════════════════════════════════════

        df = pd.DataFrame({
            "Dose": doses,
            "Mortos": deaths,
            "Mortalidade (%)": [
                (d / total) * 100
                for d in deaths
            ]
        })

        st.subheader("Tabela de Dados")

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Baixar CSV",
            csv,
            file_name="ecotox_resultados.csv",
            mime="text/csv"
        )

        st.info(
            "Métodos baseados em modelos clássicos "
            "de toxicologia aquática."
        )

    except Exception as e:

        st.error(f"Erro: {e}")
