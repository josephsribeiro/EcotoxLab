# =============================================================================
# EcotoxLab — Análise de Toxicidade Aquática
# Executar: streamlit run ecotox_app.py
# Dependências: pip install -r requirements.txt
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import ndtr, ndtri
from io import BytesIO
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcotoxLab",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp{background:#0a0e14;color:##ffffff}
  section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #2d333b}
  .stTextInput input,.stNumberInput input,.stSelectbox select{
    background:#13191f!important;border:1px solid #2d333b!important;
    color:##ffffff!important;border-radius:6px!important}
  .stButton>button{background:linear-gradient(135deg,#1a7f64,#2da677);
    color:#fff;border:none;border-radius:8px;font-weight:600;width:100%}
  .stButton>button:hover{filter:brightness(1.1)}
  div[data-testid="stMetric"]{background:#13191f;border:1px solid #2d333b;
    border-radius:10px;padding:14px 16px}
  div[data-testid="stMetricValue"]{color:#ffffff!important;font-size:1.3rem!important}
  div[data-testid="stMetricLabel"]{color:#ffffff!important;font-size:.72rem!important}
  .stTabs [data-baseweb="tab-list"]{background:#0d1117;border-bottom:1px solid #2d333b}
  .stTabs [data-baseweb="tab"]{color:#ffffff}
  .stTabs [aria-selected="true"]{color:#2da677!important;
    border-bottom:2px solid #2da677!important}
  h1,h2,h3,h4{color:##ffffff!important}
  hr{border-color:#2d333b}
  .info-box{background:#0a0e14;border:1px solid #2d333b;border-radius:8px;
    padding:12px 16px;font-size:12px;color:#;line-height:1.65}
  .warn-box{background:#2d1217;border:1px solid #e5534b60;border-radius:8px;
    padding:10px 14px;font-size:12px;color:#e5534b}
  .ok-box{background:#0d2b1f;border:1px solid #2da67760;border-radius:8px;
    padding:10px 14px;font-size:12px;color:#2da677}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA
# ─────────────────────────────────────────────────────────────────────────────

def abbott(p: float, ctrl: float) -> float:
    if ctrl >= 1.0:
        return float(p)
    return float(np.clip((p - ctrl) / (1.0 - ctrl), 0.0, 1.0))


def chi_sq_pval(chi2: float, df: int) -> float:
    if df <= 0 or np.isnan(chi2):
        return float("nan")
    return float(1.0 - stats.chi2.cdf(chi2, df))


def calc_spearman_karber(doses, deaths, totals):
    """Spearman-Kärber aparado — Wheeler et al. (2006), variância Thompson (1947)."""
    rows = sorted(
        [(d, deaths[i] / totals[i], totals[i], int(deaths[i]))
         for i, d in enumerate(doses) if d > 0],
        key=lambda r: r[0],
    )
    if len(rows) < 2:
        return None

    ld = [np.log10(r[0]) for r in rows]
    pr = [r[1] for r in rows]
    ns = [r[2] for r in rows]
    ks = [r[3] for r in rows]

    area = sum((ld[i] - ld[i-1]) * (pr[i] + pr[i-1]) / 2 for i in range(1, len(ld)))
    log_cl = ld[-1] - area

    var = sum(
        (ld[i] - ld[i-1])**2 * (pr[i] + pr[i-1]) / 2 * (1 - (pr[i] + pr[i-1]) / 2)
        / max(ns[i] - 1, 1)
        for i in range(1, len(ld))
    )
    factor = 10 ** (1.96 * np.sqrt(var))

    chi2 = sum(
        (ks[i] - ns[i] * pr[i])**2 / max(ns[i] * pr[i] * (1 - pr[i]), 1e-9)
        for i in range(len(rows))
    )
    df = max(1, len(rows) - 2)
    cl = 10 ** log_cl

    return dict(
        cl=cl, lcl=cl / factor, ucl=cl * factor, log_cl=log_cl,
        slope=None, intercept=None, z_value=None, variance=None,
        chi2=round(chi2, 4), pgof=round(chi_sq_pval(chi2, df), 4),
    )


def calc_glm(doses, deaths, totals, link: str = "probit"):
    """
    GLM binomial via IRLS.
    Equivalente a LC probit / LC logit / LT probit / LT logit
    do pacote R {ecotox} (Hlina et al. 2021) e modificado por Joseph S. Ribeiro.
    Intervalos de confiança: delta method (Finney 1971).
    """
    data = [
        (np.log10(max(d, 1e-10)), deaths[i] / totals[i], totals[i], int(deaths[i]))
        for i, d in enumerate(doses) if d > 0
    ]
    data = [(x, p, n, k) for x, p, n, k in data if np.isfinite(x)]
    if len(data) < 2:
        return None

    if link == "probit":
        ginv    = lambda eta: float(np.clip(ndtr(eta), 1e-9, 1 - 1e-9))
        gprime  = lambda mu:  float(np.exp(-0.5 * ndtri(np.clip(mu, 1e-9, 1-1e-9))**2) / np.sqrt(2 * np.pi))
        link_fn = lambda p:   float(ndtri(np.clip(p, 1e-6, 1-1e-6)))
    else:
        ginv    = lambda eta: float(np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1-1e-9))
        gprime  = lambda mu:  float(mu * (1 - mu))
        link_fn = lambda p:   float(np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6)))

    # Semente OLS
    pts = [(x, p, n, k) for x, p, n, k in data if 0 < p < 1]
    if len(pts) < 2:
        return None

    xs = [x for x, *_ in pts]
    ys = [link_fn(p) for _, p, *_ in pts]
    mx, my = float(np.mean(xs)), float(np.mean(ys))
    ssxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    ssxx = sum((v - mx) ** 2 for v in xs)
    b1 = ssxy / ssxx if ssxx > 1e-12 else 1.0
    b0 = my - b1 * mx

    # IRLS — 50 iterações
    for _ in range(50):
        sW = sWX = sXWX = sWZ = sXWZ = 0.0
        for x, p, n, k in data:
            eta = b0 + b1 * x
            mu  = ginv(eta)
            gp  = max(gprime(mu), 1e-12)
            w   = n * gp**2 / max(mu * (1 - mu), 1e-12)
            z   = eta + (p - mu) / gp
            sW   += w;       sWX  += w * x
            sXWX += w * x*x; sWZ  += w * z
            sXWZ += w * x * z
        det = sW * sXWX - sWX**2
        if abs(det) < 1e-14:
            break
        nb0 = (sXWX * sWZ - sWX * sXWZ) / det
        nb1 = (sW   * sXWZ - sWX * sWZ)  / det
        if abs(nb0 - b0) + abs(nb1 - b1) < 1e-10:
            b0, b1 = nb0, nb1
            break
        b0, b1 = nb0, nb1

    if abs(b1) < 1e-12:
        return None

    log_cl = -b0 / b1
    cl = 10 ** log_cl

    # Matriz de informação de Fisher → variância de log(CL50)
    sW = sWX = sXWX = 0.0
    for x, p, n, k in data:
        eta = b0 + b1 * x
        mu  = ginv(eta)
        gp  = max(gprime(mu), 1e-12)
        w   = n * gp**2 / max(mu * (1 - mu), 1e-12)
        sW += w; sWX += w * x; sXWX += w * x * x
    det = sW * sXWX - sWX**2
    if abs(det) < 1e-14:
        return None

    var_b0 = max(0.0, sXWX / det)
    var_b1 = max(0.0, sW   / det)
    cov_b  = -sWX / det
    var_log = (var_b0 + 2 * log_cl * cov_b + log_cl**2 * var_b1) / b1**2
    factor   = 10 ** (1.96 * np.sqrt(max(0.0, var_log)))

    se_b1  = np.sqrt(var_b1)
    z_val  = round(b1 / se_b1, 4) if se_b1 > 1e-12 else None

    # χ² de Pearson (GOF)
    chi2 = 0.0
    for x, p, n, k in data:
        mu_hat = ginv(b0 + b1 * x)
        chi2  += (k - n * mu_hat)**2 / max(n * mu_hat * (1 - mu_hat), 1e-9)
    df = max(1, len(data) - 2)

    return dict(
        cl=cl, lcl=cl / factor, ucl=cl * factor, log_cl=log_cl,
        slope=round(b1, 4), intercept=round(b0, 4),
        z_value=z_val, variance=round(var_log, 6),
        chi2=round(chi2, 4), pgof=round(chi_sq_pval(chi2, df), 4),
        _b0=b0, _b1=b1, _link=link,
    )


def predict_y(x_arr, res, link: str = "logit"):
    """Curva ajustada a partir dos parâmetros GLM."""
    out = []
    b0 = res.get("_b0")
    b1 = res.get("_b1")
    for x in x_arr:
        lx  = np.log10(max(x, 1e-10))
        if b0 is not None and b1 is not None:
            eta = b0 + b1 * lx
            v = float(ndtr(eta)) if link == "probit" else 1 / (1 + np.exp(-eta))
        else:
            v = 1 / (1 + (x / max(res["cl"], 1e-10)) ** (-4.5))
        out.append(np.clip(v * 100, 0, 100))
    return np.array(out)


def predict_from_cl(x_arr, cl: float, link: str = "logit"):
    """Curva com CL deslocado (bandas de IC)."""
    b1e = 4.5
    b0e = -b1e * np.log10(max(cl, 1e-10))
    out = []
    for x in x_arr:
        lx  = np.log10(max(x, 1e-10))
        eta = b0e + b1e * lx
        v = float(ndtr(eta)) if link == "probit" else 1 / (1 + np.exp(-eta))
        out.append(np.clip(v * 100, 0, 100))
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
UNITS_CONC = ["µg/L", "µg/g", "µg/mg", "µg/kg", "mg/L", "mg/g", "mg/kg", "ng/L", "ng/g"]
UNITS_TIME = ["h", "min", "dias", "semanas"]

METHODS = {
    "lc_probit": dict(label="LC probit", group="LC", link="probit", is_lt=False, cl_label="CL50"),
    "lc_logit":  dict(label="LC logit",  group="LC", link="logit",  is_lt=False, cl_label="CL50"),
    "lt_probit": dict(label="LT probit", group="LT", link="probit", is_lt=True,  cl_label="TL50"),
    "lt_logit":  dict(label="LT logit",  group="LT", link="logit",  is_lt=True,  cl_label="TL50"),
    "spearman":  dict(label="Spearman-Kärber", group="NP", link=None, is_lt=False, cl_label="CL50"),
}

METHOD_NOTES = {
    "lc_probit": "LC probit: GLM binomial com ligação probit (Finney 1971). Equivalente ao LC probit do pacote R {ecotox} (Hlina et al. 2021) e modificado por Joseph S. Ribeiro",
    "lc_logit":  "LC logit: GLM binomial com ligação logit. Mais robusto nos extremos da curva. Equivalente ao LC logit do {ecotox} e modificado por Joseph S. Ribeiro",
    "lt_probit": "LT probit: Mesmo algoritmo do LC probit com tempo de exposição como variável independente. Equivalente ao LT_probit do {ecotox e modificado por Joseph S. Ribeiro.",
    "lt_logit":  "LT logit: GLM logit com tempo de exposição. Equivalente ao LT_logit do {ecotox} e modificado por Joseph S. Ribeiro.",
    "spearman":  "Spearman-Kärber aparado (Wheeler et al. 2006). Variância por Thompson (1947). Correção de Abbott aplicada à mortalidade do controle e modificado por Joseph S. Ribeiro.",
}


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(res, obs_x, obs_y, x_label, cl_label, substance, method_id, link):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0a0e14")
    ax.set_facecolor("#0a0e14")

    x_min = max(min(obs_x) * 0.70, 1e-5)
    x_max = max(obs_x) * 1.35
    xs    = np.linspace(x_min, x_max, 300)

    y_curve = predict_y(xs, res, link or "logit")
    y_lower = predict_from_cl(xs, res["lcl"], link or "logit")
    y_upper = predict_from_cl(xs, res["ucl"], link or "logit")

    # Banda IC preenchida
    ax.fill_between(xs, y_lower, y_upper, color="#2da677", alpha=0.10)

    # Linhas
    ax.plot(xs, y_upper, color="#E74C3C", lw=1.3, ls="--", label="IC 95% (Sup/Inf)")
    ax.plot(xs, y_lower, color="#2da677", lw=1.3, ls="--")
    ax.plot(xs, y_curve, color="#ffffff", lw=2.5,           label=f"Curva {cl_label}")

    # Pontos observados
    ax.scatter(obs_x, obs_y, color="#e3b341", s=65, zorder=6,
               label="Observado", edgecolors="#0a0e14", lw=0.6)

    # Referências
    ax.axhline(50, color="#3a4048", lw=0.8, ls=":")
    ax.axvline(res["cl"],  color="#ffffff", lw=1.2, ls="--", alpha=0.85)
    ax.axvline(res["lcl"], color="#2da677", lw=0.8, ls=":",  alpha=0.65)
    ax.axvline(res["ucl"], color="#2da677", lw=0.8, ls=":",  alpha=0.65)

    # Anotações eixo X
    yb = ax.get_ylim()[0]
    for val, lbl, col in [(res["cl"], cl_label, "#ffffff"),
                          (res["lcl"], "LI", "#2da677"),
                          (res["ucl"], "LS", "#2da677")]:
        ax.annotate(
            f"{lbl}\n{val:.4f}",
            xy=(val, 50), xytext=(val, 53),
            color=col, fontsize=7.5, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.6),
        )

    # Estética
    ax.set_xlabel(x_label, color="#", fontsize=10)
    ax.set_ylabel("Mortalidade (%)", color="#ffffff", fontsize=10)
    ax.set_title(
        f"{substance}  —  {METHODS[method_id]['label']}",
        color="##ffffff", fontsize=11, pad=10,
    )
    ax.set_ylim(-5, 112)
    ax.set_yticks(range(0, 110, 10))
    ax.tick_params(colors="#ffffff", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#2d333b")
    ax.grid(True, color="#2d333b", lw=0.4, ls="--", alpha=0.5)

    # Legenda personalizada
    handles = [
        Line2D([0], [0], color="#ffffff", lw=2.5,         label=f"Curva {cl_label}"),
        Line2D([0], [0], color="#2da677", lw=1.3, ls="--", label="IC 95%"),
        Line2D([0], [0], marker="o", color="#e3b341", ls="none",
               markersize=6, label="Observado"),
    ]
    legend = ax.legend(handles=handles, facecolor="#13191f", edgecolor="#2d333b",
                       labelcolor="##ffffff", fontsize=9, loc="upper left")

    fig.tight_layout(pad=1.5)
    return fig


def fig_to_bytes(fig, fmt: str, dpi: int = 150) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi,
                bbox_inches="tight", facecolor="#0a0e14")
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚗ EcotoxLab")
    st.markdown(
    '<p style="color:#ffffff;">Análise de Toxicidade Aquática</p>',
    unsafe_allow_html=True
)
    st.divider()

    st.markdown("### 🔬 Experimento")
    substance = st.text_input("Substância / Espécie", value="Substância X", key="substance")

    col_r, col_i = st.columns(2)
    reps  = col_r.number_input("Repetições",  min_value=1, value=3,  step=1, key="reps")
    indiv = col_i.number_input("Indivíduos",  min_value=1, value=10, step=1, key="indiv")
    total = int(reps * indiv)
   st.markdown(
    f'<p style="color:#ffffff;">Total por grupo: <b>{total}</b> organismos</p>',
    unsafe_allow_html=True
)

    col_uc, col_ut = st.columns(2)
    unit_conc = col_uc.selectbox("Unid. Conc.", UNITS_CONC, key="unit_conc")
    unit_time = col_ut.selectbox("Unid. Tempo", UNITS_TIME, key="unit_time")
    st.divider()

    # ── Seleção de método ──
    st.markdown("### 📐 Método de Análise")
    st.caption("Baseado no pacote R **{ecotox}** — Hlina et al. (2021) e modificado por Joseph S. Ribeiro")

    st.markdown("**🔵 Concentração Letal (LC)**")
    lc_choice = st.radio(
        "lc_radio",
        options=["lc_probit", "lc_logit"],
        format_func=lambda k: METHODS[k]["label"],
        label_visibility="collapsed",
        key="lc_radio",
    )

    st.markdown("**🟡 Tempo Letal (LT)**")
    lt_choice = st.radio(
        "lt_radio",
        options=["lt_probit", "lt_logit"],
        format_func=lambda k: METHODS[k]["label"],
        label_visibility="collapsed",
        key="lt_radio",
    )

    st.markdown("**🟢 Não-paramétrico**")
    use_sk = st.checkbox("Spearman-Kärber()", key="use_sk")

    st.divider()
    analysis_mode = st.radio(
        "Modo de análise",
        options=["LC — Concentração Letal", "LT — Tempo Letal"],
        key="analysis_mode",
    )

    # Método final
    if use_sk:
        method_id = "spearman"
    elif analysis_mode.startswith("LT"):
        method_id = lt_choice
    else:
        method_id = lc_choice

    m       = METHODS[method_id]
    link    = m["link"]
    is_lt   = m["is_lt"]
    cl_label = m["cl_label"]
    unit    = unit_time if is_lt else unit_conc
    x_label = f"Tempo ({unit})" if is_lt else f"Concentração ({unit})"

    st.markdown(f"**Método ativo:** `{m['label']}`")
    st.markdown(
        f'<div class="info-box">{METHOD_NOTES[method_id]}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CABEÇALHO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"## ⚗ EcotoxLab — {m['label']}")
st.markdown(
    """
    <p style="color:#ffffff;font-size:14px;">
    Análise de Toxicidade Aquática | Baseado no pacote R {ecotox} (Hlina et al. 2021) e modificado por Joseph S. Ribeiro
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

tab_data, tab_result = st.tabs(["📋  Dados", "📊  Resultados"])


# ─────────────────────────────────────────────────────────────────────────────
# ABA DADOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:

    # ══════════════════════════════════════════
    # MODO LT — entrada por intervalo de tempo
    # ══════════════════════════════════════════
    if is_lt:
        st.markdown("#### Tempo Letal — Parâmetros do Ensaio")

        col_a, col_b, col_c = st.columns(3)
        test_duration = col_a.number_input(
            f"Duração total ({unit})",
            min_value=1, value=96, step=1, key="test_duration",
            help="Tempo total do bioensaio. Ex: 96 h (padrão EPA peixes)",
        )
        lt_interval = col_b.number_input(
            f"Intervalo de leitura ({unit})",
            min_value=1, value=24, step=1, key="lt_interval",
            help="A cada quanto tempo você conta as mortes. Ex: 24 → leituras a 24 h, 48 h, 72 h…",
        )
        lt_conc = col_c.number_input(
            f"Concentração fixa ({unit_conc})",
            min_value=0.0, value=0.0, format="%.4f", key="lt_conc",
            help="Registro da concentração usada no ensaio. Não afeta o cálculo do TL50.",
        )

        n_intervals = max(1, int(test_duration // lt_interval))
        st.caption(
            f"→ **{n_intervals}** leituras: "
            + ", ".join(str(int((i+1) * lt_interval)) for i in range(n_intervals))
            + f" {unit}"
        )
        st.divider()

        st.markdown("#### Mortes por Intervalo de Tempo")
        st.caption(
            "Digite as mortes **em cada intervalo** (ex: quantos morreram *entre* 48 h e 72 h). "
            "A mortalidade acumulada é calculada automaticamente."
        )

        # Cabeçalho
        hc = st.columns([0.5, 1.8, 2.2, 1.3, 1.3])
        for txt, col in zip(["#", f"Tempo ({unit})", f"Mortos no intervalo", "Acumulado", "Mort. %"], hc):
            col.markdown(
                f'<p style="font-size:10px;color:#ffffff;font-weight:600;margin:0">{txt}</p>',
                unsafe_allow_html=True,
            )

        lt_rows = []
        cum_dead = 0
        for i in range(n_intervals):
            rc = st.columns([0.5, 1.8, 2.2, 1.3, 1.3])
            rc[0].markdown(
                f'<p style="color:#e3b341;font-weight:600;padding-top:6px">{i+1}</p>',
                unsafe_allow_html=True,
            )
            t_val = rc[1].number_input(
                f"lt_t_{i}", label_visibility="collapsed",
                value=float((i + 1) * lt_interval), min_value=0.0,
                format="%.1f", key=f"lt_t_{i}",
            )
            d_val = rc[2].number_input(
                f"lt_d_{i}", label_visibility="collapsed",
                value=0, min_value=0, max_value=total,
                step=1, key=f"lt_d_{i}",
            )
            cum_dead += int(d_val)
            pct = cum_dead / total * 100 if total > 0 else 0.0
            col_pct = "#e5534b" if pct > 50 else "#ffffff" if pct > 25 else "#2da677"
            rc[3].markdown(
                f'<p style="padding-top:6px;color:#ffffff">{cum_dead}</p>',
                unsafe_allow_html=True,
            )
            rc[4].markdown(
                f'<p style="padding-top:6px;color:{col_pct};font-weight:600">{pct:.1f}%</p>',
                unsafe_allow_html=True,
            )
            lt_rows.append({"tempo": t_val, "mortos_intervalo": int(d_val), "acumulado": cum_dead, "pct": round(pct, 1)})

        st.divider()
        if st.button(f"▶ Calcular {cl_label} — {m['label']}", key="btn_lt"):
            erros = []
            if n_intervals < 2:
                erros.append("Mínimo 2 intervalos de tempo.")
            if cum_dead > total:
                erros.append(f"Mortos acumulados ({cum_dead}) excedem o total ({total}).")
            if cum_dead == 0:
                erros.append("Nenhuma morte registrada.")

            if erros:
                for e in erros:
                    st.markdown(f'<div class="warn-box">⚠ {e}</div>', unsafe_allow_html=True)
            else:
                axs = [r["tempo"]     for r in lt_rows]
                ade = [r["acumulado"] for r in lt_rows]
                ato = [total] * len(axs)
                res = calc_glm(axs, ade, ato, link)
                if res is None:
                    st.markdown(
                        '<div class="warn-box">⚠ Cálculo não convergiu. '
                        'Verifique se a mortalidade varia entre os tempos.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state["result"]    = res
                    st.session_state["obs_x"]     = axs
                    st.session_state["obs_y"]     = [d / total * 100 for d in ade]
                    st.session_state["method_id"] = method_id
                    st.session_state["unit"]      = unit
                    st.session_state["x_label"]   = x_label
                    st.markdown(
                        f'<div class="ok-box">✓ {cl_label} calculado: '
                        f'<b>{res["cl"]:.4f} {unit}</b> — abra a aba Resultados.</div>',
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════════
    # MODO LC / SPEARMAN — entrada por concentração
    # ══════════════════════════════════════════
    else:
        st.markdown("#### Concentração × Mortalidade")

        n_doses = st.number_input(
            "Número de concentrações (incluindo controle = 0)",
            min_value=2, max_value=20, value=5, step=1, key="n_doses",
        )

        # Cabeçalho
        hc = st.columns([0.5, 2, 2, 1.5])
        for txt, col in zip(["#", f"Concentração ({unit_conc})", "Mortos", "Mort. %"], hc):
            col.markdown(
                f'<p style="font-size:10px;color:#ffffff;font-weight:600;margin:0">{txt}</p>',
                unsafe_allow_html=True,
            )

        lc_rows = []
        for i in range(int(n_doses)):
            rc = st.columns([0.5, 2, 2, 1.5])
            rc[0].markdown(
                f'<p style="color:#ffffff;font-weight:600;padding-top:6px">{i+1}</p>',
                unsafe_allow_html=True,
            )
            dose_val = rc[1].number_input(
                f"lc_dose_{i}", label_visibility="collapsed",
                value=0.0, min_value=0.0, format="%.4f", key=f"lc_dose_{i}",
            )
            dead_val = rc[2].number_input(
                f"lc_dead_{i}", label_visibility="collapsed",
                value=0, min_value=0, max_value=total, step=1, key=f"lc_dead_{i}",
            )
            pct = dead_val / total * 100 if total > 0 else 0.0
            col_pct = "#e5534b" if pct > 50 else "#ffffff" if pct > 25 else "#2da677"
            rc[3].markdown(
                f'<p style="padding-top:6px;color:{col_pct};font-weight:600">{pct:.1f}%</p>',
                unsafe_allow_html=True,
            )
            lc_rows.append({"dose": dose_val, "dead": int(dead_val)})

        st.divider()
        if st.button(f"▶ Calcular {cl_label} — {m['label']}", key="btn_lc"):
            doses_all  = [r["dose"] for r in lc_rows]
            deaths_all = [r["dead"] for r in lc_rows]
            probs_all  = [d / total for d in deaths_all]

            # Correção de Abbott
            ctrl_idx = next((i for i, d in enumerate(doses_all) if d == 0.0), -1)
            ctrl_p   = probs_all[ctrl_idx] if ctrl_idx >= 0 else 0.0
            corr_p   = [abbott(p, ctrl_p) for p in probs_all]
            corr_d   = [round(p * total) for p in corr_p]

            # Filtra dose zero
            axs = [doses_all[i] for i in range(len(doses_all)) if doses_all[i] > 0]
            ade = [corr_d[i]    for i in range(len(doses_all)) if doses_all[i] > 0]
            ato = [total] * len(axs)

            erros = []
            if len(axs) < 2:
                erros.append("Mínimo 2 doses não-zero.")
            if all(d == 0 for d in ade):
                erros.append("Nenhuma morte registrada nas doses.")
            if all(d == total for d in ade):
                erros.append("Mortalidade 100% em todas as doses — sem variação para ajuste.")

            if erros:
                for e in erros:
                    st.markdown(f'<div class="warn-box">⚠ {e}</div>', unsafe_allow_html=True)
            else:
                if method_id == "spearman":
                    res = calc_spearman_karber(axs, ade, ato)
                else:
                    res = calc_glm(axs, ade, ato, link)

                if res is None:
                    st.markdown(
                        '<div class="warn-box">⚠ Cálculo não convergiu. '
                        'Verifique se há variação na mortalidade entre as doses.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state["result"]    = res
                    st.session_state["obs_x"]     = axs
                    st.session_state["obs_y"]     = [d / total * 100 for d in ade]
                    st.session_state["method_id"] = method_id
                    st.session_state["unit"]      = unit
                    st.session_state["x_label"]   = x_label
                    st.markdown(
                        f'<div class="ok-box">✓ {cl_label} calculado: '
                        f'<b>{res["cl"]:.4f} {unit}</b> — abra a aba Resultados.</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# ABA RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_result:
    if "result" not in st.session_state:
        st.info("Insira os dados na aba **Dados** e clique em **Calcular** para ver os resultados.")
    else:
        res      = st.session_state["result"]
        obs_x    = st.session_state["obs_x"]
        obs_y    = st.session_state["obs_y"]
        mid      = st.session_state["method_id"]
        subst    = st.session_state.get("substance", substance)
        unit_r   = st.session_state.get("unit", unit)
        x_lbl    = st.session_state.get("x_label", x_label)
        m_r      = METHODS[mid]
        link_r   = m_r["link"]
        cl_lbl   = m_r["cl_label"]

        # ── Cards principais ──
        c1, c2, c3 = st.columns(3)
        c1.metric(label=cl_lbl,               value=f"{res['cl']:.4f}",  delta=unit_r)
        c2.metric(label="Limite Inferior 95%", value=f"{res['lcl']:.4f}", delta=unit_r)
        c3.metric(label="Limite Superior 95%", value=f"{res['ucl']:.4f}", delta=unit_r)

        # ── Cards GLM ──
        if res.get("slope") is not None:
            st.divider()
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Slope (b₁)",      f"{res['slope']}")
            d2.metric("Intercept (b₀)",  f"{res['intercept']}")
            d3.metric("z-value",         str(res["z_value"]) if res["z_value"] else "—")
            d4.metric("Var(log CL)",     f"{res['variance']:.6f}" if res.get("variance") is not None else "—")

        st.divider()
        e1, e2 = st.columns(2)
        e1.metric("χ² Pearson (GOF)", str(res["chi2"]))
        pgof = res.get("pgof")
        pgof_ok  = isinstance(pgof, float) and not np.isnan(pgof) and pgof > 0.05
        pgof_str = f"{pgof:.4f}" if isinstance(pgof, float) and not np.isnan(pgof) else "—"
        e2.metric("p-valor GOF", pgof_str,
                  delta="✓ bom ajuste" if pgof_ok else "⚠ ajuste ruim")

        # ── Gráfico ──
        st.divider()
        fig = make_chart(res, obs_x, obs_y, x_lbl, cl_lbl, subst, mid, link_r)
        st.pyplot(fig, use_container_width=True)

        # ── Botões de download ──
        st.divider()
        st.markdown("#### 📥 Exportar")
        dl1, dl2, dl3 = st.columns(3)

        jpg_bytes = fig_to_bytes(fig, "jpg", dpi=300)
        dl1.download_button(
            label="⬇ JPG (300 DPI)",
            data=jpg_bytes,
            file_name=f"{subst}_{cl_lbl}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

        svg_bytes = fig_to_bytes(fig, "svg")
        dl2.download_button(
            label="⬇ SVG (vetor)",
            data=svg_bytes,
            file_name=f"{subst}_{cl_lbl}.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )

        result_rows = [
            (cl_lbl,             f"{res['cl']:.4f}",        unit_r),
            ("Lim. Inferior 95%",f"{res['lcl']:.4f}",       unit_r),
            ("Lim. Superior 95%",f"{res['ucl']:.4f}",       unit_r),
            ("log10(CL)",        f"{res['log_cl']:.4f}",    ""),
            ("Slope (b₁)",       str(res.get("slope", "—")), ""),
            ("Intercept (b₀)",   str(res.get("intercept","—")), ""),
            ("z-value",          str(res.get("z_value", "—")), ""),
            ("Var(log CL)",      str(res.get("variance","—")), ""),
            ("χ² Pearson",       str(res.get("chi2","—")),   ""),
            ("p GOF",            pgof_str,                   ""),
            ("Método",           m_r["label"],               ""),
            ("Substância",       subst,                      ""),
        ]
        csv_bytes = pd.DataFrame(result_rows, columns=["Parâmetro","Valor","Unidade"]) \
                      .to_csv(index=False).encode("utf-8")
        dl3.download_button(
            label="⬇ CSV (resultados)",
            data=csv_bytes,
            file_name=f"{subst}_{cl_lbl}_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Nota metodológica ──
        st.divider()
        st.markdown(
            f'<div class="info-box">'
            f'<b style="color:##ffffff">Referência metodológica:</b> {METHOD_NOTES[mid]}'
            f'<br><br>p GOF &gt; 0.05 indica bom ajuste do modelo aos dados.'
            f'</div>',
            unsafe_allow_html=True,
        )
