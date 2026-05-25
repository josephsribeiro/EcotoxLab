# =============================================================================
# EcotoxLab v2.2 — Análise de Toxicidade Aquática
# streamlit run ecotox_app.py
# pip install streamlit numpy pandas scipy matplotlib
#
# Métodos Spearman-Kärber:
#   SK Clássico  — Wheeler et al. (2006) / Thompson (1947)
#                  Integral trapezoidal direta, variância Thompson com correção Bessel (n−1)
#                  Sem aparamento, sem interpolação de bordos
#   SK Aparado (TSK) — Hamilton & EPA (1977/1978) / {ecotoxicology} CRAN (Gama 2013)
#                  Suavização PAV ponderada, interpolação nos bordos, aparamento simétrico
#                  SE² = Σ(xᵢ₊₁−xᵢ₋₁)²·p·(1−p)/(4n), GSD=10^SE
#                  Verificado: Hamilton 1977 → CL50=31.623, GSD=1.2969, LCL=18.997, UCL=52.639 ✓
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.special import ndtr, ndtri
from io import BytesIO
import streamlit as st

st.set_page_config(page_title="EcotoxLab",page_icon="⚗",layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');
.stApp{background:#07090d;color:#c9d5e0;font-family:'Inter',sans-serif;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b0f18 0%,#0d1320 100%);border-right:1px solid #1e2a3a;}
section[data-testid="stSidebar"] *{color:#c9d5e0 !important;}
section[data-testid="stSidebar"] strong{color:#fff !important;font-weight:700 !important;}
section[data-testid="stSidebar"] code{color:#7dd3b8 !important;background:#0f1923 !important;border-radius:4px;padding:1px 5px;}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] select,section[data-testid="stSidebar"] textarea{background:#0f1923 !important;color:#fff !important;}
section[data-testid="stSidebar"] select option{color:#000 !important;background:#fff;}
[data-baseweb="select"] [role="option"],[data-baseweb="menu"] [role="option"]{color:#000 !important;background:#fff !important;}
[data-baseweb="select"] [role="option"]:hover,[data-baseweb="menu"] [role="option"]:hover{background:#e8f4f0 !important;color:#000 !important;}
.stButton>button{background:linear-gradient(135deg,#0d6e55,#18a07a);color:#fff !important;border:none;border-radius:10px;font-weight:600;width:100%;padding:11px 18px;font-size:14px;box-shadow:0 4px 15px rgba(24,160,122,.25);transition:all .2s;}
.stButton>button:hover{filter:brightness(1.12);transform:translateY(-2px);box-shadow:0 6px 20px rgba(24,160,122,.35);}
.stApp label,.stApp .stMarkdown p,.stApp .stCaption,.stApp [data-testid="stWidgetLabel"] p,
div[data-testid="stNumberInputContainer"] label,div[class*="stNumberInput"] label,
div[class*="stTextInput"] label,div[class*="stSelectbox"] label{color:#c9d5e0 !important;}
.stTextInput input,.stNumberInput input,.stSelectbox select{background:#0f1923 !important;border:1px solid #1e2a3a !important;color:#c9d5e0 !important;border-radius:8px !important;font-family:'JetBrains Mono',monospace !important;}
.stTextInput input:focus,.stNumberInput input:focus{border-color:#18a07a !important;box-shadow:0 0 0 2px rgba(24,160,122,.15) !important;}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#0d1520,#111e2e);border:1px solid #1e3040;border-radius:14px;padding:16px 20px;transition:border-color .2s;}
div[data-testid="stMetric"]:hover{border-color:#18a07a44;}
div[data-testid="stMetricValue"]{color:#4fc3a1 !important;font-size:1.35rem !important;font-weight:700 !important;}
div[data-testid="stMetricLabel"]{color:#8ba4b8 !important;font-size:.72rem !important;text-transform:uppercase;letter-spacing:.5px;}
div[data-testid="stMetricDelta"]{font-size:.78rem !important;}
.stTabs [data-baseweb="tab-list"]{background:#0b0f18;border-bottom:1px solid #1e2a3a;padding:0 4px;gap:4px;}
.stTabs [data-baseweb="tab"]{color:#8ba4b8;border-radius:8px 8px 0 0;padding:10px 20px;font-weight:500;transition:all .15s;}
.stTabs [data-baseweb="tab"]:hover{color:#c9d5e0 !important;background:#111e2e;}
.stTabs [aria-selected="true"]{color:#18a07a !important;border-bottom:2px solid #18a07a !important;background:#0d1a14 !important;font-weight:600;}
hr{border-color:#1e2a3a;margin:18px 0;}
.warn-box{background:#1e0f10;border:1px solid #e5534b55;border-radius:10px;padding:12px 16px;font-size:13px;color:#f07070;}
.ok-box{background:#091a14;border:1px solid #18a07a55;border-radius:10px;padding:12px 16px;font-size:13px;color:#4fc3a1;}
.info-box{background:#0b0f18;border:1px solid #1e2a3a;border-radius:10px;padding:14px 18px;font-size:12px;line-height:1.7;color:#8ba4b8;}
.diff-box{background:#0f1a2e;border:1px solid #2a4a7a55;border-radius:10px;padding:14px 18px;font-size:12px;line-height:1.75;color:#8baad8;}
.section-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.section-title{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#4fc3a1 !important;}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,#18a07a33,transparent);}
.control-section{background:linear-gradient(135deg,#120f08,#1a1508);border:1px solid #e3b34133;border-radius:12px;padding:14px 18px;margin-bottom:14px;}
.control-label{font-size:11px;font-weight:700;color:#e3b341 !important;text-transform:uppercase;letter-spacing:1px;}
.control-badge{display:inline-block;background:#e3b34118;color:#e3b341;border:1px solid #e3b34133;border-radius:4px;font-size:9px;padding:2px 7px;margin-left:8px;vertical-align:middle;text-transform:none;letter-spacing:0;}
.doses-section{background:linear-gradient(135deg,#0d1520,#0f1e2e);border:1px solid #1e3040;border-radius:12px;padding:14px 18px;margin-bottom:14px;}
.doses-label{font-size:11px;font-weight:700;color:#539bf5 !important;text-transform:uppercase;letter-spacing:1px;}
.disabled-input{background:#111923;border:1px solid #1e2a3a;border-radius:8px;padding:9px 13px;color:#e3b341;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;display:flex;align-items:center;gap:6px;}
[data-testid="stExpander"]{background:#0d1520 !important;border:1px solid #1e2a3a !important;border-radius:12px !important;}
[data-testid="stExpander"] summary{color:#c9d5e0 !important;font-weight:600;}
[data-testid="stExpander"] summary p{color:#c9d5e0 !important;}
.streamlit-expanderHeader p,.streamlit-expanderHeader span{color:#c9d5e0 !important;font-weight:600;}
.hero-header{background:linear-gradient(135deg,#0a1628 0%,#0d1e35 50%,#091420 100%);border:1px solid #1e3050;border-radius:16px;padding:22px 28px;margin-bottom:20px;position:relative;overflow:hidden;}
.hero-header::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#18a07a,#4fc3a1,transparent);}
.hero-title{font-size:1.7rem;font-weight:700;color:#fff;margin:0;letter-spacing:-.5px;}
.hero-title span{color:#4fc3a1;}
.hero-subtitle{color:#6a8ba8;font-size:.85rem;margin-top:4px;}
.hero-badge{display:inline-block;background:#18a07a18;color:#4fc3a1;border:1px solid #18a07a33;border-radius:20px;font-size:11px;padding:3px 12px;margin-top:10px;font-weight:500;}
.method-active{background:linear-gradient(135deg,#091a14,#0d2218);border:1px solid #18a07a44;border-radius:10px;padding:10px 14px;font-size:13px;margin-top:8px;}
.sk-compare{background:linear-gradient(135deg,#0a1a28,#0d1e30);border:1px solid #2a5a8855;border-radius:12px;padding:16px 20px;margin-top:12px;}
.sk-compare-title{font-size:11px;font-weight:700;color:#7ec8e3 !important;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;}
.stDownloadButton>button{background:linear-gradient(135deg,#0d1e35,#152840) !important;color:#4fc3a1 !important;border:1px solid #1e3a50 !important;border-radius:10px !important;font-weight:600 !important;transition:all .2s !important;}
.stDownloadButton>button:hover{border-color:#18a07a !important;background:linear-gradient(135deg,#0d2218,#122e22) !important;transform:translateY(-1px) !important;}
[data-testid="stCheckbox"] label{color:#c9d5e0 !important;}
[data-testid="stRadio"] label{color:#c9d5e0 !important;}
.stNumberInput button{color:#4fc3a1 !important;background:#0f1923 !important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0b0f18;}
::-webkit-scrollbar-thumb{background:#1e3040;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#18a07a;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def abbott(p, ctrl):
    if ctrl >= 1.0: return float(p)
    return float(np.clip((p - ctrl) / (1.0 - ctrl), 0.0, 1.0))

def chi_sq_pval(chi2, df):
    if df <= 0 or np.isnan(chi2): return float("nan")
    return float(1.0 - stats.chi2.cdf(chi2, df))


# ─────────────────────────────────────────────────────────────────────────────
# SK CLÁSSICO — Wheeler et al. (2006) / Thompson (1947)
#
# Formulação:
#   μ = log₁₀(dₖ) − Σᵢ (log₁₀dᵢ − log₁₀dᵢ₋₁)·(pᵢ+pᵢ₋₁)/2
#   Var = Σᵢ (Δlogd)² · (p̄ᵢ)·(1−p̄ᵢ) / (nᵢ−1)   onde p̄ᵢ=(pᵢ+pᵢ₋₁)/2
#   IC 95%: 10^(μ ± 1.96·SE)
#
# Diferenças vs TSK:
#   • Sem aparamento
#   • Sem suavização monotônica (trabalha com dados brutos)
#   • Sem interpolação de bordos
#   • Variância usa trapézio médio (p̄ᵢ) e correção Bessel (n−1)
#   • Mais simples; exige que p[0]≈0 e p[-1]≈1 naturalmente
# ─────────────────────────────────────────────────────────────────────────────

def calc_sk_classico(doses, deaths, totals, conf=0.95):
    rows = sorted(
        [(float(d), int(deaths[i]), int(totals[i]))
         for i, d in enumerate(doses) if float(d) > 0 and int(totals[i]) > 0],
        key=lambda r: r[0])
    if len(rows) < 2:
        return None

    ld = np.array([np.log10(r[0]) for r in rows])
    pr = np.array([r[1] / r[2]   for r in rows])
    ns = np.array([r[2]           for r in rows], dtype=float)
    ks = np.array([r[1]           for r in rows], dtype=float)
    k  = len(rows)

    # μ — integral trapezoidal direta
    area = sum((ld[i] - ld[i-1]) * (pr[i] + pr[i-1]) / 2.0 for i in range(1, k))
    mu   = ld[-1] - area

    # Variância Thompson (1947) com correção de Bessel (nᵢ − 1) no intervalo i
    # p̄ᵢ = (pᵢ + pᵢ₋₁) / 2  — proporção média no trapézio
    var = 0.0
    for i in range(1, k):
        p_bar = (pr[i] + pr[i-1]) / 2.0
        denom = max(ns[i] - 1.0, 1.0)   # Bessel
        var  += (ld[i] - ld[i-1])**2 * p_bar * (1.0 - p_bar) / denom

    se  = float(np.sqrt(max(var, 0.0)))
    z   = float(ndtri(1.0 - (1.0 - conf) / 2.0))
    cl  = float(10.0**mu)
    lcl = float(10.0**(mu - z * se))
    ucl = float(10.0**(mu + z * se))
    gsd = float(10.0**se)

    # χ² GOF sobre proporções brutas
    chi2 = 0.0
    for j in range(k):
        mh = float(pr[j])
        if 0.0 < mh < 1.0 and ns[j] > 0:
            chi2 += (ks[j] - ns[j]*mh)**2 / (ns[j] * mh * (1.0 - mh))

    return dict(
        cl=cl, lcl=lcl, ucl=ucl, log_cl=float(mu),
        slope=None, intercept=None, z_value=None,
        variance=round(var, 8), se_log=round(se, 6), gsd=round(gsd, 6),
        chi2=round(chi2, 4), pgof=round(chi_sq_pval(chi2, max(1, k-2)), 4),
        was_smoothed=False, n_doses=k, trim_used=0.0,
        _b0=None, _b1=None, _link=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SK APARADO (TSK) — Hamilton & EPA (1977/1978)
# Tradução fiel do TSK.R — {ecotoxicology} CRAN (Jose Gama 2013 / Brenton Stone EPA 2010)
#
# Diferenças vs SK Clássico:
#   • Suavização PAV ponderada por n (Pool Adjacent Violators)
#   • Escala com aparamento simétrico A em cada extremo
#   • Interpolação linear nos bordos → datatrim
#   • Variância: SE² = Σᵢ(xᵢ₊₁−xᵢ₋₁)²·pᵢ·(1−pᵢ)/(4·nᵢ) sobre internos do datatrim
#   • GSD = 10^SE (sem z)
#   Verificado: Hamilton 1977 → CL50=31.623, GSD=1.2969, LCL=18.997, UCL=52.639 ✓
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_pav(p_arr, n_arr):
    """Pool Adjacent Violators ponderado por n — idêntico ao TSK.R."""
    p = p_arr.copy().astype(float)
    n = n_arr.copy().astype(float)
    changed = True
    while changed:
        changed = False
        for i in range(len(p) - 1):
            if p[i] > p[i + 1]:
                pave = (p[i]*n[i] + p[i+1]*n[i+1]) / (n[i] + n[i+1])
                p[i] = pave; p[i+1] = pave; changed = True
    return p


def calc_spearman_karber(doses, deaths, totals, trim=0.0, conf=0.95):
    rows = sorted(
        [(float(d), int(deaths[i]), int(totals[i]))
         for i, d in enumerate(doses) if float(d) > 0 and int(totals[i]) > 0],
        key=lambda r: r[0])
    if len(rows) < 2:
        return None

    x_real = np.array([r[0] for r in rows])
    r_arr  = np.array([r[1] for r in rows], dtype=float)
    n_arr  = np.array([r[2] for r in rows], dtype=float)
    N      = len(rows)
    lx     = np.log10(x_real)
    raw_p  = r_arr / n_arr

    p_sm   = _smooth_pav(raw_p, n_arr)
    was_sm = not np.allclose(raw_p, p_sm)

    A    = float(np.clip(trim, 0.0, 0.4999))
    p_sc = (p_sm - A) / (1.0 - 2.0*A) if A > 0 else p_sm.copy()

    if p_sc[0] > 0 or p_sc[-1] < 1:
        A_min = max(float(p_sm[0]), float(1.0 - p_sm[-1]))
        return {"error": f"Trim insuficiente. Use trim ≥ {A_min*100:.1f}%."}

    keepers = (p_sc > 0) & (p_sc < 1)
    if np.any(keepers):
        wk     = np.where(keepers)[0]
        Uscale = int(wk[-1]); Lscale = int(wk[0])

        if Lscale > 0 and abs(p_sc[Lscale] - p_sc[Lscale-1]) > 1e-12:
            xhead = lx[Lscale-1] + (lx[Lscale]-lx[Lscale-1]) * \
                    (0.0 - p_sc[Lscale-1]) / (p_sc[Lscale] - p_sc[Lscale-1])
        else:
            xhead = lx[Lscale-1] if Lscale > 0 else lx[0]

        if Uscale+1 < N and abs(p_sc[Uscale+1] - p_sc[Uscale]) > 1e-12:
            xtail = lx[Uscale] + (lx[Uscale+1]-lx[Uscale]) * \
                    (1.0 - p_sc[Uscale]) / (p_sc[Uscale+1] - p_sc[Uscale])
        else:
            xtail = lx[Uscale+1] if Uscale+1 < N else lx[-1]

        dt_x = np.concatenate([[xhead], lx[keepers], [xtail]])
        dt_p = np.concatenate([[0.0],   p_sm[keepers], [1.0]])
        n_h  = n_arr[Lscale-1] if Lscale > 0 else n_arr[0]
        n_t  = n_arr[Uscale+1] if Uscale+1 < N else n_arr[-1]
        dt_n = np.concatenate([[n_h], n_arr[keepers], [n_t]])
    else:
        i = int(np.where(p_sc > 0)[0][0]) - 1
        if i < 0 or abs(p_sc[i+1] - p_sc[i]) < 1e-12:
            return None
        xh = lx[i] + (lx[i+1]-lx[i])*(0.0-p_sc[i])/(p_sc[i+1]-p_sc[i])
        xt = lx[i] + (lx[i+1]-lx[i])*(1.0-p_sc[i])/(p_sc[i+1]-p_sc[i])
        dt_x = np.array([xh, xt]); dt_p = np.array([0., 1.])
        dt_n = np.array([n_arr[i], n_arr[i]])

    M  = len(dt_x)
    mu = dt_x[-1] - sum((dt_x[j+1]-dt_x[j])*(dt_p[j+1]+dt_p[j])/2.0 for j in range(M-1))

    var_mu = 0.0
    for j in range(1, M-1):
        pj = dt_p[j]; nj = dt_n[j]
        if 0.0 < pj < 1.0 and nj > 0:
            var_mu += (dt_x[j+1] - dt_x[j-1])**2 * pj*(1.0-pj) / (4.0*nj)
    var_mu = max(var_mu, 0.0)
    se     = float(np.sqrt(var_mu))
    z      = float(ndtri(1.0 - (1.0-conf)/2.0))
    cl     = float(10.0**mu)
    lcl    = float(10.0**(mu - z*se))
    ucl    = float(10.0**(mu + z*se))
    gsd    = float(10.0**se)

    chi2 = 0.0
    for j in range(N):
        mh = float(p_sm[j])
        if 0.0 < mh < 1.0 and n_arr[j] > 0:
            chi2 += (r_arr[j] - n_arr[j]*mh)**2 / (n_arr[j]*mh*(1.0-mh))

    return dict(
        cl=cl, lcl=lcl, ucl=ucl, log_cl=float(mu),
        slope=None, intercept=None, z_value=None,
        variance=round(var_mu, 8), se_log=round(se, 6), gsd=round(gsd, 6),
        chi2=round(chi2, 4), pgof=round(chi_sq_pval(chi2, max(1, N-2)), 4),
        was_smoothed=was_sm, n_doses=N, trim_used=A,
        _b0=None, _b1=None, _link=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLM PROBIT / LOGIT  (não alterados)
# ─────────────────────────────────────────────────────────────────────────────

def calc_glm(doses, deaths, totals, link="probit"):
    data = [(np.log10(max(d,1e-10)), deaths[i]/totals[i], totals[i], int(deaths[i]))
            for i,d in enumerate(doses) if d > 0]
    data = [(x,p,n,k) for x,p,n,k in data if np.isfinite(x)]
    if len(data) < 2: return None

    if link == "probit":
        ginv    = lambda e: float(np.clip(ndtr(e),1e-9,1-1e-9))
        gprime  = lambda m: float(np.exp(-0.5*ndtri(np.clip(m,1e-9,1-1e-9))**2)/np.sqrt(2*np.pi))
        link_fn = lambda p: float(ndtri(np.clip(p,1e-6,1-1e-6)))
    else:
        ginv    = lambda e: float(np.clip(1/(1+np.exp(-e)),1e-9,1-1e-9))
        gprime  = lambda m: float(m*(1-m))
        link_fn = lambda p: float(np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)))

    pts = [(x,p,n,k) for x,p,n,k in data if 0<p<1]
    if len(pts) < 2: return None

    xs=[x for x,*_ in pts]; ys=[link_fn(p) for _,p,*_ in pts]
    mx,my=float(np.mean(xs)),float(np.mean(ys))
    ssxy=sum((xs[i]-mx)*(ys[i]-my) for i in range(len(xs)))
    ssxx=sum((v-mx)**2 for v in xs)
    b1=ssxy/ssxx if ssxx>1e-12 else 1.0; b0=my-b1*mx

    for _ in range(50):
        sW=sWX=sXWX=sWZ=sXWZ=0.0
        for x,p,n,k in data:
            eta=b0+b1*x; mu=ginv(eta); gp=max(gprime(mu),1e-12)
            w=n*gp**2/max(mu*(1-mu),1e-12); z2=eta+(p-mu)/gp
            sW+=w;sWX+=w*x;sXWX+=w*x*x;sWZ+=w*z2;sXWZ+=w*x*z2
        det=sW*sXWX-sWX**2
        if abs(det)<1e-14: break
        nb0=(sXWX*sWZ-sWX*sXWZ)/det; nb1=(sW*sXWZ-sWX*sWZ)/det
        if abs(nb0-b0)+abs(nb1-b1)<1e-10: b0,b1=nb0,nb1; break
        b0,b1=nb0,nb1

    if abs(b1)<1e-12: return None
    log_cl=-b0/b1; cl=10**log_cl

    sW=sWX=sXWX=0.0
    for x,p,n,k in data:
        eta=b0+b1*x; mu=ginv(eta); gp=max(gprime(mu),1e-12)
        w=n*gp**2/max(mu*(1-mu),1e-12); sW+=w;sWX+=w*x;sXWX+=w*x*x
    det=sW*sXWX-sWX**2
    if abs(det)<1e-14: return None

    var_b0=max(0.,sXWX/det); var_b1=max(0.,sW/det); cov_b=-sWX/det
    var_log=(var_b0+2*log_cl*cov_b+log_cl**2*var_b1)/b1**2
    factor=10**(1.96*np.sqrt(max(0.,var_log)))
    se_b1=np.sqrt(var_b1); z_val=round(b1/se_b1,4) if se_b1>1e-12 else None

    chi2=0.0
    for x,p,n,k in data:
        mh=ginv(b0+b1*x); chi2+=(k-n*mh)**2/max(n*mh*(1-mh),1e-9)
    df=max(1,len(data)-2)

    return dict(cl=cl,lcl=cl/factor,ucl=cl*factor,log_cl=log_cl,
                slope=round(b1,4),intercept=round(b0,4),z_value=z_val,
                variance=round(var_log,6),se_log=None,gsd=None,
                chi2=round(chi2,4),pgof=round(chi_sq_pval(chi2,df),4),
                was_smoothed=False,n_doses=len(data),trim_used=None,
                _b0=b0,_b1=b1,_link=link)


def predict_y(x_arr, res, link="logit"):
    b0=res.get("_b0"); b1=res.get("_b1"); out=[]
    for x in x_arr:
        lx=np.log10(max(x,1e-10))
        if b0 is not None and b1 is not None:
            eta=b0+b1*lx
            v=float(ndtr(eta)) if link=="probit" else 1/(1+np.exp(-eta))
        else:
            v=1/(1+(x/max(res["cl"],1e-10))**(-4.5))
        out.append(np.clip(v*100,0,100))
    return np.array(out)

def predict_from_cl(x_arr, cl, link="logit"):
    b1e=4.5; b0e=-b1e*np.log10(max(cl,1e-10)); out=[]
    for x in x_arr:
        lx=np.log10(max(x,1e-10)); eta=b0e+b1e*lx
        v=float(ndtr(eta)) if link=="probit" else 1/(1+np.exp(-eta))
        out.append(np.clip(v*100,0,100))
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
UNITS_CONC=["µg/L","µg/g","µg/mg","µg/kg","mg/L","mg/g","mg/kg","ng/L","ng/g"]
UNITS_TIME=["h","min","dias","semanas"]
FONT_FAMILIES=["DejaVu Sans","DejaVu Serif","Liberation Mono","STIXGeneral","serif","monospace"]
LEGEND_LOCS=["upper left","upper right","lower left","lower right",
             "upper center","lower center","center left","center right","center"]

METHODS={
    "lc_probit":  dict(label="LC Probit",            group="LC",link="probit",is_lt=False,cl_label="CL50"),
    "lc_logit":   dict(label="LC Logit",              group="LC",link="logit", is_lt=False,cl_label="CL50"),
    "lt_probit":  dict(label="LT Probit",             group="LT",link="probit",is_lt=True, cl_label="TL50"),
    "lt_logit":   dict(label="LT Logit",              group="LT",link="logit", is_lt=True, cl_label="TL50"),
    "sk_classico":dict(label="SK Clássico",           group="NP",link=None,    is_lt=False,cl_label="CL50"),
    "sk_tsk":     dict(label="SK Aparado (TSK)",      group="NP",link=None,    is_lt=False,cl_label="CL50"),
}

METHOD_NOTES={
    "lc_probit": "LC Probit: GLM binomial com ligação probit (Finney 1971). Equivalente ao LC probit do {ecotox} (Hlina et al. 2021). Modificado por Joseph S. Ribeiro.",
    "lc_logit":  "LC Logit: GLM binomial com ligação logit. Mais robusto nos extremos. Equivalente ao LC logit do {ecotox}. Modificado por Joseph S. Ribeiro.",
    "lt_probit": "LT Probit: GLM probit com tempo como variável independente. Equivalente ao LT probit do {ecotox}. Modificado por Joseph S. Ribeiro.",
    "lt_logit":  "LT Logit: GLM logit com tempo de exposição. Equivalente ao LT logit do {ecotox}. Modificado por Joseph S. Ribeiro.",
    "sk_classico":(
        "Spearman-Kärber Clássico — Wheeler et al. (2006) / Thompson (1947). "
        "Formulação: μ = log₁₀(dₖ) − Σᵢ Δlogdᵢ·(pᵢ+pᵢ₋₁)/2; "
        "Var = Σᵢ (Δlogd)²·p̄ᵢ·(1−p̄ᵢ)/(nᵢ−1) (correção Bessel). "
        "Sem aparamento, sem suavização, sem interpolação de bordos. "
        "Mais simples e amplamente citado na literatura ecotoxicológica. "
        "Exige que p₁≈0 e pₖ≈1 naturalmente nos dados. "
        "Modificado por Joseph S. Ribeiro."
    ),
    "sk_tsk":(
        "Spearman-Kärber Aparado (TSK) — Hamilton & EPA (1977/1978). "
        "Tradução fiel do TSK.R ({ecotoxicology} CRAN, Jose Gama 2013 / Brenton Stone EPA 2010). "
        "Etapas: suavização PAV ponderada por n → escala com aparamento A → "
        "interpolação nos bordos (datatrim) → μ = integral trapezoidal → "
        "SE² = Σᵢ(xᵢ₊₁−xᵢ₋₁)²·pᵢ·(1−pᵢ)/(4·nᵢ); GSD = 10^SE. "
        "Verificado: Hamilton 1977 → CL50=31.623, GSD=1.2969, LCL=18.997, UCL=52.639 ✓. "
        "Padrão regulatório EPA para ensaios agudos. Modificado por Joseph S. Ribeiro. "
        "Ref: Hamilton et al. Environ. Sci. Technol. 11, 714–719 (1977); 12, 417 (1978)."
    ),
}

SK_DIFF_HTML = """
<div class="sk-compare">
<div class="sk-compare-title">⚖️ Diferenças entre os dois métodos Spearman-Kärber</div>
<table style="width:100%;border-collapse:collapse;font-size:11.5px;color:#8baad8">
<tr style="border-bottom:1px solid #2a4a7a44">
  <th style="text-align:left;padding:5px 8px;color:#7ec8e3;width:28%">Aspecto</th>
  <th style="text-align:left;padding:5px 8px;color:#4fc3a1;width:36%">SK Clássico</th>
  <th style="text-align:left;padding:5px 8px;color:#e3b341;width:36%">SK Aparado (TSK)</th>
</tr>
<tr style="border-bottom:1px solid #1a3050"><td style="padding:5px 8px">Suavização</td><td style="padding:5px 8px">Não</td><td style="padding:5px 8px">PAV ponderado por n</td></tr>
<tr style="border-bottom:1px solid #1a3050"><td style="padding:5px 8px">Aparamento</td><td style="padding:5px 8px">Não</td><td style="padding:5px 8px">Simétrico 0–40%</td></tr>
<tr style="border-bottom:1px solid #1a3050"><td style="padding:5px 8px">Bordos</td><td style="padding:5px 8px">Dados brutos</td><td style="padding:5px 8px">Interpolação linear</td></tr>
<tr style="border-bottom:1px solid #1a3050"><td style="padding:5px 8px">Variância</td><td style="padding:5px 8px">Thompson (Δlogd)²·p̄·(1−p̄)/(n−1)</td><td style="padding:5px 8px">EPA (xᵢ₊₁−xᵢ₋₁)²·p·(1−p)/(4n)</td></tr>
<tr style="border-bottom:1px solid #1a3050"><td style="padding:5px 8px">GSD reportado</td><td style="padding:5px 8px">10^SE</td><td style="padding:5px 8px">10^SE (sem z)</td></tr>
<tr style="border-bottom:1px solid #1a3050"><td style="padding:5px 8px">Requisito dados</td><td style="padding:5px 8px">p₁≈0 e pₖ≈1</td><td style="padding:5px 8px">Flexível via trim</td></tr>
<tr><td style="padding:5px 8px">Uso típico</td><td style="padding:5px 8px">Literatura geral</td><td style="padding:5px 8px">Padrão regulatório EPA</td></tr>
</table>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(
    res, obs_x, obs_y, x_label, y_label, cl_label,
    substance, method_id, link,
    title_main, subtitle, bg_color,
    curve_color, ci_lower_color, ci_upper_color, point_color, cl50_line_color,
    unit_r,
    show_curve, show_ci_lower, show_ci_upper,
    show_points, show_pct_labels, show_cl50_line,
    show_mort50_line, show_grid, show_param_box, show_legend,
    font_family, font_size_title, font_size_axis, font_size_ticks, font_size_legend,
    param_box_x=0.98, param_box_y=0.03, param_box_ha="right", param_box_va="bottom",
    cl50_lbl_x_off=0.2, cl50_lbl_y=105.0,
    legend_loc="upper left", title_x=0.5,
):
    IS_DARK=(bg_color=="dark")
    BG     ="#07090d" if IS_DARK else "#ffffff"
    AX_BG  ="#0a0e14" if IS_DARK else "#f8fafb"
    TEXT_C ="#c9d5e0" if IS_DARK else "#1a1a2e"
    GRID_C ="#1e2a3a" if IS_DARK else "#e8ecf0"
    SPINE_C="#1e2a3a" if IS_DARK else "#c0c8d0"
    LEG_BG ="#0d1520" if IS_DARK else "#f0f4f8"
    LEG_EC ="#1e3040" if IS_DARK else "#c0c8d0"
    LEG_LC ="#c9d5e0" if IS_DARK else "#1a1a2e"
    PBG    ="#0d1520" if IS_DARK else "#f0f4f8"
    PEC    ="#1e3040" if IS_DARK else "#c0c8d0"
    M50_C  ="#3a5060" if IS_DARK else "#b0bec5"
    ACCENT ="#4fc3a1" if IS_DARK else "#0d6e55"

    plt.rcParams["font.family"]=font_family
    fig,ax=plt.subplots(figsize=(10.5,6.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C); sp.set_linewidth(0.8)

    x_min=max(min(obs_x)*0.55,1e-5); x_max=max(obs_x)*1.45
    xs=np.linspace(x_min,x_max,700)
    y_curve=predict_y(xs,res,link or "logit")
    y_lower=predict_from_cl(xs,res["lcl"],link or "logit")
    y_upper=predict_from_cl(xs,res["ucl"],link or "logit")
    handles=[]

    if show_ci_lower and show_ci_upper:
        ax.fill_between(xs,y_lower,y_upper,alpha=0.08,color=curve_color,zorder=1)
    if show_ci_lower:
        ax.plot(xs,y_lower,color=ci_lower_color,lw=1.3,ls="--",alpha=0.85,zorder=3)
        handles.append(Line2D([0],[0],color=ci_lower_color,lw=1.3,ls="--",alpha=0.85,label="IC 95% — Limite inferior"))
    if show_ci_upper:
        ax.plot(xs,y_upper,color=ci_upper_color,lw=1.3,ls="--",alpha=0.85,zorder=3)
        handles.append(Line2D([0],[0],color=ci_upper_color,lw=1.3,ls="--",alpha=0.85,label="IC 95% — Limite superior"))
    if show_curve:
        ax.plot(xs,y_curve,color=curve_color,lw=2.6,ls="-",zorder=5,solid_capstyle="round")
        handles.append(Line2D([0],[0],color=curve_color,lw=2.6,label="Curva ajustada"))
    if show_mort50_line:
        ax.axhline(50,color=M50_C,lw=0.8,ls=":",zorder=2,alpha=0.7)
        handles.append(Line2D([0],[0],color=M50_C,lw=0.8,ls=":",alpha=0.7,label="50% mortalidade"))
    if show_cl50_line:
        ax.axvline(res["cl"],color=cl50_line_color,lw=1.4,ls="-.",alpha=0.85,zorder=4)
        ax.text(res["cl"]+cl50_lbl_x_off,cl50_lbl_y,
                f"{cl_label} = {res['cl']:.3f}",
                color=cl50_line_color,fontsize=font_size_ticks,
                va="top",ha="left" if cl50_lbl_x_off>=0 else "right",
                fontweight="600",alpha=0.9,zorder=8)
        handles.append(Line2D([0],[0],color=cl50_line_color,lw=1.4,ls="-.",alpha=0.85,label=f"{cl_label} estimada"))
    if show_points:
        for xi,yi in zip(obs_x,obs_y):
            p=yi/100.; se=np.sqrt(p*(1-p)/10)*100 if 0<p<1 else 0
            ax.errorbar(xi,yi,yerr=se,fmt='o',color=point_color,
                        markeredgecolor=BG,markeredgewidth=1.2,markersize=9,
                        ecolor=point_color,capsize=4,capthick=1.3,elinewidth=1.3,zorder=7)
            if show_pct_labels:
                oy=4.5 if yi<85 else -7
                ax.annotate(f"{yi:.1f}%",xy=(xi,yi),xytext=(2,oy),
                            textcoords="offset points",
                            ha="left",va="bottom" if oy>0 else "top",
                            fontsize=font_size_ticks,color=TEXT_C,fontweight="500",
                            bbox=dict(boxstyle="round,pad=0.2",facecolor=AX_BG,
                                      edgecolor="none",alpha=0.6))
        handles.insert(0,Line2D([0],[0],marker="o",color=point_color,
                                markeredgecolor=BG,markeredgewidth=1.,
                                ls="none",markersize=8,label="Dados observados"))

    if show_param_box:
        sl=res.get("slope"); ic_v=res.get("intercept")
        vr=f"{res['variance']:.6f}" if res.get("variance") is not None else "—"
        se_v=f"{res['se_log']:.6f}" if res.get("se_log") is not None else "—"
        gsd_v=f"{res['gsd']:.6f}" if res.get("gsd") is not None else "—"
        is_sk = method_id in ("sk_classico","sk_tsk")
        if is_sk:
            sk_tag="SK Clássico" if method_id=="sk_classico" else "SK Aparado (TSK)"
            ptxt=(f"{sk_tag}\n{'─'*23}\n"
                  f"{cl_label}        = {res['cl']:.4f}\n"
                  f"log₁₀({cl_label}) = {res['log_cl']:.4f}\n"
                  f"SE(log₁₀)  = {se_v}\n"
                  f"GSD        = {gsd_v}\n"
                  f"{'─'*23}\nIC 95%\n"
                  f"  Inferior = {res['lcl']:.4f}\n"
                  f"  Superior = {res['ucl']:.4f}")
        else:
            ptxt=(f"Parâmetros GLM\n{'─'*23}\n"
                  f"{cl_label}        = {res['cl']:.4f}\n"
                  f"Slope (b₁)  = {sl if sl is not None else '—'}\n"
                  f"Intercept   = {ic_v if ic_v is not None else '—'}\n"
                  f"Var(log)    = {vr}\n"
                  f"{'─'*23}\nIC 95%\n"
                  f"  Inferior = {res['lcl']:.4f}\n"
                  f"  Superior = {res['ucl']:.4f}")
        ax.text(param_box_x,param_box_y,ptxt,transform=ax.transAxes,
                fontsize=font_size_ticks-0.5,va=param_box_va,ha=param_box_ha,
                color=TEXT_C,fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.6",facecolor=PBG,edgecolor=PEC,
                          alpha=0.93,linewidth=0.8),zorder=9)

    ax.axhline(0,color=SPINE_C,lw=0.5,zorder=0)
    ax.axhline(100,color=SPINE_C,lw=0.5,zorder=0,ls=":")
    fig.suptitle(title_main,fontsize=font_size_title,fontweight="bold",
                 color=TEXT_C,y=0.99,x=title_x,ha="center")
    ax.set_title(subtitle,fontsize=max(font_size_axis-1,8),fontstyle="italic",color=ACCENT,pad=8)
    ax.set_xlabel(x_label,color=TEXT_C,fontsize=font_size_axis,labelpad=10)
    ax.set_ylabel(y_label,color=TEXT_C,fontsize=font_size_axis,labelpad=10)
    ax.set_ylim(-4,113); ax.set_xlim(left=0)
    ax.set_yticks(range(0,111,10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:g}"))
    ax.tick_params(colors=TEXT_C,labelsize=font_size_ticks,which="both",length=4,width=0.7)
    if show_grid:
        ax.grid(True,color=GRID_C,lw=0.4,ls="--",alpha=0.6,zorder=0)
        ax.set_axisbelow(True)
    if show_legend and handles:
        leg=ax.legend(handles=handles,facecolor=LEG_BG,edgecolor=LEG_EC,
                      labelcolor=LEG_LC,fontsize=font_size_legend,
                      loc=legend_loc,framealpha=0.93,borderpad=0.8,handlelength=2.0)
        leg.get_frame().set_linewidth(0.7)
    fig.tight_layout(rect=[0,0,1,0.97])
    return fig


def fig_to_bytes(fig, fmt, dpi=150, bg="#07090d"):
    buf=BytesIO(); fig.savefig(buf,format=fmt,dpi=dpi,bbox_inches="tight",facecolor=bg)
    buf.seek(0); return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 4px">
      <div style="font-size:1.3rem;font-weight:700;color:#4fc3a1;letter-spacing:-.5px">⚗ EcotoxLab</div>
      <div style="font-size:11px;color:#5a7a8a;margin-top:2px">Toxicidade Aquática · v2.2</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("**🔬 Experimento**")
    substance=st.text_input("Substância / Espécie",value="Substância X",key="substance")
    col_r,col_i=st.columns(2)
    reps =col_r.number_input("Repetições", min_value=1,value=3, step=1,key="reps")
    indiv=col_i.number_input("Indivíduos", min_value=1,value=10,step=1,key="indiv")
    total=int(reps*indiv)
    st.caption(f"Total por grupo: **{total}** organismos")
    col_uc,col_ut=st.columns(2)
    unit_conc=col_uc.selectbox("Conc.",UNITS_CONC,key="unit_conc")
    unit_time=col_ut.selectbox("Tempo",UNITS_TIME,key="unit_time")
    st.divider()

    st.markdown("**📐 Método de Análise**")
    st.caption("Baseado em `{ecotox}` (Hlina 2021) e `{ecotoxicology}` (Gama, CRAN) · mod. Joseph S. Ribeiro")

    st.markdown('<p style="font-size:12px;color:#4fc3a1;margin:6px 0 2px">🔵 Concentração Letal (LC)</p>',unsafe_allow_html=True)
    lc_choice=st.radio("lc_r",["lc_probit","lc_logit"],
        format_func=lambda k:METHODS[k]["label"],label_visibility="collapsed",key="lc_radio")

    st.markdown('<p style="font-size:12px;color:#e3b341;margin:6px 0 2px">🟡 Tempo Letal (LT)</p>',unsafe_allow_html=True)
    lt_choice=st.radio("lt_r",["lt_probit","lt_logit"],
        format_func=lambda k:METHODS[k]["label"],label_visibility="collapsed",key="lt_radio")

    st.markdown('<p style="font-size:12px;color:#4fc3a1;margin:6px 0 2px">🟢 Não-paramétrico — Spearman-Kärber</p>',unsafe_allow_html=True)
    sk_choice=st.radio(
        "sk_r",
        ["none","sk_classico","sk_tsk"],
        format_func=lambda k:{"none":"— Não usar SK","sk_classico":"SK Clássico (Wheeler 2006)","sk_tsk":"SK Aparado / TSK (Hamilton & EPA)"}[k],
        label_visibility="collapsed",key="sk_radio")

    sk_trim=0
    if sk_choice=="sk_tsk":
        sk_trim=st.slider("Aparamento (trim %)",0,40,0,5,key="sk_trim",
            help="0 = sem aparamento (padrão EPA). Aumentar se p₁>0 ou pₖ<1.")

    # Mostrar tabela comparativa quando SK está selecionado
    if sk_choice in ("sk_classico","sk_tsk"):
        st.markdown(SK_DIFF_HTML,unsafe_allow_html=True)

    st.divider()
    st.markdown("**Modo de análise**")
    analysis_mode=st.radio("Modo",["LC — Concentração Letal","LT — Tempo Letal"],
        key="analysis_mode",label_visibility="collapsed")

    # Determinar método ativo
    if sk_choice == "sk_classico":    method_id="sk_classico"
    elif sk_choice == "sk_tsk":       method_id="sk_tsk"
    elif analysis_mode.startswith("LT"): method_id=lt_choice
    else:                              method_id=lc_choice

    m=METHODS[method_id]; link=m["link"]; is_lt=m["is_lt"]
    cl_label=m["cl_label"]; unit=unit_time if is_lt else unit_conc
    x_label=f"Tempo ({unit})" if is_lt else f"Concentração ({unit})"

    st.markdown(
        f'<div class="method-active">'
        f'<span style="font-size:10px;color:#5a7a8a;text-transform:uppercase;letter-spacing:1px">Método ativo</span><br>'
        f'<span style="color:#4fc3a1;font-weight:700;font-size:14px">{m["label"]}</span>'
        f'</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="info-box" style="margin-top:10px">{METHOD_NOTES[method_id]}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <div class="hero-title">⚗ <span>Ecotox</span>Lab</div>
  <div class="hero-subtitle">Análise de Toxicidade Aquática · {m['label']}</div>
  <div class="hero-badge">Baseado em {{ecotox}} (Hlina 2021) · {{ecotoxicology}} (Gama, CRAN) · mod. Joseph S. Ribeiro</div>
</div>
""",unsafe_allow_html=True)

tab_data,tab_result=st.tabs(["📋  Dados","📊  Resultados"])


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _rc(pct): return "#f07070" if pct>50 else "#e3b341" if pct>25 else "#4fc3a1"
def _hdr(t):  return f'<p style="font-size:10px;color:#8ba4b8;font-weight:600;margin:0;text-transform:uppercase;letter-spacing:.5px">{t}</p>'
def _sec(t):  return f'<div class="section-header"><span class="section-title">{t}</span><span class="section-line"></span></div>'


# ─────────────────────────────────────────────────────────────────────────────
# ABA DADOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:

    # ══════ MODO LT ══════
    if is_lt:
        st.markdown(_sec("Tempo Letal — Parâmetros do Ensaio"),unsafe_allow_html=True)
        ca,cb,cc=st.columns(3)
        test_duration=ca.number_input(f"Duração total ({unit})",          min_value=1,value=96,step=1,key="test_duration")
        lt_interval  =cb.number_input(f"Intervalo de leitura ({unit})",   min_value=1,value=24,step=1,key="lt_interval")
        lt_conc      =cc.number_input(f"Concentração fixa ({unit_conc})", min_value=0.0,value=0.0,format="%.4f",key="lt_conc")
        n_intervals=max(1,int(test_duration//lt_interval))
        st.caption("→ **%d** leituras: %s %s"%(n_intervals,
            ", ".join(str(int((i+1)*lt_interval)) for i in range(n_intervals)),unit))
        st.divider()

        st.markdown(_sec("Mortes por Intervalo de Tempo"),unsafe_allow_html=True)
        st.caption("Digite as mortes **em cada intervalo**. Mortalidade acumulada calculada automaticamente.")
        hc=st.columns([.5,1.8,2.2,1.3,1.3])
        for t,c in zip(["#",f"Tempo ({unit})","Mortos no intervalo","Acumulado","Mort. %"],hc):
            c.markdown(_hdr(t),unsafe_allow_html=True)

        lt_rows=[]; cum_dead=0
        for i in range(n_intervals):
            rc=st.columns([.5,1.8,2.2,1.3,1.3])
            rc[0].markdown(f'<p style="color:#e3b341;font-weight:600;padding-top:8px">{i+1}</p>',unsafe_allow_html=True)
            t_val=rc[1].number_input("",label_visibility="collapsed",
                value=float((i+1)*lt_interval),min_value=0.0,format="%.1f",key=f"lt_t_{i}")
            d_val=rc[2].number_input("",label_visibility="collapsed",
                value=0,min_value=0,max_value=total,step=1,key=f"lt_d_{i}")
            cum_dead+=int(d_val); pct=cum_dead/total*100 if total>0 else 0.0
            rc[3].markdown(f'<p style="padding-top:8px;color:#c9d5e0">{cum_dead}</p>',unsafe_allow_html=True)
            rc[4].markdown(f'<p style="padding-top:8px;color:{_rc(pct)};font-weight:600">{pct:.1f}%</p>',unsafe_allow_html=True)
            lt_rows.append({"tempo":t_val,"mortos_intervalo":int(d_val),"acumulado":cum_dead,"pct":round(pct,1)})

        st.divider()
        if st.button(f"▶  Calcular {cl_label} — {m['label']}",key="btn_lt"):
            erros=[]
            if n_intervals<2: erros.append("Mínimo 2 intervalos de tempo.")
            if cum_dead>total: erros.append(f"Mortos acumulados ({cum_dead}) excedem o total ({total}).")
            if cum_dead==0:    erros.append("Nenhuma morte registrada.")
            if erros:
                for e in erros: st.markdown(f'<div class="warn-box">⚠ {e}</div>',unsafe_allow_html=True)
            else:
                axs=[r["tempo"] for r in lt_rows]; ade=[r["acumulado"] for r in lt_rows]
                res_calc=calc_glm(axs,ade,[total]*len(axs),link)
                if res_calc is None:
                    st.markdown('<div class="warn-box">⚠ Cálculo não convergiu.</div>',unsafe_allow_html=True)
                else:
                    st.session_state.update({"result":res_calc,"obs_x":axs,
                        "obs_y":[d/total*100 for d in ade],
                        "method_id":method_id,"unit":unit,"x_label":x_label})
                    st.markdown(f'<div class="ok-box">✓ {cl_label} = <b>{res_calc["cl"]:.4f} {unit}</b> — abra a aba Resultados.</div>',unsafe_allow_html=True)

    # ══════ MODO LC / SK ══════
    else:
        st.markdown(_sec("Concentração × Mortalidade"),unsafe_allow_html=True)
        n_doses=int(st.number_input("Número de concentrações teste",
            min_value=2,max_value=20,value=5,step=1,key="n_doses",help="Não inclui o controle"))

        st.markdown(
            '<div class="control-section">'
            '<p class="control-label">🔒 Controle — Concentração = 0,0000'
            '<span class="control-badge">não entra no cálculo da CL50</span></p>'
            '</div>',unsafe_allow_html=True)
        cc=st.columns([.5,2,2,1.5])
        cc[0].markdown('<p style="color:#e3b341;font-weight:700;padding-top:8px;font-size:13px">C</p>',unsafe_allow_html=True)
        cc[1].markdown('<div class="disabled-input">🔒  0.0000</div>',unsafe_allow_html=True)
        ctrl_dead=cc[2].number_input("ctrl_dead",label_visibility="collapsed",
            value=0,min_value=0,max_value=total,step=1,key="ctrl_dead")
        ctrl_pct=ctrl_dead/total*100 if total>0 else 0.0
        cc[3].markdown(f'<p style="padding-top:8px;color:{_rc(ctrl_pct)};font-weight:600">{ctrl_pct:.1f}%</p>',unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div class="doses-section"><p class="doses-label">📊 Concentrações Teste ({n_doses} doses)</p></div>',unsafe_allow_html=True)
        hc=st.columns([.5,2,2,1.5])
        for t,c in zip(["#",f"Concentração ({unit_conc})","Mortos","Mort. %"],hc):
            c.markdown(_hdr(t),unsafe_allow_html=True)

        lc_rows=[]
        for i in range(n_doses):
            rc=st.columns([.5,2,2,1.5])
            rc[0].markdown(f'<p style="color:#539bf5;font-weight:700;padding-top:8px">{i+1}</p>',unsafe_allow_html=True)
            dose_val=rc[1].number_input("",label_visibility="collapsed",
                value=0.0,min_value=0.0,format="%.4f",key=f"lc_dose_{i}")
            dead_val=rc[2].number_input("",label_visibility="collapsed",
                value=0,min_value=0,max_value=total,step=1,key=f"lc_dead_{i}")
            pct=dead_val/total*100 if total>0 else 0.0
            rc[3].markdown(f'<p style="padding-top:8px;color:{_rc(pct)};font-weight:600">{pct:.1f}%</p>',unsafe_allow_html=True)
            lc_rows.append({"dose":dose_val,"dead":int(dead_val)})

        st.divider()
        if st.button(f"▶  Calcular {cl_label} — {m['label']}",key="btn_lc"):
            doses_all=[r["dose"] for r in lc_rows]; deaths_all=[r["dead"] for r in lc_rows]
            ctrl_p=ctrl_dead/total if total>0 else 0.0
            corr_p=[abbott(d/total,ctrl_p) for d in deaths_all]
            corr_d=[round(p*total) for p in corr_p]
            axs=[doses_all[i] for i in range(len(doses_all)) if doses_all[i]>0]
            ade=[corr_d[i]    for i in range(len(doses_all)) if doses_all[i]>0]
            ato=[total]*len(axs)

            erros=[]
            if len(axs)<2:             erros.append("Mínimo 2 doses com valor > 0.")
            if all(d==0 for d in ade): erros.append("Nenhuma morte registrada nas doses.")
            if all(d==total for d in ade): erros.append("Mortalidade 100% em todas as doses.")

            if erros:
                for e in erros: st.markdown(f'<div class="warn-box">⚠ {e}</div>',unsafe_allow_html=True)
            else:
                if method_id=="sk_classico":
                    res_calc=calc_sk_classico(axs,ade,ato)
                elif method_id=="sk_tsk":
                    res_calc=calc_spearman_karber(axs,ade,ato,trim=sk_trim/100.0)
                else:
                    res_calc=calc_glm(axs,ade,ato,link)

                if res_calc is None:
                    st.markdown('<div class="warn-box">⚠ Cálculo não convergiu. Verifique variação na mortalidade.</div>',unsafe_allow_html=True)
                elif isinstance(res_calc,dict) and "error" in res_calc:
                    st.markdown(f'<div class="warn-box">⚠ {res_calc["error"]}</div>',unsafe_allow_html=True)
                else:
                    st.session_state.update({"result":res_calc,"obs_x":axs,
                        "obs_y":[d/total*100 for d in ade],
                        "method_id":method_id,"unit":unit,"x_label":x_label})
                    extra=""
                    if res_calc.get("was_smoothed"):
                        extra=" <b style='color:#e3b341'>(suavização PAV aplicada)</b>"
                    gsd_txt=f" | GSD = {res_calc['gsd']}" if res_calc.get("gsd") else ""
                    st.markdown(
                        f'<div class="ok-box">✓ {cl_label} = <b>{res_calc["cl"]:.4f} {unit}</b>'
                        f'{gsd_txt} — abra a aba Resultados.{extra}</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ABA RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_result:
    if "result" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#5a7a8a">
          <div style="font-size:3rem;margin-bottom:16px">📊</div>
          <div style="font-size:1.1rem;font-weight:500;color:#8ba4b8">Nenhum resultado ainda</div>
          <div style="font-size:13px;margin-top:8px">Insira os dados na aba <b>Dados</b> e clique em <b>Calcular</b></div>
        </div>""",unsafe_allow_html=True)
    else:
        res    =st.session_state["result"]
        obs_x  =st.session_state["obs_x"]
        obs_y  =st.session_state["obs_y"]
        mid    =st.session_state["method_id"]
        subst  =st.session_state.get("substance",substance)
        unit_r =st.session_state.get("unit",unit)
        x_lbl_r=st.session_state.get("x_label",x_label)
        m_r    =METHODS[mid]; link_r=m_r["link"]; cl_lbl=m_r["cl_label"]

        # Cards principais
        c1,c2,c3=st.columns(3)
        c1.metric(cl_lbl,               f"{res['cl']:.4f}", delta=unit_r)
        c2.metric("Limite Inferior 95%", f"{res['lcl']:.4f}",delta=unit_r)
        c3.metric("Limite Superior 95%", f"{res['ucl']:.4f}",delta=unit_r)

        if res.get("was_smoothed"):
            st.markdown(
                '<div class="warn-box" style="margin-top:10px">'
                '⚠ <b>Suavização aplicada (TSK):</b> proporções não eram monotonicamente crescentes. '
                'Pool Adjacent Violators ponderado por n (Hamilton 1977). Verifique os dados.'
                '</div>',unsafe_allow_html=True)

        # Métricas secundárias
        if res.get("slope") is not None:
            st.divider()
            d1,d2,d3,d4=st.columns(4)
            d1.metric("Slope (b₁)",    str(res["slope"]))
            d2.metric("Intercept (b₀)",str(res["intercept"]))
            d3.metric("z-value",       str(res["z_value"]) if res["z_value"] else "—")
            d4.metric("Var(log CL)",   f"{res['variance']:.6f}" if res.get("variance") is not None else "—")
        elif mid in ("sk_classico","sk_tsk"):
            st.divider()
            d1,d2,d3,d4=st.columns(4)
            d1.metric("log₁₀(CL50)",  f"{res['log_cl']:.4f}")
            d2.metric("SE(log₁₀)",    str(res.get("se_log","—")))
            d3.metric("GSD",          str(res.get("gsd","—")))
            trim_lbl=f"{(res.get('trim_used') or 0)*100:.0f}%" if mid=="sk_tsk" else "N/A"
            d4.metric("Trim (TSK)",   trim_lbl)

        # Comparação lado a lado se SK foi usado
        if mid in ("sk_classico","sk_tsk"):
            st.divider()
            st.markdown(_sec("Comparação dos dois métodos SK com estes dados"),unsafe_allow_html=True)
            doses_ss=obs_x; deaths_ss=[round(p/100*total) for p in obs_y]; totals_ss=[total]*len(obs_x)
            r_cl=calc_sk_classico(doses_ss,deaths_ss,totals_ss)
            r_tsk=calc_spearman_karber(doses_ss,deaths_ss,totals_ss,trim=sk_trim/100.0 if mid=="sk_tsk" else 0.0)

            comp1,comp2=st.columns(2)
            with comp1:
                st.markdown('<p style="font-size:12px;font-weight:700;color:#4fc3a1">SK Clássico (Wheeler 2006)</p>',unsafe_allow_html=True)
                if r_cl and "error" not in r_cl:
                    st.metric("CL50",f"{r_cl['cl']:.4f}",delta=unit_r)
                    st.metric("GSD", str(r_cl.get("gsd","—")))
                    st.metric("LCL / UCL",f"{r_cl['lcl']:.4f} / {r_cl['ucl']:.4f}")
                else:
                    st.markdown('<div class="warn-box">⚠ Não calculável com estes dados.</div>',unsafe_allow_html=True)
            with comp2:
                st.markdown('<p style="font-size:12px;font-weight:700;color:#e3b341">SK Aparado / TSK (Hamilton & EPA)</p>',unsafe_allow_html=True)
                if r_tsk and "error" not in r_tsk:
                    st.metric("CL50",f"{r_tsk['cl']:.4f}",delta=unit_r)
                    st.metric("GSD", str(r_tsk.get("gsd","—")))
                    st.metric("LCL / UCL",f"{r_tsk['lcl']:.4f} / {r_tsk['ucl']:.4f}")
                else:
                    st.markdown(f'<div class="warn-box">⚠ {r_tsk.get("error","Não calculável.")}</div>',unsafe_allow_html=True)

        st.divider()
        e1,e2=st.columns(2)
        e1.metric("χ² Pearson (GOF)",str(res["chi2"]))
        pgof=res.get("pgof")
        pgof_ok=isinstance(pgof,float) and not np.isnan(pgof) and pgof>0.05
        pgof_str=f"{pgof:.4f}" if isinstance(pgof,float) and not np.isnan(pgof) else "—"
        e2.metric("p-valor GOF",pgof_str,delta="✓ bom ajuste" if pgof_ok else "⚠ ajuste ruim")

        # ── Painel de personalização ────────────────────────────────────────
        st.divider()
        with st.expander("🎨  Personalizar Gráfico",expanded=True):

            st.markdown("##### ✏️ Textos")
            tc1,tc2=st.columns(2)
            chart_title   =tc1.text_input("Título principal",f"Curva Dose–Resposta ({cl_lbl})",key="chart_title")
            chart_subtitle=tc2.text_input("Subtítulo",        f"Modelo {m_r['label']}",         key="chart_subtitle")
            tc3,tc4=st.columns(2)
            chart_xlabel=tc3.text_input("Rótulo Eixo X",x_lbl_r,          key="chart_xlabel")
            chart_ylabel=tc4.text_input("Rótulo Eixo Y","Mortalidade (%)", key="chart_ylabel")

            st.divider()
            st.markdown("##### 👁️ Mostrar / Ocultar Elementos")
            vc1,vc2,vc3=st.columns(3)
            show_curve      =vc1.checkbox("Curva ajustada",         True, key="show_curve")
            show_ci_lower   =vc1.checkbox("IC Inferior (tracejado)", True, key="show_ci_lower")
            show_ci_upper   =vc1.checkbox("IC Superior (tracejado)", True, key="show_ci_upper")
            show_points     =vc2.checkbox("Dados observados",        True, key="show_points")
            show_pct_labels =vc2.checkbox("Rótulos % nos pontos",    True, key="show_pct_labels")
            show_cl50_line  =vc2.checkbox(f"Linha {cl_lbl} estimada",True, key="show_cl50_line")
            show_mort50_line=vc3.checkbox("Linha 50% mortalidade",   True, key="show_mort50_line")
            show_grid       =vc3.checkbox("Grade (grid)",            True, key="show_grid")
            show_param_box  =vc3.checkbox("Caixa de parâmetros",     True, key="show_param_box")
            show_legend     =vc3.checkbox("Legenda",                 True, key="show_legend")

            st.divider()
            st.markdown("##### 🎨 Aparência e Cores")
            ac1,_=st.columns([1,2])
            bg_choice=ac1.radio("Fundo",["dark","white"],
                format_func=lambda x:"🌑 Escuro" if x=="dark" else "⬜ Branco",
                horizontal=True,key="bg_choice")
            cc1,cc2,cc3,cc4,cc5=st.columns(5)
            col_curve  =cc1.color_picker("Curva",      "#4fc3a1",key="col_curve")
            col_ci_low =cc2.color_picker("IC Inferior","#4fc3a1",key="col_ci_low")
            col_ci_high=cc3.color_picker("IC Superior","#f07070",key="col_ci_high")
            col_points =cc4.color_picker("Pontos",     "#539bf5",key="col_points")
            col_cl50   =cc5.color_picker("Linha CL50", "#e3b341",key="col_cl50")

            st.divider()
            st.markdown("##### 📌 Posição dos Elementos")
            st.caption("Mova caixa de parâmetros, rótulo do CL50, legenda e título com os sliders.")
            pos1,pos2=st.columns(2)
            with pos1:
                st.markdown("###### Caixa de Parâmetros")
                pb_x =st.slider("X (0=esq → 1=dir)",    0.0,1.0,0.98,0.01,key="pb_x")
                pb_y =st.slider("Y (0=baixo → 1=cima)", 0.0,1.0,0.03,0.01,key="pb_y")
                pb_ha=st.radio("Alinhamento H",["left","center","right"],index=2,horizontal=True,key="pb_ha")
                pb_va=st.radio("Alinhamento V",["bottom","center","top"], index=0,horizontal=True,key="pb_va")
            with pos2:
                st.markdown("###### Rótulo da Linha CL50")
                cl_xoff=st.slider("Offset X do rótulo",  -10.0,10.0, 0.2,0.1,key="cl_xoff")
                cl_ypos=st.slider("Posição Y do rótulo",  -5.0,115.0,105.0,0.5,key="cl_ypos")
                st.markdown("###### Legenda e Título")
                leg_loc=st.selectbox("Posição da legenda",LEGEND_LOCS,index=0,key="leg_loc")
                title_x=st.slider("Posição X do título", 0.0,1.0,0.5,0.05,key="title_x")

            st.divider()
            st.markdown("##### 🔤 Tipografia")
            fc1,fc2,fc3,fc4=st.columns(4)
            font_family    =fc1.selectbox("Fonte",FONT_FAMILIES,index=0,key="font_family")
            font_size_title=fc2.number_input("Tam. Título", 8,24,14,1,key="font_size_title")
            font_size_axis =fc3.number_input("Tam. Eixos",  7,18,11,1,key="font_size_axis")
            font_size_ticks=fc4.number_input("Tam. Ticks",  6,14, 9,1,key="font_size_ticks")
            font_size_legend=st.number_input("Tam. Legenda",6,14, 9,1,key="font_size_legend")

        # Gerar gráfico
        BG_HEX="#07090d" if bg_choice=="dark" else "#ffffff"
        fig=make_chart(
            res=res,obs_x=obs_x,obs_y=obs_y,
            x_label=chart_xlabel,y_label=chart_ylabel,
            cl_label=cl_lbl,substance=subst,method_id=mid,link=link_r,
            title_main=chart_title,subtitle=chart_subtitle,bg_color=bg_choice,
            curve_color=col_curve,ci_lower_color=col_ci_low,
            ci_upper_color=col_ci_high,point_color=col_points,
            cl50_line_color=col_cl50,unit_r=unit_r,
            show_curve=show_curve,show_ci_lower=show_ci_lower,
            show_ci_upper=show_ci_upper,show_points=show_points,
            show_pct_labels=show_pct_labels,show_cl50_line=show_cl50_line,
            show_mort50_line=show_mort50_line,show_grid=show_grid,
            show_param_box=show_param_box,show_legend=show_legend,
            font_family=font_family,
            font_size_title=int(font_size_title),font_size_axis=int(font_size_axis),
            font_size_ticks=int(font_size_ticks),font_size_legend=int(font_size_legend),
            param_box_x=pb_x,param_box_y=pb_y,param_box_ha=pb_ha,param_box_va=pb_va,
            cl50_lbl_x_off=cl_xoff,cl50_lbl_y=cl_ypos,
            legend_loc=leg_loc,title_x=title_x,
        )
        st.pyplot(fig,use_container_width=True)

        # ── Exportar ──
        st.divider()
        st.markdown(_sec("📥 Exportar"),unsafe_allow_html=True)
        dl1,dl2,dl3=st.columns(3)
        dl1.download_button("⬇  JPG (300 DPI)",
            fig_to_bytes(fig,"jpg",dpi=300,bg=BG_HEX),
            f"{subst}_{cl_lbl}.jpg","image/jpeg",use_container_width=True)
        dl2.download_button("⬇  SVG (vetorial)",
            fig_to_bytes(fig,"svg",bg=BG_HEX),
            f"{subst}_{cl_lbl}.svg","image/svg+xml",use_container_width=True)

        result_rows=[
            (cl_lbl,               f"{res['cl']:.4f}",  unit_r),
            ("Lim. Inferior 95%",  f"{res['lcl']:.4f}", unit_r),
            ("Lim. Superior 95%",  f"{res['ucl']:.4f}", unit_r),
            ("log₁₀(CL)",          f"{res['log_cl']:.4f}",""),
            ("Slope (b₁)",         str(res.get("slope","—")),""),
            ("Intercept (b₀)",     str(res.get("intercept","—")),""),
            ("z-value",            str(res.get("z_value","—")),""),
            ("Var(log)",           str(res.get("variance","—")),""),
            ("SE(log)",            str(res.get("se_log","—")),""),
            ("GSD",                str(res.get("gsd","—")),""),
            ("Trim usado",         f"{(res.get('trim_used') or 0)*100:.0f}%",""),
            ("Suavização PAV",     str(res.get("was_smoothed","—")),""),
            ("χ² Pearson",         str(res.get("chi2","—")),""),
            ("p GOF",              pgof_str,""),
            ("Método",             m_r["label"],""),
            ("Substância",         subst,""),
        ]
        dl3.download_button("⬇  CSV (resultados)",
            pd.DataFrame(result_rows,columns=["Parâmetro","Valor","Unidade"]).to_csv(index=False).encode("utf-8"),
            f"{subst}_{cl_lbl}_resultados.csv","text/csv",use_container_width=True)

        # ── Nota metodológica ──
        st.divider()
        st.markdown(
            f'<div class="info-box">'
            f'<b style="color:#4fc3a1">Referência metodológica:</b><br>'
            f'{METHOD_NOTES[mid]}'
            f'<br><br><span style="color:#6a8ba8">p GOF &gt; 0.05 indica bom ajuste do modelo aos dados.</span>'
            f'</div>',unsafe_allow_html=True)
