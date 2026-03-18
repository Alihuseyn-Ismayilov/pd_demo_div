import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="Credit Risk", layout="centered")

st.markdown("""
<style>
    /* Dark background everywhere */
    .stApp {
        background-color: #0f0f0f;
        color: #e0e0e0;
    }

    /* Main content area */
    .block-container {
        background-color: #0f0f0f;
        padding-top: 40px;
        padding-bottom: 40px;
    }

    /* Input fields — slightly lighter than background */
    input[type="number"] {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
    }

    /* Input labels */
    label {
        color: #a0a0a0 !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.03em !important;
    }

    /* Number input container */
    div[data-testid="stNumberInput"] > div {
        background-color: #1e1e1e !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 4px !important;
    }

    /* Button */
    .stButton > button {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 10px 32px;
        font-size: 0.9rem;
        font-weight: 400;
        letter-spacing: 0.06em;
        width: 100%;
        margin-top: 16px;
        cursor: pointer;
        transition: border-color 0.15s, background-color 0.15s;
    }
    .stButton > button:hover {
        background-color: #252525;
        border-color: #666666;
    }

    /* Result box */
    .result-box {
        background-color: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 24px 28px;
        margin-top: 24px;
    }
    .result-label {
        color: #666666;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-value {
        color: #e0e0e0;
        font-size: 2.2rem;
        font-weight: 300;
        letter-spacing: 0.02em;
    }
    .result-note {
        color: #555555;
        font-size: 0.78rem;
        margin-top: 10px;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #222222;
        margin: 28px 0;
    }

    /* Page title */
    h1 {
        color: #d0d0d0 !important;
        font-weight: 300 !important;
        font-size: 1.4rem !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 4px !important;
    }
    p {
        color: #555555 !important;
        font-size: 0.82rem !important;
    }

    /* Coefficient table */
    .coef-table {
        background-color: #161616;
        border: 1px solid #222222;
        border-radius: 4px;
        padding: 16px 20px;
        margin-top: 8px;
        font-size: 0.82rem;
        font-family: monospace;
    }
    .coef-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px solid #1e1e1e;
        color: #606060;
    }
    .coef-row:last-child { border-bottom: none; }
    .coef-name  { color: #808080; }
    .coef-value { color: #505050; }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load coefficients ─────────────────────────────────────────────
@st.cache_data
def load_coefficients():
    return pd.read_csv("coefficients.csv").set_index("feature")["coefficient"].to_dict()

coefs = load_coefficients()

# ── Page title ────────────────────────────────────────────────────
st.title("Credit Risk — Logistic Regression")
st.write("Manual prediction using model coefficients")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────
age             = st.number_input("age",             value=40,      step=1)
dpd_30          = st.number_input("dpd_30",          value=0,       step=1)
monthly_payment = st.number_input("monthly_payment", value=500.0,   step=10.0)
salary          = st.number_input("salary",          value=3000.0,  step=100.0)

# ── Predict button ────────────────────────────────────────────────
if st.button("CALCULATE"):

    # log-odds = b0 + b1*x1 + b2*x2 + ...
    log_odds = (
        coefs["bias"]            +
        coefs["age"]             * age            +
        coefs["dpd_30"]          * dpd_30         +
        coefs["monthly_payment"] * monthly_payment +
        coefs["salary"]          * salary
    )

    # sigmoid → probability
    probability = 1 / (1 + math.exp(-log_odds))
    probability_pct = round(probability * 100, 1)

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">default probability</div>
        <div class="result-value">{probability_pct}%</div>
        <div class="result-note">log-odds: {log_odds:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Coefficients reference ────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("coefficients"):
    rows = "".join([
        f'<div class="coef-row"><span class="coef-name">{k}</span><span class="coef-value">{v:+.4f}</span></div>'
        for k, v in coefs.items()
    ])
    st.markdown(f'<div class="coef-table">{rows}</div>', unsafe_allow_html=True)
