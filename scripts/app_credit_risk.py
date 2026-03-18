import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Risk",
    page_icon="💳",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Page background ── */
    .stApp { background-color: #F0F4F8; }

    /* ── Header banner ── */
    .header {
        background: linear-gradient(135deg, #1B2A4A 0%, #0D9488 100%);
        padding: 30px 36px 24px 36px;
        border-radius: 14px;
        margin-bottom: 28px;
    }
    .header h1 { color: #FFFFFF; font-size: 2rem; font-weight: 800; margin: 0; }
    .header p  { color: #CCFBF1; font-size: 1rem; margin: 10px 0 0 0; }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748B;
        margin: 22px 0 6px 0;
    }

    /* ── Result boxes ── */
    .box-low      { background:#DCFCE7; border-left:5px solid #16A34A; border-radius:10px; padding:18px 22px; margin-top:24px; }
    .box-low h2   { color:#15803D; margin:0 0 6px 0; }
    .box-low p    { color:#166534; margin:0; }

    .box-medium   { background:#FEF9C3; border-left:5px solid #CA8A04; border-radius:10px; padding:18px 22px; margin-top:24px; }
    .box-medium h2{ color:#92400E; margin:0 0 6px 0; }
    .box-medium p { color:#78350F; margin:0; }

    .box-high     { background:#FEE2E2; border-left:5px solid #DC2626; border-radius:10px; padding:18px 22px; margin-top:24px; }
    .box-high h2  { color:#DC2626; margin:0 0 6px 0; }
    .box-high p   { color:#7F1D1D; margin:0; }

    /* ── Metric row ── */
    .metric-row { display:flex; gap:12px; margin-top:18px; }
    .metric-card {
        flex:1; background:#FFFFFF;
        border:1px solid #E2E8F0;
        border-radius:10px;
        padding:14px 10px;
        text-align:center;
    }
    .metric-card .val { font-size:1.55rem; font-weight:800; color:#1B2A4A; }
    .metric-card .lbl { font-size:0.72rem; color:#64748B; margin-top:3px; }

    /* ── Predict button ── */
    .stButton > button {
        width:100%;
        background: linear-gradient(135deg, #1B2A4A, #0D9488);
        color:#FFFFFF;
        font-size:1.05rem;
        font-weight:700;
        border:none;
        border-radius:10px;
        padding:14px;
        margin-top:22px;
        cursor:pointer;
        transition: opacity .15s;
    }
    .stButton > button:hover { opacity:.88; }

    /* ── Input label weight ── */
    label { font-weight: 600 !important; color:#1E293B !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🏦 Credit Default Risk Predictor</h1>
    <p>Enter applicant information below. The model estimates the probability
       of serious delinquency (90+ days late) within the next 2 years.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD PIPELINE  (cached — loads only once per session)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("best_pipeline_v2.pkl")

try:
    pipeline = load_pipeline()
    model_ready = True
except FileNotFoundError:
    st.warning("⚠️  Model file not found. Place **best_pipeline_v2.pkl** in the same folder as this script.")
    model_ready = False

# ─────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Financial Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    revolving_utilization = st.number_input(
        "Revolving Utilization  (0 = 0%,  1 = 100%)",
        min_value=0.0, max_value=1.0, value=0.30, step=0.01
    )
    monthly_income = st.number_input(
        "Monthly Income  ($)",
        min_value=0.0, value=5000.0, step=100.0
    )
    debt_ratio = st.number_input(
        "Debt Ratio  (monthly debts ÷ income)",
        min_value=0.0, value=0.35, step=0.01
    )

with col2:
    age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
    open_credit_lines = st.number_input("Open Credit Lines & Loans", min_value=0, value=8, step=1)
    real_estate_loans = st.number_input("Real Estate Loans or Lines", min_value=0, value=1, step=1)

st.markdown('<div class="section-label">Delinquency History</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3: delq_30_59  = st.number_input("Times 30–59 Days Late", min_value=0, value=0, step=1)
with col4: delq_60_89  = st.number_input("Times 60–89 Days Late", min_value=0, value=0, step=1)
with col5: delq_90plus = st.number_input("Times 90+ Days Late",   min_value=0, value=0, step=1)

st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
dependents = st.number_input("Number of Dependents", min_value=0, value=0, step=1)

# ─────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────
if st.button("Assess Default Risk"):

    if not model_ready:
        st.error("Cannot predict — model file is missing.")

    else:
        # ── Replicate exactly the feature engineering done in the notebook ──
        has_any_delq        = int((delq_30_59 > 0) or (delq_60_89 > 0) or (delq_90plus > 0))
        delq_severity_score = (delq_30_59 * 1) + (delq_60_89 * 2) + (delq_90plus * 3)

        input_df = pd.DataFrame([{
            "revolving_utilization" : revolving_utilization,
            "age"                   : age,
            "debt_ratio"            : debt_ratio,
            "monthly_income"        : monthly_income,
            "open_credit_lines"     : open_credit_lines,
            "real_estate_loans"     : real_estate_loans,
            "dependents"            : dependents,
            "has_any_delq"          : has_any_delq,
            "delq_severity_score"   : delq_severity_score,
        }])

        prob       = pipeline.predict_proba(input_df)[0][1]
        prediction = pipeline.predict(input_df)[0]
        prob_pct   = round(prob * 100, 1)

        # ── Result banner ──
        if prob_pct < 15:
            st.markdown(f"""
            <div class="box-low">
                <h2>✅  Low Risk</h2>
                <p>Probability of default: <strong>{prob_pct}%</strong>. 
                   The applicant shows a low likelihood of serious delinquency 
                   over the next 2 years.</p>
            </div>""", unsafe_allow_html=True)

        elif prob_pct < 40:
            st.markdown(f"""
            <div class="box-medium">
                <h2>⚠️  Moderate Risk</h2>
                <p>Probability of default: <strong>{prob_pct}%</strong>. 
                   Some risk factors are present. Further manual review is recommended.</p>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="box-high">
                <h2>❌  High Risk</h2>
                <p>Probability of default: <strong>{prob_pct}%</strong>. 
                   The applicant shows a high likelihood of serious delinquency. 
                   Loan should be reviewed carefully or declined.</p>
            </div>""", unsafe_allow_html=True)

        # ── Metric cards ──
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="val">{prob_pct}%</div>
                <div class="lbl">Default Probability</div>
            </div>
            <div class="metric-card">
                <div class="val">{"Default" if prediction == 1 else "No Default"}</div>
                <div class="lbl">Model Decision</div>
            </div>
            <div class="metric-card">
                <div class="val">{delq_severity_score}</div>
                <div class="lbl">Delinquency Score</div>
            </div>
            <div class="metric-card">
                <div class="val">{"Yes" if has_any_delq else "No"}</div>
                <div class="lbl">Any Past Late Payment</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Debug expander ──
        with st.expander("Show data sent to model"):
            st.dataframe(input_df)
