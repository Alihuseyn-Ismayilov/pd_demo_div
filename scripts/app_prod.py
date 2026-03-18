import streamlit as st
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps each column at a given lower and upper percentile.
    Boundaries are computed from training data only.
    """
    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(self.lower_pct)
        self.upper_bounds_ = X.quantile(self.upper_pct)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )
        return X.values

class MissingFlagAdder(BaseEstimator, TransformerMixin):
    """
    Adds a binary flag column for each specified column.
    1 = value was missing, 0 = value was present.
    Must run before imputation.
    """
    def __init__(self, cols_to_flag=None):
        self.cols_to_flag = cols_to_flag

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.cols_to_flag:
            for col in self.cols_to_flag:
                if col in X.columns:
                    X[str(col) + '_missing'] = X[col].isnull().astype(int)
        return X.values

class SkewnessTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p to columns whose skewness exceeds the threshold.
    Skewness is measured on training data only.
    Only applies to non-negative columns.
    """
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        skewness = X.skew()
        non_negative = (X.min() >= 0)
        self.cols_to_transform_ = skewness[
            (skewness.abs() > self.threshold) & non_negative
        ].index.tolist()
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.cols_to_transform_:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X.values

st.title("Credit Default Risk")

# pipeline = joblib.load(r"\scripts\best_pipeline_v2.pkl")
pipeline = joblib.load(os.path.join(BASE_DIR, "best_pipeline_v2.pkl"))

age             = st.number_input("Age",             value=40)
revolving       = st.number_input("Revolving Utilization (0-1)", value=0.3)
debt_ratio      = st.number_input("Debt Ratio",      value=0.35)
monthly_income  = st.number_input("Monthly Income",  value=5000)
open_credit     = st.number_input("Open Credit Lines", value=8)
real_estate     = st.number_input("Real Estate Loans", value=1)
dependents      = st.number_input("Dependents",      value=0)
dpd_30          = st.number_input("Times 30-59 Days Late", value=0)
dpd_60          = st.number_input("Times 60-89 Days Late", value=0)
dpd_90          = st.number_input("Times 90+ Days Late",   value=0)

if st.button("Predict"):

    has_any_delq = int(dpd_30 > 0 or dpd_60 > 0 or dpd_90 > 0)
    delq_score   = dpd_30 * 1 + dpd_60 * 2 + dpd_90 * 3

    input_df = pd.DataFrame([{
        "revolving_utilization" : revolving,
        "age"                   : age,
        "debt_ratio"            : debt_ratio,
        "monthly_income"        : monthly_income,
        "open_credit_lines"     : open_credit,
        "real_estate_loans"     : real_estate,
        "dependents"            : dependents,
        "has_any_delq"          : has_any_delq,
        "delq_severity_score"   : delq_score,
    }])

    prob = pipeline.predict_proba(input_df)[0][1]
    st.write(f"Default Probability: {prob:.1%}")
