# -*- coding: utf-8 -*-
"""
PPDA High-Value Tender Prediction — Streamlit App

Run locally:
    streamlit run app.py
    (place ppda_full_pipeline.pkl in this same folder to skip the download)

Run on Streamlit Cloud:
    Push this file + requirements.txt to GitHub, deploy via share.streamlit.io.
    If ppda_full_pipeline.pkl is NOT committed to the repo, the app will
    automatically download it from the Hugging Face Hub repo configured below.
"""

import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILENAME = "ppda_full_pipeline.pkl"

# TODO: set this to your actual Hugging Face Hub repo id, e.g. "Josephine-Analytics/ppda-tender-model"
# Only used if the .pkl is not found locally / not committed to the GitHub repo.
HUGGING_FACE_REPO_ID = "Josephine-Analytics/ppda-tender-value-status-prediction"

st.set_page_config(page_title="PPDA Tender Value Prediction", page_icon="💰", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling — Uganda-inspired accent palette (black / gold / red)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ppda-gold: #FCD116;
        --ppda-red: #D90000;
        --ppda-black: #111111;
    }
    .stApp header {background: transparent;}
    div[data-testid="stForm"] {
        border: 1px solid rgba(217, 0, 0, 0.15);
        border-radius: 14px;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        background: #FFFFFF;
    }
    .ppda-hero {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(252,209,22,0.25), rgba(217,0,0,0.08));
        border: 1px solid rgba(217,0,0,0.15);
        margin-bottom: 1.25rem;
    }
    .ppda-hero h1 {
        margin: 0;
        font-size: clamp(1.1rem, 2.6vw, 1.8rem);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .ppda-hero p {
        margin: 0.25rem 0 0 0;
        opacity: 0.85;
    }
    div.stButton > button, button[kind="formSubmit"] {
        background: var(--ppda-red);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button:hover, button[kind="formSubmit"]:hover {
        background: #b30000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Preprocessor class — MUST match the class used when the pipeline was
# trained/pickled (same attributes, same transform logic), and must be
# defined here in app.py's own module namespace so joblib can unpickle it.
# ---------------------------------------------------------------------------
class PPDA_Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_vectorizer, conversion_rates, cat_cols_to_encode):
        self.tfidf = tfidf_vectorizer
        self.conversion_rates = conversion_rates
        self.cat_cols = cat_cols_to_encode
        self.numeric_feature_names = [
            "tender_value_amount_ugx",
            "tender_duration",
            "year",
            "month",
            "quarter",
            "day_of_week",
        ]
        self.ohe_feature_names = None
        self.tfidf_feature_names = None

    def fit(self, X, y=None):
        X_text_combined = X["tender_title"].astype(str) + " " + X["tender_description"].astype(str)
        self.tfidf.fit(X_text_combined)
        self.tfidf_feature_names = [f"tfidf_{i}" for i in range(self.tfidf.idf_.shape[0])]

        temp_ohe_df = pd.get_dummies(X[self.cat_cols], prefix=self.cat_cols)
        self.ohe_feature_names = list(temp_ohe_df.columns)
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. tender_value_amount_ugx
        X_copy["tender_value_amount"] = pd.to_numeric(
            X_copy["tender_value_amount"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        ).fillna(0).astype("Int64")

        X_copy["tender_value_amount_ugx"] = X_copy.apply(
            lambda row: row["tender_value_amount"] * self.conversion_rates.get(row["tender_value_currency"], 1),
            axis=1,
        )
        X_copy["tender_value_amount_ugx"] = pd.to_numeric(X_copy["tender_value_amount_ugx"], errors="coerce")

        # 2. tender_duration
        for col in ["tender_tenderperiod_enddate", "tender_tenderperiod_startdate"]:
            X_copy[col] = pd.to_datetime(X_copy[col], format="%d/%m/%Y", errors="coerce")
        X_copy["tender_duration"] = (
            X_copy["tender_tenderperiod_enddate"] - X_copy["tender_tenderperiod_startdate"]
        ).dt.days.fillna(0)

        # 3. Date features
        X_copy["date"] = pd.to_datetime(X_copy["date"], format="%d/%m/%Y", errors="coerce")
        X_copy["year"] = X_copy["date"].dt.year.fillna(0).astype(int)
        X_copy["month"] = X_copy["date"].dt.month.fillna(0).astype(int)
        X_copy["day"] = X_copy["date"].dt.day.fillna(0).astype(int)
        X_copy["day_of_week"] = X_copy["date"].dt.dayofweek.fillna(0).astype(int)
        X_copy["quarter"] = X_copy["date"].dt.quarter.fillna(0).astype(int)

        # 4. Text features (TF-IDF)
        X_text_combined = X_copy["tender_title"].astype(str) + " " + X_copy["tender_description"].astype(str)
        text_features_sparse = self.tfidf.transform(X_text_combined)
        text_features_df = pd.DataFrame(
            text_features_sparse.toarray(), columns=self.tfidf_feature_names, index=X_copy.index
        )

        # 5. One-hot encoding
        ohe_features_df = pd.get_dummies(X_copy[self.cat_cols], prefix=self.cat_cols)
        if self.ohe_feature_names:
            ohe_features_df = ohe_features_df.reindex(columns=self.ohe_feature_names, fill_value=0)

        numeric_features = X_copy[self.numeric_feature_names]

        final_features = pd.concat([numeric_features, ohe_features_df, text_features_df], axis=1)
        final_features = final_features.fillna(0)
        return final_features


# ---------------------------------------------------------------------------
# Model loading — local file first, Hugging Face Hub fallback
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model_path = MODEL_FILENAME

    if not os.path.exists(model_path):
        st.info(f"Model file '{MODEL_FILENAME}' not found locally. Downloading from Hugging Face Hub...")
        try:
            model_path = hf_hub_download(repo_id=HUGGING_FACE_REPO_ID, filename=MODEL_FILENAME)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(
                f"Failed to download model from Hugging Face Hub ('{HUGGING_FACE_REPO_ID}'): {e}\n\n"
                "Check that HUGGING_FACE_REPO_ID in app.py is correct and that the repo/file are public "
                "(or that HF auth is configured)."
            )
            st.stop()

    if os.path.getsize(model_path) < 100_000:
        st.error(
            f"Model file at '{model_path}' is only {os.path.getsize(model_path)} bytes — "
            "this looks like a corrupted download or a Git LFS pointer file, not the real model."
        )
        st.stop()

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the pipeline: {e}")
        st.stop()


loaded_pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
unique_procurement_methods = ["OPEN", "LIMITED", "SELECTIVE", "DIRECT"]
unique_tender_statuses = ["complete", "active"]
currencies = ["UGX", "USD", "KES", "EUR", "GBP", "JPY"]

st.markdown(
    """
    <div class="ppda-hero">
        <h1>💰 PPDA Tender Value Status Prediction</h1>
        <p>Enter the tender details below to predict if a tender is <b>High-Value</b> or <b>Normal-Value</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Adjust these parameters to customize your prediction.")
    buyer_name = st.text_input(
        "Buyer Name", "Uganda National Roads Authority",
        help="The procuring entity. High-value thresholds are calculated per entity."
    )
    tender_procurementmethod = st.selectbox(
        "Procurement Method", unique_procurement_methods,
        help="OPEN tenders are typically competitive; DIRECT/SELECTIVE are more restricted."
    )
    tender_status = st.selectbox("Tender Status", unique_tender_statuses)

with st.form("tender_prediction_form"):
    st.header("📝 Tender Details")

    col1, col2 = st.columns(2)
    with col1:
        tender_title = st.text_input(
            "Tender Title", "Procurement of Office Supplies",
            help="Short title as it would appear on the tender notice."
        )
    with col2:
        tender_description = st.text_area(
            "Tender Description",
            "Supply and delivery of various office stationery and equipment for the financial year.",
            help="More detail helps the model's text features pick up relevant signals."
        )

    st.subheader("Value")
    col3, col4 = st.columns(2)
    with col3:
        tender_value_amount = st.number_input(
            "Tender Value Amount (in selected currency)",
            min_value=0.0, value=10000000.0, step=1000000.0,
            help="Enter the raw amount in the currency selected on the right; it's converted to UGX internally."
        )
    with col4:
        tender_value_currency = st.selectbox("Tender Value Currency", currencies)

    if tender_value_amount <= 0:
        st.warning("⚠️ Tender value is 0 — the prediction will likely default to Normal-Value regardless of other inputs.")

    st.subheader("🗓️ Dates")
    with st.expander("View/Edit Tender Dates", expanded=False):
        today = datetime.date.today()
        tender_date = st.date_input("Date", today, help="Publication date — used to derive year/month/quarter features.")
        tender_period_start_date = st.date_input("Tender Period Start Date", today - datetime.timedelta(days=30))
        tender_period_end_date = st.date_input(
            "Tender Period End Date", today + datetime.timedelta(days=60),
            help="Must be after the start date — duration is a key model feature."
        )

    date_error = tender_period_end_date <= tender_period_start_date
    if date_error:
        st.error("🚫 Tender Period End Date must be after the Start Date.")

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict Tender Value Status", disabled=date_error)

    if submitted:
        input_data = pd.DataFrame(
            {
                "date": [tender_date.strftime("%d/%m/%Y")],
                "buyer_name": [buyer_name],
                "tender_procurementmethod": [tender_procurementmethod],
                "tender_title": [tender_title],
                "tender_description": [tender_description],
                "tender_status": [tender_status],
                "tender_value_amount": [float(tender_value_amount)],
                "tender_value_currency": [tender_value_currency],
                "tender_tenderperiod_enddate": [tender_period_end_date.strftime("%d/%m/%Y")],
                "tender_tenderperiod_startdate": [tender_period_start_date.strftime("%d/%m/%Y")],
                "link": ["dummy_link"],
                "id": ["dummy_id"],
                "tag": ["active"],
                "ocid": ["dummy_ocid"],
                "buyer_id": [9999],
                "tender_id": [99999],
                "tender_bidopening_date": [tender_period_end_date.strftime("%d/%m/%Y")],
                "tender_bidopening_address_streetaddress": ["dummy address"],
                "tender_bidopening_location_description": ["dummy location"],
            }
        )

        try:
            prediction_numerical = loaded_pipeline.predict(input_data)[0]
            prediction_label = "Yes" if prediction_numerical == 1 else "No"

            # Confidence score, if the underlying model supports predict_proba
            proba = None
            try:
                proba = loaded_pipeline.predict_proba(input_data)[0]  # [P(No), P(Yes)]
            except Exception:
                pass

            st.markdown("---")
            result_col, chart_col = st.columns([1, 1])

            with result_col:
                if prediction_label == "Yes":
                    st.success(f"### 🟢 Predicted: **High-Value**")
                    st.info("Consider strategic allocation of resources and closer oversight for this tender. 📈")
                else:
                    st.info(f"### 🔵 Predicted: **Normal-Value**")
                    st.caption("Standard procurement handling is likely sufficient. 📊")

                if proba is not None:
                    confidence = proba[1] if prediction_label == "Yes" else proba[0]
                    st.metric("Model Confidence", f"{confidence * 100:.1f}%")
                    st.progress(float(confidence))

            with chart_col:
                if proba is not None:
                    proba_df = pd.DataFrame(
                        {"Outcome": ["Normal-Value (No)", "High-Value (Yes)"], "Probability": [proba[0], proba[1]]}
                    ).set_index("Outcome")
                    st.caption("Prediction probability breakdown")
                    st.bar_chart(proba_df, height=220)
                else:
                    st.caption("This model doesn't expose probability scores.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
