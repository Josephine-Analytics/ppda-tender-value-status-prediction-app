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

import io
import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from huggingface_hub import hf_hub_download


# Configuration
MODEL_FILENAME = "ppda_full_pipeline.pkl"

# Hugging Face repo that holds ppda_full_pipeline.pkl (used when the file is not found locally)
# Only used if the .pkl is not found locally / not committed to the GitHub repo.
HUGGING_FACE_REPO_ID = "JosephineNamyalo/Tender-Value-Status-Prediction-Model"

st.set_page_config(page_title="PPDA High-Value Tender Prediction", page_icon="💰", layout="wide")

# Custom styling — Uganda-inspired accent palette (black / gold / red)
st.markdown(
    """
    <style>
    :root {
        --ppda-gold: #FCD116;
        --ppda-red: #D90000;
        --ppda-black: #111111;
        --ppda-bg: #F7F4EC;
        --ppda-panel: #FFFFFF;
        --ppda-text: #1F1F1F;
    }

    /* Force light background + dark text everywhere, regardless of browser theme override */
    .stApp {
        background-color: var(--ppda-bg) !important;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div,
    h1, h2, h3, h4, h5, h6 {
        color: var(--ppda-text) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #EFEADC !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--ppda-text) !important;
    }

    /* Input fields — force readable light backgrounds */
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: var(--ppda-panel) !important;
        color: var(--ppda-text) !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
    }

    .stApp header {background: transparent;}
    div[data-testid="stForm"] {
        border: 1px solid rgba(217, 0, 0, 0.15);
        border-radius: 14px;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        background: var(--ppda-panel) !important;
    }
    .ppda-hero {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(252,209,22,0.35), rgba(217,0,0,0.12));
        border: 1px solid rgba(217,0,0,0.15);
        margin-bottom: 1.25rem;
    }
    .ppda-hero h1, .ppda-hero p {
        color: var(--ppda-black) !important;
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
    div.stButton > button,
    button[kind="formSubmit"],
    div[data-testid="stFormSubmitButton"] button,
    button[data-testid="stFormSubmitButton"],
    div[data-testid="stForm"] button {
        background-color: var(--ppda-red) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--ppda-red) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    div.stButton > button *,
    button[kind="formSubmit"] *,
    div[data-testid="stFormSubmitButton"] button *,
    button[data-testid="stFormSubmitButton"] *,
    div[data-testid="stForm"] button * {
        color: #FFFFFF !important;
    }
    div.stButton > button:hover,
    button[kind="formSubmit"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    button[data-testid="stFormSubmitButton"]:hover,
    div[data-testid="stForm"] button:hover {
        background-color: #b30000 !important;
        border: 1px solid #b30000 !important;
        color: #FFFFFF !important;
    }
    div.stButton > button:disabled,
    div[data-testid="stFormSubmitButton"] button:disabled,
    button[data-testid="stFormSubmitButton"]:disabled {
        background-color: #d9a3a3 !important;
        border: 1px solid #d9a3a3 !important;
        color: #FFFFFF !important;
        opacity: 0.7 !important;
    }
    div[data-testid="stDownloadButton"] button {
        background-color: var(--ppda-red) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--ppda-red) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDownloadButton"] button * {
        color: #FFFFFF !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #b30000 !important;
        border: 1px solid #b30000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Preprocessor class — MUST match the class used when the pipeline was
# trained/pickled (same attributes, same transform logic), and must be
# defined here in app.py's own module namespace so joblib can unpickle it.
class PPDA_Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_vectorizer, conversion_rates, cat_cols_to_encode):
        self.tfidf = tfidf_vectorizer
        self.conversion_rates = conversion_rates
        self.cat_cols = cat_cols_to_encode
        self.numeric_feature_names = [
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


# Model loading — local file first, Hugging Face Hub fallback
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
# Batch scoring helpers: score a whole file of tenders instead of one at a time
# ---------------------------------------------------------------------------
TEXT_COLUMNS = [
    "buyer_name",
    "tender_procurementmethod",
    "tender_status",
    "tender_title",
    "tender_description",
]
DATE_COLUMNS = [
    "date",
    "tender_tenderperiod_startdate",
    "tender_tenderperiod_enddate",
]
REQUIRED_COLUMNS = TEXT_COLUMNS + DATE_COLUMNS

MAX_BATCH_ROWS = 20000   # keeps one upload from using up the app's memory
CHUNK_SIZE = 250         # tenders scored per step (the text features are large)
PROB_COL = "Probability of high value (%)"
FLAG_COL = "High-value tender?"


def normalise_column_name(col):
    """Same renaming the notebook applies to the raw PPDA export."""
    col = str(col)
    if col == "_link":
        return "link"
    return col.strip().lower().replace(" ", "_").replace("-", "_").replace("__", "_")


def read_uploaded_file(uploaded):
    """Read a CSV or Excel upload into a DataFrame."""
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    try:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin1")


def prepare_batch(raw):
    """Clean an uploaded table into what the model expects.

    Returns (tidy table, model input, missing column names).
    The tender value is never read: the model must not see it.
    """
    df = raw.copy()
    df.columns = [normalise_column_name(c) for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return df, None, missing

    model_input = pd.DataFrame(index=df.index)
    for col in TEXT_COLUMNS:
        model_input[col] = df[col].fillna("").astype(str).str.strip()
    for col in DATE_COLUMNS:
        parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        model_input[col] = parsed.dt.strftime("%d/%m/%Y")
    # the preprocessor still looks for these two columns, then discards them
    model_input["tender_value_amount"] = 0.0
    model_input["tender_value_currency"] = "UGX"
    return df, model_input, []


def known_buyers(pipeline):
    """Buyers the model saw during training (None if it can't be read)."""
    try:
        names = pipeline.named_steps["preprocess"].ohe_feature_names
        return {n[len("buyer_name_"):] for n in names if n.startswith("buyer_name_")}
    except Exception:
        return None


def score_batch(pipeline, model_input, on_progress=None):
    """Probability of 'Yes' for every row, scored in small chunks."""
    probs = []
    total = len(model_input)
    for start in range(0, total, CHUNK_SIZE):
        chunk = model_input.iloc[start:start + CHUNK_SIZE].reset_index(drop=True)
        probs.extend(pipeline.predict_proba(chunk)[:, 1])
        if on_progress:
            on_progress(min(start + CHUNK_SIZE, total) / total)
    return np.array(probs)


def build_results(df, probs, threshold_pct):
    """Add the probability and the Yes/No flag, ranked from most to least likely."""
    probs = np.asarray(probs)
    results = df.copy()
    results[PROB_COL] = (probs * 100).round(1)
    results[FLAG_COL] = np.where(probs * 100 >= threshold_pct, "Yes", "No")
    results = results.sort_values(PROB_COL, ascending=False, kind="stable").reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    return results


def template_csv():
    """A small example file people can copy the layout from."""
    example = pd.DataFrame(
        [
            {
                "tender_id": 1001,
                "buyer_name": "Office of the Prime Minister",
                "tender_procurementmethod": "LIMITED",
                "tender_status": "active",
                "tender_title": "Supply of Fuel and Lubricants",
                "tender_description": "Supply and delivery of diesel, petrol and lubricants for vehicles and generators for one financial year.",
                "date": "09/06/2022",
                "tender_tenderperiod_startdate": "09/06/2022",
                "tender_tenderperiod_enddate": "07/10/2022",
            },
            {
                "tender_id": 1002,
                "buyer_name": "Uganda National Roads Authority",
                "tender_procurementmethod": "OPEN",
                "tender_status": "complete",
                "tender_title": "Procurement of Office Supplies",
                "tender_description": "Supply and delivery of various office stationery and equipment for the financial year.",
                "date": "01/03/2023",
                "tender_tenderperiod_startdate": "15/03/2023",
                "tender_tenderperiod_enddate": "01/05/2023",
            },
        ]
    )
    return example.to_csv(index=False).encode("utf-8-sig")


# Streamlit UI
unique_procurement_methods = ["OPEN", "LIMITED", "SELECTIVE", "DIRECT"]
unique_tender_statuses = ["complete", "active"]

st.markdown(
    """
    <div class="ppda-hero">
        <h1>💰 PPDA High-Value Tender Prediction</h1>
        <p>Check one tender, or upload a whole procurement period and rank every tender by how likely it is to be <b>high-value</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Adjust these parameters to customize your prediction.")
    st.caption("These three fields are used by the Single tender tab.")
    buyer_name = st.text_input(
        "Buyer Name", "Uganda National Roads Authority",
        help="The procuring entity. High-value thresholds are calculated per entity."
    )
    tender_procurementmethod = st.selectbox(
        "Procurement Method", unique_procurement_methods,
        help="OPEN tenders are typically competitive; DIRECT/SELECTIVE are more restricted."
    )
    tender_status = st.selectbox("Tender Status", unique_tender_statuses)

tab_single, tab_batch = st.tabs(["🧾 Single tender", "📂 Batch: upload a file"])

with tab_single:
    with st.form("tender_prediction_form"):
        st.header("📝 Tender Details")

        col1, col2 = st.columns(2)
        with col1:
            tender_title = st.text_area(
                "Tender Title", "Procurement of Office Supplies",
                height=100,
                help="Short title as it would appear on the tender notice."
            )
            st.caption(
                "💡 **Tip:** Good titles name the works/goods/services and, where "
                "relevant, the location or scope — e.g. *'Rehabilitation of Regional "
                "Highway, Northern Uganda'* rather than just *'Road Works'*. Clearer "
                "titles help the model's text features pick up stronger signals."
            )
        with col2:
            tender_description = st.text_area(
                "Tender Description",
                "Supply and delivery of various office stationery and equipment for the financial year.",
                height=180,
                help="More detail helps the model's text features pick up relevant signals."
            )

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
        submitted = st.form_submit_button("🚀 Predict High-Value Tender Status", disabled=date_error)

        if submitted:
            input_data = pd.DataFrame(
                {
                    "date": [tender_date.strftime("%d/%m/%Y")],
                    "buyer_name": [buyer_name],
                    "tender_procurementmethod": [tender_procurementmethod],
                    "tender_title": [tender_title],
                    "tender_description": [tender_description],
                    "tender_status": [tender_status],
                    # Not used as a model feature (removed to fix target leakage) —
                    # kept here only because the preprocessor's transform() still
                    # references these columns internally before discarding them.
                    "tender_value_amount": [0.0],
                    "tender_value_currency": ["UGX"],
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
                    st.markdown("#### High-Value Tender")
                    if prediction_label == "Yes":
                        st.success("### 🟢 High-Value Tender: **Yes**")
                        st.info("Consider strategic allocation of resources and closer oversight for this tender. 📈")
                    else:
                        st.info("### 🔵 High-Value Tender: **No**")
                        st.caption("Standard procurement handling is likely sufficient. 📊")

                    if proba is not None:
                        confidence = proba[1] if prediction_label == "Yes" else proba[0]
                        st.metric("Model Confidence", f"{confidence * 100:.1f}%")
                        st.progress(float(confidence))

                with chart_col:
                    if proba is not None:
                        proba_df = pd.DataFrame(
                            {"Outcome": ["High-Value Tender: No", "High-Value Tender: Yes"], "Probability": [proba[0], proba[1]]}
                        ).set_index("Outcome")
                        st.caption("Prediction probability breakdown")
                        st.bar_chart(proba_df, height=220)
                    else:
                        st.caption("This model doesn't expose probability scores.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")



with tab_batch:
    st.header("📂 Score a whole procurement period")
    st.markdown(
        "Upload the list of tenders submitted in a period, as a CSV or Excel file. "
        "The model scores every tender and ranks them from most to least likely to be high-value, "
        "so reviewers can start at the top of the list. It is a screening aid, not a decision: "
        "people still review the flagged tenders. Any tender value in your file is ignored, "
        "because the model never uses it."
    )

    with st.expander("Which columns does the file need?"):
        st.markdown(
            "Your file needs these columns (names are matched ignoring capitals and spaces, "
            "so the raw PPDA export works as it is):\n\n"
            "- `buyer_name`, `tender_procurementmethod`, `tender_status`\n"
            "- `tender_title`, `tender_description`\n"
            "- `date`, `tender_tenderperiod_startdate`, `tender_tenderperiod_enddate` "
            "(day/month/year, for example 09/06/2022)\n\n"
            "Extra columns, such as `tender_id`, are kept and shown in the results."
        )
        st.download_button(
            "Download an example file (CSV)",
            data=template_csv(),
            file_name="ppda_tender_example.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Tender list (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded is not None:
        key = f"{uploaded.name}:{uploaded.size}"

        # score once per file; moving the slider below must not re-score everything
        if st.session_state.get("batch_key") != key:
            st.session_state.pop("batch_key", None)
            raw = None
            try:
                raw = read_uploaded_file(uploaded)
            except Exception as e:
                st.error(f"I couldn't read that file: {e}")

            if raw is not None:
                if len(raw) == 0:
                    st.error("The file has no rows.")
                elif len(raw) > MAX_BATCH_ROWS:
                    st.error(f"The file has {len(raw):,} rows. Please upload {MAX_BATCH_ROWS:,} or fewer at a time.")
                else:
                    df, model_input, missing = prepare_batch(raw)
                    if missing:
                        st.error("These required columns are missing: " + ", ".join(f"`{c}`" for c in missing))
                    else:
                        bar = st.progress(0.0, text="Scoring tenders...")
                        probs = score_batch(
                            loaded_pipeline, model_input,
                            on_progress=lambda f: bar.progress(f, text=f"Scoring tenders... {f:.0%}"),
                        )
                        bar.empty()

                        buyers = known_buyers(loaded_pipeline)
                        unseen = int((~model_input["buyer_name"].isin(buyers)).sum()) if buyers is not None else 0
                        bad_dates = int(model_input[DATE_COLUMNS].isna().any(axis=1).sum())
                        st.session_state.update(
                            batch_key=key, batch_df=df, batch_probs=probs,
                            batch_unseen=unseen, batch_bad_dates=bad_dates,
                        )

        if st.session_state.get("batch_key") == key:
            df = st.session_state["batch_df"]
            probs = st.session_state["batch_probs"]

            if st.session_state["batch_unseen"]:
                st.warning(
                    f"{st.session_state['batch_unseen']} tender(s) come from buyers the model did not see in training. "
                    "Their scores rely on the procurement method, wording and dates instead."
                )
            if st.session_state["batch_bad_dates"]:
                st.warning(
                    f"{st.session_state['batch_bad_dates']} tender(s) have a missing or unreadable date, "
                    "so their duration and timing were treated as unknown."
                )

            threshold = st.slider(
                "Flag a tender as high-value when its probability is at least (%)",
                min_value=10, max_value=90, value=50, step=5,
                help="50% is the normal cut-off. Lower it to catch more high-value tenders, "
                     "at the cost of more false alarms.",
            )
            results = build_results(df, probs, threshold)
            flagged = int((results[FLAG_COL] == "Yes").sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Tenders scored", f"{len(results):,}")
            m2.metric("Flagged as high-value", f"{flagged:,}")
            m3.metric("Share flagged", f"{flagged / len(results):.0%}")

            st.caption("Ranked from most to least likely to be high-value. Start reviewing from the top.")
            show_cols = [c for c in ["Rank", "tender_id", "id", "buyer_name", "tender_title",
                                     "tender_procurementmethod", "tender_status", "date"] if c in results.columns]
            show_cols += [PROB_COL, FLAG_COL]
            st.dataframe(
                results[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    PROB_COL: st.column_config.ProgressColumn(
                        PROB_COL, format="%.0f%%", min_value=0, max_value=100
                    ),
                },
            )

            st.download_button(
                "Download the full results (CSV)",
                data=results.to_csv(index=False).encode("utf-8-sig"),
                file_name="ppda_ranked_tenders.csv",
                mime="text/csv",
            )
