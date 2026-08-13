# PPDA High-Value Tender Prediction

A machine learning system that predicts whether a public procurement tender is likely to be **High-Value**, using Uganda's Public Procurement and Disposal of Public Assets Authority (PPDA) tender records. Built as an end-to-end pipeline — from raw data to a live, interactive web app.

🔗 **Live App:** https://ppda-tender-value-status-prediction-app-j54tsnepkszjzdxcf2vura.streamlit.app/
🎥 **Video Demo:** (https://drive.google.com/file/d/10fAYZ_qvv73-aYVp6UsdCIzIjZeHN4qn/view)
🤗 **Model on Hugging Face Hub:** [huggingface.co/JosephineNamyalo/Tender-Value-Status-Prediction-Model](https://huggingface.co/JosephineNamyalo/Tender-Value-Status-Prediction-Model)

---

## Problem Statement

Public procurement entities issue thousands of tenders every year, but only a fraction are high-value or high-risk enough to need close oversight. Procurement teams have limited time and can't review every tender in equal depth.

**ML Goal:** predict, from a tender's basic characteristics (buyer, procurement method, title, description, dates), whether it is likely to be a **High-Value** tender — a binary classification problem — so oversight resources can be prioritized before the tender is awarded.

> **Note on the target:** the tender's monetary value is used *only* to construct the target label, not as a model input feature. Each tender is labeled "high-value" relative to its own buyer's typical tender size (see [Target Definition](#target-definition) below), not a global cutoff.

---

## Dataset

- **Source:** Uganda's Public Procurement and Disposal of Public Assets Authority (PPDA) — real, publicly-issued tender notices.
- **Size:** 20,212 records, 20 raw columns, 0 duplicate rows.
- **Key fields:** buyer entity, procurement method, tender title & description, tender value & currency, publication and bid-opening dates.

---

## Target Definition

A tender is labeled **High-Value = Yes** if its converted value (in UGX) is at or above the **75th percentile of tender values for that specific buyer**, rather than a single global threshold. This means "high-value" is relative to what's unusually large *for that entity* — a large tender for a small district office and a large tender for a national roads authority are judged against their own respective baselines.

```python
entity_thresholds = df.groupby("buyer_name")["tender_value_amount_ugx"].quantile(0.75)
df["high_value_tender"] = df.apply(
    lambda row: "Yes" if row["tender_value_amount_ugx"] >= row["entity_threshold"] else "No",
    axis=1
)
```

---

## Data Pipeline

1. **Clean** — fixed inconsistent column names, dropped a fully-empty column, standardized text fields, removed duplicates.
2. **Standardize currency** — converted every tender value into a single common currency (UGX) using fixed exchange rates.
3. **Engineer features** — tender duration (days between tender period start/end), publication year/month/quarter/day-of-week, per-entity value threshold.
4. **Vectorize text** — TF-IDF on combined tender title + description.
5. **Encode categoricals** — one-hot encoding for buyer name, procurement method, and tender status.

The full pipeline (cleaning → feature engineering → model) is wrapped in a single scikit-learn `Pipeline` object (`PPDA_Preprocessor` + classifier) and persisted as one `.pkl` file, so the exact same transformations run identically at inference time as at training time.

## Exploratory Data Analysis

- **Tender value distribution** is heavily right-skewed; a log transform reveals an approximately normal underlying distribution — confirming why a per-entity percentile threshold (rather than one global cutoff) was the right approach for labeling.
- **Tender duration** is also right-skewed — most tenders complete within roughly 20–90 days, with a smaller number extending much longer.
- **Correlation analysis** shows that, individually, the numeric/date features (year, month, quarter, day-of-week, duration) have very weak linear correlation with the target — including the raw tender value itself (0.01), since the label is entity-relative rather than absolute. This confirms the model needs buyer identity, procurement method, and tender text (via TF-IDF) to predict well, not numeric features alone.

---

## Models Compared

Four models were trained and evaluated on the same train/test split and feature set:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 75.9% | 52.3% | 67.6% | 58.9% |
| **Random Forest** | **83.3%** | **75.1%** | 52.1% | **61.6%** |
| XGBoost | 81.9% | 75.2% | 44.2% | 55.7% |
| Neural Network (MLPClassifier) | 81.6% | 73.4% | 44.5% | 55.4% |

**Random Forest** was selected as the production model — the best balance of accuracy, precision, and F1-score among the four.

- **Precision** answers: "When the model predicts High-Value, how often is it right?"
- **Recall** answers: "Of all the actual High-Value tenders, how many did the model catch?"

---

## Deployment Architecture

```
Raw PPDA CSV → Cleaning & Feature Engineering → Trained Pipeline (.pkl)
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    │                                       │
                          Hugging Face Hub                            GitHub Repo
                       (hosts the .pkl model file)              (app.py, requirements.txt)
                                    │                                       │
                                    └───────────────┬───────────────────────┘
                                                     │
                                            Streamlit Cloud
                                          (live prediction app)
```

- **Model hosting:** Hugging Face Hub (the trained pipeline exceeds typical GitHub file-size practices, so it's hosted separately and downloaded at runtime).
- **App hosting:** Streamlit Cloud, deployed directly from this GitHub repository.
- **Fallback logic:** `app.py` checks for the model file locally first (for fast local development) and falls back to downloading it from Hugging Face Hub if not found — so the same code runs both locally and in the cloud.

---

## Repository Structure

```
├── app.py                    # Streamlit application (inference + UI)
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes the large .pkl from git tracking
├── .streamlit/
│   └── config.toml           # App theme configuration
├── PPDA_HighValueTender_Model.ipynb   # Full training notebook (data → model)
└── README.md
```

*(The trained `ppda_full_pipeline.pkl` is not committed to this repository — see [Deployment Architecture](#deployment-architecture).)*

---

## Running Locally

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Optionally, place `ppda_full_pipeline.pkl` in the same folder as `app.py` to skip the Hugging Face download step.

---

## Tech Stack

- **Data processing:** pandas, NumPy
- **Modeling:** scikit-learn (Logistic Regression, Random Forest, MLPClassifier), XGBoost
- **NLP features:** TF-IDF (scikit-learn)
- **Visualization:** Matplotlib, Seaborn
- **App framework:** Streamlit
- **Model hosting:** Hugging Face Hub
- **Deployment:** Streamlit Cloud, GitHub

---

## Author

**Josephine Namyalo**
Refactory Data Science And Analytics Capstone Project — 2026
