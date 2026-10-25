# PPDA High-Value Tender Prediction

A machine learning app that predicts whether a public procurement tender is likely to be **high-value**, built on tender records from Uganda's Public Procurement and Disposal of Public Assets Authority (PPDA). It goes from raw data to a live web app, and it works two ways: check **one tender** at a time, or **upload a whole procurement period** and get every tender ranked by how likely it is to be high-value.

- 🔗 **Live app:** https://ppda-tender-value-status-prediction-app-j54tsnepkszjzdxcf2vura.streamlit.app/
- 🎥 **Demo video:** [Watch on Google Drive](https://drive.google.com/file/d/10fAYZ_qvv73-aYVp6UsdCIzIjZeHN4qn/view)
- 🤗 **Model on Hugging Face:** [JosephineNamyalo/Tender-Value-Status-Prediction-Model](https://huggingface.co/JosephineNamyalo/Tender-Value-Status-Prediction-Model)

---

## The problem

Public procurement bodies issue thousands of tenders a year, but only some are large or risky enough to need close oversight. Procurement teams have limited time and can't review every tender in the same depth, and entering tenders into a tool one by one is almost as slow as reviewing them by hand.

**The goal:** from what is known about a tender (the buyer, the procurement method, the title and description, the dates), predict whether it is likely to be high-value, so oversight goes where it matters most. This is a binary classification problem (Yes / No).

> **A rule I kept:** the tender's monetary value is only used to build the label. It is never an input to the model. Each tender is labelled "high-value" relative to its own buyer's tenders, not against one global cutoff (see [Target definition](#target-definition)).

---

## What the app does

The app has two tabs.

### 1. Single tender
Fill in the buyer, procurement method, status, title, description and the three dates (the form starts empty), then press **Predict**. The result appears beside the form, so everything fits on one screen. If anything is missing, the app tells you which fields to fill in. You get a Yes/No answer, the model's confidence, and a chart of the probabilities.

### 2. Batch: upload a file
Upload the list of tenders for a period, as a **CSV or Excel (.xlsx)** file. The app scores every tender and ranks them from most to least likely to be high-value, so reviewers can start at the top of the list.

- **Ranked results** with a probability bar for each tender and a Yes/No flag.
- **A cut-off slider** (50% by default). Lowering it flags more tenders. That catches more of the real high-value ones, at the cost of more false alarms.
- **Summary figures:** tenders scored, tenders flagged, and the share flagged.
- **Download** the full ranked results as a CSV.
- **Warnings** for tenders from buyers the model never saw in training, and for rows with missing or unreadable dates.
- Up to **20,000 rows** per upload. The raw PPDA export works as it is.

**Columns the file needs** (matched ignoring capital letters and spaces):

| Column | Example |
|---|---|
| `buyer_name` | Office of the Prime Minister |
| `tender_procurementmethod` | LIMITED |
| `tender_status` | active |
| `tender_title` | Supply of Fuel and Lubricants |
| `tender_description` | Supply and delivery of diesel, petrol and lubricants... |
| `date` | 09/06/2022 |
| `tender_tenderperiod_startdate` | 09/06/2022 |
| `tender_tenderperiod_enddate` | 07/10/2022 |

Dates are day/month/year. Any other columns, such as `tender_id`, are kept and shown in the results. If your file has a tender value column, it is ignored. The app also offers a small example file you can download and copy the layout from.

---

## Try it

These two examples were checked on the live app. They show the model's pattern. They are not real tenders.

The form starts empty, so type these in. All three dates are required. Type each one as year/month/day and press Enter.

**A "No"**

| Field | Value |
|---|---|
| Buyer | Uganda National Roads Authority |
| Method / Status | OPEN / complete |
| Title | Procurement of Office Supplies |
| Description | Supply and delivery of various office stationery and equipment for the financial year. |
| Date Published, Period Start, Period End | 2026/10/24, 2026/09/24, 2026/12/23 |

The model answers **No** with about 74% confidence when I ran it. The exact figure moves a little with the dates.

**A "Yes"**

| Field | Value |
|---|---|
| Buyer | Office of the Prime Minister |
| Method / Status | LIMITED / active |
| Title | Supply of Fuel and Lubricants |
| Description | Supply and delivery of diesel, petrol and lubricants for vehicles and generators for one financial year. |
| Date Published, Period Start, Period End | 2022/06/09, 2022/06/09, 2022/10/07 |

The model answers **Yes** with about 86% confidence.

---

## Dataset

- **Source:** real tender notices from the PPDA portal (Uganda).
- **Size:** 20,212 records and 20 raw columns, with no duplicate rows.
- **Key fields:** buyer, procurement method, title and description, tender value and currency, publication and bid-opening dates.

### Target definition

A tender is labelled **high-value (Yes)** if its value, converted to UGX, is at or above the **75th percentile of that buyer's own tenders**. A large tender for a small district office and a large tender for a national roads authority are therefore judged against their own baselines. About one tender in four (5,186 of 20,212) ends up labelled Yes.

```python
entity_thresholds = df.groupby("buyer_name")["tender_value_amount_ugx"].quantile(0.75)
df["high_value_tender"] = df.apply(
    lambda row: "Yes" if row["tender_value_amount_ugx"] >= row["entity_threshold"] else "No",
    axis=1
)
```

---

## How the data is prepared

1. **Clean:** fix inconsistent column names, drop a completely empty column, trim stray spaces in the text columns, and check for duplicates (there were none).
2. **Convert currency:** every tender value goes to UGX using fixed exchange rates.
3. **Build features:** how long the tender stays open, the publication year, month, quarter and day of the week, and each buyer's own threshold.
4. **Text:** the title and description are joined and turned into TF-IDF scores for the 5,000 most useful terms.
5. **Categories:** the buyer, procurement method and tender status are one-hot encoded.

All of this lives in one scikit-learn `Pipeline` (a custom `PPDA_Preprocessor` plus the classifier), saved as a single file. The same steps run at prediction time as at training time.

### A leakage bug I fixed
An earlier version of the preprocessor also fed the converted tender value into the model as a feature. The label is built from that same value, so the model could nearly read the answer off its input. I removed it and retrained. The numeric features are now just the duration and the date parts.

---

## What the data shows

- Raw tender values are heavily skewed. On a log scale they look roughly bell-shaped, which is why the threshold is set per buyer.
- Most tenders stay open for a few weeks up to about three months, with a smaller number running much longer.
- Procurement method carries real signal. Open tenders are the bulk of the data (roughly one in four is high-value). Limited tenders are the only method where high-value ones outnumber the rest.
- No numeric feature correlates strongly with the label. The best is duration at 0.08, and the raw tender value is only 0.01. The model has to lean on the buyer, the procurement method and the tender text.

---

## Models compared

Four models were trained and tested on the same 80/20 split (4,043 test tenders). All figures are for the Yes class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression (baseline) | 75.9% | 52.3% | 67.6% | 59.0% |
| **Random forest** | **83.3%** | 75.2% | 52.2% | **61.6%** |
| XGBoost | 81.9% | 75.4% | 43.7% | 55.3% |
| Neural network (MLP) | 81.6% | 73.4% | 44.5% | 55.4% |

- **Precision:** when the model says Yes, how often is it right?
- **Recall:** of the tenders that really are high-value, how many does it catch?

The **random forest** is the production model. It has the best accuracy and F1 of the four, its precision is within a fraction of XGBoost's, and it catches more of the high-value tenders. The saved pipeline is fitted on all 20,212 records, so the test-set scores above are the honest estimate of how it does on tenders it hasn't seen.

---

## Using it responsibly

- **It's a screening aid, not a decision-maker.** It ranks tenders so people know where to look first, and people still review the flagged ones.
- **It misses some.** At the 50% cut-off the model catches about half of the high-value tenders. Lowering the slider in batch mode catches more, with more false alarms.
- **It learns from past data.** Buyers it never saw in training are scored from the method, wording and dates only, and the app says so.

---

## Architecture

```
Raw PPDA CSV → cleaning and features → trained pipeline (.pkl)
                                              │
                          ┌───────────────────┴───────────────────┐
                          │                                       │
                 Hugging Face Hub                          GitHub repo
              (hosts the model file)              (app.py, requirements.txt)
                          │                                       │
                          └───────────────────┬───────────────────┘
                                              │
                                       Streamlit Cloud
                                       (the live app)
```

- **Model hosting:** the saved pipeline is about 270 MB, which is too big for GitHub, so it lives on Hugging Face and `app.py` downloads it when it starts.
- **App hosting:** Streamlit Cloud deploys straight from this repository.
- **One codebase, two places:** `app.py` uses a local `ppda_full_pipeline.pkl` if it finds one and otherwise downloads the model, so it runs the same way on a laptop and in the cloud.
- **Updating the model:** upload the new file to the Hugging Face repo under the same filename, then reboot the app from Manage app and clear its cache. The app keeps the old model in memory until you do. The Hugging Face repo must stay public, with gated access switched off.

---

## Repository structure

```
├── app.py                              # Streamlit app: single tender + batch upload
├── requirements.txt                    # Python dependencies
├── .gitignore                          # keeps the large .pkl out of git
├── .streamlit/
│   └── config.toml                     # app theme
├── PPDA_HighValueTender_Model.ipynb    # full notebook: data to model
└── README.md
```

The trained `ppda_full_pipeline.pkl` is not in this repository. See [Architecture](#architecture).

---

## Running it locally

```bash
git clone https://github.com/Josephine-Analytics/ppda-tender-value-status-prediction-app.git
cd ppda-tender-value-status-prediction-app
pip install -r requirements.txt
streamlit run app.py
```

Put `ppda_full_pipeline.pkl` next to `app.py` to skip the download. Reading `.xlsx` uploads needs `openpyxl`, which is already in `requirements.txt`.

---

## Tech stack

- **Data:** pandas, NumPy
- **Models:** scikit-learn (logistic regression, random forest, MLP), XGBoost
- **Text features:** TF-IDF (scikit-learn)
- **Charts:** Matplotlib, Seaborn
- **App:** Streamlit, with `openpyxl` for Excel uploads
- **Hosting:** Hugging Face Hub, GitHub, Streamlit Cloud

---

## What's next

- Show *why* a tender was flagged, not just that it was.
- Retrain automatically on fresh PPDA data.
- Let reviewers confirm or correct each flag, and learn from their feedback.
- Serve the model through an API, so a procurement or financial management system can call it directly.

---

## Author

**Josephine Namyalo**
Refactory Data Science and Analytics Capstone Project, 2026
