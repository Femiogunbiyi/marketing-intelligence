# 📊 Marketing Intelligence & ML Analytics Pipeline

![CI Status](https://github.com)
![Python](https://shields.io)
![Streamlit](https://shields.io)
![Scikit-Learn](https://shields.io)

An end-to-end Python & Machine Learning marketing analytics engine and interactive Streamlit dashboard. Designed to ingest raw ad-platform and payment-gateway exports (Meta Ads, Google Ads, TikTok, Paystack, Shopify), clean messy currency/text data, compute weighted marketing KPIs, segment customers via K-Means clustering, flag underperforming campaigns, and export executive reports.

> **Live Interactive Demo:** [Launch Dashboard on Streamlit Cloud](https://marketing-intelligence-vhz62q8hyxvwiqdx5mhzuu.streamlit.app/) *(Replace `#` with your `share.streamlit.io` URL after deploying)*

---

## ✨ Key Capabilities

* **Automated Ingestion & Platform Mapping (`data_loader.py`, `column_mapper.py`, `cleaning.py`):**
  * Normalizes varied platform headers (`Amount Spent (NGN)`, `Purchase Conversion Value`, `Customer Email`, `Link Clicks`) into a unified internal schema.
  * Strips currency symbols (`₦`, `$`, `£`, `€`), parses dates, deduplicates rows, and handles missing values automatically.
* **Weighted Marketing KPIs (`metrics.py` & `campaign_analysis.py`):**
  * Computes row-level and properly weighted aggregate **CTR, CPC, CPM, Conversion Rate, CPA, ROAS, ROI, Net Profit, and AOV** with built-in divide-by-zero protection.
* **RFM & Machine Learning Customer Segmentation (`customer_analysis.py` & `segmentation.py`):**
  * Calculates **Recency, Frequency, Monetary (RFM)** scores, **Historical LTV**, and **Repeat Purchase Rate**.
  * Trains a `StandardScaler` + `KMeans` pipeline (with log-transformed skew handling) to classify buyers into actionable tiers (**Champions**, **Loyal Customers**, **Potential / Recent**, **At-Risk / Dormant**) and persists model artifacts via `joblib`.
* **Automated Threshold Alerts & Executive Reporting (`alerts.py` & `reporting.py`):**
  * Flags campaigns or overall accounts breaching target ROAS, CTR, or net-profit thresholds.
  * Exports timestamped **JSON**, **CSV**, and **Markdown** executive summaries with one click.
* **Interactive Web Dashboard (`app.py`) & CI Testing (`tests/`):**
  * Live Streamlit interface with multi-currency support (`₦`, `$`, `£`, `€`), interactive threshold sliders, and automated `pytest` verification via GitHub Actions.

---

## 🗂️ Project Structure

```text
marketing_intelligence/
├── .github/workflows/
│   └── ci.yml                  # Automated GitHub Actions pytest pipeline
├── data/
│   ├── raw/                    # Raw input CSVs (includes synthetic demo dataset)
│   └── processed/              # Cleaned, standardized datasets
├── models/                     # Saved scikit-learn clustering pipelines (.joblib)
├── reports/                    # Exported JSON, CSV & Markdown executive reports
├── src/
│   ├── data_loader.py          # Safe CSV/Excel ingestion with validation
│   ├── column_mapper.py        # Meta/Google/Paystack/Shopify header normalization
│   ├── cleaning.py             # Modular pandas .pipe() cleaning workflow
│   ├── metrics.py              # Zero-safe marketing KPI & weighted total formulas
│   ├── campaign_analysis.py    # Campaign/channel aggregation & top/bottom ranking
│   ├── customer_analysis.py    # RFM scoring, Historical LTV & retention KPIs
│   ├── segmentation.py         # K-Means customer clustering & segment profiling
│   ├── alerts.py               # Rule-based KPI threshold & anomaly detection
│   └── reporting.py            # Multi-format executive report generator
├── tests/
│   └── test_pipeline.py        # End-to-end pytest unit test suite
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # CLI batch execution pipeline
└── requirements.txt            # Pinned Python dependencies
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository & Activate Virtual Environment
```bash
git clone https://github.com
cd marketing-intelligence

python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to explore the pre-loaded sample dataset or upload your own raw marketing CSV in the sidebar.

### 3. Run the CLI Pipeline or Unit Tests
```bash
# Execute the full batch pipeline and export reports to reports/
python main.py

# Run the automated pytest suite
pytest -v
```

---

## 🛠️ Tech Stack

* **Data Processing:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`Pipeline`, `StandardScaler`, `KMeans`), Joblib
* **Dashboard & Reporting:** Streamlit, Tabulate
* **Testing & CI/CD:** Pytest, GitHub Actions

---

## 📬 Contact & Services

Built by **Femi Ogunbiyi** — Marketing Data & Analytics Specialist.  
Interested in a **48-Hour Marketing & Ad Spend Profit Audit** or custom analytics pipeline for your business? Connect on [GitHub](https://github.com).
