# 💳 Finance Intelligence Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-006AFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Prophet-1.1-4285F4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Power_BI-Ready-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <strong>An end-to-end, automated personal finance analytics system — from raw bank PDF exports to ML-powered spending predictions and live dashboards.</strong>
</p>

---

## 🎯 What This Project Does

This pipeline ingests raw Polish bank transaction PDFs (Erste Bank format), parses and categorizes every transaction, stores them in a structured MySQL data warehouse, runs full exploratory analysis, trains an ML model to predict future monthly spending, and schedules all of this to run automatically.

**Built for two purposes:**
1. A portfolio-grade data engineering + ML project demonstrating real production skills
2. A personal finance intelligence tool producing actionable insights from real bank data

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FINANCE INTELLIGENCE PIPELINE                   │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │ PDF Bank │───▶│  Ingestion   │───▶│   MySQL Staging Layer   │   │
│  │  Export  │    │  (pdfplumber)│    │  (raw_transactions)     │   │
│  └──────────┘    └──────────────┘    └────────────┬────────────┘   │
│                                                   │                 │
│                                      ┌────────────▼────────────┐   │
│                                      │   Transform Layer       │   │
│                                      │  - Smart Categorizer    │   │
│                                      │  - Feature Engineering  │   │
│                                      │  - Deduplication        │   │
│                                      └────────────┬────────────┘   │
│                                                   │                 │
│                                      ┌────────────▼────────────┐   │
│                                      │   MySQL Data Mart       │   │
│                                      │  - fact_transactions    │   │
│                                      │  - dim_categories       │   │
│                                      │  - agg_monthly_spending │   │
│                                      └────┬───────────┬────────┘   │
│                                           │           │             │
│                              ┌────────────▼──┐  ┌─────▼──────────┐ │
│                              │  EDA Engine   │  │  ML Predictor  │ │
│                              │  + Reports    │  │  XGBoost +     │ │
│                              │  + Power BI   │  │  Prophet SHAP  │ │
│                              └───────────────┘  └────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              APScheduler — Fully Automated                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔍 Intelligent PDF Parsing
- Regex-powered parser tuned to Erste Bank Poland statement format
- Handles multi-line merchant names, Polish characters (UTF-8)
- Differentiates: BLIK payments, VISA card transactions, transfers, salary, refunds
- Idempotent ingestion — no duplicates on re-run

### 🗄️ Two-Layer MySQL Data Warehouse
- **Staging layer**: Raw, untransformed data for auditability and re-processing
- **Data Mart layer**: Clean, categorized, feature-rich tables ready for BI tools
- Stored procedures for incremental loads
- Full referential integrity with foreign keys

### 🏷️ Smart Transaction Categorizer
- 14 spending categories mapped from 80+ known Polish merchants
- Regex-based fallback for unknown merchants
- Extensible merchant dictionary (`config/merchants.py`)

### 📊 EDA Engine
- 12 automated visualizations saved to `reports/figures/`
- Monthly spending trends, category breakdowns, daily heatmaps
- Spending velocity analysis, income vs expense curves
- Power BI `.pbix` connection guide included

### 🤖 ML Spending Predictor
- **Prophet**: Time series decomposition — trend, weekly, monthly seasonality
- **XGBoost**: Category-level regression with lag features
- **SHAP**: Feature explainability for every prediction
- Model serialization to `models/` with versioning

### ⚙️ Full Automation
- APScheduler-based scheduler (configurable: daily / weekly)
- Watches a target folder for new PDFs and auto-processes
- Sends optional email summary (SMTP configurable)
- Logs everything to `outputs/pipeline.log`

---

## 📁 Project Structure

```
finance-intelligence-pipeline/
│
├── 📄 README.md                    ← You are here
├── 📄 EXPLANATION.md               ← Deep-dive: what every file does and why
├── 📄 pipeline.py                  ← Main entry point — runs the full pipeline
├── 📄 requirements.txt             ← All Python dependencies
├── 📄 .env.example                 ← Template for environment variables
├── 📄 .gitignore
│
├── 📁 config/
│   ├── settings.py                 ← Central config (DB, paths, flags)
│   └── merchants.py                ← Merchant → category mapping dictionary
│
├── 📁 sql/
│   ├── 01_staging_schema.sql       ← Staging layer DDL
│   ├── 02_datamart_schema.sql      ← Data mart DDL
│   └── 03_stored_procedures.sql    ← Stored procedures for transforms
│
├── 📁 src/
│   ├── 📁 ingestion/
│   │   └── pdf_parser.py           ← PDF text extraction + transaction parsing
│   ├── 📁 staging/
│   │   └── loader.py               ← Staging MySQL loader (idempotent)
│   ├── 📁 transform/
│   │   ├── categorizer.py          ← Merchant → category classification
│   │   └── feature_engineering.py  ← Date/lag/rolling features
│   ├── 📁 datamart/
│   │   └── builder.py              ← Data mart population logic
│   ├── 📁 eda/
│   │   └── analysis.py             ← Automated EDA visualizations
│   ├── 📁 ml/
│   │   ├── predictor.py            ← Prediction inference
│   │   └── model_trainer.py        ← Training: XGBoost + Prophet + SHAP
│   └── 📁 automation/
│       └── scheduler.py            ← APScheduler pipeline orchestration
│
├── 📁 notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── 📁 docs/
│   └── architecture.md             ← Detailed system design notes
│
├── 📁 models/                      ← Serialized ML models (gitignored)
├── 📁 reports/figures/             ← Auto-generated EDA charts
└── 📁 outputs/                     ← Logs and run summaries
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# MySQL 8.0+ running locally or remote
mysql --version
```

### 2. Clone & Install

```bash
git clone https://github.com/samdrzpolsk26/finance-intelligence-pipeline.git
cd finance-intelligence-pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your MySQL credentials and settings
nano .env
```

### 4. Initialize Database

```bash
# Create the database
mysql -u root -p -e "CREATE DATABASE finance_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# Run schemas in order
mysql -u root -p finance_db < sql/01_staging_schema.sql
mysql -u root -p finance_db < sql/02_datamart_schema.sql
mysql -u root -p finance_db < sql/03_stored_procedures.sql
```

### 5. Run the Pipeline

```bash
# Full pipeline on a single PDF
python pipeline.py --pdf path/to/your_bank_statement.pdf

# Full pipeline with ML training
python pipeline.py --pdf path/to/statement.pdf --train-ml

# Start the automated scheduler (watches folder for new PDFs)
python pipeline.py --schedule

# EDA only (assumes data already in DB)
python pipeline.py --eda-only
```

---

## 📊 Dashboard Setup (Power BI)

1. Open Power BI Desktop
2. **Get Data → MySQL Database**
3. Connect to your MySQL instance / `finance_db`
4. Import tables: `fact_transactions`, `agg_monthly_spending`, `dim_categories`
5. See `docs/architecture.md` for pre-built DAX measures

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🗺️ Roadmap

- [ ] Telegram bot integration for daily spending alerts
- [ ] Multi-bank PDF support (PKO, mBank, ING)
- [ ] Streamlit dashboard as Power BI alternative
- [ ] Docker Compose deployment
- [ ] Anomaly detection layer (Isolation Forest)
- [ ] Currency normalization for EUR/USD transactions

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| PDF Parsing | `pdfplumber` | Superior text extraction with coordinate awareness |
| Data Storage | `MySQL 8.0` | Production-grade relational DB, Power BI native connector |
| ORM / Queries | `SQLAlchemy` + raw SQL | Best of both worlds — ORM for CRUD, raw SQL for analytics |
| Data Processing | `pandas`, `numpy` | Industry standard for tabular data |
| ML - Time Series | `Prophet` | Handles seasonality, holidays naturally |
| ML - Regression | `XGBoost` | Best-in-class for tabular features |
| Explainability | `SHAP` | Model transparency, not black boxes |
| Visualization | `matplotlib`, `seaborn` | Publication-quality static charts |
| Scheduling | `APScheduler` | Lightweight, Pythonic cron-like scheduler |
| Config | `python-dotenv` | 12-factor app compliance |

---

## 👤 Author

**Angel** — Junior Data Professional | Mechatronics + Web Dev background  
GitHub: [@samdrzpolsk26](https://github.com/samdrzpolsk26)

---

## 📝 License

MIT License — see `LICENSE` file for details.
