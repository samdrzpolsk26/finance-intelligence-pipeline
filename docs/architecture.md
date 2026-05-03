# Architecture — Finance Intelligence Pipeline

## Overview

This document describes the technical design of every layer in the pipeline, the reasoning behind each decision, and the Power BI connection guide.

---

## Layer 1 — Ingestion (PDF Parsing)

**File:** `src/ingestion/pdf_parser.py`  
**Library:** `pdfplumber`

### Why pdfplumber over PyPDF2 or pdfminer?

pdfplumber preserves spatial layout when extracting text. Erste Bank PDFs use a column-based layout where amounts appear to the right of descriptions. pdfplumber's coordinate-aware extraction keeps these aligned in the raw text output, while PyPDF2 often scrambles the column order.

### Parsing Strategy

The parser works in three passes:

1. **Text extraction** — All pages extracted into a single list of text lines
2. **Block grouping** — Lines are grouped into per-transaction blocks. Each block starts with a date line (`\d{1,2} \w{3} \d{4}`) and ends just before the next date line
3. **Block parsing** — Each block is parsed individually:
   - Line 0 → transaction date
   - Last line → amount + balance (regex: `(-?\s*[\d\s]+,\d{2})\s*(PLN|EUR)`)
   - Middle lines → description (joined, deduplicated)

### Idempotency

Each PDF is hashed with SHA-256 before loading. If the hash already exists in `stg_pdf_loads`, the load is aborted. This means re-running the pipeline on the same file is always safe.

---

## Layer 2 — Staging (MySQL)

**File:** `src/staging/loader.py`  
**Tables:** `stg_pdf_loads`, `stg_raw_transactions`

### Design Principle: Immutability

Raw staging rows are **never modified** after insert. The `is_processed` flag is the only field that changes (set to 1 after the row reaches the data mart). This means:

- If the categorization logic changes, you can re-process from staging without re-parsing the PDF
- Every pipeline run is auditable via `stg_pdf_loads`
- The `v_staging_summary` view gives instant health monitoring

### Batch Inserts

Rows are inserted in batches of 500 using SQLAlchemy's `execute()` with a list of dicts. This is 10-50× faster than row-by-row inserts for large PDFs.

---

## Layer 3 — Transformation

### Categorizer (`src/transform/categorizer.py`)

**Strategy:** Rule-based regex matching with priority ordering.

Categories are checked in this priority order to prevent false positives:

```
INCOME → RENT → SUBSCRIPTIONS → TRANSPORT → GROCERIES → DINING
→ HEALTHCARE → BEAUTY → CLOTHING → ELECTRONICS → ENTERTAINMENT
→ UTILITIES → HOTEL_TRAVEL → TRANSFER → OTHER
```

INCOME is checked first because "ZWROT" (return) appears in both income descriptions and pharmacy names. RENT is second because it's a short keyword that could match inside longer descriptions.

**LRU Cache:** The `categorize()` method is decorated with `@lru_cache(maxsize=2048)`. Bank statements contain many repeated merchants (e.g., the same Biedronka branch dozens of times). The cache ensures each unique description string is only regex-matched once per session.

**Extension:** To add a new merchant, simply add a regex entry to the appropriate list in `config/merchants.py`. No code changes needed anywhere else.

### Feature Engineer (`src/transform/feature_engineering.py`)

**Rolling Windows:** Rolling 7-day and 30-day spend are computed on a daily-resampled series, then mapped back to transaction rows by date. This handles the irregular distribution of transactions correctly (e.g., 5 transactions on Friday, 0 on Saturday).

**Lag Features:** Used exclusively by the ML layer. `create_monthly_ml_features()` produces one row per (year, month, category) with 3 historical lag values — the minimum needed for XGBoost to learn trends.

---

## Layer 4 — Data Mart (MySQL)

**File:** `src/datamart/builder.py`  
**Tables:** `fact_transactions`, `agg_monthly_spending`, `agg_daily_spending`  
**Views:** `v_monthly_cashflow`, `v_category_totals`, `v_merchant_leaderboard`

### Star Schema Design

```
dim_categories (15 rows)
      │
      │  category_id (FK)
      ▼
fact_transactions  ◀──── agg_monthly_spending
      │                         ▲
      └──────────────────────── agg_daily_spending
```

`fact_transactions` is the central fact table. All date dimensions are denormalized into it (year, month, day, day_of_week, quarter, is_weekend) to avoid joins in Power BI — this is standard star schema practice for analytical workloads.

### INSERT IGNORE for Idempotency

`fact_transactions` has a `UNIQUE KEY` on `raw_id`. Using `INSERT IGNORE` means re-running the pipeline after a partial failure will skip already-inserted rows without errors.

### Stored Procedures

`sp_full_refresh()` calls `sp_refresh_monthly_aggregates()` + `sp_refresh_daily_aggregates()`. These use `ON DUPLICATE KEY UPDATE` so they're safe to run multiple times. Called automatically after every data mart load.

---

## Layer 5 — EDA Engine

**File:** `src/eda/analysis.py`  
**Output:** `reports/figures/*.png`

All charts use a dark theme with a `#0F1117` background to match modern BI tools. `matplotlib.use("Agg")` forces the non-interactive backend — critical for server-side execution where there's no display.

The `EDAEngine` loads data from the data mart views (not from raw staging), ensuring charts always reflect the clean, categorized data.

---

## Layer 6 — Machine Learning

### Model Selection Rationale

| Model | Why it's used | What it handles well |
|-------|--------------|----------------------|
| XGBoost | Tabular features with lag columns | Non-linear patterns, category interactions |
| Prophet | Time series with dates only | Trend + seasonality, missing months |
| SHAP | Explainability layer | "Why this prediction?" |

Both models are complementary: Prophet gives the interval (lower/upper bounds), XGBoost gives the point estimate using richer features.

### Why TimeSeriesSplit for Cross-Validation?

Standard k-fold cross-validation shuffles data randomly. For financial time series, this causes **data leakage** — the model would see future data during training. `TimeSeriesSplit` always trains on past data and validates on future data, respecting chronological order.

### Model Versioning

Every training run saves models with a timestamp suffix (`xgb_spending_20260430_1430.pkl`). The predictor always loads the most recently created file. Old models are kept for rollback.

---

## Layer 7 — Automation

**File:** `src/automation/scheduler.py`

Two parallel mechanisms:

1. **Watchdog** — Reacts immediately when a PDF lands in the watch folder (event-driven, zero polling delay)
2. **APScheduler** — Runs a scan every N minutes as a fallback for files placed while the watcher was down

The `_processing` set in `PDFFileHandler` prevents double-triggers from OS-level duplicate events (common on some Linux filesystems).

---

## Power BI Connection Guide

### Step 1 — Get Data

1. Open Power BI Desktop
2. **Home → Get Data → MySQL Database**
3. Enter: Server = `localhost`, Database = `finance_db`

### Step 2 — Import Tables

Import these tables/views:

| Name | Type | Use |
|------|------|-----|
| `fact_transactions` | Table | All transaction details |
| `dim_categories` | Table | Category lookup + colors |
| `agg_monthly_spending` | Table | Pre-aggregated monthly data + ML predictions |
| `v_monthly_cashflow` | View | Income vs expense by month |
| `v_category_totals` | View | All-time category breakdown |
| `v_merchant_leaderboard` | View | Top merchants |

### Step 3 — Relationships

Power BI should auto-detect:
- `fact_transactions[category_id]` → `dim_categories[category_id]`

### Step 4 — Recommended DAX Measures

```dax
Total Expenses =
CALCULATE(
    SUM(fact_transactions[abs_amount]),
    fact_transactions[is_expense] = 1
)

Total Income =
CALCULATE(
    SUM(fact_transactions[abs_amount]),
    fact_transactions[is_expense] = 0
)

Savings Rate % =
DIVIDE(
    [Total Income] - [Total Expenses],
    [Total Income],
    0
) * 100

MoM Change % =
VAR CurrentMonth = SELECTEDVALUE(agg_monthly_spending[month])
VAR CurrentYear  = SELECTEDVALUE(agg_monthly_spending[year])
VAR PrevSpend    = CALCULATE(
    SUM(agg_monthly_spending[total_spend]),
    agg_monthly_spending[year] = CurrentYear,
    agg_monthly_spending[month] = CurrentMonth - 1
)
RETURN
DIVIDE([Total Expenses] - PrevSpend, PrevSpend, 0) * 100
```

### Step 5 — Suggested Visuals

- **Line chart:** `v_monthly_cashflow[month_label]` × `total_income` + `total_expenses`
- **Donut chart:** `v_category_totals[category_label]` × `total_spent`
- **Matrix:** `fact_transactions[year]` × `fact_transactions[month]` × `Total Expenses`
- **Bar chart:** `v_merchant_leaderboard[merchant_name]` × `total_spent`
- **Card KPIs:** Total Expenses, Total Income, Savings Rate %, Current Balance
- **Forecast line:** `agg_monthly_spending[total_spend]` + `predicted_spend` with confidence band from `prediction_lower`/`prediction_upper`

### Step 6 — Scheduled Refresh (Power BI Service)

If publishing to Power BI Service:
1. Install the **On-Premises Data Gateway** on the same machine as MySQL
2. Configure gateway connection to `finance_db`
3. Set refresh schedule to **Daily at 06:00** (after the pipeline scheduler runs)

---

## Database ERD (Text)

```
stg_pdf_loads (load_id PK, file_hash UNIQUE, ...)
    │
    │ 1:N
    ▼
stg_raw_transactions (raw_id PK, load_id FK, is_processed, ...)
    │
    │ 1:1  (raw_id UNIQUE in fact_transactions)
    ▼
fact_transactions (transaction_id PK, raw_id UNIQUE FK, category_id FK, ...)
    │
    │ N:1
    ▼
dim_categories (category_id PK, category_code UNIQUE, ...)

fact_transactions → agg_monthly_spending (populated by stored procedure)
fact_transactions → agg_daily_spending   (populated by stored procedure)
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | MySQL host |
| `DB_PORT` | `3306` | MySQL port |
| `DB_USER` | `root` | MySQL user |
| `DB_PASSWORD` | *(required)* | MySQL password |
| `DB_NAME` | `finance_db` | Database name |
| `PDF_WATCH_FOLDER` | `./data/inbox` | Drop PDFs here for auto-processing |
| `PDF_ARCHIVE_FOLDER` | `./data/archive` | Processed PDFs moved here |
| `SCHEDULER_INTERVAL_MINUTES` | `60` | How often to scan for new PDFs |
| `ML_RETRAIN_THRESHOLD` | `30` | Min new transactions to trigger retrain |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `SMTP_HOST` | *(optional)* | Email notifications host |
| `NOTIFY_EMAIL` | *(optional)* | Email to receive pipeline summaries |

---

*Architecture document — Finance Intelligence Pipeline v1.0*
