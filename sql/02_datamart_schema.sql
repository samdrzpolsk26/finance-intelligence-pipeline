-- ============================================================
-- sql/02_datamart_schema.sql
-- Finance Intelligence Pipeline — Data Mart Layer
-- 
-- Purpose: Clean, categorized, feature-rich tables optimized
--          for analytics and Power BI consumption.
-- Run AFTER: 01_staging_schema.sql
-- ============================================================

USE finance_db;

-- ── Dimension: Categories ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_categories (
    category_id     TINYINT UNSIGNED    AUTO_INCREMENT PRIMARY KEY,
    category_code   VARCHAR(30)         NOT NULL UNIQUE,    -- e.g. 'GROCERIES'
    category_label  VARCHAR(50)         NOT NULL,           -- e.g. '🛒 Groceries'
    category_color  CHAR(7)             NOT NULL DEFAULT '#7F8C8D',  -- Hex color
    is_expense      TINYINT(1)          NOT NULL DEFAULT 1,
    sort_order      TINYINT UNSIGNED    NOT NULL DEFAULT 99,
    created_at      DATETIME            NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Lookup table for transaction categories';

-- Seed categories
INSERT IGNORE INTO dim_categories (category_code, category_label, category_color, is_expense, sort_order) VALUES
('GROCERIES',    '🛒 Groceries',       '#2ECC71', 1, 1),
('DINING',       '🍽️ Dining & Cafes',  '#E74C3C', 1, 2),
('TRANSPORT',    '🚌 Transport',        '#3498DB', 1, 3),
('HEALTHCARE',   '💊 Healthcare',       '#1ABC9C', 1, 4),
('BEAUTY',       '💄 Beauty & Care',    '#E91E8C', 1, 5),
('CLOTHING',     '👗 Clothing',         '#9B59B6', 1, 6),
('ENTERTAINMENT','🎬 Entertainment',    '#F39C12', 1, 7),
('UTILITIES',    '📱 Utilities & Bills','#95A5A6', 1, 8),
('RENT',         '🏠 Rent',             '#C0392B', 1, 9),
('ELECTRONICS',  '💻 Electronics',      '#2980B9', 1, 10),
('HOTEL_TRAVEL', '✈️ Travel & Hotels',  '#D35400', 1, 11),
('SUBSCRIPTIONS','📡 Subscriptions',    '#8E44AD', 1, 12),
('TRANSFER',     '↔️ Transfer',         '#BDC3C7', 0, 13),
('INCOME',       '💰 Income',           '#27AE60', 0, 14),
('OTHER',        '❓ Other',            '#7F8C8D', 1, 15);


-- ── Fact: Transactions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_transactions (
    transaction_id      BIGINT UNSIGNED     AUTO_INCREMENT PRIMARY KEY,
    raw_id              BIGINT UNSIGNED     NOT NULL UNIQUE,    -- Link back to staging

    -- Date dimensions
    transaction_date    DATE                NOT NULL,
    year                SMALLINT UNSIGNED   NOT NULL,
    month               TINYINT UNSIGNED    NOT NULL,
    day                 TINYINT UNSIGNED    NOT NULL,
    day_of_week         TINYINT UNSIGNED    NOT NULL,   -- 0=Monday, 6=Sunday
    week_of_year        TINYINT UNSIGNED    NOT NULL,
    quarter             TINYINT UNSIGNED    NOT NULL,
    is_weekend          TINYINT(1)          NOT NULL DEFAULT 0,
    is_month_start      TINYINT(1)          NOT NULL DEFAULT 0,
    is_month_end        TINYINT(1)          NOT NULL DEFAULT 0,

    -- Financial amounts
    amount_pln          DECIMAL(12, 2)      NOT NULL,
    balance_pln         DECIMAL(12, 2)      NOT NULL,
    abs_amount          DECIMAL(12, 2)      NOT NULL,           -- ABS(amount_pln)
    is_expense          TINYINT(1)          NOT NULL,           -- 1 if amount < 0
    currency            CHAR(3)             NOT NULL DEFAULT 'PLN',

    -- Transaction details
    transaction_type    VARCHAR(20)         NOT NULL,
    description_clean   VARCHAR(512)        NOT NULL,
    merchant_name       VARCHAR(200),
    city                VARCHAR(100),

    -- Category (FK to dim_categories)
    category_code       VARCHAR(30)         NOT NULL DEFAULT 'OTHER',
    category_id         TINYINT UNSIGNED,

    -- Rolling aggregates (pre-computed at transform time)
    rolling_7d_spend    DECIMAL(12, 2),     -- Total expenses in past 7 days
    rolling_30d_spend   DECIMAL(12, 2),     -- Total expenses in past 30 days

    -- Source tracking
    load_id             INT UNSIGNED        NOT NULL,
    created_at          DATETIME            NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (category_id) REFERENCES dim_categories(category_id),
    INDEX idx_date          (transaction_date),
    INDEX idx_year_month    (year, month),
    INDEX idx_category      (category_code),
    INDEX idx_is_expense    (is_expense),
    INDEX idx_amount        (amount_pln),
    INDEX idx_load_id       (load_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Cleaned, enriched transactions — primary analytics table';


-- ── Aggregate: Monthly Spending ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agg_monthly_spending (
    agg_id              INT UNSIGNED        AUTO_INCREMENT PRIMARY KEY,
    year                SMALLINT UNSIGNED   NOT NULL,
    month               TINYINT UNSIGNED    NOT NULL,
    category_code       VARCHAR(30)         NOT NULL,

    -- Financial summary
    total_spend         DECIMAL(12, 2)      NOT NULL DEFAULT 0.00,
    transaction_count   INT UNSIGNED        NOT NULL DEFAULT 0,
    avg_transaction     DECIMAL(12, 2),
    max_transaction     DECIMAL(12, 2),
    min_transaction     DECIMAL(12, 2),

    -- Month-over-month change
    mom_change_pct      DECIMAL(8, 4),      -- % change vs previous month
    mom_change_abs      DECIMAL(12, 2),     -- Absolute change PLN

    -- ML prediction (populated by ml/predictor.py)
    predicted_spend     DECIMAL(12, 2),
    prediction_lower    DECIMAL(12, 2),
    prediction_upper    DECIMAL(12, 2),
    prediction_mape     DECIMAL(8, 4),

    last_updated        DATETIME            NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uq_month_category (year, month, category_code),
    INDEX idx_year_month (year, month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Pre-aggregated monthly spending per category — optimized for BI';


-- ── Aggregate: Daily Spending ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agg_daily_spending (
    agg_id              INT UNSIGNED        AUTO_INCREMENT PRIMARY KEY,
    spend_date          DATE                NOT NULL UNIQUE,
    year                SMALLINT UNSIGNED   NOT NULL,
    month               TINYINT UNSIGNED    NOT NULL,
    day_of_week         TINYINT UNSIGNED    NOT NULL,

    total_expenses      DECIMAL(12, 2)      NOT NULL DEFAULT 0.00,
    total_income        DECIMAL(12, 2)      NOT NULL DEFAULT 0.00,
    net_flow            DECIMAL(12, 2)      NOT NULL DEFAULT 0.00,
    transaction_count   INT UNSIGNED        NOT NULL DEFAULT 0,
    closing_balance     DECIMAL(12, 2),

    last_updated        DATETIME            NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_date      (spend_date),
    INDEX idx_year_month (year, month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Daily cash flow summary — used for heatmap and trend charts';


-- ── View: Monthly Income vs Expenses ──────────────────────────────────────────
CREATE OR REPLACE VIEW v_monthly_cashflow AS
SELECT
    year,
    month,
    DATE_FORMAT(STR_TO_DATE(CONCAT(year, '-', LPAD(month, 2, '0'), '-01'), '%Y-%m-%d'), '%b %Y') AS month_label,
    SUM(CASE WHEN is_expense = 0 THEN abs_amount ELSE 0 END)  AS total_income,
    SUM(CASE WHEN is_expense = 1 THEN abs_amount ELSE 0 END)  AS total_expenses,
    SUM(CASE WHEN is_expense = 0 THEN abs_amount ELSE -abs_amount END) AS net_savings,
    COUNT(*)                                                   AS transaction_count,
    AVG(CASE WHEN is_expense = 1 THEN abs_amount END)         AS avg_expense_amount
FROM fact_transactions
GROUP BY year, month
ORDER BY year DESC, month DESC;


-- ── View: Category Breakdown (All Time) ───────────────────────────────────────
CREATE OR REPLACE VIEW v_category_totals AS
SELECT
    ft.category_code,
    dc.category_label,
    dc.category_color,
    COUNT(*)                AS transaction_count,
    SUM(ft.abs_amount)      AS total_spent,
    AVG(ft.abs_amount)      AS avg_transaction,
    MAX(ft.abs_amount)      AS max_transaction,
    MIN(ft.transaction_date) AS first_seen,
    MAX(ft.transaction_date) AS last_seen
FROM fact_transactions ft
LEFT JOIN dim_categories dc ON ft.category_code = dc.category_code
WHERE ft.is_expense = 1
GROUP BY ft.category_code, dc.category_label, dc.category_color
ORDER BY total_spent DESC;


-- ── View: Merchant Leaderboard ─────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_merchant_leaderboard AS
SELECT
    merchant_name,
    category_code,
    COUNT(*)            AS visits,
    SUM(abs_amount)     AS total_spent,
    AVG(abs_amount)     AS avg_spend,
    MIN(transaction_date) AS first_visit,
    MAX(transaction_date) AS last_visit
FROM fact_transactions
WHERE is_expense = 1
  AND merchant_name IS NOT NULL
  AND merchant_name != ''
GROUP BY merchant_name, category_code
ORDER BY total_spent DESC
LIMIT 50;
