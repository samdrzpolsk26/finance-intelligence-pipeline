-- ============================================================
-- sql/01_staging_schema.sql
-- Finance Intelligence Pipeline — Staging Layer
-- 
-- Purpose: Store raw, unmodified transaction data exactly as
--          parsed from the PDF. This layer is immutable and
--          used for re-processing if transform logic changes.
-- Run: mysql -u root -p finance_db < sql/01_staging_schema.sql
-- ============================================================

USE finance_db;

-- ── PDF Ingestion Audit Log ────────────────────────────────────────────────────
-- Tracks every PDF file processed by the pipeline.
CREATE TABLE IF NOT EXISTS stg_pdf_loads (
    load_id         INT UNSIGNED    AUTO_INCREMENT PRIMARY KEY,
    file_name       VARCHAR(512)    NOT NULL,
    file_hash       CHAR(64)        NOT NULL,    -- SHA-256 of file content
    file_size_bytes INT UNSIGNED    NOT NULL,
    account_number  VARCHAR(50),
    bank_format     VARCHAR(50)     NOT NULL DEFAULT 'erste_poland',
    transactions_parsed INT UNSIGNED DEFAULT 0,
    loaded_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status          ENUM('SUCCESS', 'FAILED', 'PARTIAL') NOT NULL DEFAULT 'SUCCESS',
    error_message   TEXT,

    UNIQUE KEY uq_file_hash (file_hash)     -- Prevents reloading same file
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Audit trail for every PDF processed by the pipeline';


-- ── Raw Transactions (Staging) ─────────────────────────────────────────────────
-- Exactly as parsed from PDF, no transformations applied.
CREATE TABLE IF NOT EXISTS stg_raw_transactions (
    raw_id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    load_id             INT UNSIGNED    NOT NULL,
    
    -- Raw parsed fields
    transaction_date    DATE            NOT NULL,
    booking_date        DATE,
    raw_description     TEXT            NOT NULL,    -- Full description line from PDF
    amount_raw          VARCHAR(30)     NOT NULL,    -- Original string e.g. "-101,42 PLN"
    balance_raw         VARCHAR(30)     NOT NULL,    -- Original string e.g. "4 324,47 PLN"
    
    -- Parsed numeric values (NULL if parsing failed)
    amount_pln          DECIMAL(12, 2),              -- Negative = expense, Positive = income
    balance_pln         DECIMAL(12, 2),
    currency            CHAR(3)         DEFAULT 'PLN',
    
    -- Transaction type detected at parse time
    transaction_type    ENUM(
        'CARD_PAYMENT',     -- DOP. VISA / PŁATNOŚĆ KARTĄ
        'BLIK',             -- Zakup BLIK
        'TRANSFER_OUT',     -- Outgoing transfers
        'TRANSFER_IN',      -- Incoming transfers (salary, returns)
        'BANK_FEE',         -- Account/card fees
        'UNKNOWN'
    ) NOT NULL DEFAULT 'UNKNOWN',
    
    -- Reference number (extracted from BLIK transactions)
    reference_number    VARCHAR(50),
    
    -- Merchant name (cleaned from description)
    merchant_raw        VARCHAR(512),
    
    -- Source location
    city_raw            VARCHAR(100),
    
    -- Processing metadata
    inserted_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_processed        TINYINT(1)      NOT NULL DEFAULT 0,  -- 1 = moved to data mart
    
    FOREIGN KEY (load_id) REFERENCES stg_pdf_loads(load_id) ON DELETE CASCADE,
    INDEX idx_transaction_date  (transaction_date),
    INDEX idx_is_processed      (is_processed),
    INDEX idx_amount            (amount_pln),
    INDEX idx_load_id           (load_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Raw transactions exactly as parsed from PDF. Never modified after insert.';


-- ── Staging Summary View ───────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_staging_summary AS
SELECT
    l.load_id,
    l.file_name,
    l.loaded_at,
    l.transactions_parsed,
    l.status,
    COUNT(t.raw_id)                     AS rows_in_staging,
    SUM(CASE WHEN t.is_processed = 1 THEN 1 ELSE 0 END) AS rows_processed,
    MIN(t.transaction_date)             AS earliest_date,
    MAX(t.transaction_date)             AS latest_date,
    SUM(CASE WHEN t.amount_pln < 0 THEN t.amount_pln ELSE 0 END) AS total_expenses,
    SUM(CASE WHEN t.amount_pln > 0 THEN t.amount_pln ELSE 0 END) AS total_income
FROM stg_pdf_loads l
LEFT JOIN stg_raw_transactions t ON l.load_id = t.load_id
GROUP BY l.load_id, l.file_name, l.loaded_at, l.transactions_parsed, l.status;
