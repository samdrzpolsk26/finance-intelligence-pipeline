-- ============================================================
-- sql/03_stored_procedures.sql
-- Finance Intelligence Pipeline — Stored Procedures
-- 
-- Automates the aggregation refresh logic inside MySQL.
-- Called by src/datamart/builder.py after each pipeline run.
-- ============================================================

USE finance_db;

DELIMITER $$

-- ── Refresh Monthly Aggregates ─────────────────────────────────────────────────
DROP PROCEDURE IF EXISTS sp_refresh_monthly_aggregates$$

CREATE PROCEDURE sp_refresh_monthly_aggregates()
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;

    START TRANSACTION;

    -- Delete existing aggregates for months that have new data
    DELETE FROM agg_monthly_spending
    WHERE CONCAT(year, LPAD(month, 2, '0')) IN (
        SELECT DISTINCT CONCAT(year, LPAD(month, 2, '0'))
        FROM fact_transactions
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
    );

    -- Rebuild aggregates for affected months
    INSERT INTO agg_monthly_spending (
        year, month, category_code,
        total_spend, transaction_count,
        avg_transaction, max_transaction, min_transaction
    )
    SELECT
        year, month, category_code,
        SUM(abs_amount)             AS total_spend,
        COUNT(*)                    AS transaction_count,
        AVG(abs_amount)             AS avg_transaction,
        MAX(abs_amount)             AS max_transaction,
        MIN(abs_amount)             AS min_transaction
    FROM fact_transactions
    WHERE is_expense = 1
      AND CONCAT(year, LPAD(month, 2, '0')) IN (
          SELECT DISTINCT CONCAT(year, LPAD(month, 2, '0'))
          FROM fact_transactions
          WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
      )
    GROUP BY year, month, category_code
    ON DUPLICATE KEY UPDATE
        total_spend         = VALUES(total_spend),
        transaction_count   = VALUES(transaction_count),
        avg_transaction     = VALUES(avg_transaction),
        max_transaction     = VALUES(max_transaction),
        min_transaction     = VALUES(min_transaction),
        last_updated        = NOW();

    -- Calculate month-over-month changes
    UPDATE agg_monthly_spending a
    INNER JOIN agg_monthly_spending b ON (
        b.category_code = a.category_code AND
        (b.year * 12 + b.month) = (a.year * 12 + a.month) - 1
    )
    SET
        a.mom_change_abs = a.total_spend - b.total_spend,
        a.mom_change_pct = CASE
            WHEN b.total_spend = 0 THEN NULL
            ELSE ROUND(((a.total_spend - b.total_spend) / b.total_spend) * 100, 4)
        END;

    COMMIT;
END$$


-- ── Refresh Daily Aggregates ───────────────────────────────────────────────────
DROP PROCEDURE IF EXISTS sp_refresh_daily_aggregates$$

CREATE PROCEDURE sp_refresh_daily_aggregates()
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;

    START TRANSACTION;

    INSERT INTO agg_daily_spending (
        spend_date, year, month, day_of_week,
        total_expenses, total_income, net_flow,
        transaction_count, closing_balance
    )
    SELECT
        transaction_date                                        AS spend_date,
        year, month,
        DAYOFWEEK(transaction_date) - 1                         AS day_of_week,
        SUM(CASE WHEN is_expense = 1 THEN abs_amount ELSE 0 END) AS total_expenses,
        SUM(CASE WHEN is_expense = 0 THEN abs_amount ELSE 0 END) AS total_income,
        SUM(amount_pln)                                         AS net_flow,
        COUNT(*)                                                AS transaction_count,
        -- Get closing balance: last recorded balance for that day
        (SELECT balance_pln FROM fact_transactions ft2
         WHERE ft2.transaction_date = ft.transaction_date
         ORDER BY ft2.transaction_id DESC LIMIT 1)             AS closing_balance
    FROM fact_transactions ft
    GROUP BY transaction_date, year, month, day_of_week
    ON DUPLICATE KEY UPDATE
        total_expenses    = VALUES(total_expenses),
        total_income      = VALUES(total_income),
        net_flow          = VALUES(net_flow),
        transaction_count = VALUES(transaction_count),
        closing_balance   = VALUES(closing_balance),
        last_updated      = NOW();

    COMMIT;
END$$


-- ── Full Refresh (calls both) ──────────────────────────────────────────────────
DROP PROCEDURE IF EXISTS sp_full_refresh$$

CREATE PROCEDURE sp_full_refresh()
BEGIN
    CALL sp_refresh_monthly_aggregates();
    CALL sp_refresh_daily_aggregates();
    SELECT 'Aggregates refreshed successfully' AS status;
END$$

DELIMITER ;
