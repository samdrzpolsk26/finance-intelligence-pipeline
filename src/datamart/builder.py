"""
src/datamart/builder.py
========================
Data Mart builder — populates fact_transactions and triggers aggregate refresh.

Takes transformed, feature-engineered DataFrames and writes them to
the clean data mart layer (fact_transactions, then triggers stored procedures
to refresh agg_monthly_spending and agg_daily_spending).

Why a separate builder?
    The data mart is the "source of truth" for analytics and Power BI.
    It should always be consistent, complete, and indexed correctly.
    This class ensures that data flows correctly from staging → mart
    and that aggregates are always current.

Usage:
    from src.datamart.builder import DataMartBuilder
    builder = DataMartBuilder()
    builder.upsert_transactions(enriched_df, load_id=42)
    builder.refresh_aggregates()
"""

import pandas as pd
from sqlalchemy import create_engine, text, Engine
from loguru import logger

from config.settings import settings


class DataMartBuilder:
    """
    Populates and maintains the MySQL data mart layer.
    
    Handles:
    - Upserting transactions into fact_transactions
    - Refreshing pre-aggregated tables via stored procedures
    - Linking category dimension lookups
    """

    def __init__(self):
        self.engine = create_engine(
            settings.db.url,
            pool_pre_ping=True,
            echo=False,
        )
        self._category_map = self._load_category_map()

    def _load_category_map(self) -> dict[str, int]:
        """Load category_code → category_id lookup from dim_categories."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT category_code, category_id FROM dim_categories")
            ).fetchall()
        return {row[0]: row[1] for row in result}

    def upsert_transactions(
        self,
        df: pd.DataFrame,
        load_id: int,
    ) -> int:
        """
        Insert enriched transactions into fact_transactions.
        
        Uses INSERT IGNORE to handle re-runs gracefully — if a raw_id
        already exists in fact_transactions, it won't be duplicated.
        
        Args:
            df:      Feature-engineered DataFrame (output of FeatureEngineer)
            load_id: The load_id from stg_pdf_loads for traceability
            
        Returns:
            Number of rows inserted.
        """
        if df.empty:
            logger.warning("Empty DataFrame — nothing to insert into data mart")
            return 0

        logger.info(f"Building data mart rows for load_id={load_id}...")

        rows = self._df_to_fact_rows(df, load_id)
        inserted = 0

        BATCH_SIZE = 500
        with self.engine.begin() as conn:
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                result = conn.execute(
                    text("""
                        INSERT IGNORE INTO fact_transactions (
                            raw_id, transaction_date, year, month, day,
                            day_of_week, week_of_year, quarter,
                            is_weekend, is_month_start, is_month_end,
                            amount_pln, balance_pln, abs_amount, is_expense,
                            currency, transaction_type,
                            description_clean, merchant_name, city,
                            category_code, category_id,
                            rolling_7d_spend, rolling_30d_spend,
                            load_id
                        ) VALUES (
                            :raw_id, :transaction_date, :year, :month, :day,
                            :day_of_week, :week_of_year, :quarter,
                            :is_weekend, :is_month_start, :is_month_end,
                            :amount_pln, :balance_pln, :abs_amount, :is_expense,
                            :currency, :transaction_type,
                            :description_clean, :merchant_name, :city,
                            :category_code, :category_id,
                            :rolling_7d_spend, :rolling_30d_spend,
                            :load_id
                        )
                    """),
                    batch
                )
                inserted += result.rowcount
                logger.debug(f"Data mart insert: {inserted}/{len(rows)} rows")

        logger.success(f"Data mart: {inserted} new rows inserted from load_id={load_id}")
        return inserted

    def _df_to_fact_rows(self, df: pd.DataFrame, load_id: int) -> list[dict]:
        """Convert DataFrame rows to dicts matching fact_transactions schema."""
        rows = []
        for _, row in df.iterrows():
            cat_code = str(row.get("category_code", "OTHER"))
            rows.append({
                "raw_id":            int(row["raw_id"]) if "raw_id" in row else None,
                "transaction_date":  row["transaction_date"].date() if hasattr(row["transaction_date"], "date") else row["transaction_date"],
                "year":              int(row.get("year", 0)),
                "month":             int(row.get("month", 0)),
                "day":               int(row.get("day", 0)),
                "day_of_week":       int(row.get("day_of_week", 0)),
                "week_of_year":      int(row.get("week_of_year", 0)),
                "quarter":           int(row.get("quarter", 0)),
                "is_weekend":        int(row.get("is_weekend", 0)),
                "is_month_start":    int(row.get("is_month_start", 0)),
                "is_month_end":      int(row.get("is_month_end", 0)),
                "amount_pln":        float(row.get("amount_pln", 0)),
                "balance_pln":       float(row.get("balance_pln", 0)),
                "abs_amount":        float(row.get("abs_amount", 0)),
                "is_expense":        int(row.get("is_expense", 0)),
                "currency":          str(row.get("currency", "PLN")),
                "transaction_type":  str(row.get("transaction_type", "UNKNOWN")),
                "description_clean": str(row.get("raw_description", ""))[:512],
                "merchant_name":     str(row.get("merchant_raw", "") or "")[:200] or None,
                "city":              str(row.get("city_raw", "") or "")[:100] or None,
                "category_code":     cat_code,
                "category_id":       self._category_map.get(cat_code),
                "rolling_7d_spend":  float(row.get("rolling_7d_spend", 0) or 0),
                "rolling_30d_spend": float(row.get("rolling_30d_spend", 0) or 0),
                "load_id":           load_id,
            })
        return rows

    def refresh_aggregates(self):
        """
        Trigger stored procedures to rebuild monthly and daily aggregates.
        
        This should be called after every successful data mart upsert.
        The stored procedures are designed to be idempotent.
        """
        logger.info("Refreshing aggregate tables via stored procedures...")
        with self.engine.begin() as conn:
            conn.execute(text("CALL sp_full_refresh()"))
        logger.success("Aggregate refresh complete")

    def get_summary(self) -> dict:
        """Return data mart health summary for monitoring."""
        with self.engine.connect() as conn:
            facts_count = conn.execute(
                text("SELECT COUNT(*) FROM fact_transactions")
            ).scalar()
            date_range = conn.execute(
                text("SELECT MIN(transaction_date), MAX(transaction_date) FROM fact_transactions")
            ).fetchone()
            monthly_rows = conn.execute(
                text("SELECT COUNT(*) FROM agg_monthly_spending")
            ).scalar()

        return {
            "total_transactions": facts_count,
            "date_range": {
                "min": str(date_range[0]) if date_range[0] else None,
                "max": str(date_range[1]) if date_range[1] else None,
            },
            "monthly_aggregate_rows": monthly_rows,
        }
