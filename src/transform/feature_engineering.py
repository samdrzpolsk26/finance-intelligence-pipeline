"""
src/transform/feature_engineering.py
======================================
Feature engineering for the Finance Intelligence Pipeline.

Transforms raw categorized transactions into a feature-rich dataset
ready for both the data mart and ML model training.

Features generated:
    ─ Date/Time Features ─────────────────────────────────────────
    year, month, day, day_of_week, week_of_year, quarter
    is_weekend, is_month_start, is_month_end
    days_since_payday (heuristic: 3rd of month)

    ─ Financial Features ─────────────────────────────────────────
    abs_amount (|amount|)
    rolling_7d_spend, rolling_30d_spend
    cumulative_monthly_spend (by category)
    spending_velocity (change in rolling 7d vs prior 7d)
    category_share_this_month (% of total monthly spend)
    days_since_last_transaction (spending gap)

    ─ Balance Features ───────────────────────────────────────────
    balance_change (balance delta between transactions)
    balance_pct_change

    ─ Lag Features (for ML) ──────────────────────────────────────
    lag_1m_spend, lag_2m_spend, lag_3m_spend (previous months same category)
    lag_1m_total (previous month total spending)

Usage:
    from src.transform.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    enriched_df = fe.transform(df)
"""

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """
    Adds derived features to a categorized transactions DataFrame.
    
    Input DataFrame must have columns:
        transaction_date, amount_pln, balance_pln, category_code, is_expense
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the DataFrame.
        
        Args:
            df: Categorized transactions (output of Categorizer.categorize_dataframe)
            
        Returns:
            Enriched DataFrame with all derived features.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to FeatureEngineer")
            return df

        logger.info(f"Feature engineering on {len(df)} transactions...")

        df = df.copy()

        # Ensure correct types
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["amount_pln"] = pd.to_numeric(df["amount_pln"], errors="coerce").fillna(0)
        df["balance_pln"] = pd.to_numeric(df["balance_pln"], errors="coerce").fillna(0)

        # Sort chronologically before computing rolling features
        df = df.sort_values("transaction_date").reset_index(drop=True)

        # Apply each feature group
        df = self._add_date_features(df)
        df = self._add_amount_features(df)
        df = self._add_rolling_features(df)
        df = self._add_balance_features(df)
        df = self._add_gap_features(df)

        logger.success(
            f"Feature engineering complete. "
            f"Columns added: {[c for c in df.columns if c not in ['transaction_date', 'amount_pln']]}"
        )

        return df

    # ── Date Features ──────────────────────────────────────────────────────────

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar-based features from transaction_date."""
        dt = df["transaction_date"].dt

        df["year"]          = dt.year.astype("int16")
        df["month"]         = dt.month.astype("int8")
        df["day"]           = dt.day.astype("int8")
        df["day_of_week"]   = dt.dayofweek.astype("int8")    # 0=Mon, 6=Sun
        df["week_of_year"]  = dt.isocalendar().week.astype("int8")
        df["quarter"]       = dt.quarter.astype("int8")

        # Boolean flags
        df["is_weekend"]        = (df["day_of_week"] >= 5).astype("int8")
        df["is_month_start"]    = (df["day"] <= 3).astype("int8")
        df["is_month_end"]      = (df["day"] >= 28).astype("int8")

        # Days since payday heuristic:
        # Most Polish salaries arrive on the 3rd-5th of the month.
        # We compute days elapsed since the most recent "3rd of month" date.
        def days_since_payday(row):
            payday = row["transaction_date"].replace(day=3)
            if payday > row["transaction_date"]:
                # Last month's payday
                if payday.month == 1:
                    payday = payday.replace(year=payday.year - 1, month=12)
                else:
                    payday = payday.replace(month=payday.month - 1)
            return (row["transaction_date"] - payday).days

        df["days_since_payday"] = df.apply(days_since_payday, axis=1).astype("int16")

        # Day of week name (for visualization only, not ML features)
        df["day_name"] = df["transaction_date"].dt.day_name()

        return df

    # ── Amount Features ────────────────────────────────────────────────────────

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive amount-based features."""
        df["abs_amount"] = df["amount_pln"].abs().round(2)

        # Amount buckets for visualization
        conditions = [
            df["abs_amount"] <= 10,
            (df["abs_amount"] > 10) & (df["abs_amount"] <= 50),
            (df["abs_amount"] > 50) & (df["abs_amount"] <= 200),
            (df["abs_amount"] > 200) & (df["abs_amount"] <= 500),
            df["abs_amount"] > 500,
        ]
        labels = ["micro (<10)", "small (10-50)", "medium (50-200)", "large (200-500)", "major (500+)"]
        df["amount_bucket"] = np.select(conditions, labels, default="other")

        return df

    # ── Rolling Window Features ────────────────────────────────────────────────

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling spending windows.
        
        Uses a date-indexed approach to handle irregular daily transaction counts:
        - rolling_7d_spend: total expenses in the past 7 calendar days
        - rolling_30d_spend: total expenses in the past 30 calendar days
        - spending_velocity: how much faster/slower spending is vs prior 7 days
        """
        # Work with expenses only for rolling calculations
        expenses = df[df["is_expense"] == 1].copy()
        expenses = expenses.set_index("transaction_date")["abs_amount"]

        # Create daily sum series
        daily_spend = expenses.resample("D").sum()
        full_range = pd.date_range(
            start=df["transaction_date"].min(),
            end=df["transaction_date"].max(),
            freq="D"
        )
        daily_spend = daily_spend.reindex(full_range, fill_value=0)

        # Rolling windows
        rolling_7d  = daily_spend.rolling(window=7,  min_periods=1).sum()
        rolling_30d = daily_spend.rolling(window=30, min_periods=1).sum()

        # Spending velocity: 7d window vs prior 7d window
        prior_7d = daily_spend.rolling(window=7, min_periods=1).sum().shift(7).fillna(0)
        velocity = rolling_7d - prior_7d  # Positive = spending more than last week

        # Map back to transaction rows by date
        df["rolling_7d_spend"] = df["transaction_date"].map(rolling_7d).round(2)
        df["rolling_30d_spend"] = df["transaction_date"].map(rolling_30d).round(2)
        df["spending_velocity"] = df["transaction_date"].map(velocity).round(2)

        # Category share of monthly spend
        monthly_total = (
            df[df["is_expense"] == 1]
            .groupby(["year", "month"])["abs_amount"]
            .transform("sum")
        )
        # Avoid division by zero
        df["monthly_category_share"] = (
            df["abs_amount"] / monthly_total.where(monthly_total > 0, other=1)
        ).round(4)
        df["monthly_category_share"] = df["monthly_category_share"].where(df["is_expense"] == 1, 0)

        return df

    # ── Balance Features ───────────────────────────────────────────────────────

    def _add_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track balance trajectory features."""
        df["balance_change"] = df["balance_pln"].diff().round(2)
        df["balance_pct_change"] = (
            df["balance_pln"].pct_change() * 100
        ).round(4)

        # Balance health: normalize balance to [0, 1] range within dataset
        b_min = df["balance_pln"].min()
        b_max = df["balance_pln"].max()
        if b_max > b_min:
            df["balance_normalized"] = (
                (df["balance_pln"] - b_min) / (b_max - b_min)
            ).round(4)
        else:
            df["balance_normalized"] = 0.5

        return df

    # ── Gap Features ───────────────────────────────────────────────────────────

    def _add_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time gaps between transactions (spending cadence)."""
        df["days_since_prev_tx"] = (
            df["transaction_date"].diff().dt.days.fillna(0).astype("int16")
        )
        return df

    # ── ML Lag Features ────────────────────────────────────────────────────────

    def create_monthly_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a monthly-aggregated DataFrame with lag features for ML training.
        
        This produces one row per (year, month, category_code) with:
        - total spend that month
        - lag_1m, lag_2m, lag_3m (previous month spends)
        - date calendar features
        
        Returns:
            Monthly aggregated DataFrame ready for XGBoost training.
        """
        logger.info("Building monthly ML features...")

        # Aggregate to monthly level per category
        monthly = (
            df[df["is_expense"] == 1]
            .groupby(["year", "month", "category_code"])
            .agg(
                total_spend=("abs_amount", "sum"),
                tx_count=("abs_amount", "count"),
                avg_spend=("abs_amount", "mean"),
                max_spend=("abs_amount", "max"),
            )
            .reset_index()
        )

        # Create year-month period for lag computation
        monthly["period"] = monthly["year"] * 12 + monthly["month"]

        # Create lag features per category
        monthly = monthly.sort_values(["category_code", "period"])
        for lag in [1, 2, 3]:
            monthly[f"lag_{lag}m_spend"] = monthly.groupby("category_code")["total_spend"].shift(lag)

        # Total monthly spend (across all categories)
        monthly_total = (
            monthly.groupby(["year", "month"])["total_spend"]
            .sum()
            .reset_index()
            .rename(columns={"total_spend": "total_monthly_spend"})
        )
        monthly = monthly.merge(monthly_total, on=["year", "month"], how="left")
        monthly["lag_1m_total"] = monthly.groupby("category_code")["total_monthly_spend"].shift(1)

        # Drop rows without at least 1 lag (need history for prediction)
        monthly = monthly.dropna(subset=["lag_1m_spend"])

        logger.success(f"Monthly ML features: {len(monthly)} rows, {len(monthly.columns)} features")
        return monthly
