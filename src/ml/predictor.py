"""
src/ml/predictor.py
====================
Inference engine — loads trained models and generates spending predictions.

This module is the "deployment" side of the ML layer.
The trainer (model_trainer.py) builds and saves models.
This predictor loads the latest saved models and runs inference.

Workflow:
    1. Load latest XGBoost model from models/
    2. Load Prophet models per category
    3. Generate next N months of predictions
    4. Write predictions back to agg_monthly_spending (prediction columns)
    5. Return a clean summary DataFrame

Usage:
    from src.ml.predictor import SpendingPredictor
    predictor = SpendingPredictor()
    predictions = predictor.predict_next_months(n_months=3)
"""

import pickle
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger

from config.settings import settings


class SpendingPredictor:
    """
    Loads saved models and generates spending predictions for upcoming months.
    """

    def __init__(self):
        self.engine = create_engine(settings.db.url, echo=False)
        self.models_dir = settings.paths.models_dir
        self._xgb_bundle: dict | None = None
        self._prophet_models: dict = {}
        self._load_latest_models()

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_latest_models(self):
        """Load the most recently trained models from disk."""
        # XGBoost — find latest .pkl
        xgb_files = sorted(self.models_dir.glob("xgb_spending_*.pkl"))
        if xgb_files:
            with open(xgb_files[-1], "rb") as f:
                self._xgb_bundle = pickle.load(f)
            logger.info(f"XGBoost loaded: {xgb_files[-1].name}")
        else:
            logger.warning("No XGBoost model found — run pipeline.py --train-ml first")

        # Prophet — find latest per category
        prophet_files = sorted(self.models_dir.glob("prophet_*.pkl"))
        for pf in prophet_files:
            # Filename: prophet_{CATEGORY}_{VERSION}.pkl
            parts = pf.stem.split("_")
            if len(parts) >= 3:
                # version is last two parts (YYYYMMDD_HHMM), category is everything in between
                category = "_".join(parts[1:-2])
                # Only load if it's the most recent version for this category
                if category not in self._prophet_models:
                    with open(pf, "rb") as f:
                        self._prophet_models[category] = pickle.load(f)
        if self._prophet_models:
            logger.info(f"Prophet models loaded: {list(self._prophet_models.keys())}")

    def _load_historical_features(self) -> pd.DataFrame:
        """Pull monthly aggregated features from DB for inference."""
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT year, month, category_code,
                           total_spend, transaction_count
                    FROM agg_monthly_spending
                    ORDER BY year, month, category_code
                """),
                conn
            )
        return df

    # ── XGBoost Predictions ────────────────────────────────────────────────────

    def _predict_xgboost(
        self,
        hist_df: pd.DataFrame,
        n_months: int = 3
    ) -> pd.DataFrame:
        """
        Use XGBoost to predict spending for next N months per category.
        
        Strategy: iterative forecasting — use each prediction as a lag
        for the next month's prediction.
        """
        if not self._xgb_bundle:
            return pd.DataFrame()

        model = self._xgb_bundle["model"]
        le = self._xgb_bundle["label_encoder"]
        feature_cols = self._xgb_bundle["feature_cols"]

        # Build lag table from history
        hist = hist_df.copy()
        hist["period"] = hist["year"] * 12 + hist["month"]
        hist = hist.sort_values(["category_code", "period"])

        results = []
        categories = hist["category_code"].unique()

        # Get last known period
        last_period = hist["period"].max()
        last_year = last_period // 12
        last_month = last_period % 12
        if last_month == 0:
            last_month = 12
            last_year -= 1

        for category in categories:
            cat_hist = hist[hist["category_code"] == category].copy()
            if len(cat_hist) < 1:
                continue

            # Encode category (skip if not seen during training)
            try:
                cat_encoded = le.transform([category])[0]
            except ValueError:
                continue

            # Seed lags from last 3 months of history
            lags = cat_hist.sort_values("period")["total_spend"].tail(3).tolist()
            while len(lags) < 3:
                lags.insert(0, 0.0)

            total_hist = hist.groupby("period")["total_spend"].sum()
            last_total = total_hist.tail(1).values[0] if len(total_hist) else 0

            current_year = last_year
            current_month = last_month

            for _ in range(n_months):
                # Advance month
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1

                quarter = ((current_month - 1) // 3) + 1

                X_pred = pd.DataFrame([{
                    "lag_1m_spend":   lags[-1],
                    "lag_2m_spend":   lags[-2],
                    "lag_3m_spend":   lags[-3],
                    "month":          current_month,
                    "quarter":        quarter,
                    "category_encoded": cat_encoded,
                    "lag_1m_total":   last_total,
                }])

                # Ensure correct column order
                X_pred = X_pred[feature_cols]
                predicted = float(model.predict(X_pred)[0])
                predicted = max(0, predicted)  # No negative spending

                results.append({
                    "year":           current_year,
                    "month":          current_month,
                    "category_code":  category,
                    "predicted_spend": round(predicted, 2),
                    "model":          "xgboost",
                })

                # Roll lags forward
                lags = lags[1:] + [predicted]
                last_total = sum(
                    r["predicted_spend"] for r in results
                    if r["year"] == current_year and r["month"] == current_month
                )

        return pd.DataFrame(results)

    # ── Prophet Predictions ────────────────────────────────────────────────────

    def _predict_prophet(self, n_months: int = 3) -> pd.DataFrame:
        """Generate Prophet forecasts for each category."""
        if not self._prophet_models:
            return pd.DataFrame()

        results = []
        for category, model in self._prophet_models.items():
            try:
                future = model.make_future_dataframe(periods=n_months, freq="MS")
                forecast = model.predict(future)
                # Only return future rows
                last_train_date = model.history["ds"].max()
                future_rows = forecast[forecast["ds"] > last_train_date]

                for _, row in future_rows.iterrows():
                    results.append({
                        "year":             row["ds"].year,
                        "month":            row["ds"].month,
                        "category_code":    category,
                        "predicted_spend":  max(0, round(row["yhat"], 2)),
                        "prediction_lower": max(0, round(row["yhat_lower"], 2)),
                        "prediction_upper": max(0, round(row["yhat_upper"], 2)),
                        "model":            "prophet",
                    })
            except Exception as exc:
                logger.warning(f"Prophet prediction failed for {category}: {exc}")

        return pd.DataFrame(results)

    # ── Main Prediction Interface ──────────────────────────────────────────────

    def predict_next_months(self, n_months: int = 3) -> pd.DataFrame:
        """
        Generate spending predictions for the next N months.
        
        Combines XGBoost point estimates with Prophet confidence intervals.
        Writes predictions back to agg_monthly_spending table.
        
        Args:
            n_months: How many future months to predict (default: 3)
            
        Returns:
            DataFrame with predicted spending per category per month.
        """
        logger.info(f"Generating {n_months}-month predictions...")

        hist_df = self._load_historical_features()
        if hist_df.empty:
            logger.warning("No historical data in DB — cannot predict")
            return pd.DataFrame()

        xgb_preds = self._predict_xgboost(hist_df, n_months)
        prophet_preds = self._predict_prophet(n_months)

        # Merge: use XGBoost for point estimate, Prophet for intervals
        if xgb_preds.empty and prophet_preds.empty:
            logger.warning("No predictions generated — are models trained?")
            return pd.DataFrame()

        if not xgb_preds.empty and not prophet_preds.empty:
            merged = xgb_preds.merge(
                prophet_preds[["year", "month", "category_code",
                               "prediction_lower", "prediction_upper"]],
                on=["year", "month", "category_code"],
                how="left"
            )
        elif not xgb_preds.empty:
            merged = xgb_preds.copy()
            merged["prediction_lower"] = merged["predicted_spend"] * 0.8
            merged["prediction_upper"] = merged["predicted_spend"] * 1.2
        else:
            merged = prophet_preds.rename(columns={"predicted_spend": "predicted_spend"})

        merged["prediction_lower"] = merged.get("prediction_lower", merged["predicted_spend"] * 0.8)
        merged["prediction_upper"] = merged.get("prediction_upper", merged["predicted_spend"] * 1.2)

        # Write to DB
        self._write_predictions_to_db(merged)

        logger.success(f"Predictions generated: {len(merged)} rows")
        return merged

    def _write_predictions_to_db(self, df: pd.DataFrame):
        """Write predictions back into agg_monthly_spending."""
        with self.engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO agg_monthly_spending
                            (year, month, category_code, total_spend,
                             predicted_spend, prediction_lower, prediction_upper)
                        VALUES
                            (:year, :month, :cat, 0,
                             :pred, :lower, :upper)
                        ON DUPLICATE KEY UPDATE
                            predicted_spend   = :pred,
                            prediction_lower  = :lower,
                            prediction_upper  = :upper
                    """),
                    {
                        "year":  int(row["year"]),
                        "month": int(row["month"]),
                        "cat":   row["category_code"],
                        "pred":  float(row["predicted_spend"]),
                        "lower": float(row.get("prediction_lower", 0)),
                        "upper": float(row.get("prediction_upper", 0)),
                    }
                )
        logger.info("Predictions written to agg_monthly_spending")

    def get_next_month_summary(self) -> dict:
        """
        Return a simple dict with next month's predicted total spending.
        Used by automation/scheduler.py for the email notification.
        """
        preds = self.predict_next_months(n_months=1)
        if preds.empty:
            return {}

        total = preds["predicted_spend"].sum()
        by_category = preds.set_index("category_code")["predicted_spend"].to_dict()

        return {
            "total_predicted": round(total, 2),
            "by_category": {k: round(v, 2) for k, v in by_category.items()},
            "period": f"{int(preds.iloc[0]['year'])}-{int(preds.iloc[0]['month']):02d}",
        }
