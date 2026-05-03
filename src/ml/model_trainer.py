"""
src/ml/model_trainer.py
========================
ML model training for monthly spending prediction.

Two complementary models:
─────────────────────────────────────────────────────────────────
1. PROPHET (per-category time series)
   - Best for: overall trend + seasonality detection
   - Handles missing months, holidays, changepoints
   - Produces: trend, weekly seasonality, forecast intervals
   - One Prophet model per spending category

2. XGBOOST (feature-based regression)
   - Best for: explaining what drives spending changes
   - Uses: lag features, calendar, rolling windows
   - Produces: point predictions + feature importance
   - One XGBoost model for all categories combined (category as feature)

3. SHAP EXPLAINABILITY
   - Explains every XGBoost prediction
   - "Why does it predict 800 PLN for groceries next month?"

Models are serialized to the models/ directory with version stamps.

Usage:
    from src.ml.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    results = trainer.train(df)
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from loguru import logger

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not installed — skipping time series models")
    PROPHET_AVAILABLE = False

from config.settings import settings


class ModelTrainer:
    """
    Trains XGBoost and Prophet models for spending prediction.
    
    Attributes:
        models_dir (Path): Where trained models are saved.
        version (str):     Training run version stamp (ISO datetime).
    """

    def __init__(self):
        self.models_dir = settings.paths.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        self._label_encoder = LabelEncoder()
        self._xgb_model: XGBRegressor | None = None
        self._prophet_models: dict[str, "Prophet"] = {}
        self._shap_explainer = None

    # ── Main Training Entry Point ──────────────────────────────────────────────

    def train(self, monthly_df: pd.DataFrame) -> dict:
        """
        Train all models on the monthly aggregated feature DataFrame.
        
        Args:
            monthly_df: Output of FeatureEngineer.create_monthly_ml_features()
            
        Returns:
            Dictionary with training results and model metrics.
        """
        logger.info(f"Starting ML training (version {self.version})...")

        if len(monthly_df) < 3:
            logger.warning("Insufficient data for training (need ≥ 3 months). Skipping.")
            return {"status": "insufficient_data", "min_required": 3, "available": len(monthly_df)}

        results = {}

        # ── Train XGBoost ──────────────────────────────────────────────────────
        xgb_results = self._train_xgboost(monthly_df)
        results["xgboost"] = xgb_results

        # ── Train Prophet (one per category) ──────────────────────────────────
        if PROPHET_AVAILABLE:
            prophet_results = self._train_prophet_models(monthly_df)
            results["prophet"] = prophet_results
        else:
            results["prophet"] = {"status": "unavailable"}

        # ── SHAP Analysis ─────────────────────────────────────────────────────
        if self._xgb_model:
            shap_results = self._compute_shap(monthly_df)
            results["shap"] = shap_results

        # ── Save metadata ──────────────────────────────────────────────────────
        metadata = {
            "version": self.version,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(monthly_df),
            "categories": monthly_df["category_code"].unique().tolist(),
            "date_range": {
                "min": f"{monthly_df['year'].min()}-{monthly_df['month'].min():02d}",
                "max": f"{monthly_df['year'].max()}-{monthly_df['month'].max():02d}",
            },
            "results": results,
        }
        self._save_json(metadata, "model_metadata.json")

        logger.success(f"Training complete — version {self.version}")
        return results

    # ── XGBoost Training ───────────────────────────────────────────────────────

    def _train_xgboost(self, df: pd.DataFrame) -> dict:
        """
        Train XGBoost regressor to predict monthly spend per category.
        
        Feature set:
            - Lag features: lag_1m_spend, lag_2m_spend, lag_3m_spend
            - Calendar: month, quarter
            - Category (label-encoded)
            - Total monthly spend lag
        """
        logger.info("Training XGBoost regressor...")

        feature_cols = [
            "lag_1m_spend", "lag_2m_spend", "lag_3m_spend",
            "month", "quarter", "category_encoded",
            "lag_1m_total",
        ]

        df = df.copy()
        df["quarter"] = ((df["month"] - 1) // 3) + 1

        # Encode categories
        df["category_encoded"] = self._label_encoder.fit_transform(df["category_code"])

        # Drop rows with NaN in any feature
        df_clean = df.dropna(subset=feature_cols + ["total_spend"])
        if len(df_clean) < 3:
            return {"status": "insufficient_clean_data", "rows": len(df_clean)}

        X = df_clean[feature_cols]
        y = df_clean["total_spend"]

        # XGBoost params from config
        cfg = settings.ml
        self._xgb_model = XGBRegressor(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=0,
            objective="reg:squarederror",
        )

        # Time Series Cross-Validation (don't shuffle financial data!)
        tscv = TimeSeriesSplit(n_splits=min(cfg.cv_folds, len(df_clean) // 2))
        cv_scores = cross_val_score(
            self._xgb_model, X, y,
            cv=tscv, scoring="neg_mean_absolute_percentage_error"
        )
        cv_mape = -cv_scores.mean()

        # Final fit on all data
        self._xgb_model.fit(X, y)

        # In-sample metrics
        y_pred = self._xgb_model.predict(X)
        train_mae = mean_absolute_error(y, y_pred)
        train_mape = mean_absolute_percentage_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)

        # Save model
        model_path = self.models_dir / f"xgb_spending_{self.version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self._xgb_model,
                "label_encoder": self._label_encoder,
                "feature_cols": feature_cols,
                "version": self.version,
            }, f)

        # Feature importance chart
        self._plot_feature_importance(feature_cols)

        logger.success(
            f"XGBoost trained | CV MAPE: {cv_mape:.4f} | "
            f"Train MAE: {train_mae:.2f} PLN | R²: {train_r2:.4f}"
        )

        return {
            "status": "success",
            "model_path": str(model_path),
            "n_samples": len(df_clean),
            "cv_mape": round(cv_mape, 6),
            "train_mae": round(train_mae, 2),
            "train_mape": round(train_mape, 6),
            "train_r2": round(train_r2, 6),
        }

    def _plot_feature_importance(self, feature_cols: list[str]):
        """Save feature importance bar chart."""
        importances = self._xgb_model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        ax.barh(
            [feature_cols[i] for i in sorted_idx],
            importances[sorted_idx],
            color="#4CC9F0", alpha=0.85
        )
        ax.set_title("🤖 XGBoost — Feature Importance", fontsize=14, color="white", pad=10)
        ax.set_xlabel("Importance Score", fontsize=10, color="#C8CDD6")
        ax.tick_params(colors="#C8CDD6")
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        path = settings.paths.reports_dir / "ml_feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
        plt.close(fig)
        logger.info("Feature importance chart saved")

    # ── Prophet Training ───────────────────────────────────────────────────────

    def _train_prophet_models(self, df: pd.DataFrame) -> dict:
        """
        Train one Prophet model per spending category.
        
        Prophet expects a DataFrame with columns 'ds' (date) and 'y' (value).
        We create monthly dates and fit per category.
        """
        logger.info("Training Prophet time series models...")
        results = {}

        for category in df["category_code"].unique():
            cat_data = df[df["category_code"] == category].copy()
            if len(cat_data) < 3:
                continue

            # Build Prophet-format DataFrame
            prophet_df = pd.DataFrame({
                "ds": pd.to_datetime(cat_data.apply(
                    lambda r: f"{int(r['year'])}-{int(r['month']):02d}-01", axis=1
                )),
                "y": cat_data["total_spend"].values
            }).sort_values("ds").reset_index(drop=True)

            model = Prophet(
                changepoint_prior_scale=settings.ml.prophet_changepoint_prior_scale,
                seasonality_prior_scale=settings.ml.prophet_seasonality_prior_scale,
                yearly_seasonality=len(prophet_df) >= 12,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.80,
                growth="linear",
            )

            # Suppress Prophet's verbose output
            import logging
            logging.getLogger("prophet").setLevel(logging.WARNING)
            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

            model.fit(prophet_df)
            self._prophet_models[category] = model

            # 3-month forecast for evaluation
            future = model.make_future_dataframe(periods=3, freq="MS")
            forecast = model.predict(future)

            # Save model
            model_path = self.models_dir / f"prophet_{category}_{self.version}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            results[category] = {
                "status": "success",
                "n_samples": len(prophet_df),
                "model_path": str(model_path),
                "next_3_months_forecast": forecast[forecast["ds"] > prophet_df["ds"].max()][
                    ["ds", "yhat", "yhat_lower", "yhat_upper"]
                ].round(2).to_dict("records")
            }
            logger.debug(f"Prophet model trained for {category} ({len(prophet_df)} months)")

        logger.success(f"Prophet: {len(results)} category models trained")
        return results

    # ── SHAP Analysis ──────────────────────────────────────────────────────────

    def _compute_shap(self, df: pd.DataFrame) -> dict:
        """
        Compute SHAP values to explain model predictions.
        
        SHAP (SHapley Additive exPlanations) answers:
        "Why did the model predict X for this month?"
        
        Generates a summary plot saved to reports/figures/.
        """
        logger.info("Computing SHAP values...")

        feature_cols = [
            "lag_1m_spend", "lag_2m_spend", "lag_3m_spend",
            "month", "quarter", "category_encoded", "lag_1m_total"
        ]

        df = df.copy()
        df["quarter"] = ((df["month"] - 1) // 3) + 1
        df["category_encoded"] = self._label_encoder.transform(df["category_code"])
        df_clean = df.dropna(subset=feature_cols)

        if df_clean.empty:
            return {"status": "no_data"}

        X = df_clean[feature_cols]

        try:
            explainer = shap.TreeExplainer(self._xgb_model)
            shap_values = explainer.shap_values(X)
            self._shap_explainer = explainer

            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0F1117")
            shap.summary_plot(
                shap_values, X,
                feature_names=feature_cols,
                plot_type="bar",
                show=False,
                color="#4CC9F0"
            )
            plt.title("🔍 SHAP Feature Impact on Spending Prediction",
                      fontsize=14, color="white", pad=10)
            plt.tight_layout()

            path = settings.paths.reports_dir / "ml_shap_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
            plt.close()

            # Compute mean |SHAP| per feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            feature_shap = dict(zip(feature_cols, mean_shap.round(4).tolist()))

            logger.success("SHAP analysis complete")
            return {
                "status": "success",
                "plot_path": str(path),
                "mean_abs_shap": feature_shap,
                "top_feature": max(feature_shap, key=feature_shap.get),
            }

        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
            return {"status": "failed", "error": str(exc)}

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _save_json(self, data: dict, filename: str):
        """Save a dictionary as JSON to models directory."""
        path = self.models_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Saved metadata: {filename}")
