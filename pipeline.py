"""
pipeline.py
===========
Finance Intelligence Pipeline — Main Entry Point

This is the single command you run for everything.
Orchestrates all stages in order:

    PDF → Parse → Stage → Categorize → Engineer → DataMart → EDA → ML

CLI Usage:
    # Full pipeline on a PDF
    python pipeline.py --pdf ./data/inbox/statement.pdf

    # Full pipeline + train ML models
    python pipeline.py --pdf ./data/inbox/statement.pdf --train-ml

    # EDA only (data already in DB from a previous run)
    python pipeline.py --eda-only

    # Run predictions only (models already trained)
    python pipeline.py --predict-only

    # Start automated scheduler + file watcher
    python pipeline.py --schedule

    # Health check (test DB connection + summarize data mart)
    python pipeline.py --health

Author: github.com/samdrzpolsk26
"""

import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import settings, configure_logging

console = Console()

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║          FINANCE INTELLIGENCE PIPELINE  v1.0                  ║
║          by github.com/samdrzpolsk26                          ║
╚═══════════════════════════════════════════════════════════════╝
"""


def run_pipeline(pdf_path: Path, train_ml: bool = False) -> dict:
    """
    Execute the full ETL pipeline on a single PDF file.
    
    Stages:
        1. PDF Parsing       — Extract transactions from PDF
        2. Staging Load      — Write raw data to MySQL staging
        3. Categorization    — Map merchants to categories
        4. Feature Eng.      — Derive date/rolling/lag features
        5. Data Mart         — Write enriched data to fact tables
        6. Aggregate Refresh — Rebuild monthly/daily aggregates
        7. EDA               — Generate visualization charts
        8. ML Training       — (Optional) Train/retrain models
        9. Predictions       — Generate next-month forecasts
    
    Args:
        pdf_path:   Path to the bank statement PDF
        train_ml:   Whether to retrain ML models after loading
        
    Returns:
        Summary dict with stats from each stage.
    """
    from src.ingestion.pdf_parser import ErsteParser
    from src.staging.loader import StagingLoader
    from src.transform.categorizer import Categorizer
    from src.transform.feature_engineering import FeatureEngineer
    from src.datamart.builder import DataMartBuilder
    from src.eda.analysis import EDAEngine

    summary = {"pdf": str(pdf_path), "stages": {}}

    # ══ STAGE 1: PDF Parsing ══════════════════════════════════════════════════
    console.print("\n[bold cyan]▶ Stage 1/9 — PDF Parsing[/bold cyan]")
    parser = ErsteParser(pdf_path)
    raw_df = parser.parse()

    if raw_df.empty:
        console.print("[bold red]✗ No transactions parsed — check PDF format[/bold red]")
        return summary

    summary["stages"]["parse"] = {
        "transactions": len(raw_df),
        "date_range": f"{raw_df['transaction_date'].min().date()} → {raw_df['transaction_date'].max().date()}",
        "account": parser.account_number,
    }
    console.print(f"  [green]✓ {len(raw_df)} transactions parsed[/green]")

    # ══ STAGE 2: Staging Load ══════════════════════════════════════════════════
    console.print("[bold cyan]▶ Stage 2/9 — Staging Load[/bold cyan]")
    loader = StagingLoader()
    load_id = loader.load(
        df=raw_df,
        file_name=pdf_path.name,
        file_hash=parser.file_hash,
        file_size=parser.file_size,
        account_number=parser.account_number,
    )

    if load_id is None:
        console.print("  [yellow]⚠ PDF already loaded — skipping duplicate[/yellow]")
        return summary

    summary["stages"]["staging"] = {"load_id": load_id}
    console.print(f"  [green]✓ Staging load_id={load_id}[/green]")

    # ══ STAGE 3: Categorization ════════════════════════════════════════════════
    console.print("[bold cyan]▶ Stage 3/9 — Categorization[/bold cyan]")
    categorizer = Categorizer()
    categorized_df = categorizer.categorize_dataframe(raw_df)

    summary["stages"]["categorize"] = {
        "categories": categorized_df["category_code"].value_counts().to_dict()
    }
    console.print(f"  [green]✓ {categorized_df['category_code'].nunique()} categories assigned[/green]")

    # ══ STAGE 4: Feature Engineering ═══════════════════════════════════════════
    console.print("[bold cyan]▶ Stage 4/9 — Feature Engineering[/bold cyan]")
    fe = FeatureEngineer()
    enriched_df = fe.transform(categorized_df)
    # Attach raw_id from staging
    staging_df = loader.get_unprocessed(load_id)
    if not staging_df.empty:
        enriched_df["raw_id"] = staging_df["raw_id"].values[:len(enriched_df)]

    summary["stages"]["features"] = {"feature_columns": len(enriched_df.columns)}
    console.print(f"  [green]✓ {len(enriched_df.columns)} features engineered[/green]")

    # ══ STAGE 5 + 6: Data Mart ══════════════════════════════════════════════════
    console.print("[bold cyan]▶ Stage 5/9 — Data Mart Load[/bold cyan]")
    builder = DataMartBuilder()
    inserted = builder.upsert_transactions(enriched_df, load_id)

    console.print("[bold cyan]▶ Stage 6/9 — Aggregate Refresh[/bold cyan]")
    builder.refresh_aggregates()

    summary["stages"]["datamart"] = {"rows_inserted": inserted}
    console.print(f"  [green]✓ {inserted} rows inserted, aggregates refreshed[/green]")

    # Mark staging rows as processed
    staging_ids = staging_df["raw_id"].tolist() if not staging_df.empty else []
    if staging_ids:
        loader.mark_as_processed(staging_ids)

    # ══ STAGE 7: EDA ════════════════════════════════════════════════════════════
    console.print("[bold cyan]▶ Stage 7/9 — EDA & Visualizations[/bold cyan]")
    try:
        eda = EDAEngine()
        charts = eda.run_full_eda()
        summary["stages"]["eda"] = {"charts_generated": len(charts)}
        console.print(f"  [green]✓ {len(charts)} charts saved to reports/figures/[/green]")
    except Exception as exc:
        logger.warning(f"EDA stage failed (non-fatal): {exc}")
        console.print(f"  [yellow]⚠ EDA skipped: {exc}[/yellow]")

    # ══ STAGE 8: ML Training (Optional) ═════════════════════════════════════════
    if train_ml:
        console.print("[bold cyan]▶ Stage 8/9 — ML Training[/bold cyan]")
        try:
            from src.ml.model_trainer import ModelTrainer
            monthly_features = fe.create_monthly_ml_features(enriched_df)
            trainer = ModelTrainer()
            ml_results = trainer.train(monthly_features)
            summary["stages"]["ml_training"] = ml_results
            console.print(
                f"  [green]✓ Models trained | "
                f"XGBoost MAPE: {ml_results.get('xgboost', {}).get('cv_mape', 'N/A')}[/green]"
            )
        except Exception as exc:
            logger.warning(f"ML training failed (non-fatal): {exc}")
            console.print(f"  [yellow]⚠ ML training skipped: {exc}[/yellow]")
    else:
        console.print("[bold dim]▶ Stage 8/9 — ML Training [skipped — use --train-ml][/bold dim]")

    # ══ STAGE 9: Predictions ════════════════════════════════════════════════════
    console.print("[bold cyan]▶ Stage 9/9 — Spending Predictions[/bold cyan]")
    try:
        from src.ml.predictor import SpendingPredictor
        predictor = SpendingPredictor()
        pred_summary = predictor.get_next_month_summary()
        summary["stages"]["predictions"] = pred_summary

        if pred_summary:
            total = pred_summary.get("total_predicted", 0)
            period = pred_summary.get("period", "N/A")
            console.print(
                f"  [green]✓ Next month ({period}): predicted {total:,.2f} PLN[/green]"
            )
        else:
            console.print("  [yellow]⚠ No predictions — train models first with --train-ml[/yellow]")
    except Exception as exc:
        logger.warning(f"Prediction stage failed (non-fatal): {exc}")
        console.print(f"  [yellow]⚠ Predictions skipped: {exc}[/yellow]")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--pdf",         type=click.Path(exists=True), help="Path to PDF bank statement")
@click.option("--train-ml",    is_flag=True, default=False,  help="Retrain ML models after load")
@click.option("--eda-only",    is_flag=True, default=False,  help="Run EDA only (no PDF needed)")
@click.option("--predict-only",is_flag=True, default=False,  help="Run predictions only")
@click.option("--schedule",    is_flag=True, default=False,  help="Start automated scheduler")
@click.option("--health",      is_flag=True, default=False,  help="Health check & DB summary")
def main(pdf, train_ml, eda_only, predict_only, schedule, health):
    """Finance Intelligence Pipeline — Automated Bank Statement Analytics."""

    # Configure logging
    configure_logging()

    console.print(f"[bold cyan]{BANNER}[/bold cyan]")

    # ── Health Check ────────────────────────────────────────────────────────────
    if health:
        from src.datamart.builder import DataMartBuilder
        builder = DataMartBuilder()
        info = builder.get_summary()
        console.print(Panel(
            f"[green]✓ Database connected[/green]\n"
            f"Total transactions: [bold]{info['total_transactions']}[/bold]\n"
            f"Date range: [bold]{info['date_range']['min']} → {info['date_range']['max']}[/bold]\n"
            f"Monthly aggregate rows: [bold]{info['monthly_aggregate_rows']}[/bold]",
            title="🏥 Health Check",
            border_style="green"
        ))
        return

    # ── EDA Only ────────────────────────────────────────────────────────────────
    if eda_only:
        from src.eda.analysis import EDAEngine
        eda = EDAEngine()
        charts = eda.run_full_eda()
        console.print(f"[green]✓ EDA complete: {len(charts)} charts[/green]")
        return

    # ── Predict Only ────────────────────────────────────────────────────────────
    if predict_only:
        from src.ml.predictor import SpendingPredictor
        predictor = SpendingPredictor()
        preds = predictor.predict_next_months(n_months=3)
        console.print(preds.to_string())
        return

    # ── Scheduler Mode ───────────────────────────────────────────────────────────
    if schedule:
        from src.automation.scheduler import PipelineScheduler

        def scheduled_run(pdf_path: Path):
            run_pipeline(pdf_path, train_ml=False)

        scheduler = PipelineScheduler(pipeline_fn=scheduled_run)
        scheduler.start(block=True)
        return

    # ── Full Pipeline Mode ───────────────────────────────────────────────────────
    if not pdf:
        console.print("[bold red]✗ Provide --pdf <path> or use --help[/bold red]")
        sys.exit(1)

    pdf_path = Path(pdf)
    summary = run_pipeline(pdf_path, train_ml=train_ml)

    # Final summary panel
    stages_ok = [k for k, v in summary.get("stages", {}).items() if v]
    console.print(Panel(
        f"[green]Pipeline complete![/green]\n"
        f"Stages: {' → '.join(stages_ok)}\n"
        f"Transactions: {summary['stages'].get('parse', {}).get('transactions', 0)}\n"
        f"Charts: {summary['stages'].get('eda', {}).get('charts_generated', 0)}\n"
        f"See [bold]reports/figures/[/bold] for visualizations",
        title="✅ Done",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
