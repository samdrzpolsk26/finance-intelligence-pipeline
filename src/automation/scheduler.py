"""
src/automation/scheduler.py
============================
APScheduler-based pipeline orchestration.

Watches a folder for new PDF files and automatically triggers
the full pipeline whenever a new bank statement is dropped.

Two modes:
─────────────────────────────────────────────────────────────────
1. FOLDER WATCHER (watchdog)
   - Monitors PDF_WATCH_FOLDER for new .pdf files
   - Triggers pipeline on each new file immediately
   - Moves processed files to PDF_ARCHIVE_FOLDER

2. INTERVAL SCHEDULER (APScheduler)
   - Runs pipeline every N minutes (configurable in .env)
   - Scans watch folder on each interval
   - Also triggers a weekly ML retraining job (Sundays 03:00)

3. EMAIL NOTIFICATIONS (optional)
   - Sends a spending summary email after successful run
   - Requires SMTP credentials in .env

Usage:
    # Start watcher + scheduler
    python pipeline.py --schedule

    # Or import directly:
    from src.automation.scheduler import PipelineScheduler
    scheduler = PipelineScheduler()
    scheduler.start()
"""

import shutil
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

from config.settings import settings


class PDFFileHandler(FileSystemEventHandler):
    """
    Watchdog event handler — triggers pipeline when a new PDF appears.
    """

    def __init__(self, pipeline_fn: Callable[[Path], None]):
        """
        Args:
            pipeline_fn: Callable that accepts a PDF Path and runs the pipeline.
        """
        self._pipeline_fn = pipeline_fn
        self._processing = set()  # Track in-flight files to avoid double-trigger

    def on_created(self, event: FileCreatedEvent):
        """Called when a new file is created in the watched folder."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return

        if str(path) in self._processing:
            logger.debug(f"Already processing {path.name} — skipping duplicate event")
            return

        self._processing.add(str(path))
        logger.info(f"New PDF detected: {path.name}")

        try:
            # Small delay to ensure file is fully written before reading
            time.sleep(2)
            self._pipeline_fn(path)
        except Exception as exc:
            logger.error(f"Pipeline failed for {path.name}: {exc}")
        finally:
            self._processing.discard(str(path))


class PipelineScheduler:
    """
    Orchestrates automated pipeline execution.

    Combines a file system watcher with a time-based scheduler.
    """

    def __init__(self, pipeline_fn: Callable[[Path], None]):
        """
        Args:
            pipeline_fn: The main pipeline callable (from pipeline.py).
                         Signature: pipeline_fn(pdf_path: Path) -> None
        """
        self._pipeline_fn = pipeline_fn
        self._scheduler = BackgroundScheduler(
            max_instances=settings.scheduler.max_instances,
            coalesce=settings.scheduler.coalesce,
            timezone="Europe/Warsaw",
        )
        self._observer: Observer | None = None

        # Ensure watch and archive folders exist
        settings.paths.pdf_watch_folder.mkdir(parents=True, exist_ok=True)
        settings.paths.pdf_archive_folder.mkdir(parents=True, exist_ok=True)

    # ── Jobs ───────────────────────────────────────────────────────────────────

    def _scan_folder_job(self):
        """
        Interval job: scan watch folder for any PDFs that weren't
        caught by the file watcher (e.g., files placed while scheduler was down).
        """
        watch_folder = settings.paths.pdf_watch_folder
        pdfs = list(watch_folder.glob("*.pdf"))

        if not pdfs:
            logger.debug("Folder scan: no new PDFs found")
            return

        logger.info(f"Folder scan found {len(pdfs)} PDF(s) to process")
        for pdf_path in pdfs:
            try:
                self._pipeline_fn(pdf_path)
                self._archive_file(pdf_path)
            except Exception as exc:
                logger.error(f"Scheduled pipeline failed for {pdf_path.name}: {exc}")

    def _weekly_retrain_job(self):
        """
        Weekly cron job: retrain ML models with accumulated data.
        Runs every Sunday at 03:00 Warsaw time.
        """
        logger.info("Weekly ML retraining job triggered...")
        try:
            # Import here to avoid circular imports
            from src.ml.model_trainer import ModelTrainer
            from src.transform.feature_engineering import FeatureEngineer
            from sqlalchemy import create_engine, text
            import pandas as pd

            engine = create_engine(settings.db.url, echo=False)
            with engine.connect() as conn:
                df = pd.read_sql(
                    text("SELECT * FROM fact_transactions ORDER BY transaction_date"),
                    conn, parse_dates=["transaction_date"]
                )

            if df.empty:
                logger.warning("No data for retraining — skipping")
                return

            fe = FeatureEngineer()
            monthly_features = fe.create_monthly_ml_features(df)

            trainer = ModelTrainer()
            results = trainer.train(monthly_features)
            logger.success(f"Weekly retrain complete: {results.get('xgboost', {}).get('status')}")

        except Exception as exc:
            logger.error(f"Weekly retraining failed: {exc}")

    # ── File Archiving ─────────────────────────────────────────────────────────

    def _archive_file(self, pdf_path: Path):
        """Move processed PDF to archive folder."""
        dest = settings.paths.pdf_archive_folder / pdf_path.name
        # If file with same name exists in archive, append timestamp
        if dest.exists():
            stem = pdf_path.stem
            suffix = pdf_path.suffix
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = settings.paths.pdf_archive_folder / f"{stem}_{ts}{suffix}"
        shutil.move(str(pdf_path), str(dest))
        logger.info(f"Archived: {pdf_path.name} → {dest.name}")

    # ── Email Notification ─────────────────────────────────────────────────────

    def send_email_summary(self, summary: dict):
        """
        Send an HTML email with the pipeline run summary.
        Only runs if SMTP credentials are configured in .env.
        """
        cfg = settings.notifications
        if not cfg.enabled:
            logger.debug("Email notifications not configured — skipping")
            return

        total = summary.get("total_predicted", 0)
        period = summary.get("period", "N/A")
        by_cat = summary.get("by_category", {})

        rows = "\n".join(
            f"<tr><td style='padding:6px;border-bottom:1px solid #333'>{cat}</td>"
            f"<td style='padding:6px;border-bottom:1px solid #333;text-align:right'>"
            f"<b>{val:,.2f} PLN</b></td></tr>"
            for cat, val in sorted(by_cat.items(), key=lambda x: -x[1])
        )

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0F1117;color:#C8CDD6;padding:24px">
        <h2 style="color:#4CC9F0">💳 Finance Intelligence Pipeline</h2>
        <p>Pipeline run completed. Predictions for <b style="color:white">{period}</b>:</p>
        <table style="border-collapse:collapse;width:400px;background:#1A1D27">
          <thead>
            <tr style="background:#2D3142">
              <th style="padding:8px;text-align:left">Category</th>
              <th style="padding:8px;text-align:right">Predicted</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
          <tfoot>
            <tr style="background:#2D3142">
              <td style="padding:8px"><b>TOTAL</b></td>
              <td style="padding:8px;text-align:right"><b>{total:,.2f} PLN</b></td>
            </tr>
          </tfoot>
        </table>
        <p style="color:#8B92A5;font-size:12px;margin-top:24px">Finance Intelligence Pipeline</p>
        </body></html>
        """

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"💳 Finance Report — {period}"
            msg["From"] = cfg.smtp_user
            msg["To"] = cfg.notify_email
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
                server.starttls()
                server.login(cfg.smtp_user, cfg.smtp_password)
                server.sendmail(cfg.smtp_user, cfg.notify_email, msg.as_string())

            logger.info(f"Email notification sent to {cfg.notify_email}")
        except Exception as exc:
            logger.warning(f"Email failed: {exc}")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, block: bool = True):
        """
        Start both the file watcher and the APScheduler.
        
        Args:
            block: If True, blocks the main thread (for standalone use).
                   Set False when integrating into an existing event loop.
        """
        # ── APScheduler jobs ───────────────────────────────────────────────────
        self._scheduler.add_job(
            self._scan_folder_job,
            trigger=IntervalTrigger(minutes=settings.scheduler.interval_minutes),
            id="folder_scan",
            name="PDF Folder Scanner",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self._weekly_retrain_job,
            trigger=CronTrigger(day_of_week="sun", hour=3, minute=0),
            id="weekly_retrain",
            name="Weekly ML Retraining",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info(
            f"Scheduler started — scanning every {settings.scheduler.interval_minutes} min | "
            f"ML retrain: Sundays 03:00 Warsaw"
        )

        # ── Watchdog file observer ─────────────────────────────────────────────
        handler = PDFFileHandler(pipeline_fn=self._pipeline_fn)
        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(settings.paths.pdf_watch_folder),
            recursive=False
        )
        self._observer.start()
        logger.info(f"File watcher active: {settings.paths.pdf_watch_folder}")

        if block:
            logger.info("Scheduler running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(5)
            except (KeyboardInterrupt, SystemExit):
                self.stop()

    def stop(self):
        """Gracefully shut down scheduler and file watcher."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped gracefully")
