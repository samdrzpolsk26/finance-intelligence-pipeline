"""
config/settings.py
==================
Central configuration hub for the Finance Intelligence Pipeline.

Uses python-dotenv to load .env variables and Pydantic for validation.
All other modules import from here — never hardcode credentials elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from loguru import logger

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class DatabaseConfig(BaseModel):
    """MySQL connection parameters."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 3306))
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "")
    name: str = os.getenv("DB_NAME", "finance_db")

    @property
    def url(self) -> str:
        """SQLAlchemy connection string."""
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
            f"?charset=utf8mb4"
        )

    @property
    def url_safe(self) -> str:
        """Connection string with masked password (for logging)."""
        return (
            f"mysql+pymysql://{self.user}:****"
            f"@{self.host}:{self.port}/{self.name}"
        )


class PathConfig(BaseModel):
    """File system paths."""
    base_dir: Path = BASE_DIR
    pdf_watch_folder: Path = Path(os.getenv("PDF_WATCH_FOLDER", str(BASE_DIR / "data/inbox")))
    pdf_archive_folder: Path = Path(os.getenv("PDF_ARCHIVE_FOLDER", str(BASE_DIR / "data/archive")))
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports" / "figures"
    outputs_dir: Path = BASE_DIR / "outputs"
    notebooks_dir: Path = BASE_DIR / "notebooks"

    model_config = {"arbitrary_types_allowed": True}

    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for path in [
            self.pdf_watch_folder,
            self.pdf_archive_folder,
            self.models_dir,
            self.reports_dir,
            self.outputs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class MLConfig(BaseModel):
    """Machine learning parameters."""
    retrain_threshold: int = int(os.getenv("ML_RETRAIN_THRESHOLD", 30))
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5


class SchedulerConfig(BaseModel):
    """APScheduler settings."""
    interval_minutes: int = int(os.getenv("SCHEDULER_INTERVAL_MINUTES", 60))
    max_instances: int = 1
    coalesce: bool = True  # If multiple runs were missed, run only once


class NotificationConfig(BaseModel):
    """Email notification settings."""
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", 587))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    notify_email: str = os.getenv("NOTIFY_EMAIL", "")

    @property
    def enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.notify_email)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = Path(os.getenv("LOG_FILE", str(BASE_DIR / "outputs/pipeline.log")))

    model_config = {"arbitrary_types_allowed": True}


class Settings(BaseModel):
    """Master settings object — import this in all modules."""
    db: DatabaseConfig = DatabaseConfig()
    paths: PathConfig = PathConfig()
    ml: MLConfig = MLConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    notifications: NotificationConfig = NotificationConfig()
    logging: LoggingConfig = LoggingConfig()
    bank_format: str = os.getenv("BANK_FORMAT", "erste_poland")

    model_config = {"arbitrary_types_allowed": True}


# ── Singleton instance ─────────────────────────────────────────────────────────
settings = Settings()


def configure_logging():
    """Setup loguru with file + console handlers."""
    settings.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.logging.level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(
        settings.logging.log_file,
        level=settings.logging.level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
    )
