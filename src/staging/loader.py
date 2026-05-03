"""
src/staging/loader.py
======================
MySQL staging layer loader for the Finance Intelligence Pipeline.

Responsibilities:
1. Create a load record in stg_pdf_loads (audit trail)
2. Insert raw parsed transactions into stg_raw_transactions
3. Enforce idempotency — the same PDF cannot be loaded twice
   (checked via SHA-256 file hash)
4. Handle partial failures gracefully — rollback on error

Why staging?
    Separating raw data from transformed data is a core data engineering
    principle. If our transform logic changes later, we can re-process
    raw data without re-parsing the PDF. It's also an audit trail.

Usage:
    from src.staging.loader import StagingLoader
    loader = StagingLoader()
    load_id = loader.load(df, file_name="statement.pdf", file_hash="abc...", file_size=102400)
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text, Engine
from loguru import logger

from config.settings import settings


class StagingLoader:
    """
    Loads parsed transaction DataFrames into MySQL staging tables.
    
    Attributes:
        engine (Engine): SQLAlchemy connection engine.
    """

    def __init__(self):
        self.engine = self._create_engine()

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with connection pooling."""
        engine = create_engine(
            settings.db.url,
            pool_pre_ping=True,     # Test connection before using from pool
            pool_recycle=3600,      # Recycle connections every hour
            echo=False,
        )
        logger.debug(f"Database engine: {settings.db.url_safe}")
        return engine

    def check_already_loaded(self, file_hash: str) -> Optional[int]:
        """
        Check if a PDF was already loaded (by SHA-256 hash).
        
        Returns:
            load_id if already loaded, None otherwise.
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT load_id FROM stg_pdf_loads WHERE file_hash = :hash"),
                {"hash": file_hash}
            ).fetchone()
        if result:
            logger.warning(f"PDF with hash {file_hash[:16]}... already loaded (load_id={result[0]}). Skipping.")
            return result[0]
        return None

    def _create_load_record(
        self,
        conn,
        file_name: str,
        file_hash: str,
        file_size: int,
        account_number: Optional[str],
        n_transactions: int,
        status: str = "SUCCESS",
        error_msg: Optional[str] = None
    ) -> int:
        """
        Insert a record into stg_pdf_loads and return the load_id.
        This creates the audit trail for each pipeline run.
        """
        result = conn.execute(
            text("""
                INSERT INTO stg_pdf_loads
                    (file_name, file_hash, file_size_bytes, account_number,
                     transactions_parsed, status, error_message)
                VALUES
                    (:file_name, :file_hash, :file_size, :account,
                     :n_tx, :status, :error)
            """),
            {
                "file_name":  file_name,
                "file_hash":  file_hash,
                "file_size":  file_size,
                "account":    account_number,
                "n_tx":       n_transactions,
                "status":     status,
                "error":      error_msg,
            }
        )
        load_id = result.lastrowid
        logger.info(f"Created load record: load_id={load_id}, file={file_name}")
        return load_id

    def _df_to_staging_rows(self, df: pd.DataFrame, load_id: int) -> list[dict]:
        """
        Transform parsed DataFrame rows into dicts matching stg_raw_transactions schema.
        """
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "load_id":           load_id,
                "transaction_date":  row["transaction_date"].date() if pd.notna(row["transaction_date"]) else None,
                "booking_date":      row["booking_date"].date() if pd.notna(row.get("booking_date")) else None,
                "raw_description":   str(row["raw_description"])[:65535],
                "amount_raw":        str(row["amount_raw"])[:30],
                "balance_raw":       str(row["balance_raw"])[:30],
                "amount_pln":        float(row["amount_pln"]) if pd.notna(row["amount_pln"]) else None,
                "balance_pln":       float(row["balance_pln"]) if pd.notna(row["balance_pln"]) else None,
                "currency":          str(row.get("currency", "PLN")),
                "transaction_type":  str(row.get("transaction_type", "UNKNOWN")),
                "reference_number":  str(row["reference_number"]) if row.get("reference_number") else None,
                "merchant_raw":      str(row["merchant_raw"])[:512] if row.get("merchant_raw") else None,
                "city_raw":          str(row["city_raw"])[:100] if row.get("city_raw") else None,
            })
        return rows

    def load(
        self,
        df: pd.DataFrame,
        file_name: str,
        file_hash: str,
        file_size: int,
        account_number: Optional[str] = None,
    ) -> Optional[int]:
        """
        Load a parsed transactions DataFrame into the staging layer.
        
        Args:
            df:             Parsed transactions from ErsteParser.parse()
            file_name:      Original PDF filename (for audit log)
            file_hash:      SHA-256 of the file (for deduplication)
            file_size:      File size in bytes
            account_number: Account number extracted from PDF
            
        Returns:
            load_id (int) on success, None on duplicate/error.
        """
        # ── Idempotency check ──────────────────────────────────────────────────
        existing = self.check_already_loaded(file_hash)
        if existing:
            return None

        if df is None or df.empty:
            logger.warning("Empty DataFrame — nothing to load")
            return None

        n_transactions = len(df)
        logger.info(f"Loading {n_transactions} transactions to staging...")

        # ── Transactional insert ───────────────────────────────────────────────
        with self.engine.begin() as conn:  # auto-commit or rollback
            try:
                load_id = self._create_load_record(
                    conn, file_name, file_hash, file_size, account_number, n_transactions
                )

                rows = self._df_to_staging_rows(df, load_id)

                # Batch insert for performance
                BATCH_SIZE = 500
                inserted = 0
                for i in range(0, len(rows), BATCH_SIZE):
                    batch = rows[i:i + BATCH_SIZE]
                    conn.execute(
                        text("""
                            INSERT INTO stg_raw_transactions (
                                load_id, transaction_date, booking_date,
                                raw_description, amount_raw, balance_raw,
                                amount_pln, balance_pln, currency,
                                transaction_type, reference_number,
                                merchant_raw, city_raw
                            ) VALUES (
                                :load_id, :transaction_date, :booking_date,
                                :raw_description, :amount_raw, :balance_raw,
                                :amount_pln, :balance_pln, :currency,
                                :transaction_type, :reference_number,
                                :merchant_raw, :city_raw
                            )
                        """),
                        batch
                    )
                    inserted += len(batch)
                    logger.debug(f"Inserted batch: {inserted}/{n_transactions}")

                logger.success(f"Staging load complete: {inserted} rows, load_id={load_id}")
                return load_id

            except Exception as exc:
                logger.error(f"Staging load failed: {exc}")
                raise

    def get_unprocessed(self, load_id: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve rows from staging that haven't been moved to the data mart yet.
        
        Args:
            load_id: If provided, only return rows from this load.
                     If None, returns all unprocessed rows.
        """
        query = "SELECT * FROM stg_raw_transactions WHERE is_processed = 0"
        params = {}
        if load_id:
            query += " AND load_id = :load_id"
            params["load_id"] = load_id

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        logger.info(f"Retrieved {len(df)} unprocessed staging rows")
        return df

    def mark_as_processed(self, raw_ids: list[int]):
        """Mark staging rows as processed after successful data mart load."""
        if not raw_ids:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE stg_raw_transactions SET is_processed = 1 WHERE raw_id IN :ids"),
                {"ids": tuple(raw_ids)}
            )
        logger.debug(f"Marked {len(raw_ids)} rows as processed")

    def get_load_summary(self) -> pd.DataFrame:
        """Return summary of all PDF loads for monitoring."""
        with self.engine.connect() as conn:
            return pd.read_sql(
                text("SELECT * FROM v_staging_summary ORDER BY loaded_at DESC"),
                conn
            )
