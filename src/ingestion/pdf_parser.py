"""
src/ingestion/pdf_parser.py
============================
Parser corregido para el formato real de Erste Bank Poland.
Formato real: fecha + descripcion + monto + balance en UNA sola linea.
"""

import re
import hashlib
from pathlib import Path
from typing import Optional

import pdfplumber
import pandas as pd
from loguru import logger


AMOUNT_PATTERN = re.compile(
    r"(-[\d\s]+,\d{2})\s+PLN\s+([\d\s]+,\d{2})\s+PLN\s*$"
)

DATE_PREFIX = re.compile(
    r"^(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\s+(.*)",
    re.IGNORECASE
)

POLISH_MONTHS = {
    "Jan":"01","Feb":"02","Mar":"03","Apr":"04",
    "May":"05","Jun":"06","Jul":"07","Aug":"08",
    "Sep":"09","Oct":"10","Nov":"11","Dec":"12"
}

SKIP_LINES = {
    "Transaction list", "Transaction date", "Booking date",
    "Account:", "Document on:", "Page"
}


def parse_date(date_str: str) -> Optional[str]:
    parts = date_str.strip().split()
    if len(parts) != 3:
        return None
    try:
        day = parts[0].zfill(2)
        month = POLISH_MONTHS.get(parts[1][:3].capitalize(), "01")
        year = parts[2]
        return f"{year}-{month}-{day}"
    except Exception:
        return None


def parse_amount(amount_str: str) -> Optional[float]:
    if not amount_str:
        return None
    try:
        cleaned = amount_str.strip().replace(" ", "").replace(",", ".")
        return float(cleaned)
    except Exception:
        return None


def detect_type(description: str) -> str:
    d = description.upper()
    if "BLIK" in d and "ZAKUP" in d:
        return "BLIK"
    if "PLATNOSC KARTA" in d or "PŁATNOŚĆ KARTĄ" in d or "DOP. VISA" in d:
        return "CARD_PAYMENT"
    if any(k in d for k in ["WYNAGRODZ", "ZWROT Z PODATKU", "ZFSS", "PREMIA"]):
        return "TRANSFER_IN"
    if "TRANSFER TO" in d or "PRZELEW" in d:
        return "TRANSFER_OUT"
    if "OPLATA" in d or "OPŁATA" in d:
        return "BANK_FEE"
    return "UNKNOWN"


class ErsteParser:
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.file_hash = self._sha256()
        self.file_size = self.pdf_path.stat().st_size
        self.account_number = None
        logger.info(f"ErsteParser: {self.pdf_path.name} ({self.file_size:,} bytes)")

    def _sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def parse(self) -> pd.DataFrame:
        logger.info(f"Starting parse: {self.pdf_path.name}")
        all_lines = []

        with pdfplumber.open(self.pdf_path) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages")
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    all_lines.extend(text.split("\n"))

        # Extract account number
        for line in all_lines[:5]:
            m = re.search(r"Account:\s*([\d\s]+)", line)
            if m:
                self.account_number = m.group(1).replace(" ", "")
                logger.info(f"Account: {self.account_number[:4]}...{self.account_number[-4:]}")
                break

        transactions = []
        pending_extra = None  # "Booking date <extra description>" from next line

        for i, line in enumerate(all_lines):
            line = line.strip()
            if not line:
                continue

            # Skip header/footer lines
            if any(line.startswith(s) for s in SKIP_LINES):
                # But capture extra description after "Booking date"
                if line.startswith("Booking date"):
                    extra = line.replace("Booking date", "").strip()
                    if extra and transactions:
                        transactions[-1]["raw_description"] += " " + extra
                continue

            # Try to match: DATE + description + -amount PLN balance PLN
            date_match = DATE_PREFIX.match(line)
            if not date_match:
                continue

            date_str = date_match.group(1)
            rest = date_match.group(2)

            amount_match = AMOUNT_PATTERN.search(rest)
            if not amount_match:
                continue

            amount_raw = amount_match.group(1).strip()
            balance_raw = amount_match.group(2).strip()
            description = rest[:amount_match.start()].strip()

            transaction_date = parse_date(date_str)
            if not transaction_date:
                continue

            amount_pln = parse_amount(amount_raw)
            balance_pln = parse_amount(balance_raw)

            transactions.append({
                "transaction_date": transaction_date,
                "booking_date":     transaction_date,
                "raw_description":  description,
                "amount_raw":       f"{amount_raw} PLN",
                "balance_raw":      f"{balance_raw} PLN",
                "amount_pln":       amount_pln,
                "balance_pln":      balance_pln,
                "currency":         "PLN",
                "transaction_type": detect_type(description),
                "reference_number": re.search(r"ref:(\d+)", description, re.I) and
                                    re.search(r"ref:(\d+)", description, re.I).group(1),
                "merchant_raw":     None,
                "city_raw":         None,
            })

        if not transactions:
            logger.warning("No transactions parsed")
            return pd.DataFrame()

        df = pd.DataFrame(transactions)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["booking_date"]     = pd.to_datetime(df["booking_date"],     errors="coerce")
        df = df.sort_values("transaction_date").reset_index(drop=True)

        logger.success(
            f"Parsed {len(df)} transactions | "
            f"{df['transaction_date'].min().date()} -> {df['transaction_date'].max().date()}"
        )
        return df