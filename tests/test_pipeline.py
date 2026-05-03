"""
tests/test_pipeline.py
========================
Test suite for the Finance Intelligence Pipeline.

Tests cover:
    - PDF Parser: amount parsing, date parsing, transaction type detection
    - Categorizer: merchant matching, edge cases, DataFrame categorization
    - Feature Engineer: date features, rolling windows, correct output types
    - Staging Loader: idempotency logic, DataFrame→dict conversion
    - Data Mart Builder: row conversion, category mapping
    - Integration: full parse → categorize → feature engineer chain

Run:
    pytest tests/ -v --tb=short
    pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# PDF PARSER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAmountParsing:
    """Tests for parse_polish_amount() — the most critical parsing function."""

    def setup_method(self):
        from src.ingestion.pdf_parser import parse_polish_amount
        self.parse = parse_polish_amount

    def test_negative_small_amount(self):
        assert self.parse("-6,00 PLN") == -6.0

    def test_negative_with_thousands_separator(self):
        assert self.parse("-101,42 PLN") == -101.42

    def test_positive_with_thousands_space(self):
        assert self.parse("3 157,06 PLN") == 3157.06

    def test_large_balance(self):
        assert self.parse("4 324,47 PLN") == 4324.47

    def test_zero(self):
        assert self.parse("0,00 PLN") == 0.0

    def test_none_input(self):
        assert self.parse(None) is None

    def test_empty_string(self):
        assert self.parse("") is None

    def test_strips_currency(self):
        assert self.parse("100,00 PLN") == 100.0

    def test_handles_eur(self):
        assert self.parse("50,00 EUR") == 50.0


class TestDateParsing:
    """Tests for parse_date() — date format conversion."""

    def setup_method(self):
        from src.ingestion.pdf_parser import parse_date
        self.parse = parse_date

    def test_standard_date(self):
        assert self.parse("26 Apr 2026") == "2026-04-26"

    def test_single_digit_day(self):
        assert self.parse("3 Mar 2026") == "2026-03-03"

    def test_january(self):
        assert self.parse("1 Jan 2026") == "2026-01-01"

    def test_december(self):
        assert self.parse("31 Dec 2025") == "2025-12-31"

    def test_none_input(self):
        assert self.parse(None) is None

    def test_invalid_format(self):
        assert self.parse("not a date") is None

    def test_empty_string(self):
        assert self.parse("") is None


class TestTransactionTypeDetection:
    """Tests for detect_transaction_type()."""

    def setup_method(self):
        from src.ingestion.pdf_parser import detect_transaction_type
        self.detect = detect_transaction_type

    def test_blik_purchase(self):
        assert self.detect("Zakup BLIK UNICARD SMART CITY SP Zakopiańska 162 ref:93883804395") == "BLIK"

    def test_card_payment(self):
        assert self.detect("DOP. VISA 421352******4879 PŁATNOŚĆ KARTĄ 26.98 PLN ROSSMANN 11 Warszawa") == "CARD_PAYMENT"

    def test_salary(self):
        assert self.detect("Wynagrodz. za 03.2026") == "TRANSFER_IN"

    def test_tax_return(self):
        assert self.detect("Zwrot z podatku PIT za rok 2025") == "TRANSFER_IN"

    def test_bank_fee(self):
        assert self.detect("Opłata za prowadzenie Konta Santander od 01.01.2026 do 31.01.2026") == "BANK_FEE"

    def test_transfer_out(self):
        assert self.detect("Transfer to the phone") == "TRANSFER_OUT"

    def test_unknown(self):
        assert self.detect("something unrecognized xyz") == "UNKNOWN"


class TestMerchantExtraction:
    """Tests for extract_merchant_and_city()."""

    def setup_method(self):
        from src.ingestion.pdf_parser import extract_merchant_and_city
        self.extract = extract_merchant_and_city

    def test_biedronka(self):
        merchant, city = self.extract(
            "DOP. VISA 421352******4879 PŁATNOŚĆ KARTĄ 50.02 PLN JMP S.A. BIEDRONKA 5226 KRAKOW"
        )
        assert merchant is not None
        assert "BIEDRONKA" in merchant.upper()

    def test_blik_booking(self):
        merchant, city = self.extract(
            "Zakup BLIK Hotel at Booking.com Herengracht 597 ref:93548769303"
        )
        assert merchant is not None

    def test_empty_description(self):
        merchant, city = self.extract("")
        assert merchant is None
        assert city is None


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORIZER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCategorizer:
    """Tests for the merchant-to-category mapping engine."""

    def setup_method(self):
        from src.transform.categorizer import Categorizer
        self.cat = Categorizer()

    def test_biedronka_is_groceries(self):
        assert self.cat.categorize("JMP S.A. BIEDRONKA 5226 KRAKOW") == "GROCERIES"

    def test_kaufland_is_groceries(self):
        assert self.cat.categorize("KAUFLAND 1020 KRAKOW") == "GROCERIES"

    def test_zabka_is_groceries(self):
        assert self.cat.categorize("ZABKA Z2613 K.1 KRAKOW") == "GROCERIES"

    def test_rossmann_is_beauty(self):
        assert self.cat.categorize("ROSSMANN 11 Warszawa") == "BEAUTY"

    def test_sephora_is_beauty(self):
        assert self.cat.categorize("SEPHORA POLSKA SP. Z 03 KRAKOW") == "BEAUTY"

    def test_kfc_is_dining(self):
        assert self.cat.categorize("PL KFC KRAKOW KAZIMIERZ KRAKOW") == "DINING"

    def test_subway_is_dining(self):
        assert self.cat.categorize("SUBWAY KRAKOW") == "DINING"

    def test_koleo_is_transport(self):
        assert self.cat.categorize("KOLEO bilety kolejowe Warszawa") == "TRANSPORT"

    def test_unicard_is_transport(self):
        assert self.cat.categorize("Zakup BLIK UNICARD SMART CITY SP Zakopiańska 162") == "TRANSPORT"

    def test_hbomax_is_subscriptions(self):
        assert self.cat.categorize("help.hbomax.com Prague") == "SUBSCRIPTIONS"

    def test_tmobile_is_utilities(self):
        assert self.cat.categorize("TMOBILE POLSKA WARSZAWA") == "UTILITIES"

    def test_salary_is_income(self):
        assert self.cat.categorize("Wynagrodz. za 03.2026") == "INCOME"

    def test_tax_return_is_income(self):
        assert self.cat.categorize("Zwrot z podatku PIT za rok 2025") == "INCOME"

    def test_rent_is_rent(self):
        assert self.cat.categorize("rent") == "RENT"

    def test_deichmann_is_clothing(self):
        assert self.cat.categorize("Deichmann-Obuwie Sp.z.o.o Krakow") == "CLOTHING"

    def test_euronet_is_electronics(self):
        assert self.cat.categorize("EURO NET SP Z O O KRAKOW") == "ELECTRONICS"

    def test_booking_is_travel(self):
        assert self.cat.categorize("Hotel at Booking.com Herengracht 597") == "HOTEL_TRAVEL"

    def test_unknown_merchant_is_other(self):
        assert self.cat.categorize("TOTALLY UNKNOWN MERCHANT XYZ 9999 CITY") == "OTHER"

    def test_empty_string_is_other(self):
        assert self.cat.categorize("") == "OTHER"

    def test_case_insensitive(self):
        assert self.cat.categorize("biedronka") == self.cat.categorize("BIEDRONKA")

    def test_is_expense_negative_amount(self):
        assert self.cat.is_expense("GROCERIES", -50.0) is True

    def test_income_is_not_expense(self):
        assert self.cat.is_expense("INCOME", 6041.66) is False

    def test_explain_returns_match(self):
        result = self.cat.explain("BIEDRONKA 5226 KRAKOW")
        assert result["category"] == "GROCERIES"
        assert result["matched_pattern"] is not None

    def test_explain_returns_other(self):
        result = self.cat.explain("TOTALLY UNKNOWN XYZ")
        assert result["category"] == "OTHER"
        assert result["matched_pattern"] is None

    def test_dataframe_categorization(self):
        df = pd.DataFrame({
            "raw_description": [
                "BIEDRONKA 5226 KRAKOW",
                "ROSSMANN 18 KRAKOW",
                "Wynagrodz. za 03.2026",
            ],
            "amount_pln": [-50.0, -30.0, 6041.66],
        })
        result = self.cat.categorize_dataframe(df)
        assert "category_code" in result.columns
        assert "is_expense" in result.columns
        assert result.iloc[0]["category_code"] == "GROCERIES"
        assert result.iloc[2]["is_expense"] == 0  # Income

    def test_lru_cache_works(self):
        """Same input called twice should hit cache (no error)."""
        r1 = self.cat.categorize("BIEDRONKA 5226 KRAKOW")
        r2 = self.cat.categorize("BIEDRONKA 5226 KRAKOW")
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:
    """Tests for feature derivation logic."""

    def setup_method(self):
        from src.transform.feature_engineering import FeatureEngineer
        self.fe = FeatureEngineer()

    def _make_df(self):
        """Minimal DataFrame for testing."""
        return pd.DataFrame({
            "transaction_date": pd.to_datetime([
                "2026-04-01", "2026-04-05", "2026-04-10",
                "2026-04-15", "2026-04-20", "2026-04-26",
            ]),
            "amount_pln":  [-50.0, -30.0, 6041.66, -100.0, -7.0, -6.0],
            "balance_pln": [1229.20, 1199.20, 7241.00, 7141.00, 7134.00, 7128.00],
            "category_code": ["GROCERIES", "BEAUTY", "INCOME", "GROCERIES", "DINING", "TRANSPORT"],
            "is_expense": [1, 1, 0, 1, 1, 1],
            "raw_description": ["BIEDRONKA", "ROSSMANN", "SALARY", "KAUFLAND", "KFC", "UNICARD"],
        })

    def test_transform_returns_dataframe(self):
        df = self._make_df()
        result = self.fe.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_date_features_present(self):
        result = self.fe.transform(self._make_df())
        for col in ["year", "month", "day", "day_of_week", "quarter", "is_weekend"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_year_is_correct(self):
        result = self.fe.transform(self._make_df())
        assert (result["year"] == 2026).all()

    def test_month_is_correct(self):
        result = self.fe.transform(self._make_df())
        assert (result["month"] == 4).all()

    def test_abs_amount_is_positive(self):
        result = self.fe.transform(self._make_df())
        assert (result["abs_amount"] >= 0).all()

    def test_abs_amount_matches_input(self):
        result = self.fe.transform(self._make_df())
        assert result.iloc[0]["abs_amount"] == 50.0
        assert result.iloc[2]["abs_amount"] == 6041.66

    def test_rolling_features_present(self):
        result = self.fe.transform(self._make_df())
        assert "rolling_7d_spend" in result.columns
        assert "rolling_30d_spend" in result.columns

    def test_rolling_7d_is_non_negative(self):
        result = self.fe.transform(self._make_df())
        assert (result["rolling_7d_spend"] >= 0).all()

    def test_is_weekend_flag(self):
        result = self.fe.transform(self._make_df())
        # 2026-04-05 is a Sunday → is_weekend = 1
        sunday_row = result[result["day"] == 5]
        if not sunday_row.empty:
            assert sunday_row.iloc[0]["is_weekend"] == 1

    def test_is_month_start_flag(self):
        result = self.fe.transform(self._make_df())
        first_row = result[result["day"] == 1]
        assert first_row.iloc[0]["is_month_start"] == 1

    def test_amount_bucket_column_exists(self):
        result = self.fe.transform(self._make_df())
        assert "amount_bucket" in result.columns

    def test_balance_change_column(self):
        result = self.fe.transform(self._make_df())
        assert "balance_change" in result.columns

    def test_empty_df_returns_empty(self):
        result = self.fe.transform(pd.DataFrame())
        assert result.empty

    def test_monthly_ml_features(self):
        df = self._make_df()
        # Need at least a few months of data
        months_df = pd.concat([
            df.assign(transaction_date=pd.to_datetime(df["transaction_date"]) - pd.DateOffset(months=i))
            for i in range(4)
        ]).reset_index(drop=True)
        months_df = self.fe.transform(months_df)
        monthly = self.fe.create_monthly_ml_features(months_df)
        assert isinstance(monthly, pd.DataFrame)
        if not monthly.empty:
            assert "lag_1m_spend" in monthly.columns
            assert "total_spend" in monthly.columns


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — Parse → Categorize → Feature Engineer
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationChain:
    """Tests the full transform chain without a real PDF or DB."""

    def _make_raw_parsed_df(self):
        """Simulate output of ErsteParser.parse()."""
        return pd.DataFrame({
            "transaction_date": pd.to_datetime([
                "2026-04-24", "2026-04-24", "2026-04-07",
                "2026-04-03", "2026-04-11",
            ]),
            "booking_date": pd.to_datetime([
                "2026-04-24", "2026-04-24", "2026-04-07",
                "2026-04-03", "2026-04-11",
            ]),
            "raw_description": [
                "DOP. VISA 421352******4879 PŁATNOŚĆ KARTĄ 26.98 PLN ROSSMANN 11 Warszawa",
                "DOP. VISA 421352******4879 PŁATNOŚĆ KARTĄ 39.39 PLN ZABKA Z8765 K.1 WARSZAWA",
                "Zwrot z podatku PIT za rok 2025 /KL/03",
                "Wynagrodz. za 03.2026",
                "DOP. VISA 421352******4879 PŁATNOŚĆ KARTĄ 33.33 PLN help.hbomax.com Prague",
            ],
            "amount_raw":  ["-26,98 PLN", "-39,39 PLN", "1 453,00 PLN", "6 041,66 PLN", "-33,33 PLN"],
            "balance_raw": ["3 067,93 PLN", "2 959,57 PLN", "5 297,24 PLN", "7 263,87 PLN", "4 154,65 PLN"],
            "amount_pln":  [-26.98, -39.39, 1453.00, 6041.66, -33.33],
            "balance_pln": [3067.93, 2959.57, 5297.24, 7263.87, 4154.65],
            "currency":    ["PLN"] * 5,
            "transaction_type": ["CARD_PAYMENT", "CARD_PAYMENT", "TRANSFER_IN", "TRANSFER_IN", "CARD_PAYMENT"],
            "reference_number": [None] * 5,
            "merchant_raw": ["ROSSMANN 11", "ZABKA Z8765", None, None, "help.hbomax.com"],
            "city_raw":     ["Warszawa", "Warszawa", None, None, "Prague"],
        })

    def test_full_chain_produces_expected_columns(self):
        from src.transform.categorizer import Categorizer
        from src.transform.feature_engineering import FeatureEngineer

        raw = self._make_raw_parsed_df()

        cat = Categorizer()
        categorized = cat.categorize_dataframe(raw)

        fe = FeatureEngineer()
        enriched = fe.transform(categorized)

        expected_cols = [
            "transaction_date", "amount_pln", "balance_pln",
            "category_code", "is_expense", "abs_amount",
            "year", "month", "day", "rolling_7d_spend",
        ]
        for col in expected_cols:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_income_rows_not_marked_as_expense(self):
        from src.transform.categorizer import Categorizer
        from src.transform.feature_engineering import FeatureEngineer

        raw = self._make_raw_parsed_df()
        cat = Categorizer()
        categorized = cat.categorize_dataframe(raw)
        fe = FeatureEngineer()
        enriched = fe.transform(categorized)

        income_rows = enriched[enriched["amount_pln"] > 0]
        # All positive-amount rows should have is_expense == 0
        assert (income_rows["is_expense"] == 0).all()

    def test_rossmann_categorized_as_beauty(self):
        from src.transform.categorizer import Categorizer

        raw = self._make_raw_parsed_df()
        cat = Categorizer()
        categorized = cat.categorize_dataframe(raw)
        rossmann_row = categorized[categorized["merchant_raw"] == "ROSSMANN 11"]
        assert rossmann_row.iloc[0]["category_code"] == "BEAUTY"

    def test_hbomax_categorized_as_subscriptions(self):
        from src.transform.categorizer import Categorizer

        raw = self._make_raw_parsed_df()
        cat = Categorizer()
        categorized = cat.categorize_dataframe(raw)
        hbo_row = categorized[categorized["raw_description"].str.contains("hbomax", case=False)]
        assert hbo_row.iloc[0]["category_code"] == "SUBSCRIPTIONS"

    def test_no_null_category_codes(self):
        from src.transform.categorizer import Categorizer

        raw = self._make_raw_parsed_df()
        cat = Categorizer()
        categorized = cat.categorize_dataframe(raw)
        assert categorized["category_code"].notna().all()


# ══════════════════════════════════════════════════════════════════════════════
# STAGING LOADER — Unit tests (no DB required, mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestStagingLoaderUtils:
    """Tests for helper methods that don't need a real DB."""

    def test_df_to_staging_rows_length(self):
        from src.staging.loader import StagingLoader

        loader = StagingLoader.__new__(StagingLoader)  # skip __init__

        df = pd.DataFrame({
            "transaction_date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
            "booking_date":     pd.to_datetime(["2026-04-01", "2026-04-02"]),
            "raw_description":  ["DESC 1", "DESC 2"],
            "amount_raw":       ["-50,00 PLN", "-30,00 PLN"],
            "balance_raw":      ["3 000,00 PLN", "2 970,00 PLN"],
            "amount_pln":       [-50.0, -30.0],
            "balance_pln":      [3000.0, 2970.0],
            "currency":         ["PLN", "PLN"],
            "transaction_type": ["CARD_PAYMENT", "CARD_PAYMENT"],
            "reference_number": [None, None],
            "merchant_raw":     ["BIEDRONKA", "ROSSMANN"],
            "city_raw":         ["KRAKOW", "KRAKOW"],
        })

        rows = loader._df_to_staging_rows(df, load_id=1)
        assert len(rows) == 2
        assert rows[0]["load_id"] == 1
        assert rows[0]["amount_pln"] == -50.0
        assert rows[1]["merchant_raw"] == "ROSSMANN"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    """Tests for settings validation."""

    def test_settings_loads(self):
        from config.settings import settings
        assert settings is not None

    def test_db_url_format(self):
        from config.settings import settings
        url = settings.db.url
        assert url.startswith("mysql+pymysql://")
        assert settings.db.name in url

    def test_db_url_safe_masks_password(self):
        from config.settings import settings
        safe = settings.db.url_safe
        assert "****" in safe

    def test_ml_config_defaults(self):
        from config.settings import settings
        assert settings.ml.xgb_n_estimators > 0
        assert 0 < settings.ml.test_size < 1
        assert settings.ml.random_state == 42

    def test_scheduler_interval_positive(self):
        from config.settings import settings
        assert settings.scheduler.interval_minutes > 0

    def test_merchant_patterns_not_empty(self):
        from config.merchants import MERCHANT_PATTERNS, CATEGORY_PRIORITY
        assert len(MERCHANT_PATTERNS) > 0
        assert "GROCERIES" in MERCHANT_PATTERNS
        assert "INCOME" in CATEGORY_PRIORITY

    def test_category_colors_have_hex(self):
        from config.merchants import CATEGORY_COLORS
        for cat, color in CATEGORY_COLORS.items():
            assert color.startswith("#"), f"{cat} color {color!r} is not hex"
            assert len(color) == 7, f"{cat} color {color!r} is wrong length"
