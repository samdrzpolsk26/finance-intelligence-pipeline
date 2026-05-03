"""
src/transform/categorizer.py
=============================
Transaction categorization engine.

Takes a raw description string and maps it to one of 15 spending
categories using regex pattern matching against a merchant dictionary.

Design:
    - Pattern matching is case-insensitive
    - Categories are checked in priority order (INCOME and RENT first
      to avoid false positives from partial matches)
    - Falls back to 'OTHER' if no pattern matches
    - is_expense flag is derived from the category type

Why regex instead of ML?
    For a personal dataset of ~200-500 transactions/month, regex with
    a well-curated merchant dictionary is more reliable, interpretable,
    and fast than a trained classifier. An ML classifier would need
    labeled training data we don't have yet. As this dataset grows,
    a fine-tuned text classifier could replace this module.

Usage:
    from src.transform.categorizer import Categorizer
    cat = Categorizer()
    result = cat.categorize("DOP. VISA ... BIEDRONKA 5226 KRAKOW")
    # Returns: {'category_code': 'GROCERIES', 'is_expense': True}
"""

import re
from typing import Optional
from functools import lru_cache

import pandas as pd
from loguru import logger

from config.merchants import (
    MERCHANT_PATTERNS,
    CATEGORY_PRIORITY,
    CATEGORY_LABELS,
    CATEGORY_COLORS,
)

# Non-expense categories (income / neutral transfers)
NON_EXPENSE_CATEGORIES = {"INCOME", "TRANSFER"}


class Categorizer:
    """
    Rule-based transaction categorizer.
    
    Pre-compiles all regex patterns at initialization for performance.
    A single Categorizer instance should be reused across the pipeline.
    """

    def __init__(self):
        # Pre-compile all patterns grouped by category
        self._compiled: dict[str, list[re.Pattern]] = {}
        for category, patterns in MERCHANT_PATTERNS.items():
            self._compiled[category] = [
                re.compile(p, re.IGNORECASE | re.UNICODE)
                for p in patterns
            ]
        logger.debug(
            f"Categorizer initialized: {len(self._compiled)} categories, "
            f"{sum(len(v) for v in self._compiled.values())} patterns"
        )

    @lru_cache(maxsize=2048)
    def categorize(self, description: str) -> str:
        """
        Map a transaction description to a category code.
        
        Args:
            description: Raw transaction description string.
            
        Returns:
            Category code string (e.g., 'GROCERIES', 'DINING', 'OTHER').
        """
        if not description or not description.strip():
            return "OTHER"

        # Check categories in priority order
        for category in CATEGORY_PRIORITY:
            patterns = self._compiled.get(category, [])
            for pattern in patterns:
                if pattern.search(description):
                    return category

        return "OTHER"

    def is_expense(self, category_code: str, amount_pln: float) -> bool:
        """
        Determine if a transaction is an expense.
        
        Primary rule: negative amount = expense.
        Secondary rule: non-expense categories (INCOME, TRANSFER) override.
        """
        if category_code in NON_EXPENSE_CATEGORIES:
            return False
        return amount_pln < 0

    def categorize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply categorization to an entire DataFrame.
        
        Expects columns: 'raw_description', 'amount_pln'
        Adds columns: 'category_code', 'category_label', 'category_color', 'is_expense'
        
        Args:
            df: DataFrame with transaction data (from staging or transform).
            
        Returns:
            DataFrame with new category columns added.
        """
        if df.empty:
            return df

        logger.info(f"Categorizing {len(df)} transactions...")

        df = df.copy()

        # Apply categorization
        df["category_code"] = df["raw_description"].apply(self.categorize)

        # is_expense: True for negative amounts (expenses)
        df["is_expense"] = df.apply(
            lambda row: self.is_expense(row["category_code"], row.get("amount_pln", 0)),
            axis=1
        ).astype(int)

        # Add display labels and colors
        df["category_label"] = df["category_code"].map(CATEGORY_LABELS).fillna("❓ Other")
        df["category_color"] = df["category_code"].map(CATEGORY_COLORS).fillna("#7F8C8D")

        # Distribution summary for logging
        dist = df["category_code"].value_counts()
        logger.info(f"Category distribution:\n{dist.to_string()}")

        uncategorized = (df["category_code"] == "OTHER").sum()
        if uncategorized > 0:
            logger.warning(
                f"{uncategorized} transactions fell into 'OTHER'. "
                "Consider adding patterns to config/merchants.py"
            )

        return df

    def get_unknown_merchants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return transactions that couldn't be categorized (category = OTHER).
        Useful for extending the merchant dictionary.
        """
        unknown = df[df["category_code"] == "OTHER"][
            ["transaction_date", "raw_description", "amount_pln", "merchant_raw"]
        ].copy()
        return unknown.sort_values("amount_pln")

    def explain(self, description: str) -> dict:
        """
        Debug helper: shows which pattern matched (or didn't) for a description.
        
        Returns:
            dict with 'category', 'matched_pattern', 'checked_categories'
        """
        checked = []
        for category in CATEGORY_PRIORITY:
            patterns = self._compiled.get(category, [])
            for pattern in patterns:
                if pattern.search(description):
                    return {
                        "category": category,
                        "matched_pattern": pattern.pattern,
                        "checked_categories": checked,
                    }
            checked.append(category)

        return {
            "category": "OTHER",
            "matched_pattern": None,
            "checked_categories": checked,
        }
