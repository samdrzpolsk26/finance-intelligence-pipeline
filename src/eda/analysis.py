"""
src/eda/analysis.py
====================
Automated Exploratory Data Analysis engine.

Generates 12 publication-quality charts and a text summary report
from the data mart. All charts are saved to reports/figures/.

Charts generated:
    01_monthly_cashflow.png         ─ Income vs expenses bar chart
    02_category_pie.png             ─ Spending breakdown donut chart
    03_daily_heatmap.png            ─ Calendar heatmap (spending intensity)
    04_spending_trend.png           ─ Rolling 30d spend + balance overlay
    05_top_merchants.png            ─ Top 15 merchants by total spend
    06_category_trend.png           ─ Stacked bar: category mix over time
    07_weekday_patterns.png         ─ Avg spend by day of week
    08_amount_distribution.png      ─ Expense amount histogram
    09_balance_timeline.png         ─ Account balance over time
    10_velocity_chart.png           ─ Spending velocity (7d vs prior 7d)
    11_income_sources.png           ─ Income breakdown
    12_savings_rate.png             ─ Monthly savings rate %

Usage:
    from src.eda.analysis import EDAEngine
    engine = EDAEngine()
    engine.run_full_eda()
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from rich.console import Console
from rich.table import Table

from config.settings import settings
from config.merchants import CATEGORY_COLORS, CATEGORY_LABELS

console = Console()

# ── Visualization Styling ──────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":    "#0F1117",
    "axes.facecolor":      "#1A1D27",
    "axes.edgecolor":      "#2D3142",
    "axes.labelcolor":     "#C8CDD6",
    "axes.titlecolor":     "#FFFFFF",
    "axes.titlesize":      14,
    "axes.labelsize":      11,
    "xtick.color":         "#8B92A5",
    "ytick.color":         "#8B92A5",
    "text.color":          "#C8CDD6",
    "grid.color":          "#2D3142",
    "grid.linewidth":      0.6,
    "legend.facecolor":    "#1A1D27",
    "legend.edgecolor":    "#2D3142",
    "legend.labelcolor":   "#C8CDD6",
}

ACCENT   = "#4CC9F0"
POSITIVE = "#06D6A0"
NEGATIVE = "#EF476F"
NEUTRAL  = "#FFD166"

plt.rcParams.update(STYLE)
sns.set_theme(style="dark", rc=STYLE)


class EDAEngine:
    """
    Runs automated EDA on the finance data mart and exports charts.
    """

    def __init__(self):
        self.engine = create_engine(settings.db.url, echo=False)
        self.output_dir = settings.paths.reports_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame | None = None
        self._monthly: pd.DataFrame | None = None

    # ── Data Loading ───────────────────────────────────────────────────────────

    def _load_data(self):
        """Load transactions and monthly aggregates from data mart."""
        with self.engine.connect() as conn:
            self._df = pd.read_sql(
                text("SELECT * FROM fact_transactions ORDER BY transaction_date"),
                conn, parse_dates=["transaction_date"]
            )
            self._monthly = pd.read_sql(
                text("SELECT * FROM v_monthly_cashflow ORDER BY year, month"),
                conn
            )
        logger.info(f"EDA loaded: {len(self._df)} transactions, {len(self._monthly)} months")

    def _save(self, fig: plt.Figure, filename: str, dpi: int = 150):
        """Save figure and close to free memory."""
        path = self.output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved: {filename}")

    # ── Chart 01: Monthly Cashflow ─────────────────────────────────────────────

    def chart_monthly_cashflow(self) -> str:
        """Grouped bar chart: Income vs Expenses per month."""
        df = self._monthly.copy()
        if df.empty:
            return ""

        x = np.arange(len(df))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        bars_income = ax.bar(x - width/2, df["total_income"], width,
                             color=POSITIVE, alpha=0.85, label="Income", zorder=3)
        bars_expense = ax.bar(x + width/2, df["total_expenses"], width,
                              color=NEGATIVE, alpha=0.85, label="Expenses", zorder=3)

        # Net savings line
        ax2 = ax.twinx()
        ax2.plot(x, df["net_savings"], color=ACCENT, linewidth=2.5,
                 marker="o", markersize=5, label="Net Savings", zorder=4)
        ax2.axhline(0, color=NEUTRAL, linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_ylabel("Net Savings (PLN)", color=ACCENT, fontsize=10)
        ax2.tick_params(colors=ACCENT)

        ax.set_title("💰 Monthly Cash Flow — Income vs Expenses", fontsize=16, pad=15, color="white")
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Amount (PLN)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df["month_label"], rotation=45, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.legend(loc="upper left", fontsize=10)
        ax2.legend(loc="upper right", fontsize=10)

        fig.tight_layout()
        self._save(fig, "01_monthly_cashflow.png")
        return str(self.output_dir / "01_monthly_cashflow.png")

    # ── Chart 02: Category Donut ───────────────────────────────────────────────

    def chart_category_donut(self) -> str:
        """Donut chart of spending by category."""
        df = self._df[self._df["is_expense"] == 1].copy()
        category_totals = df.groupby("category_code")["abs_amount"].sum().sort_values(ascending=False)

        # Only show top 10, group rest as "OTHER"
        top_n = 10
        top = category_totals.head(top_n)
        rest = category_totals.iloc[top_n:].sum()
        if rest > 0:
            top["OTHER_REST"] = rest

        labels = [CATEGORY_LABELS.get(c, c) for c in top.index]
        colors = [CATEGORY_COLORS.get(c, "#7F8C8D") for c in top.index]

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0F1117")
        ax.set_facecolor("#0F1117")

        wedges, texts, autotexts = ax.pie(
            top.values, labels=None, colors=colors,
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="#0F1117", linewidth=2)
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(9)

        # Center label
        total = category_totals.sum()
        ax.text(0, 0, f"Total\n{total:,.0f} PLN", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")

        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.05, 0.5),
                  fontsize=10, framealpha=0.5)
        ax.set_title("🏷️ Spending Breakdown by Category", fontsize=16, pad=20, color="white")

        fig.tight_layout()
        self._save(fig, "02_category_donut.png")
        return str(self.output_dir / "02_category_donut.png")

    # ── Chart 03: Calendar Heatmap ─────────────────────────────────────────────

    def chart_daily_heatmap(self) -> str:
        """Calendar heatmap showing spending intensity per day."""
        df = self._df[self._df["is_expense"] == 1].copy()
        daily = df.groupby("transaction_date")["abs_amount"].sum().reset_index()
        daily.columns = ["date", "spend"]
        daily["year"] = daily["date"].dt.year
        daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
        daily["dow"]  = daily["date"].dt.dayofweek

        years = sorted(daily["year"].unique())
        fig, axes = plt.subplots(len(years), 1, figsize=(18, 4 * len(years)), facecolor="#0F1117")
        if len(years) == 1:
            axes = [axes]

        for ax, year in zip(axes, years):
            year_data = daily[daily["year"] == year]
            pivot = year_data.pivot_table(index="dow", columns="week", values="spend", aggfunc="sum")
            pivot = pivot.reindex(index=range(7), columns=range(1, 54))

            sns.heatmap(
                pivot, ax=ax, cmap="YlOrRd", linewidths=0.5,
                linecolor="#0F1117", cbar=True,
                cbar_kws={"shrink": 0.4, "label": "PLN"},
                fmt=".0f", annot=False,
            )
            ax.set_title(f"🔥 Daily Spending Heatmap — {year}", fontsize=13, color="white", pad=10)
            ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0, fontsize=9)
            ax.set_xlabel("Week of Year", fontsize=10)
            ax.set_ylabel("")

        fig.tight_layout(pad=2)
        self._save(fig, "03_daily_heatmap.png", dpi=120)
        return str(self.output_dir / "03_daily_heatmap.png")

    # ── Chart 04: Spending Trend + Balance ────────────────────────────────────

    def chart_spending_trend(self) -> str:
        """Dual-axis: rolling 30d spend + account balance over time."""
        df = self._df.sort_values("transaction_date").copy()
        expenses = df[df["is_expense"] == 1][["transaction_date", "abs_amount"]].copy()
        expenses = expenses.set_index("transaction_date").resample("D")["abs_amount"].sum()
        rolling_30 = expenses.rolling(30, min_periods=1).sum()

        balance = df[["transaction_date", "balance_pln"]].drop_duplicates("transaction_date").set_index("transaction_date")

        fig, ax1 = plt.subplots(figsize=(16, 6), facecolor="#0F1117")
        ax1.set_facecolor("#1A1D27")

        ax1.fill_between(rolling_30.index, rolling_30.values, alpha=0.25, color=NEGATIVE)
        ax1.plot(rolling_30.index, rolling_30.values, color=NEGATIVE, linewidth=2, label="30d Rolling Spend")

        ax2 = ax1.twinx()
        ax2.plot(balance.index, balance["balance_pln"], color=ACCENT, linewidth=2.5,
                 alpha=0.9, label="Account Balance")
        ax2.fill_between(balance.index, balance["balance_pln"], alpha=0.08, color=ACCENT)

        ax1.set_title("📈 Spending Trend & Account Balance", fontsize=16, pad=15, color="white")
        ax1.set_ylabel("30-Day Rolling Spend (PLN)", color=NEGATIVE, fontsize=11)
        ax2.set_ylabel("Account Balance (PLN)", color=ACCENT, fontsize=11)
        ax1.tick_params(colors=NEGATIVE)
        ax2.tick_params(colors=ACCENT)
        ax1.grid(alpha=0.2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

        fig.tight_layout()
        self._save(fig, "04_spending_trend.png")
        return str(self.output_dir / "04_spending_trend.png")

    # ── Chart 05: Top Merchants ────────────────────────────────────────────────

    def chart_top_merchants(self) -> str:
        """Horizontal bar chart of top 15 merchants by total spend."""
        df = self._df[(self._df["is_expense"] == 1) & (self._df["merchant_name"].notna())].copy()
        top = (
            df.groupby(["merchant_name", "category_code"])["abs_amount"]
            .sum()
            .sort_values(ascending=True)
            .tail(15)
        )

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        colors = [CATEGORY_COLORS.get(cat, "#7F8C8D") for _, cat in top.index]
        bars = ax.barh(
            [m for m, _ in top.index],
            top.values, color=colors, alpha=0.85, zorder=3
        )

        for bar, value in zip(bars, top.values):
            ax.text(value + 5, bar.get_y() + bar.get_height()/2,
                    f"{value:,.0f} PLN", va="center", fontsize=9, color="#C8CDD6")

        ax.set_title("🏪 Top 15 Merchants by Total Spend", fontsize=16, pad=15, color="white")
        ax.set_xlabel("Total Spend (PLN)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, top.values.max() * 1.2)

        fig.tight_layout()
        self._save(fig, "05_top_merchants.png")
        return str(self.output_dir / "05_top_merchants.png")

    # ── Chart 06: Category Stacked Bar ────────────────────────────────────────

    def chart_category_trend(self) -> str:
        """Stacked bar chart showing category mix evolution over months."""
        df = self._df[self._df["is_expense"] == 1].copy()
        monthly_cat = (
            df.groupby(["year", "month", "category_code"])["abs_amount"]
            .sum()
            .reset_index()
        )
        monthly_cat["period"] = monthly_cat.apply(
            lambda r: f"{r['year']}-{int(r['month']):02d}", axis=1
        )

        pivot = monthly_cat.pivot_table(
            index="period", columns="category_code", values="abs_amount", fill_value=0
        )

        top_cats = pivot.sum().sort_values(ascending=False).head(8).index.tolist()
        pivot_top = pivot[top_cats]

        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        bottom = np.zeros(len(pivot_top))
        for cat in top_cats:
            color = CATEGORY_COLORS.get(cat, "#7F8C8D")
            label = CATEGORY_LABELS.get(cat, cat)
            ax.bar(range(len(pivot_top)), pivot_top[cat].values,
                   bottom=bottom, label=label, color=color, alpha=0.85, zorder=3)
            bottom += pivot_top[cat].values

        ax.set_title("📊 Monthly Spending by Category (Stacked)", fontsize=16, pad=15, color="white")
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Total Spend (PLN)", fontsize=11)
        ax.set_xticks(range(len(pivot_top)))
        ax.set_xticklabels(pivot_top.index, rotation=45, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=1)

        fig.tight_layout()
        self._save(fig, "06_category_trend.png")
        return str(self.output_dir / "06_category_trend.png")

    # ── Chart 07: Weekday Patterns ─────────────────────────────────────────────

    def chart_weekday_patterns(self) -> str:
        """Bar chart of average daily spend by day of week."""
        df = self._df[self._df["is_expense"] == 1].copy()
        day_avg = df.groupby("day_name")["abs_amount"].mean()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_avg = day_avg.reindex(order)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        colors = [NEGATIVE if day in ["Saturday", "Sunday"] else ACCENT for day in order]
        bars = ax.bar(order, day_avg.values, color=colors, alpha=0.85, zorder=3)

        for bar, val in zip(bars, day_avg.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                        f"{val:.0f}", ha="center", fontsize=9, color="white")

        ax.set_title("📅 Average Spend by Day of Week", fontsize=16, pad=15, color="white")
        ax.set_ylabel("Avg. Transaction Amount (PLN)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=1)

        weekend_patch = mpatches.Patch(color=NEGATIVE, alpha=0.85, label="Weekend")
        weekday_patch = mpatches.Patch(color=ACCENT, alpha=0.85, label="Weekday")
        ax.legend(handles=[weekday_patch, weekend_patch], fontsize=10)

        fig.tight_layout()
        self._save(fig, "07_weekday_patterns.png")
        return str(self.output_dir / "07_weekday_patterns.png")

    # ── Chart 09: Balance Timeline ─────────────────────────────────────────────

    def chart_balance_timeline(self) -> str:
        """Account balance over time with income/expense markers."""
        df = self._df.sort_values("transaction_date").copy()

        fig, ax = plt.subplots(figsize=(16, 6), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        ax.plot(df["transaction_date"], df["balance_pln"],
                color=ACCENT, linewidth=2, alpha=0.9, zorder=3)
        ax.fill_between(df["transaction_date"], df["balance_pln"],
                        alpha=0.12, color=ACCENT)

        # Mark income events
        income_events = df[df["is_expense"] == 0]
        ax.scatter(income_events["transaction_date"], income_events["balance_pln"],
                   color=POSITIVE, s=50, zorder=5, alpha=0.8, label="Income")

        # Mark large expenses (top 10% by amount)
        threshold = df[df["is_expense"] == 1]["abs_amount"].quantile(0.90)
        large_exp = df[(df["is_expense"] == 1) & (df["abs_amount"] >= threshold)]
        ax.scatter(large_exp["transaction_date"], large_exp["balance_pln"],
                   color=NEGATIVE, s=60, zorder=5, alpha=0.8, marker="v", label="Large Expense")

        ax.set_title("💳 Account Balance Timeline", fontsize=16, pad=15, color="white")
        ax.set_ylabel("Balance (PLN)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(alpha=0.2, zorder=1)
        ax.legend(fontsize=10)

        fig.tight_layout()
        self._save(fig, "09_balance_timeline.png")
        return str(self.output_dir / "09_balance_timeline.png")

    # ── Chart 12: Savings Rate ─────────────────────────────────────────────────

    def chart_savings_rate(self) -> str:
        """Monthly savings rate as percentage of income."""
        df = self._monthly.copy()
        df["savings_rate"] = (df["net_savings"] / df["total_income"].where(df["total_income"] > 0) * 100).fillna(0)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")

        colors = [POSITIVE if v >= 0 else NEGATIVE for v in df["savings_rate"]]
        bars = ax.bar(range(len(df)), df["savings_rate"], color=colors, alpha=0.85, zorder=3)

        ax.axhline(0, color="white", linewidth=1, alpha=0.5)
        ax.axhline(20, color=NEUTRAL, linewidth=1.5, linestyle="--", alpha=0.7, label="20% goal")

        for bar, val in zip(bars, df["savings_rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + (1 if val >= 0 else -3),
                    f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8, color="white")

        ax.set_title("💹 Monthly Savings Rate", fontsize=16, pad=15, color="white")
        ax.set_ylabel("Savings Rate (%)", fontsize=11)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["month_label"], rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.legend(fontsize=10)

        fig.tight_layout()
        self._save(fig, "12_savings_rate.png")
        return str(self.output_dir / "12_savings_rate.png")

    # ── Full EDA Run ───────────────────────────────────────────────────────────

    def run_full_eda(self) -> list[str]:
        """
        Run all charts and print a summary report.
        Returns list of output file paths.
        """
        logger.info("Starting full EDA run...")
        self._load_data()

        if self._df is None or self._df.empty:
            logger.error("No data found in data mart — run the pipeline first")
            return []

        generated = []
        chart_methods = [
            self.chart_monthly_cashflow,
            self.chart_category_donut,
            self.chart_daily_heatmap,
            self.chart_spending_trend,
            self.chart_top_merchants,
            self.chart_category_trend,
            self.chart_weekday_patterns,
            self.chart_balance_timeline,
            self.chart_savings_rate,
        ]

        for method in chart_methods:
            try:
                path = method()
                if path:
                    generated.append(path)
            except Exception as exc:
                logger.warning(f"Chart {method.__name__} failed: {exc}")

        self._print_text_summary()
        logger.success(f"EDA complete: {len(generated)} charts saved to {self.output_dir}")
        return generated

    def _print_text_summary(self):
        """Print a Rich-formatted text summary to console."""
        df = self._df
        expenses = df[df["is_expense"] == 1]
        income = df[df["is_expense"] == 0]

        table = Table(title="📊 Finance Intelligence — Summary Report", style="cyan")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Transactions", str(len(df)))
        table.add_row("Date Range", f"{df['transaction_date'].min().date()} → {df['transaction_date'].max().date()}")
        table.add_row("Total Expenses", f"{expenses['abs_amount'].sum():,.2f} PLN")
        table.add_row("Total Income", f"{income['abs_amount'].sum():,.2f} PLN")
        table.add_row("Net Savings", f"{(income['abs_amount'].sum() - expenses['abs_amount'].sum()):,.2f} PLN")
        table.add_row("Avg. Transaction", f"{expenses['abs_amount'].mean():,.2f} PLN")
        table.add_row("Biggest Expense", f"{expenses['abs_amount'].max():,.2f} PLN")
        table.add_row("Most Active Category", expenses.groupby("category_code")["abs_amount"].sum().idxmax())
        table.add_row("Current Balance", f"{df.sort_values('transaction_date').iloc[-1]['balance_pln']:,.2f} PLN")

        console.print(table)
