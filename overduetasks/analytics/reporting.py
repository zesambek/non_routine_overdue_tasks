"""Advanced analytics, statistical modeling, and visualization helpers for open task data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

try:  # Matplotlib and seaborn are optional but strongly recommended.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None  # type: ignore[assignment]

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - optional dependency
    scipy_stats = None  # type: ignore[assignment]

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover - optional dependency
    smf = None  # type: ignore[assignment]


DATE_COLUMNS = {
    "Due": "due_date",
    "Found on Date": "found_on_date",
    "Next Shop Visit": "next_shop_visit",
}


@dataclass(slots=True)
class TaskAnalyticsSummary:
    """Lightweight summary payload for feeding reports and dashboards."""

    total_tasks: int
    overdue_tasks: int
    completion_rate: float
    mean_days_until_due: float
    mean_days_overdue: float
    severity_distribution: Dict[str, float]
    status_distribution: Dict[str, float]
    fleet_distribution: Dict[str, float]


def prepare_task_dataframe(df: pd.DataFrame, *, tz: str | None = None) -> pd.DataFrame:
    """Return a normalized copy of the open tasks table with enriched analytics columns."""
    data = df.copy(deep=True)

    if data.empty:
        return data

    timestamp_now = pd.Timestamp.now(tz=tz).normalize()

    for column, alias in DATE_COLUMNS.items():
        if column in data.columns:
            data[alias] = pd.to_datetime(data[column], errors="coerce", utc=False)
        else:
            data[alias] = pd.NaT

    if "due_date" in data.columns:
        delta = (data["due_date"] - timestamp_now).dt.total_seconds() / 86400.0
        data["days_until_due"] = delta
        data["is_overdue"] = delta < 0
        data["days_overdue"] = np.where(data["is_overdue"], -delta, 0.0)
    else:
        data["days_until_due"] = np.nan
        data["is_overdue"] = False
        data["days_overdue"] = 0.0

    if "found_on_date" in data.columns:
        data["days_since_found"] = (timestamp_now - data["found_on_date"]).dt.total_seconds() / 86400.0
    else:
        data["days_since_found"] = np.nan

    if "Severity" in data.columns:
        data["severity"] = (
            data["Severity"].astype("string").str.strip().str.title().replace({"": "Unknown", "Nan": "Unknown"})
        )
    else:
        data["severity"] = "Unknown"

    if "Status" in data.columns:
        data["status"] = data["Status"].astype("string").str.strip().replace({"": "Unknown", "Nan": "Unknown"})
    else:
        data["status"] = "Unknown"

    if "Fleet Type" in data.columns:
        data["fleet"] = data["Fleet Type"].astype("string").str.strip().replace({"": "Unknown", "Nan": "Unknown"})
    else:
        data["fleet"] = "Unknown"

    numeric_candidates = [
        "days_until_due",
        "days_overdue",
        "days_since_found",
    ]
    for column in numeric_candidates:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def summarize_tasks(df: pd.DataFrame) -> TaskAnalyticsSummary:
    """Generate KPI-style aggregates for the prepared task frame."""
    if df.empty:
        return TaskAnalyticsSummary(
            total_tasks=0,
            overdue_tasks=0,
            completion_rate=0.0,
            mean_days_until_due=float("nan"),
            mean_days_overdue=float("nan"),
            severity_distribution={},
            status_distribution={},
            fleet_distribution={},
        )

    overdue_tasks = int(df["is_overdue"].sum()) if "is_overdue" in df.columns else 0
    total_tasks = int(len(df))

    completion_rate = 0.0
    if "status" in df.columns and total_tasks > 0:
        closed = df["status"].str.lower().str.contains("close").sum()
        completion_rate = float(closed / total_tasks)

    mean_days_until_due = float(df["days_until_due"].mean()) if "days_until_due" in df else float("nan")
    mean_days_overdue = float(df.loc[df.get("is_overdue", False), "days_overdue"].mean()) if overdue_tasks else 0.0

    return TaskAnalyticsSummary(
        total_tasks=total_tasks,
        overdue_tasks=overdue_tasks,
        completion_rate=completion_rate,
        mean_days_until_due=mean_days_until_due,
        mean_days_overdue=mean_days_overdue,
        severity_distribution=_normalised_counts(df, "severity"),
        status_distribution=_normalised_counts(df, "status"),
        fleet_distribution=_normalised_counts(df, "fleet"),
    )


def identify_outliers(df: pd.DataFrame, *, column: str = "days_overdue", z_threshold: float = 2.5) -> pd.DataFrame:
    """Return rows that exceed the provided Z-score threshold."""
    if df.empty or column not in df:
        return df.iloc[0:0]

    if scipy_stats is None:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for outlier detection. Install it with `pip install scipy`.")

    subset = df[[column]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty:
        return df.iloc[0:0]

    scores = scipy_stats.zscore(subset[column].to_numpy())
    mask = np.abs(scores) >= z_threshold
    index = subset.index[mask]
    return df.loc[index].copy()


def train_due_date_model(df: pd.DataFrame):
    """Fit a lightweight linear model that explains days-until-due variance."""
    if smf is None:  # pragma: no cover - optional dependency
        raise ImportError("statsmodels is required for regression modeling. Install it with `pip install statsmodels`.")

    if df.empty:
        raise ValueError("Cannot train due date model on an empty dataframe.")

    required = {"days_until_due"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing required columns for modeling: {missing}")

    model_df = df.copy()
    if "severity" in model_df.columns:
        model_df["severity_enc"] = model_df["severity"].astype("category")
    if "status" in model_df.columns:
        model_df["status_enc"] = model_df["status"].astype("category")
    if "fleet" in model_df.columns:
        model_df["fleet_enc"] = model_df["fleet"].astype("category")

    formula_terms: list[str] = []
    if "severity_enc" in model_df:
        formula_terms.append("C(severity_enc)")
    if "status_enc" in model_df:
        formula_terms.append("C(status_enc)")
    if "fleet_enc" in model_df:
        formula_terms.append("C(fleet_enc)")
    if "days_since_found" in model_df:
        formula_terms.append("days_since_found")

    if not formula_terms:
        raise ValueError("Insufficient feature columns available for modeling.")

    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["days_until_due"])
    if model_df.empty:
        raise ValueError("No valid rows remain after cleaning for modeling.")

    formula = "days_until_due ~ " + " + ".join(formula_terms)
    result = smf.ols(formula=formula, data=model_df).fit()
    return result


def create_visualizations(df: pd.DataFrame) -> Dict[str, "plt.Figure"]:
    """Generate a suite of matplotlib figures for downstream dashboards."""
    if df.empty:
        return {}

    if plt is None or sns is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib and seaborn are required for visualization. Install them with `pip install matplotlib seaborn`."
        )

    sns.set_theme(style="whitegrid", palette="deep")

    figures: Dict[str, plt.Figure] = {}

    if {"severity", "is_overdue"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x="severity", hue="is_overdue", ax=ax)
        ax.set_title("Task Severity vs Overdue Status")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Count")
        ax.legend(title="Overdue", labels=["No", "Yes"])
        figures["severity_overdue"] = fig

    if {"fleet", "days_overdue"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_overdue = df.groupby("fleet", as_index=False)["days_overdue"].mean()
        sns.barplot(data=avg_overdue, x="fleet", y="days_overdue", ax=ax, errorbar=None)
        ax.set_title("Average Days Overdue by Fleet")
        ax.set_xlabel("Fleet")
        ax.set_ylabel("Average Days Overdue")
        figures["fleet_overdue"] = fig

    if "due_date" in df.columns and df["due_date"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        timeline = (
            df.dropna(subset=["due_date"])
            .assign(due_day=lambda frame: frame["due_date"].dt.to_period("D").dt.to_timestamp())
            .groupby("due_day")
            .size()
            .rename("count")
            .reset_index()
        )
        sns.lineplot(data=timeline, x="due_day", y="count", marker="o", ax=ax)
        ax.set_title("Daily Volume of Open Tasks")
        ax.set_xlabel("Due Date")
        ax.set_ylabel("Tasks")
        figures["due_timeline"] = fig

    if "days_until_due" in df.columns and df["days_until_due"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=df, x="days_until_due", fill=True, ax=ax)
        ax.set_title("Distribution of Days Until Due")
        ax.set_xlabel("Days Until Due")
        figures["due_distribution"] = fig

    return figures


def _normalised_counts(df: pd.DataFrame, column: str) -> Dict[str, float]:
    if column not in df:
        return {}

    series = df[column].replace("", "Unknown").fillna("Unknown").astype(str)
    counts = series.value_counts(normalize=True, dropna=False).sort_values(ascending=False)
    return {index: round(float(value), 4) for index, value in counts.items()}
