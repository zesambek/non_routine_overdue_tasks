"""Summary dataclasses and KPI aggregations for overdue task analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .preparation import SEVERITY_HIGH_KEYWORDS, STATUS_RESPONSE_KEYWORDS
from .utils import coerce_to_naive_timestamp, share_of_keywords


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
    report_date: datetime | None


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
            report_date=None,
        )

    overdue_tasks = int(df.get("is_overdue", False).sum()) if "is_overdue" in df else 0
    total_tasks = int(len(df))

    completion_rate = 0.0
    if "status" in df.columns and total_tasks > 0:
        closed = df["status"].str.lower().str.contains("close").sum()
        completion_rate = float(closed / total_tasks)

    mean_days_until_due = float(df["days_until_due"].mean()) if "days_until_due" in df else float("nan")
    mean_days_overdue = float(df.loc[df.get("is_overdue", False), "days_overdue"].mean()) if overdue_tasks else 0.0

    report_date: datetime | None = None
    if "run_date" in df.columns:
        run_dates = df["run_date"].dropna()
        if not run_dates.empty:
            report_date = coerce_to_naive_timestamp(run_dates.max())

    return TaskAnalyticsSummary(
        total_tasks=total_tasks,
        overdue_tasks=overdue_tasks,
        completion_rate=completion_rate,
        mean_days_until_due=mean_days_until_due,
        mean_days_overdue=mean_days_overdue,
        severity_distribution=_normalised_counts(df, "severity"),
        status_distribution=_normalised_counts(df, "status"),
        fleet_distribution=_normalised_counts(df, "fleet"),
        report_date=report_date,
    )


def build_fleet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fleet-level metrics for dashboard tables."""
    if df.empty or "fleet" not in df:
        return pd.DataFrame(columns=["Fleet", "Index", "Δ Index", "Tasks", "Overdue", "On-Time %", "High Severity %"])

    grouped = df.groupby("fleet", dropna=False)
    metrics = grouped.agg(
        tasks=("fleet", "size"),
        overdue=("is_overdue", "sum"),
        mean_days_until_due=("days_until_due", "mean"),
    ).reset_index()

    metrics["on_time_pct"] = 100.0 - (metrics["overdue"] / metrics["tasks"].replace({0: np.nan})) * 100.0
    metrics["on_time_pct"] = metrics["on_time_pct"].fillna(0.0)
    metrics["index"] = metrics["on_time_pct"].clip(0, 100)
    metrics["delta_index"] = 0.0  # Placeholder until historical data is available.

    high_severity_pct = grouped["severity"].apply(lambda col: share_of_keywords(col, SEVERITY_HIGH_KEYWORDS))
    metrics["high_severity_pct"] = metrics["fleet"].map(high_severity_pct).fillna(0.0)

    metrics["Fleet"] = metrics["fleet"]
    metrics["Index"] = metrics["index"].round(1)
    metrics["Δ Index"] = metrics["delta_index"].round(1)
    metrics["Tasks"] = metrics["tasks"].astype(int)
    metrics["Overdue"] = metrics["overdue"].astype(int)
    metrics["On-Time %"] = metrics["on_time_pct"].round(1)
    metrics["High Severity %"] = metrics["high_severity_pct"].round(1)

    table = metrics[["Fleet", "Index", "Δ Index", "Tasks", "Overdue", "On-Time %", "High Severity %"]].sort_values(
        by="Index", ascending=False
    )
    return table.reset_index(drop=True)


def build_fleet_overdue_share(df: pd.DataFrame) -> pd.DataFrame:
    """Return fleet-level totals with overdue share metrics for dashboards and reports."""
    if df.empty or "fleet" not in df.columns:
        return pd.DataFrame(
            columns=[
                "Fleet",
                "Total Tasks",
                "Overdue Tasks",
                "Overdue Rate %",
                "Task Share %",
                "Overdue Share %",
            ]
        )

    working = df.copy()
    working["_overdue_flag"] = working.get("is_overdue", False).fillna(False).astype(bool)

    grouped = (
        working.groupby("fleet", dropna=False)
        .agg(
            total_tasks=("fleet", "size"),
            overdue_tasks=("_overdue_flag", "sum"),
        )
        .reset_index()
        .rename(columns={"fleet": "Fleet"})
    )

    grouped["Fleet"] = grouped["Fleet"].astype("string").replace({"<NA>": "Unknown", "": "Unknown"}).fillna("Unknown")

    total_tasks_overall = grouped["total_tasks"].sum()
    overdue_total_overall = grouped["overdue_tasks"].sum()

    grouped["Overdue Rate %"] = (
        (grouped["overdue_tasks"] / grouped["total_tasks"].replace({0: np.nan})) * 100.0
    ).fillna(0.0)
    if total_tasks_overall:
        grouped["Task Share %"] = (grouped["total_tasks"] / total_tasks_overall) * 100.0
    else:
        grouped["Task Share %"] = 0.0

    if overdue_total_overall:
        grouped["Overdue Share %"] = (grouped["overdue_tasks"] / overdue_total_overall) * 100.0
    else:
        grouped["Overdue Share %"] = 0.0

    result = grouped.assign(
        **{
            "Total Tasks": grouped["total_tasks"].astype(int),
            "Overdue Tasks": grouped["overdue_tasks"].astype(int),
            "Overdue Rate %": grouped["Overdue Rate %"].round(1),
            "Task Share %": grouped["Task Share %"].round(1),
            "Overdue Share %": grouped["Overdue Share %"].round(1),
        }
    )[
        [
            "Fleet",
            "Total Tasks",
            "Overdue Tasks",
            "Overdue Rate %",
            "Task Share %",
            "Overdue Share %",
        ]
    ]

    return result.sort_values(["Overdue Share %", "Overdue Tasks"], ascending=[False, False]).reset_index(drop=True)


def _normalised_counts(df: pd.DataFrame, column: str) -> Dict[str, float]:
    if column not in df:
        return {}

    series = df[column].replace("", "Unknown").fillna("Unknown").astype(str)
    counts = series.value_counts(normalize=True, dropna=False).sort_values(ascending=False)
    return {index: round(float(value), 4) for index, value in counts.items()}


__all__ = [
    "TaskAnalyticsSummary",
    "build_fleet_summary",
    "build_fleet_overdue_share",
    "summarize_tasks",
]
