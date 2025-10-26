"""Visualization helpers for overdue task analytics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .preparation import SEVERITY_HIGH_KEYWORDS, STATUS_RESPONSE_KEYWORDS
from .summaries import build_fleet_overdue_share
from .utils import clamp, share_of_keywords

try:  # Matplotlib stack is optional but strongly recommended.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None  # type: ignore[assignment]

try:
    from wordcloud import WordCloud
except ImportError:  # pragma: no cover - optional dependency
    WordCloud = None  # type: ignore[assignment]


@dataclass
class VisualizationBundle:
    """Container bundling a rendered matplotlib figure with its source datasets."""

    figure: Any
    datasets: "OrderedDict[str, pd.DataFrame]"


def create_visualizations(df: pd.DataFrame) -> "OrderedDict[str, VisualizationBundle]":
    """Generate a suite of matplotlib figures for downstream dashboards."""
    if df.empty:
        return OrderedDict()

    if plt is None or sns is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib and seaborn are required for visualization. Install them with `pip install matplotlib seaborn`."
        )

    sns.set_theme(style="whitegrid", palette="deep")

    bundles: "OrderedDict[str, VisualizationBundle]" = OrderedDict()
    overdue_order = ["On Time", "Overdue"]

    if {"severity_display", "is_overdue"}.issubset(df.columns):
        plot_df = df[["severity_display", "is_overdue"]].copy()
        plot_df.rename(columns={"severity_display": "Severity"}, inplace=True)
        plot_df["Severity"] = (
            plot_df["Severity"].astype("string").replace({"": "UNKNOWN", "<NA>": "UNKNOWN"}).fillna("UNKNOWN")
        )
        plot_df["Overdue Status"] = plot_df["is_overdue"].map({True: "Overdue", False: "On Time"}).fillna("On Time")

        severity_counts = (
            plot_df.groupby(["Severity", "Overdue Status"], dropna=False).size().rename("Count").reset_index()
        )
        if not severity_counts.empty:
            severity_order = (
                severity_counts.groupby("Severity")["Count"].sum().sort_values(ascending=False).index.tolist()
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                data=plot_df,
                x="Severity",
                hue="Overdue Status",
                order=severity_order,
                hue_order=overdue_order,
                ax=ax,
            )
            ax.set_title("Task Severity vs Overdue Status")
            ax.set_xlabel("Severity (Standardized)")
            ax.set_ylabel("Task Count")
            ax.legend(title="Overdue Status")
            ax.tick_params(axis="x", rotation=35, labelsize=10)
            fig.tight_layout()

            datasets = OrderedDict(
                [("Severity vs Overdue", severity_counts.sort_values(["Severity", "Overdue Status"]).reset_index(drop=True))]
            )
            bundles["severity_overdue"] = VisualizationBundle(fig, datasets)

    if {"deferral_class", "is_overdue"}.issubset(df.columns):
        plot_df = df[["deferral_class", "is_overdue"]].copy()
        plot_df.rename(columns={"deferral_class": "Deferral Class"}, inplace=True)
        plot_df["Deferral Class"] = (
            plot_df["Deferral Class"]
            .astype("string")
            .replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})
            .fillna("UNKNOWN")
        )
        plot_df["Overdue Status"] = plot_df["is_overdue"].map({True: "Overdue", False: "On Time"}).fillna("On Time")

        deferral_counts = (
            plot_df.groupby(["Deferral Class", "Overdue Status"], dropna=False).size().rename("Count").reset_index()
        )
        if not deferral_counts.empty:
            deferral_order = (
                deferral_counts.groupby("Deferral Class")["Count"].sum().sort_values(ascending=False).index.tolist()
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                data=plot_df,
                x="Deferral Class",
                hue="Overdue Status",
                order=deferral_order,
                hue_order=overdue_order,
                ax=ax,
            )
            ax.set_title("Deferral Class vs Overdue Status")
            ax.set_xlabel("Deferral Class")
            ax.set_ylabel("Task Count")
            ax.legend(title="Overdue Status")
            ax.tick_params(axis="x", rotation=35, labelsize=10)
            fig.tight_layout()

            datasets = OrderedDict(
                [
                    (
                        "Deferral Class vs Overdue",
                        deferral_counts.sort_values(["Deferral Class", "Overdue Status"]).reset_index(drop=True),
                    )
                ]
            )
            bundles["deferral_class_overdue"] = VisualizationBundle(fig, datasets)

    if {"fleet", "days_overdue"}.issubset(df.columns):
        plot_df = df[["fleet", "days_overdue"]].copy()
        plot_df.rename(columns={"fleet": "Fleet"}, inplace=True)
        plot_df["Fleet"] = (
            plot_df["Fleet"].astype("string").replace({"": "UNKNOWN", "<NA>": "UNKNOWN"}).fillna("UNKNOWN")
        )
        avg_overdue = (
            plot_df.dropna(subset=["Fleet"])
            .groupby("Fleet", as_index=False)["days_overdue"]
            .mean()
            .rename(columns={"days_overdue": "Average Days Overdue"})
            .sort_values("Average Days Overdue", ascending=False)
        )

        if not avg_overdue.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(
                data=avg_overdue,
                x="Fleet",
                y="Average Days Overdue",
                ax=ax,
                errorbar=None,
            )
            ax.set_title("Average Days Overdue by Fleet")
            ax.set_xlabel("Fleet")
            ax.set_ylabel("Average Days Overdue")
            ax.tick_params(axis="x", rotation=35, labelsize=10)
            fig.tight_layout()

            datasets = OrderedDict([("Average Days Overdue by Fleet", avg_overdue.reset_index(drop=True))])
            bundles["fleet_overdue"] = VisualizationBundle(fig, datasets)

    if "due_date" in df.columns and df["due_date"].notna().any():
        timeline = (
            df.dropna(subset=["due_date"])
            .assign(due_day=lambda frame: frame["due_date"].dt.to_period("D").dt.to_timestamp())
            .groupby("due_day")
            .size()
            .rename("Task Count")
            .reset_index()
            .rename(columns={"due_day": "Due Date"})
        )
        if not timeline.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=timeline, x="Due Date", y="Task Count", marker="o", ax=ax)
            ax.set_title("Daily Volume of Open Tasks")
            ax.set_xlabel("Due Date")
            ax.set_ylabel("Task Count")
            fig.autofmt_xdate()
            fig.tight_layout()

            datasets = OrderedDict([("Daily Task Volume", timeline.reset_index(drop=True))])
            bundles["due_timeline"] = VisualizationBundle(fig, datasets)

    if "days_until_due" in df.columns and df["days_until_due"].notna().any():
        non_null = df[["days_until_due"]].dropna().rename(columns={"days_until_due": "Days Until Due"})
        if not non_null.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.kdeplot(data=non_null, x="Days Until Due", fill=True, ax=ax)
            ax.set_title("Distribution of Days Until Due")
            ax.set_xlabel("Days Until Due")
            fig.tight_layout()

            datasets = OrderedDict([("Days Until Due", non_null.reset_index(drop=True))])
            bundles["due_distribution"] = VisualizationBundle(fig, datasets)

    if {"fleet", "severity_display"}.issubset(df.columns):
        pivot_source = df[["fleet", "severity_display"]].copy()
        pivot_source.rename(columns={"fleet": "Fleet", "severity_display": "Severity"}, inplace=True)
        pivot_source["Fleet"] = (
            pivot_source["Fleet"].astype("string").replace({"": "UNKNOWN", "<NA>": "UNKNOWN"}).fillna("UNKNOWN")
        )
        pivot_source["Severity"] = (
            pivot_source["Severity"].astype("string").replace({"": "UNKNOWN", "<NA>": "UNKNOWN"}).fillna("UNKNOWN")
        )

        heatmap_counts = (
            pivot_source.groupby(["Fleet", "Severity"]).size().rename("Task Count").reset_index()
        )
        if not heatmap_counts.empty:
            pivot_table = (
                heatmap_counts.pivot_table(
                    index="Fleet",
                    columns="Severity",
                    values="Task Count",
                    fill_value=0,
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Task Count"})
            ax.set_title("Fleet vs Severity Heatmap")
            ax.set_xlabel("Severity")
            ax.set_ylabel("Fleet")
            fig.tight_layout()

            datasets = OrderedDict(
                [
                    ("Fleet vs Severity (pivot)", pivot_table.reset_index()),
                    ("Fleet vs Severity (detailed)", heatmap_counts.sort_values(["Fleet", "Severity"]).reset_index(drop=True)),
                ]
            )
            bundles["fleet_severity_heatmap"] = VisualizationBundle(fig, datasets)

    fleet_share = build_fleet_overdue_share(df)
    if not fleet_share.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.3))
        palette = getattr(plt, "colormaps", None)
        if palette is not None and not callable(palette):
            cmap = palette.get("viridis")
        else:  # pragma: no cover - older Matplotlib fallback
            cmap = plt.get_cmap("viridis")
        if cmap is None:  # pragma: no cover - defensive fallback
            cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.25, 0.85, len(fleet_share)))

        ax.barh(
            fleet_share["Fleet"],
            fleet_share["Overdue Share %"],
            color=colors,
            edgecolor="#2b2d42",
        )
        ax.set_xlabel("Share of total overdue tasks (%)")
        ax.set_ylabel("Fleet")
        ax.set_title("Fleet contribution to overdue backlog")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#ced4da", alpha=0.9)

        for bar, (_, row) in zip(ax.patches, fleet_share.iterrows()):
            ax.text(
                bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{row['Overdue Share %']:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
                color="#1c1c1c",
            )

        ax_secondary = ax.twiny()
        ax_secondary.scatter(
            fleet_share["Overdue Rate %"],
            fleet_share["Fleet"],
            color="#495057",
            s=60,
            zorder=3,
        )
        ax_secondary.set_xlabel("Fleet overdue rate (%)")
        ax_secondary.set_xlim(left=0, right=max(1.0, fleet_share["Overdue Rate %"].max() * 1.1))
        ax_secondary.grid(False)

        fig.tight_layout()

        datasets = OrderedDict([("Fleet Overdue Share", fleet_share.copy())])
        bundles["fleet_overdue_share"] = VisualizationBundle(fig, datasets)

    return bundles


def create_summary_indicators(df: pd.DataFrame, summary) -> List[Dict[str, float | str]]:
    """Derive KPI indicators suitable for gauge-style widgets."""
    total = summary.total_tasks or len(df)
    total = max(total, 0)

    if total > 0:
        on_time_rate = 100.0 - (summary.overdue_tasks / total) * 100.0
        completion_pct = summary.completion_rate * 100.0
    else:
        on_time_rate = 0.0
        completion_pct = 0.0

    high_severity_pct = share_of_keywords(df.get("severity"), SEVERITY_HIGH_KEYWORDS) if total else 0.0
    response_rate = share_of_keywords(df.get("status"), STATUS_RESPONSE_KEYWORDS) if total else 0.0

    return [
        {
            "label": "On-Time Index",
            "value": round(clamp(on_time_rate), 1),
            "tooltip": "Share of active tasks that are still on time.",
        },
        {
            "label": "Completion Rate",
            "value": round(clamp(completion_pct), 1),
            "tooltip": "Tasks marked closed divided by total tasks.",
        },
        {
            "label": "High Severity Share",
            "value": round(clamp(high_severity_pct), 1),
            "tooltip": "Proportion of tasks flagged with high/critical severity.",
        },
        {
            "label": "Response Progress",
            "value": round(clamp(response_rate), 1),
            "tooltip": "Statuses mentioning response/acknowledgement/closure.",
        },
    ]


def create_summary_gauge_figures(indicators: List[Dict[str, float | str]]) -> "OrderedDict[str, go.Figure]":
    """Return Plotly gauge figures for dashboards."""
    if go is None:  # pragma: no cover - optional dependency
        raise ImportError("plotly is required for gauge visualizations. Install it with `pip install plotly`.")

    colors = ["#2b8a3e", "#1971c2", "#d9480f", "#6741d9"]
    figures: "OrderedDict[str, go.Figure]" = OrderedDict()

    for idx, metric in enumerate(indicators):
        color = colors[idx % len(colors)]
        value = float(metric["value"])
        tooltip = str(metric.get("tooltip", ""))

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": metric["label"], "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#fce0dd"},
                        {"range": [40, 70], "color": "#fff4e6"},
                        {"range": [70, 100], "color": "#e6fcf5"},
                    ],
                    "threshold": {"line": {"color": color, "width": 4}, "value": value},
                },
            )
        )
        fig.update_layout(
            margin=dict(t=30, b=30, l=30, r=30),
            height=260,
            paper_bgcolor="white",
            hoverlabel={"bgcolor": "white"},
        )
        if tooltip:
            fig.update_layout(annotations=[dict(text=tooltip, x=0.5, y=0.0, showarrow=False, font={"size": 11})])

        figures[str(metric["label"])] = fig

    return figures


def create_timeline_figure(df: pd.DataFrame) -> "go.Figure":
    """Build a Plotly area/line chart of open tasks over time."""
    if go is None:  # pragma: no cover - optional dependency
        raise ImportError("plotly is required for timeline visualization. Install it with `pip install plotly`.")

    if "due_date" not in df or df["due_date"].dropna().empty:
        raise ValueError("Timeline visualization requires a `due_date` column with values.")

    timeline = (
        df.dropna(subset=["due_date"])
        .assign(due_day=lambda frame: frame["due_date"].dt.to_period("D").dt.to_timestamp())
        .groupby("due_day")
        .size()
        .rename("count")
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["due_day"],
            y=timeline["count"],
            mode="lines+markers",
            fill="tozeroy",
            line={"color": "#1c7ed6", "width": 3},
            marker={"size": 6},
            name="Open Tasks",
        )
    )
    fig.update_layout(
        title="Open Task Volume by Due Date",
        xaxis_title="Due Date",
        yaxis_title="Tasks",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=320,
        margin=dict(t=50, b=40, l=50, r=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f3f5")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f3f5")
    return fig


def create_wordcloud_figure(summary) -> "plt.Figure":
    """Generate a qualitative word cloud from severity/status/fleet mix."""
    if WordCloud is None or plt is None:  # pragma: no cover - optional dependency
        raise ImportError("wordcloud and matplotlib are required for the key drivers cloud.")

    frequencies: Dict[str, float] = {}
    for bucket in (summary.severity_distribution, summary.status_distribution, summary.fleet_distribution):
        for key, value in bucket.items():
            if not key or key.lower() in {"unknown", "nan"}:
                continue
            frequencies[key] = frequencies.get(key, 0.0) + float(value)

    if not frequencies:
        raise ValueError("Insufficient categorical data to build a word cloud.")

    wc = WordCloud(width=1200, height=600, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


__all__ = [
    "VisualizationBundle",
    "create_summary_gauge_figures",
    "create_summary_indicators",
    "create_timeline_figure",
    "create_visualizations",
    "create_wordcloud_figure",
]
