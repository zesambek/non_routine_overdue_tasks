"""Streamlit UI helpers for analytics and visualization dashboards."""

from __future__ import annotations

import io
from typing import Dict

import pandas as pd
import streamlit as st

from overduetasks.analytics import (
    TaskAnalyticsSummary,
    create_visualizations,
    identify_outliers,
    prepare_task_dataframe,
    summarize_tasks,
    train_due_date_model,
)


def render_analysis_dashboard(raw_df: pd.DataFrame) -> None:
    """Render interactive analytics, statistical modeling, and charts in Streamlit."""
    st.header("Analytics & Insights")

    if raw_df.empty:
        st.info("No task data available. Scrape tasks first to unlock analytics.")
        return

    prepared = prepare_task_dataframe(raw_df)
    summary = summarize_tasks(prepared)

    _render_kpis(summary)
    _render_distribution_tables(summary)
    _render_visualizations(prepared)
    _render_outlier_section(prepared)
    _render_model_section(prepared)
    _render_download_section(prepared)


def _render_kpis(summary: TaskAnalyticsSummary) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", f"{summary.total_tasks:,}")
    col2.metric("Overdue Tasks", f"{summary.overdue_tasks:,}")
    col3.metric("Completion Proxy", f"{summary.completion_rate * 100:.1f}%")

    col4, col5 = st.columns(2)
    col4.metric("Mean Days Until Due", f"{summary.mean_days_until_due:.1f}")
    col5.metric("Mean Days Overdue", f"{summary.mean_days_overdue:.1f}")


def _render_distribution_tables(summary: TaskAnalyticsSummary) -> None:
    with st.container():
        st.subheader("Composition")
        dist_tabs = st.tabs(["Severity", "Status", "Fleet"])

        with dist_tabs[0]:
            st.dataframe(_dict_to_dataframe(summary.severity_distribution), use_container_width=True, hide_index=True)
        with dist_tabs[1]:
            st.dataframe(_dict_to_dataframe(summary.status_distribution), use_container_width=True, hide_index=True)
        with dist_tabs[2]:
            st.dataframe(_dict_to_dataframe(summary.fleet_distribution), use_container_width=True, hide_index=True)


def _render_visualizations(prepared: pd.DataFrame) -> None:
    try:
        figures = create_visualizations(prepared)
    except ImportError as exc:
        st.warning(str(exc))
        return

    if not figures:
        st.caption("No visualizations available for the current dataset.")
        return

    st.subheader("Visual Storytelling")
    viz_tabs = st.tabs([_prettify_key(name) for name in figures.keys()])
    for tab, (name, figure) in zip(viz_tabs, figures.items()):
        with tab:
            st.pyplot(figure, clear_figure=True)


def _render_outlier_section(prepared: pd.DataFrame) -> None:
    st.subheader("Outlier Spotlight")
    try:
        outliers = identify_outliers(prepared)
    except ImportError as exc:
        st.warning(str(exc))
        return

    if outliers.empty:
        st.caption("No statistically significant outliers detected.")
        return

    st.dataframe(outliers, use_container_width=True)


def _render_model_section(prepared: pd.DataFrame) -> None:
    st.subheader("Predictive Insight (OLS)")
    try:
        model = train_due_date_model(prepared)
    except (ImportError, ValueError) as exc:
        st.warning(str(exc))
        return

    st.markdown("###### Model Summary")
    st.code(model.summary().as_text())


def _render_download_section(prepared: pd.DataFrame) -> None:
    st.subheader("Take It Offline")

    with st.expander("Download enriched analytics dataframe"):
        buffer = io.StringIO()
        prepared.to_csv(buffer, index=False)
        st.download_button(
            "Download CSV",
            data=buffer.getvalue().encode("utf-8"),
            file_name="open_tasks_enriched.csv",
            mime="text/csv",
        )


def _dict_to_dataframe(values: Dict[str, float]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=["Category", "Share (%)"])
    data = pd.DataFrame(
        [(category, share * 100) for category, share in values.items()],
        columns=["Category", "Share (%)"],
    )
    return data.sort_values("Share (%)", ascending=False, ignore_index=True)


def _prettify_key(name: str) -> str:
    return name.replace("_", " ").title()
