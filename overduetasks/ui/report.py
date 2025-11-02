"""Streamlit UI helpers for analytics-driven dashboards and exports."""

from __future__ import annotations

import io
from collections import OrderedDict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from overduetasks.analytics import (
    TaskAnalyticsSummary,
    build_fleet_overdue_share,
    build_fleet_summary,
    build_overdue_breakdowns,
    build_overdue_breakdown_details,
    build_time_series_bundle,
    create_timeline_figure,
    create_summary_gauge_figures,
    create_summary_indicators,
    create_visualizations,
    VisualizationBundle,
    create_wordcloud_figure,
    dataframe_to_excel_bytes,
    generate_excel_report,
    generate_pdf_report,
    identify_outliers,
    profile_dataframe,
    prepare_task_dataframe,
    summarize_tasks,
    train_due_date_model,
)
from overduetasks.analytics.style import DEFAULT_PALETTE

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.pdfgen import canvas
except ImportError:  # pragma: no cover - optional dependency
    canvas = None  # type: ignore[assignment]
    letter = None  # type: ignore[assignment]
    colors = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    TableStyle = None  # type: ignore[assignment]
    Paragraph = None  # type: ignore[assignment]
    SimpleDocTemplate = None  # type: ignore[assignment]
    Spacer = None  # type: ignore[assignment]
    getSampleStyleSheet = None  # type: ignore[assignment]


def render_analysis_dashboard(raw_df: pd.DataFrame) -> None:
    """Render interactive analytics, statistical modeling, and export actions in Streamlit."""
    st.header("Analytics & Insights")

    if raw_df.empty:
        st.info("No task data available. Scrape tasks first to unlock analytics.")
        return

    prepared = prepare_task_dataframe(raw_df)
    summary = summarize_tasks(prepared)
    ts_bundle = None
    ts_error = None
    try:
        ts_bundle = build_time_series_bundle(prepared)
    except ImportError as exc:
        ts_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - surface unexpected analytics errors
        ts_error = f"Time-series analytics unavailable: {exc}" 

    _render_reporting_banner(summary, prepared)
    _render_summary_section(prepared, summary)
    _render_time_series_section(ts_bundle, ts_error)
    _render_overdue_breakdowns(prepared, summary)
    _render_evolution_section(prepared)
    _render_fleet_table(prepared)
    _render_additional_visuals(prepared)
    _render_fleet_deferral_heatmap(prepared)
    _render_outlier_section(prepared)
    _render_model_section(prepared)
    _render_key_drivers(summary)
    _render_download_section(prepared, summary)


def _render_reporting_banner(summary: TaskAnalyticsSummary, prepared: pd.DataFrame) -> None:
    run_date = summary.report_date
    run_caption = "Reporting snapshot date unavailable."

    if run_date is not None:
        run_caption = f"Reporting snapshot as of {run_date:%d %B %Y}."
        run_dates = prepared.get("run_date")
        if run_dates is not None:
            normalized = pd.to_datetime(run_dates, errors="coerce").dropna().dt.normalize()
            if not normalized.empty and normalized.nunique() > 1:
                run_caption += f" Latest of {normalized.nunique()} captured run dates."
    else:
        run_dates = prepared.get("run_date")
        if run_dates is not None:
            normalized = pd.to_datetime(run_dates, errors="coerce").dropna()
            if not normalized.empty:
                latest = normalized.max()
                if pd.notna(latest):
                    run_caption = f"Reporting snapshot as of {latest:%d %B %Y}."

    st.caption(run_caption)


def _render_summary_section(prepared: pd.DataFrame, summary: TaskAnalyticsSummary) -> None:
    st.subheader("Summary")

    _render_metric_row(summary)
    _render_summary_gauges(prepared, summary)
    _render_snapshot_highlights(prepared, summary)

    data_profile = profile_dataframe(prepared)
    quality_notes = _build_data_quality_notes(data_profile)
    if quality_notes:
        st.markdown("**Data Quality Watchlist**")
        for note in quality_notes:
            st.markdown(f"- {note}")
    else:
        st.caption("No significant data quality anomalies detected in the current snapshot.")

    _render_fleet_overdue_share(prepared)
    _render_summary_highlights(prepared, summary)


def _render_time_series_section(bundle, error: str | None) -> None:
    st.subheader("Time-Series Analytics")

    if error:
        st.info(error)
        return

    if bundle is None:
        st.caption("Time-series analytics are unavailable.")
        return

    if bundle.note:
        st.info(bundle.note)

    if bundle.weekly.empty and bundle.monthly.empty and bundle.quarterly.empty:
        st.caption("Insufficient indexed data to compute resampled analytics.")
        return

    try:
        tabs = st.tabs(["Weekly", "Monthly", "Quarterly", "Correlations"])

        with tabs[0]:
            _plot_frequency_tab(bundle.weekly, bundle.weekly_rolling, window_labels=(4, 12), title="Weekly cadence")

        with tabs[1]:
            _plot_frequency_tab(bundle.monthly, bundle.monthly_rolling, window_labels=(3, 6), title="Monthly cadence")

        with tabs[2]:
            _plot_basic_frequency(bundle.quarterly, "Quarterly cadence")

        with tabs[3]:
            if bundle.correlations.empty:
                st.caption("Not enough numeric metrics to derive correlations.")
            else:
                corr_fig = px.imshow(
                    bundle.correlations,
                    text_auto=True,
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    title="Pearson Correlations",
                )
                st.plotly_chart(corr_fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Time-series charts unavailable: {exc}")


def _render_overdue_breakdowns(prepared: pd.DataFrame, summary: TaskAnalyticsSummary) -> None:
    _inject_breakdown_theme()
    breakdowns = build_overdue_breakdowns(prepared)
    breakdown_details = build_overdue_breakdown_details(prepared)

    fleets = prepared.get("fleet")
    fleet_label = "ALL FLEETS"
    if fleets is not None and not fleets.dropna().empty:
        unique_fleets = sorted({str(value).strip() for value in fleets.dropna().unique() if str(value).strip()})
        cleaned = [fleet for fleet in unique_fleets if fleet.lower() not in {"unknown", "nan"}]
        target = cleaned or unique_fleets
        if target:
            if len(target) <= 3:
                fleet_label = " / ".join(target).upper()
            else:
                fleet_label = f"{' / '.join(target[:3]).upper()} (+{len(target) - 3} MORE)"

    st.subheader(f"NON ROUTINE OVERDUE TASKS STATUS FOR {fleet_label}")

    if not breakdowns:
        st.caption("No overdue tasks detected for the current snapshot.")
        return

    fleet_table = breakdowns.get("Fleet Type")
    if fleet_table is not None and not fleet_table.empty:
        _render_fleet_overdue_highlights(fleet_table)

    if summary.report_date:
        st.caption(f"Breakdowns below reflect overdue tasks recorded through {summary.report_date:%d %b %Y}.")
    else:
        st.caption("Breakdowns below reflect overdue tasks in the latest available snapshot.")

    tab_labels = list(breakdowns.keys())
    viz_tabs = st.tabs(tab_labels)

    for tab, (label, table) in zip(viz_tabs, breakdowns.items()):
        with tab:
            column_config = {
                "Fault Count": st.column_config.NumberColumn("Fault Count", format="%d"),
                "Fault Share %": st.column_config.ProgressColumn(
                    "Fault Share %",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "Top Deferral Classes": st.column_config.TextColumn(
                    "Top Deferral Classes",
                    help="Most common deferral classes within the bucket (share of bucket).",
                ),
            }

            augmented = _append_deferral_class_summary(table, label, breakdown_details.get(label, {}))

            trimmed = augmented
            if len(augmented) > 10:
                show_full = st.toggle(
                    "Show full list",
                    value=False,
                    key=f"{label}_show_all",
                    help="Toggle to reveal the complete table.",
                )
                if not show_full:
                    trimmed = augmented.head(10)
                    st.caption(f"Showing top {len(trimmed)} of {len(augmented)} groups by fault count.")

            display_columns = [label, "Fault Count", "Fault Share %", "Top Deferral Classes"]
            display_columns = [col for col in display_columns if col in trimmed.columns]

            with st.container():
                st.dataframe(
                    trimmed[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )
            insight_text = _breakdown_insight(trimmed, label)
            if insight_text:
                st.caption(insight_text)

            selected_bucket = trimmed[label].iloc[0] if not trimmed.empty else None
            if not trimmed.empty and len(trimmed) > 1:
                selected_bucket = st.selectbox(
                    f"Visualise deferral mix for {label.lower()}",
                    trimmed[label].tolist(),
                    index=0,
                    key=f"{label}_deferral_viz",
                )

            charts = st.columns((3, 2, 2))

            with charts[0]:
                chart_fig = _build_breakdown_bar_chart(trimmed, label)
                st.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})

            with charts[1]:
                share_fig = _build_share_pie_chart(trimmed, label)
                st.plotly_chart(share_fig, use_container_width=True, config={"displayModeBar": False})

            with charts[2]:
                mix = {}
                if selected_bucket is not None and "_deferral_mix" in augmented.columns:
                    row = augmented.loc[augmented[label] == selected_bucket]
                    if not row.empty:
                        mix = row.iloc[0].get("_deferral_mix", {}) or {}
                deferral_fig = _build_deferral_pie_chart(mix, bucket=str(selected_bucket) if selected_bucket else label)
                st.plotly_chart(deferral_fig, use_container_width=True, config={"displayModeBar": False})

            excel_ready = augmented.drop(columns=["_deferral_mix"], errors="ignore")
            excel_bytes = dataframe_to_excel_bytes(excel_ready, sheet_name=label)
            safe_name = str(label).lower().replace(" ", "_").replace("/", "_")
            st.download_button(
                f"Download {label} Breakdown (.xlsx)",
                data=excel_bytes,
                file_name=f"overdue_{safe_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"{label}_agg_download",
            )
            if canvas is not None and SimpleDocTemplate is not None:
                pdf_bytes = _breakdown_to_pdf_bytes(
                    label=label,
                    table=excel_ready[display_columns],
                    insight=insight_text,
                    share_table=trimmed[[label, "Fault Share %", "Fault Count"]] if "Fault Share %" in trimmed else None,
                    mix=mix,
                    selected_bucket=str(selected_bucket) if selected_bucket is not None else None,
                )
                st.download_button(
                    f"Download {label} Breakdown (.pdf)",
                    data=pdf_bytes,
                    file_name=f"overdue_{safe_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"{label}_agg_pdf_download",
                )
            elif canvas is None:
                st.caption("Install reportlab to enable PDF exports for this breakdown.")

            buckets = breakdown_details.get(label, {})
            if buckets:
                st.markdown("**Deep dive**")
                options = list(buckets.keys())
                focus = st.selectbox(
                    f"Select {label.lower()} to inspect",
                    options,
                    key=f"{label}_detail_select",
                )

                detail_df = buckets.get(focus, pd.DataFrame())
                if detail_df.empty:
                    st.caption("No detailed records available for the selected bucket.")
                else:
                    safe_focus = str(focus)
                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    detail_excel = dataframe_to_excel_bytes(detail_df, sheet_name=f"{label}_{safe_focus}")
                    file_stub = str(safe_focus).lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
                    detail_file = f"overdue_{safe_name}_{file_stub}.xlsx"
                    st.download_button(
                        f"Download {focus} tasks (.xlsx)",
                        data=detail_excel,
                        file_name=detail_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key=f"{label}_detail_download",
                    )

                    if "Config Prefix" in detail_df.columns:
                        prefix_series = detail_df["Config Prefix"].astype("string").replace("", "Unknown").fillna("Unknown")
                        prefix_counts = prefix_series.value_counts()
                        if not prefix_counts.empty:
                            prefix_table = (
                                prefix_counts.rename_axis("Config Prefix").reset_index(name="Fault Count")
                            )
                            st.caption("Config prefix concentration for the selected bucket")
                            st.dataframe(prefix_table, use_container_width=True, hide_index=True)
                            st.bar_chart(prefix_table.set_index("Config Prefix"), use_container_width=True)

    st.caption("Need a consolidated PDF? Use the Export section below to grab the full report, now enhanced with these breakdowns.")


def _append_deferral_class_summary(
    table: pd.DataFrame,
    label: str,
    detail_mapping: OrderedDict[str, pd.DataFrame] | dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return a copy of ``table`` augmented with deferral class bullet summaries."""
    if table.empty or label not in table.columns:
        return table
    if label.lower() == "deferral class":
        return table

    mapped = table.copy()
    if not detail_mapping:
        return mapped

    mix_lookup: dict[str, dict[str, float]] = {}
    text_lookup: dict[str, str] = {}

    for bucket, detail_df in detail_mapping.items():
        if detail_df.empty:
            continue
        summary_text, mix = _summarise_deferral_classes(detail_df)
        key = str(bucket)
        text_lookup[key] = summary_text
        mix_lookup[key] = mix

    if not text_lookup:
        return mapped

    mapped["_deferral_mix"] = (
        mapped[label]
        .astype("string")
        .map(lambda value: mix_lookup.get(str(value), {}))
    )

    mapped["Top Deferral Classes"] = (
        mapped[label]
        .astype("string")
        .map(lambda value: text_lookup.get(str(value), "—"))
        .fillna("—")
    )
    return mapped


def _summarise_deferral_classes(
    detail_df: pd.DataFrame,
    *,
    max_items: int = 3,
) -> tuple[str, dict[str, float]]:
    """Summarise top deferral classes for a given bucket."""
    for candidate in ("Deferral Class", "deferral_class"):
        if candidate in detail_df.columns:
            series = detail_df[candidate]
            break
    else:
        return "—"

    normalised = (
        series.astype("string")
        .str.strip()
        .replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})
        .fillna("UNKNOWN")
    )
    counts = normalised.value_counts()
    if counts.empty:
        return "—", {}

    total = counts.sum()
    percentage = (counts / total * 100.0) if total else counts * 0.0
    parts = []
    for value, share in percentage.head(max_items).items():
        parts.append(f"{value} ({share:.1f}%)")
    if len(percentage) > max_items:
        parts.append("…")
    return ", ".join(parts), percentage.to_dict()


def _breakdown_insight(table: pd.DataFrame, label: str) -> str | None:
    if table.empty or "Fault Share %" not in table or label not in table.columns:
        return None

    top_row = table.iloc[0]
    top_share = float(top_row.get("Fault Share %", 0.0))
    top_count = int(top_row.get("Fault Count", 0))
    top_label = str(top_row[label])

    cumulative = table["Fault Share %"].cumsum()
    coverage_idx = min(2, len(cumulative) - 1)
    coverage = float(cumulative.iloc[coverage_idx]) if coverage_idx >= 0 else top_share

    return (
        f"{top_label} leads with {top_count:,} overdue tasks ({top_share:.1f}% share); "
        f"top {coverage_idx + 1} buckets account for {coverage:.1f}% of volume."
    )


def _build_breakdown_bar_chart(table: pd.DataFrame, label: str):
    if table.empty or label not in table.columns or "Fault Count" not in table.columns:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(t=20, b=40, l=40, r=40),
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            font=dict(color="#1c1f24", size=12),
        )
        return fig

    fig = px.bar(
        table,
        x=label,
        y="Fault Count",
        text="Fault Count",
    )
    primary_colour = DEFAULT_PALETTE[1] if len(DEFAULT_PALETTE) > 1 else "#1c7293"
    fig.update_traces(
        marker_color=primary_colour,
        marker_line_color="#1c1f24",
        marker_line_width=0.6,
        texttemplate="%{y:.0f}",
        textposition="outside",
    )
    if "Fault Share %" in table.columns:
        fig.update_traces(
            customdata=table[["Fault Share %"]].to_numpy(),
            hovertemplate=f"{label}: %{{x}}<br>Fault Count: %{{y:,}}<br>Fault Share: %{{customdata[0]:.1f}}%",
        )
    else:
        fig.update_traces(
            hovertemplate=f"{label}: %{{x}}<br>Fault Count: %{{y:,}}",
            customdata=None,
        )

    fig.update_layout(
        margin=dict(t=20, b=60, l=60, r=20),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(color="#1c1f24", size=12),
        xaxis=dict(title=label, showline=False, showgrid=False, tickangle=-35),
        yaxis=dict(title="Fault Count", gridcolor="#d0d5dd", zerolinecolor="#d0d5dd"),
    )
    fig.update_yaxes(range=[0, table["Fault Count"].max() * 1.1])
    return fig


def _build_share_pie_chart(table: pd.DataFrame, label: str):
    if table.empty or "Fault Share %" not in table.columns:
        return go.Figure()

    pie_df = table[[label, "Fault Share %"]].dropna()
    if pie_df.empty:
        return go.Figure()

    fig = px.pie(
        pie_df,
        names=label,
        values="Fault Share %",
        hole=0.35,
        color_discrete_sequence=DEFAULT_PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        title=f"{label} share",
        margin=dict(t=20, b=20, l=20, r=20),
        legend_title_text=label,
        showlegend=True,
        paper_bgcolor="white",
        font=dict(color="#1c1f24", size=12),
    )
    return fig


def _build_deferral_pie_chart(mix: dict[str, float], *, bucket: str) -> go.Figure:
    if not mix:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="white",
            annotations=[
                dict(text="No deferral data", x=0.5, y=0.5, showarrow=False, font=dict(size=12, color="#6c757d"))
            ],
        )
        return fig

    mix_df = (
        pd.DataFrame(mix.items(), columns=["Deferral Class", "Share"])
        .sort_values("Share", ascending=False)
        .reset_index(drop=True)
    )

    fig = px.pie(
        mix_df,
        names="Deferral Class",
        values="Share",
        hole=0.35,
        color_discrete_sequence=DEFAULT_PALETTE,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        title=f"Deferral mix: {bucket}",
        margin=dict(t=40, b=20, l=20, r=20),
        legend_title_text="Deferral Class",
        paper_bgcolor="white",
        font=dict(color="#1c1f24", size=12),
    )
    return fig


def _plot_frequency_tab(base_frame: pd.DataFrame, rolling_frame: pd.DataFrame, *, window_labels: tuple[int, int], title: str) -> None:
    if base_frame.empty:
        st.caption("No datapoints available for this cadence.")
        return

    required = {"run_date", "total_tasks", "overdue_tasks", "overdue_rate"}
    if not required.issubset(base_frame.columns):
        st.caption("Cadence data missing required metrics.")
        return

    st.caption(title)

    tasks_fig = px.line(
        base_frame,
        x="run_date",
        y=["total_tasks", "overdue_tasks"],
        labels={"value": "Tasks", "run_date": "Period", "variable": "Series"},
        title="Task volume trend",
    )
    tasks_fig.update_layout(legend_title_text="")
    st.plotly_chart(tasks_fig, use_container_width=True, config={"displayModeBar": False})

    rate_fig = px.line(
        base_frame,
        x="run_date",
        y="overdue_rate",
        labels={"overdue_rate": "Overdue rate", "run_date": "Period"},
        title="Overdue rate trend",
    )
    st.plotly_chart(rate_fig, use_container_width=True, config={"displayModeBar": False})

    if not rolling_frame.empty and {"run_date"}.issubset(rolling_frame.columns):
        win_a, win_b = window_labels
        task_roll_cols = [col for col in rolling_frame.columns if col.startswith("tasks_roll_")]
        if task_roll_cols:
            roll_fig = px.line(
                rolling_frame,
                x="run_date",
                y=task_roll_cols,
                labels={"value": "Tasks", "run_date": "Period"},
                title=f"Rolling task averages ({win_a}/{win_b} periods)",
            )
            st.plotly_chart(roll_fig, use_container_width=True, config={"displayModeBar": False})

        rate_roll_cols = [col for col in rolling_frame.columns if col.startswith("overdue_rate_roll_")]
        if rate_roll_cols:
            rate_roll_fig = px.line(
                rolling_frame,
                x="run_date",
                y=rate_roll_cols,
                labels={"value": "Overdue rate", "run_date": "Period"},
                title=f"Rolling overdue rate ({win_a}/{win_b} periods)",
            )
            st.plotly_chart(rate_roll_fig, use_container_width=True, config={"displayModeBar": False})

    excel_bytes = dataframe_to_excel_bytes(base_frame, sheet_name="Time Series")
    st.download_button(
        "Download cadence data (.xlsx)",
        data=excel_bytes,
        file_name=f"timeseries_{title.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _plot_basic_frequency(base_frame: pd.DataFrame, title: str) -> None:
    if base_frame.empty:
        st.caption("No datapoints available for this cadence.")
        return

    if not {"run_date", "total_tasks", "overdue_rate"}.issubset(base_frame.columns):
        st.caption("Quarterly cadence missing required metrics.")
        return

    st.caption(title)
    fig = px.bar(
        base_frame,
        x="run_date",
        y="total_tasks",
        color="overdue_rate",
        labels={"total_tasks": "Tasks", "run_date": "Period", "overdue_rate": "Overdue rate"},
        title="Quarterly task distribution",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    excel_bytes = dataframe_to_excel_bytes(base_frame, sheet_name="Quarterly")
    st.download_button(
        "Download quarterly data (.xlsx)",
        data=excel_bytes,
        file_name="timeseries_quarterly.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _render_fleet_overdue_highlights(table: pd.DataFrame) -> None:
    if table.empty:
        return

    cards = st.columns(3)

    leader = table.sort_values("Fault Count", ascending=False).iloc[0]
    cards[0].metric(
        "Largest Overdue Fleet",
        leader.get("Fleet Type", "Unknown"),
        f"{int(leader['Fault Count'])} tasks" if pd.notna(leader.get("Fault Count")) else "",
    )

    if "Avg Days Overdue" in table.columns and not table["Avg Days Overdue"].dropna().empty:
        slowest = table.sort_values("Avg Days Overdue", ascending=False).iloc[0]
        cards[1].metric(
            "Slowest Recovery",
            slowest.get("Fleet Type", "Unknown"),
            f"{slowest['Avg Days Overdue']:.1f} days overdue" if pd.notna(slowest.get("Avg Days Overdue")) else "",
        )
    else:
        cards[1].metric("Slowest Recovery", "—", "")

    if "Avg Days Active" in table.columns and not table["Avg Days Active"].dropna().empty:
        aging = table.sort_values("Avg Days Active", ascending=False).iloc[0]
        cards[2].metric(
            "Oldest Findings",
            aging.get("Fleet Type", "Unknown"),
            f"{aging['Avg Days Active']:.1f} days active" if pd.notna(aging.get("Avg Days Active")) else "",
        )
    else:
        cards[2].metric("Oldest Findings", "—", "")


def _inject_breakdown_theme() -> None:
    if st.session_state.get("_breakdown_theme_injected"):
        return

    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] button {
            gap: 0.5rem;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            font-weight: 600;
            background-color: #f1f3f5;
            color: #495057;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(90deg, #364fc7, #2f9e44);
            color: #ffffff;
        }
        .stProgress > div > div > div > div {
            background-color: #2f9e44;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_breakdown_theme_injected"] = True


def _render_snapshot_highlights(prepared: pd.DataFrame, summary: TaskAnalyticsSummary) -> None:
    if summary.total_tasks == 0:
        st.info("No tasks captured for the current snapshot. Scrape the latest data to populate analytics.")
        return

    bullets: list[str] = []
    total = summary.total_tasks
    overdue = summary.overdue_tasks

    on_time_pct = summary.on_time_rate * 100.0
    completion_pct = summary.completion_rate * 100.0
    response_pct = summary.response_rate * 100.0

    headline = f"{total:,} tasks tracked with {overdue:,} overdue — on-time index at {on_time_pct:.1f}%."
    if response_pct > 0:
        headline += f" Response/acknowledgement progress stands at {response_pct:.1f}%."
    bullets.append(headline)

    runway_parts: list[str] = [f"Completion rate {completion_pct:.1f}%"]
    if not pd.isna(summary.mean_days_until_due):
        runway_parts.append(f"avg days until due {summary.mean_days_until_due:.1f}")
    if summary.overdue_tasks and not pd.isna(summary.mean_days_overdue):
        runway_parts.append(f"overdue aging {summary.mean_days_overdue:.1f} days")
    bullets.append(", ".join(runway_parts) + ".")

    if summary.high_severity_share > 0:
        bullets.append(
            f"High/critical severity work represents {summary.high_severity_share * 100:.1f}% of the active backlog."
        )

    status_dist = sorted(summary.status_distribution.items(), key=lambda item: item[1], reverse=True)[:3]
    if status_dist:
        status_text = ", ".join(f"{name} {share * 100:.1f}%" for name, share in status_dist)
        bullets.append(f"Status mix concentration: {status_text}.")

    st.markdown("**Snapshot Highlights**")
    for text in bullets[:4]:
        st.markdown(f"- {text}")


def _build_data_quality_notes(profile) -> list[str]:
    issues: list[str] = []
    added: set[str] = set()

    for column in profile.columns:
        message = ""
        if column.missing_pct >= 30.0:
            message = f"{column.name} is missing {column.missing_pct:.1f}% of values."
        elif column.semantic_type == "mixed":
            message = f"{column.name} mixes data types (inferred as {column.inferred_dtype})."
        elif profile.row_count > 0 and column.unique_count <= 1 and column.semantic_type not in {"identifier", "boolean"}:
            message = f"{column.name} shows little variation — verify that the feed is up to date."

        if message and message not in added:
            issues.append(message)
            added.add(message)

        if len(issues) >= 4:
            break

    for warning in profile.warnings:
        if warning and warning not in added:
            issues.append(warning)
            added.add(warning)
        if len(issues) >= 4:
            break

    return issues[:4]


def _render_metric_row(summary: TaskAnalyticsSummary) -> None:
    primary = st.columns(4)
    primary[0].metric("Total Tasks", f"{summary.total_tasks:,}")
    primary[1].metric("Overdue Tasks", f"{summary.overdue_tasks:,}")
    primary[2].metric("On-Time Rate", f"{summary.on_time_rate * 100:.1f}%")
    primary[3].metric("Completion Rate", f"{summary.completion_rate * 100:.1f}%")

    secondary = st.columns(4)
    secondary[0].metric("Response Progress", f"{summary.response_rate * 100:.1f}%")
    secondary[1].metric("High Severity Share", f"{summary.high_severity_share * 100:.1f}%")
    avg_overdue = "—" if pd.isna(summary.mean_days_overdue) else f"{summary.mean_days_overdue:.1f}"
    avg_until = "—" if pd.isna(summary.mean_days_until_due) else f"{summary.mean_days_until_due:.1f}"
    secondary[2].metric("Avg Days Overdue", avg_overdue)
    secondary[3].metric("Avg Days Until Due", avg_until)


def _render_summary_gauges(prepared: pd.DataFrame, summary: TaskAnalyticsSummary) -> None:
    if summary.total_tasks <= 0:
        return

    try:
        indicators = create_summary_indicators(prepared, summary)
        figures = create_summary_gauge_figures(indicators)
    except ImportError as exc:
        st.info(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Summary gauges unavailable: {exc}")
        return

    if not figures:
        return

    columns = st.columns(len(figures))
    for column, figure in zip(columns, figures.values()):
        column.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_fleet_overdue_share(prepared: pd.DataFrame) -> None:
    if "fleet" not in prepared.columns or prepared.empty:
        return
    
    fleet_summary = build_fleet_overdue_share(prepared)
    if fleet_summary.empty:
        return

    st.markdown("**Fleet Overdue Share**")
    st.caption(
        "Contribution of each fleet to overall overdue volume, alongside fleet-specific overdue rates and task shares."
    )

    st.dataframe(
        fleet_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total Tasks": st.column_config.NumberColumn("Total Tasks", format="%d"),
            "Overdue Tasks": st.column_config.NumberColumn("Overdue Tasks", format="%d"),
            "Overdue Share %": st.column_config.ProgressColumn(
                "Overdue Share %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                help="Percentage contribution of this fleet to the total overdue tasks across all fleets.",
            ),
            "Overdue Rate %": st.column_config.NumberColumn(
                "Overdue Rate %",
                format="%.1f%%",
                help="Overdue tasks divided by total tasks within the fleet.",
            ),
            "Task Share %": st.column_config.NumberColumn(
                "Task Share %",
                format="%.1f%%",
                help="Fleet's share of overall tasks in the dataset.",
            ),
        },
    )

    chart_df = fleet_summary.set_index("Fleet")[["Overdue Share %"]]
    st.bar_chart(chart_df, use_container_width=True)

    download_cols = st.columns([2, 1])
    with download_cols[0]:
        summary_label = "Fleet Overdue Share Summary"
        all_excel = _datasets_to_excel_bytes(OrderedDict({summary_label: fleet_summary}))
        st.download_button(
            "Download summary (.xlsx)",
            data=all_excel,
            file_name="fleet_overdue_share.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="fleet_overdue_share_excel",
        )

    fleets = fleet_summary["Fleet"].astype("string").tolist()
    if not fleets:
        return

    with download_cols[1]:
        selected_fleet = st.selectbox("Fleet", options=fleets, key="fleet_detail_select")

    fleet_row = fleet_summary.loc[fleet_summary["Fleet"] == selected_fleet].iloc[0]
    metrics = st.columns(3)
    metrics[0].metric("Fleet Tasks", f"{int(fleet_row['Total Tasks']):,}")
    metrics[1].metric("Fleet Overdue", f"{int(fleet_row['Overdue Tasks']):,}")
    metrics[2].metric("Fleet Overdue Rate", f"{fleet_row['Overdue Rate %']:.1f}%")

    detail_cols = st.columns(2)
    with detail_cols[0]:
        fleet_rows = prepared.loc[prepared["fleet"].astype("string") == selected_fleet]
        if fleet_rows.empty:
            st.caption("No tasks for selection.")
        else:
            excel_bytes = dataframe_to_excel_bytes(
                fleet_rows,
                sheet_name=_safe_sheet_name(f"{selected_fleet} Tasks", set(), fallback_index=1),
            )
            safe_stub = _safe_filename_fragment(selected_fleet)
            st.download_button(
                "Download fleet (.xlsx)",
                data=excel_bytes,
                file_name=f"fleet_{safe_stub}_tasks.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="fleet_download_excel",
            )

    with detail_cols[1]:
        if canvas is None or letter is None:
            st.caption("Install reportlab to export PDF summaries.")
        else:
            pdf_bytes = _fleet_summary_to_pdf(selected_fleet, fleet_row)
            safe_stub = _safe_filename_fragment(selected_fleet)
            st.download_button(
                "Download summary (.pdf)",
                data=pdf_bytes,
                file_name=f"fleet_{safe_stub}_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"fleet_{safe_stub}_pdf",
            )


def _render_summary_highlights(prepared: pd.DataFrame, summary: TaskAnalyticsSummary) -> None:
    bullets: list[str] = []

    fleet_share = build_fleet_overdue_share(prepared)
    if not fleet_share.empty:
        leader = fleet_share.iloc[0]
        leader_text = (
            f"{leader['Fleet']} carries {leader['Overdue Share %']:.1f}% of overdue tasks "
            f"({leader['Overdue Tasks']:,} backlog items)."
        )

        if len(fleet_share) > 1:
            runner_up = fleet_share.iloc[1]
            spread = fleet_share["Overdue Share %"].head(3).sum()
            leader_text += (
                f" Next fleets: {runner_up['Fleet']} at {runner_up['Overdue Share %']:.1f}% and "
                f"{spread:.1f}% of all overdues sit within the top three fleets."
            )
        bullets.append(leader_text)

    overdue_df = prepared.loc[prepared.get("is_overdue", False)].copy()
    if not overdue_df.empty:
        mean_days = overdue_df["days_overdue"].mean()
        median_days = overdue_df["days_overdue"].median()
        p90_days = overdue_df["days_overdue"].quantile(0.9)
        chronic = int((overdue_df["days_overdue"] >= 180).sum())
        bullets.append(
            f"Overdue aging averages {mean_days:,.0f} days (median {median_days:,.0f}, 90th percentile {p90_days:,.0f}); "
            f"{chronic} tasks have been open for ≥180 days."
        )

    severity_counts = (
        prepared["severity_display"]
        .astype("string")
        .replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})
        .str.upper()
        .value_counts(normalize=True)
        .head(3)
    )
    if not severity_counts.empty:
        entries = [f"{name.title()} {share * 100:.1f}%" for name, share in severity_counts.items()]
        bullets.append(f"Task severity mix is dominated by {', '.join(entries)}.")

    if "deferral_class" in prepared.columns:
        deferral_share = (
            prepared["deferral_class"]
            .astype("string")
            .replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})
            .value_counts(normalize=True)
            .head(3)
        )
        if not deferral_share.empty:
            parts = [f"{name} {share * 100:.1f}%" for name, share in deferral_share.items()]
            bullets.append(f"Top deferral classes: {', '.join(parts)}.")

    if bullets:
        st.markdown("**Insight Highlights**")
        for text in bullets[:4]:
            st.markdown(f"- {text}")
        st.caption(
            "Highlights use descriptive statistics (mean, median, quantiles) and concentration measures inspired by "
            "modern analytics practices from pandas- and matplotlib-focused workflows."
        )


def _render_evolution_section(prepared: pd.DataFrame) -> None:
    st.subheader("Indexes Evolution")
    try:
        timeline_fig = create_timeline_figure(prepared)
    except (ImportError, ValueError) as exc:
        st.warning(str(exc))
        return

    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})


def _render_fleet_table(prepared: pd.DataFrame) -> None:
    st.subheader("Fleet Health Snapshot")
    fleet_df = build_fleet_summary(prepared)

    if fleet_df.empty:
        st.caption("No fleet-level data available.")
        return

    st.dataframe(
        fleet_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Index": st.column_config.NumberColumn("Index", help="On-time index (0-100).", format="%.1f"),
            "Δ Index": st.column_config.NumberColumn("Δ Index", help="Change vs prior run (placeholder)", format="%.1f"),
            "Tasks": st.column_config.NumberColumn("Tasks", format="%d"),
            "Overdue": st.column_config.NumberColumn("Overdue", format="%d"),
            "On-Time %": st.column_config.ProgressColumn(
                "On-Time %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                help="Percentage of tasks that are not overdue.",
            ),
            "High Severity %": st.column_config.ProgressColumn(
                "High Severity %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                help="Share of tasks tagged High/Critical.",
            ),
        },
    )


def _render_fleet_deferral_heatmap(prepared: pd.DataFrame) -> None:
    st.subheader("Fleet vs Deferral Class Heatmap")

    required = {"fleet", "deferral_class"}
    if prepared.empty or not required.issubset(prepared.columns):
        st.caption("Fleet/deferral class data not available in this snapshot.")
        return

    base = prepared[list(required)].copy()
    base["fleet"] = (
        base["fleet"].astype("string").str.strip().replace({"": "UNKNOWN", "<NA>": "UNKNOWN"}).fillna("UNKNOWN")
    )
    base["deferral_class"] = (
        base["deferral_class"].astype("string").str.strip().replace({"": "UNKNOWN", "<NA>": "UNKNOWN"}).fillna("UNKNOWN")
    )

    if base.empty:
        st.caption("No fleet/deferral combinations found.")
        return

    counts = (
        base.groupby(["fleet", "deferral_class"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    if counts.empty:
        st.caption("Insufficient combinations to render the heatmap.")
        return

    fleet_order = counts.groupby("fleet")["count"].sum().sort_values(ascending=False).index.tolist()
    deferral_order = counts.groupby("deferral_class")["count"].sum().sort_values(ascending=False).index.tolist()

    pivot = (
        counts.pivot(index="fleet", columns="deferral_class", values="count")
        .reindex(index=fleet_order, columns=deferral_order)
        .fillna(0)
    )

    total = counts["count"].sum()
    top_pairs = counts.sort_values("count", ascending=False).head(3)
    if total and not top_pairs.empty:
        top_share = top_pairs["count"].sum() / total * 100.0
        highlights = ", ".join(
            f"{row['fleet']}×{row['deferral_class']} ({int(row['count']):,})"
            for _, row in top_pairs.iterrows()
        )
        st.caption(f"Top three fleet/deferral combos cover {top_share:.1f}% of backlog: {highlights}.")

    matrix = pivot.to_numpy()
    text_matrix = [[f"{val:.0f}" for val in row] for row in matrix]
    max_value = matrix.max() if matrix.size else 0
    threshold = max_value * 0.55 if max_value else 0

    fig = px.imshow(
        matrix,
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="YlGnBu",
        aspect="auto",
        origin="upper",
    )
    fig.update_layout(
        title="Deferral pressure across fleets",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1c1f24", size=12),
        margin=dict(t=70, b=60, l=80, r=60),
        coloraxis_colorbar=dict(title="Task Count"),
    )
    fig.update_xaxes(title="Deferral Class", side="top", tickangle=-40)
    fig.update_yaxes(title="Fleet")
    fig.update_traces(
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(color="#1c1f24", size=11),
        hovertemplate="Fleet: %{y}<br>Deferral: %{x}<br>Tasks: %{z:.0f}<extra></extra>",
    )

    if matrix.size and threshold:
        for y_index, fleet in enumerate(pivot.index):
            for x_index, deferral in enumerate(pivot.columns):
                value = matrix[y_index][x_index]
                if value >= threshold:
                    fig.add_annotation(
                        x=deferral,
                        y=fleet,
                        text=f"{int(value)}",
                        showarrow=False,
                        font=dict(color="#f8f9fa", size=11, family="Helvetica-Bold"),
                    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    export_frame = pivot.reset_index().rename(columns={"index": "Fleet"})
    excel_bytes = dataframe_to_excel_bytes(export_frame, sheet_name="Fleet vs Deferral")
    st.download_button(
        "Download heatmap data (.xlsx)",
        data=excel_bytes,
        file_name="fleet_deferral_heatmap.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="fleet_deferral_heatmap_excel",
    )

    if SimpleDocTemplate is not None:
        pdf_bytes = _table_to_pdf_bytes(
            title="Fleet vs Deferral Class Heatmap",
            subtitle="Cross-tabulation of task counts helping prioritise deferral relief by fleet.",
            table=export_frame,
        )
        st.download_button(
            "Download heatmap table (.pdf)",
            data=pdf_bytes,
            file_name="fleet_deferral_heatmap.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="fleet_deferral_heatmap_pdf",
        )

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError:
        chart_pdf = None
    else:
        sns.set_theme(style="white", context="talk")
        fig, ax = plt.subplots(
            figsize=(max(6.0, pivot.shape[1] * 0.55), max(4.0, pivot.shape[0] * 0.6))
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            cbar_kws={"label": "Task Count"},
            linewidths=0.6,
            linecolor="#e9ecef",
            ax=ax,
        )
        ax.set_title("Deferral pressure across fleets", fontsize=16, fontweight="bold")
        ax.set_xlabel("Deferral Class", fontsize=12)
        ax.set_ylabel("Fleet", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        chart_pdf = _figure_to_pdf_bytes(fig)
        plt.close(fig)

    if chart_pdf:
        st.download_button(
            "Download heatmap chart (.pdf)",
            data=chart_pdf,
            file_name="fleet_deferral_heatmap_chart.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="fleet_deferral_heatmap_chart_pdf",
        )
    elif chart_pdf is None:
        st.caption("Install matplotlib and seaborn to export the heatmap graphic as PDF.")


def _render_additional_visuals(prepared: pd.DataFrame) -> None:
    try:
        bundles = create_visualizations(prepared)
    except ImportError as exc:
        st.warning(str(exc))
        return

    if not bundles:
        st.caption("No additional visualizations available for the current dataset.")
        return

    st.subheader("Visual Storytelling")

    combined_bytes = _build_combined_visual_workbook(bundles)
    if combined_bytes:
        st.download_button(
            "Download all chart data (.xlsx)",
            data=combined_bytes,
            file_name="analytics_charts_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="visuals_all_data_download",
        )

    viz_tabs = st.tabs([_prettify_key(name) for name in bundles.keys()])
    for tab, (name, bundle) in zip(viz_tabs, bundles.items()):
        with tab:
            pdf_bytes = _figure_to_pdf_bytes(bundle.figure)
            st.pyplot(bundle.figure, clear_figure=True)

            total_tasks, overdue_tasks = _chart_counts_from_bundle(bundle, prepared)
            st.caption(f"Tasks represented: {total_tasks:,} • Overdue tasks: {overdue_tasks:,}")

            st.download_button(
                "Download chart (.pdf)",
                data=pdf_bytes,
                file_name=f"{name}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"{name}_pdf_download",
            )

            if bundle.datasets:
                excel_bytes = _datasets_to_excel_bytes(bundle.datasets, prefix=_prettify_key(name))
                st.download_button(
                    "Download data (.xlsx)",
                    data=excel_bytes,
                    file_name=f"{name}_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"{name}_excel_download",
                )

                with st.expander("Preview data tables", expanded=False):
                    for dataset_label, dataset_df in bundle.datasets.items():
                        st.markdown(f"**{_prettify_key(dataset_label)}**")
                        preview_rows = dataset_df.head(200)
                        st.dataframe(preview_rows, use_container_width=True)
                        if len(dataset_df) > len(preview_rows):
                            st.caption(f"Showing first {len(preview_rows)} of {len(dataset_df)} rows.")
            else:
                st.caption("No source data available for this visualization.")


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


def _render_key_drivers(summary: TaskAnalyticsSummary) -> None:
    st.subheader("Key Drivers")
    try:
        figure = create_wordcloud_figure(summary)
    except ImportError as exc:
        st.warning(str(exc))
        return
    except ValueError as exc:
        st.caption(str(exc))
        return

    st.pyplot(figure, clear_figure=True)


def _render_download_section(prepared: pd.DataFrame, summary: TaskAnalyticsSummary) -> None:
    st.subheader("Export")

    with st.expander("Download enriched analytics"):
        csv_buffer = io.StringIO()
        prepared.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="open_tasks_enriched.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_csv",
        )

        try:
            workbook_bytes = generate_excel_report(prepared, summary)
        except ImportError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Excel export failed: {exc}")
        else:
            st.download_button(
                "Download Excel Workbook",
                data=workbook_bytes,
                file_name="open_tasks_analytics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_excel",
            )

        try:
            pdf_bytes = generate_pdf_report(prepared, summary)
        except ImportError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF export failed: {exc}")
        else:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="open_tasks_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf",
            )


def _figure_to_pdf_bytes(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf", bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def _table_to_pdf_bytes(*, title: str, subtitle: str | None, table: pd.DataFrame) -> bytes:
    if (
        SimpleDocTemplate is None
        or getSampleStyleSheet is None
        or Table is None
        or TableStyle is None
        or colors is None
        or letter is None
    ):
        raise ImportError("reportlab is required for PDF export.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=48, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"])]
    if subtitle:
        story.append(Spacer(1, 6))
        story.append(Paragraph(subtitle, styles["Normal"]))
        story.append(Spacer(1, 12))

    table_obj = _dataframe_to_reportlab_table(table)
    story.append(table_obj)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _dataframe_to_reportlab_table(frame: pd.DataFrame) -> "Table":
    prepared = frame.copy()
    for column in prepared.columns:
        series = prepared[column]
        if series.dtype.kind in {"f", "i"}:
            header = column.lower()
            if "share" in header or "%" in header:
                prepared[column] = series.map(lambda val: f"{float(val):.1f}%" if pd.notna(val) else "—")
            elif any(keyword in header for keyword in ("count", "tasks")):
                prepared[column] = series.map(lambda val: f"{int(val):,}" if pd.notna(val) else "—")
            else:
                prepared[column] = series.map(lambda val: f"{float(val):.1f}" if pd.notna(val) else "—")
        else:
            prepared[column] = series.fillna("—").astype(str)

    data = [prepared.columns.tolist()] + prepared.values.tolist()
    table_obj = Table(data, hAlign="LEFT")
    header_colour = DEFAULT_PALETTE[1] if len(DEFAULT_PALETTE) > 1 else "#1c7293"
    table_obj.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_colour)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f1f3f5")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dee2e6")),
            ]
        )
    )
    return table_obj


def _datasets_to_excel_bytes(
    datasets: "OrderedDict[str, pd.DataFrame]",
    *,
    prefix: str | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:  # type: ignore[arg-type]
        existing: set[str] = set()
        for idx, (label, dataset) in enumerate(datasets.items(), start=1):
            sheet_base = f"{prefix} - {label}" if prefix else label
            sheet_name = _safe_sheet_name(sheet_base, existing, fallback_index=idx)
            frame = dataset if isinstance(dataset, pd.DataFrame) else pd.DataFrame(dataset)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def _safe_sheet_name(candidate: str, existing: set[str], *, fallback_index: int) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {" ", "_", "-"} else "_" for ch in candidate).strip()
    if not sanitized:
        sanitized = f"Sheet{fallback_index}"

    sheet_name = sanitized[:31]
    suffix = 1
    while sheet_name.lower() in existing:
        base = sanitized[: max(0, 31 - len(str(suffix)) - 1)]
        sheet_name = f"{base}_{suffix}"[:31] if base else f"SHEET_{suffix}"[:31]
        suffix += 1

    existing.add(sheet_name.lower())
    return sheet_name


def _build_combined_visual_workbook(bundles: "OrderedDict[str, VisualizationBundle]") -> bytes | None:
    combined: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for name, bundle in bundles.items():
        prefix = _prettify_key(name)
        for dataset_label, dataset_df in bundle.datasets.items():
            sheet_label = f"{prefix} - {_prettify_key(dataset_label)}"
            combined[sheet_label] = dataset_df

    if not combined:
        return None

    return _datasets_to_excel_bytes(combined)


def _chart_counts_from_bundle(bundle: VisualizationBundle, prepared: pd.DataFrame) -> tuple[int, int]:
    total_tasks = len(prepared)
    overdue_tasks = int(prepared.get("is_overdue", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())

    count_columns = ("Task Count", "Count", "Fault Count", "Tasks", "Total Tasks")
    for dataset in bundle.datasets.values():
        if not isinstance(dataset, pd.DataFrame) or dataset.empty:
            continue

        chosen = next((col for col in count_columns if col in dataset.columns), None)
        if chosen:
            total_tasks = int(dataset[chosen].sum())
            if "Overdue Status" in dataset.columns:
                mask = dataset["Overdue Status"].astype("string").str.lower() == "overdue"
                overdue_tasks = int(dataset.loc[mask, chosen].sum())
            elif "Status" in dataset.columns:
                mask = dataset["Status"].astype("string").str.lower().str.contains("overdue")
                overdue_tasks = int(dataset.loc[mask, chosen].sum())
            elif "Overdue Tasks" in dataset.columns:
                overdue_tasks = int(dataset["Overdue Tasks"].sum())
            else:
                overdue_tasks = min(overdue_tasks, total_tasks)
            break

    return max(total_tasks, 0), max(overdue_tasks, 0)


def _breakdown_to_pdf_bytes(
    *,
    label: str,
    table: pd.DataFrame,
    insight: str | None,
    share_table: pd.DataFrame | None,
    mix: dict[str, float] | None,
    selected_bucket: str | None,
) -> bytes:
    if (
        SimpleDocTemplate is None
        or getSampleStyleSheet is None
        or Table is None
        or TableStyle is None
        or colors is None
        or letter is None
    ):
        raise ImportError("reportlab is required for PDF export.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=48, bottomMargin=36)
    styles = getSampleStyleSheet()

    story = [Paragraph(f"{label} Breakdown", styles["Title"]), Spacer(1, 8)]
    if insight:
        story.append(Paragraph(insight, styles["Normal"]))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Top groups", styles["Heading3"]))
    story.append(_dataframe_to_reportlab_table(table.head(10)))
    story.append(Spacer(1, 12))

    if share_table is not None and not share_table.empty:
        story.append(Paragraph(f"{label} share snapshot", styles["Heading3"]))
        share_snapshot = share_table.copy().head(8)
        story.append(_dataframe_to_reportlab_table(share_snapshot))
        story.append(Spacer(1, 12))

    if mix:
        story.append(Paragraph(f"Deferral mix for {selected_bucket or 'selection'}", styles["Heading3"]))
        mix_df = (
            pd.DataFrame(sorted(mix.items(), key=lambda item: item[1], reverse=True), columns=["Deferral", "Share"])
            .assign(Share=lambda frame: frame["Share"].astype(float))
        )
        story.append(_dataframe_to_reportlab_table(mix_df.head(10)))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _fleet_summary_to_pdf(fleet: str, row: pd.Series) -> bytes:
    if canvas is None or letter is None:  # pragma: no cover - optional dependency
        raise ImportError("reportlab is required for PDF export.")

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setTitle(f"Fleet Summary - {fleet}")
    margin = 50
    y = height - margin

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, y, f"Fleet Summary: {fleet}")
    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(margin, y, f"Generated from Maintenix analytics dashboard.")
    y -= 25

    metrics = [
        ("Total Tasks", f"{int(row['Total Tasks']):,}"),
        ("Overdue Tasks", f"{int(row['Overdue Tasks']):,}"),
        ("Overdue Rate %", f"{row['Overdue Rate %']:.1f}%"),
        ("Overdue Share %", f"{row['Overdue Share %']:.1f}%"),
        ("Task Share %", f"{row['Task Share %']:.1f}%"),
    ]

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(margin, y, "Key Metrics")
    y -= 20

    pdf.setFont("Helvetica", 12)
    for label, value in metrics:
        pdf.drawString(margin, y, f"{label}: {value}")
        y -= 18

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _safe_filename_fragment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "fleet"


def _prettify_key(name: str) -> str:
    return name.replace("_", " ").title()
