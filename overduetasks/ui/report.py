"""Streamlit UI helpers for analytics-driven dashboards and exports."""

from __future__ import annotations

import io
from collections import OrderedDict

import pandas as pd
import plotly.express as px
import streamlit as st

from overduetasks.analytics import (
    TaskAnalyticsSummary,
    build_fleet_overdue_share,
    build_fleet_summary,
    build_overdue_breakdowns,
    build_overdue_breakdown_details,
    build_time_series_bundle,
    create_summary_gauge_figures,
    create_summary_indicators,
    create_timeline_figure,
    create_visualizations,
    VisualizationBundle,
    create_wordcloud_figure,
    dataframe_to_excel_bytes,
    generate_excel_report,
    generate_pdf_report,
    identify_outliers,
    prepare_task_dataframe,
    summarize_tasks,
    train_due_date_model,
)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:  # pragma: no cover - optional dependency
    canvas = None  # type: ignore[assignment]
    letter = None  # type: ignore[assignment]


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

    indicators = create_summary_indicators(prepared, summary)
    try:
        gauge_figs = create_summary_gauge_figures(indicators)
    except ImportError as exc:
        st.warning(str(exc))
        _render_metric_row(summary)
    else:
        cols = st.columns(len(gauge_figs))
        for col, (label, figure) in zip(cols, gauge_figs.items()):
            with col:
                st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        st.caption("Gauge cards show the share of on-time, completed, high severity, and responsive tasks.")

    _render_metric_row(summary)
    _render_fleet_overdue_share(prepared)


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
                "Avg Days Overdue": st.column_config.NumberColumn(
                    "Avg Days Overdue",
                    format="%.1f",
                    help="Mean overdue duration for tasks in this bucket.",
                ),
                "Median Days Overdue": st.column_config.NumberColumn("Median Days Overdue", format="%.1f"),
                "Max Days Overdue": st.column_config.NumberColumn("Max Days Overdue", format="%.1f"),
                "Avg Days Active": st.column_config.NumberColumn(
                    "Avg Days Active",
                    format="%.1f",
                    help="Average days elapsed since the fault was first found.",
                ),
            }

            trimmed = table
            if len(table) > 10:
                show_full = st.toggle(
                    "Show full list",
                    value=False,
                    key=f"{label}_show_all",
                    help="Toggle to reveal the complete table.",
                )
                if not show_full:
                    trimmed = table.head(10)
                    st.caption(f"Showing top {len(trimmed)} of {len(table)} groups by fault count.")

            display_columns = [label, "Fault Count", "Fault Share %", "Avg Days Overdue", "Median Days Overdue", "Max Days Overdue", "Avg Days Active"]
            display_columns = [col for col in display_columns if col in trimmed.columns]

            with st.container():
                st.dataframe(
                    trimmed[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )

            chart_data = trimmed[[label, "Fault Count"]].set_index(label)
            st.bar_chart(chart_data, use_container_width=True)

            excel_bytes = dataframe_to_excel_bytes(table, sheet_name=label)
            safe_name = str(label).lower().replace(" ", "_").replace("/", "_")
            st.download_button(
                f"Download {label} Breakdown (.xlsx)",
                data=excel_bytes,
                file_name=f"overdue_{safe_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"{label}_agg_download",
            )

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


def _render_metric_row(summary: TaskAnalyticsSummary) -> None:
    cols = st.columns(4)
    cols[0].metric("Total Tasks", f"{summary.total_tasks:,}")
    cols[1].metric("Overdue Tasks", f"{summary.overdue_tasks:,}")
    cols[2].metric("Completion Rate", f"{summary.completion_rate * 100:.1f}%")
    cols[3].metric("Avg Days Overdue", f"{summary.mean_days_overdue:.1f}")


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
