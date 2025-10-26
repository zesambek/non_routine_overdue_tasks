from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import pytest

from overduetasks.analytics import (
    build_fleet_overdue_share,
    build_fleet_summary,
    build_overdue_breakdown_details,
    build_overdue_breakdowns,
    build_time_series_bundle,
    create_visualizations,
    VisualizationBundle,
    dataframe_to_excel_bytes,
    generate_excel_report,
    prepare_task_dataframe,
    summarize_tasks,
)


def _sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Aircraft Registration": "ET-AAA",
                "Fleet Type": "B787",
                "Assigned Engineer ID": "ENG1",
                "Run Date": "2025-10-22",
                "Fault Name": "Hydraulic leak",
                "Fault ID": "F001",
                "Config Position": "29-30-00",
                "Due": "2025-10-30 23:59",
                "Found on Date": "2025-09-01",
                "Severity": "HIGH",
                "Status": "Open",
                "Deferral Class": "MEL C",
            },
            {
                "Aircraft Registration": "ET-BBB",
                "Fleet Type": "B787",
                "Assigned Engineer ID": "ENG2",
                "Run Date": "2025-10-22",
                "Fault Name": "Seat issue",
                "Fault ID": "F002",
                "Config Position": "25-10-00",
                "Due": "2025-09-15 23:59",
                "Found on Date": "2025-03-20",
                "Severity": "LOW",
                "Status": "Deferred",
                "Deferral Class": "NEFMEL C",
            },
        ]
    )


def test_prepare_task_dataframe_enriches_fields():
    prepared = prepare_task_dataframe(_sample_raw_frame(), tz="UTC")
    assert "run_date" in prepared
    assert prepared.loc[0, "is_overdue"] is False
    assert prepared.loc[1, "is_overdue"] is True
    assert prepared.loc[0, "config_prefix"] == "29"
    assert prepared.loc[0, "found_quarter_label"] == "Q3"
    assert prepared.loc[0, "severity_display"] == "HIGH"


def test_summaries_and_fleet_snapshot():
    prepared = prepare_task_dataframe(_sample_raw_frame())
    summary = summarize_tasks(prepared)
    assert summary.total_tasks == 2
    assert summary.overdue_tasks == 1
    assert summary.report_date.date() == datetime(2025, 10, 22).date()

    fleet = build_fleet_summary(prepared)
    assert not fleet.empty
    assert "Avg Days Active" not in fleet.columns  # remains focused subset


def test_fleet_overdue_share_metrics():
    prepared = prepare_task_dataframe(_sample_raw_frame())
    share = build_fleet_overdue_share(prepared)
    assert {"Fleet", "Total Tasks", "Overdue Share %"}.issubset(share.columns)
    b787 = share.loc[share["Fleet"] == "B787"].iloc[0]
    assert b787["Total Tasks"] == 2
    assert pytest.approx(50.0, abs=0.1) == b787["Overdue Rate %"]


def test_breakdowns_and_details_provide_expected_columns():
    prepared = prepare_task_dataframe(_sample_raw_frame())
    breakdowns = build_overdue_breakdowns(prepared)
    assert "Fleet Type" in breakdowns
    fleet_table = breakdowns["Fleet Type"]
    assert "Avg Days Active" in fleet_table.columns

    details = build_overdue_breakdown_details(prepared)
    fleet_details = details["Fleet Type"]["B787"]
    assert {"Fault Name", "Fault ID", "Config Prefix"}.issubset(fleet_details.columns)


def test_time_series_bundle_outputs_expected_shapes():
    prepared = prepare_task_dataframe(_sample_raw_frame())
    bundle = build_time_series_bundle(prepared)
    assert not bundle.weekly.empty
    assert "overdue_rate" in bundle.weekly.columns
    assert bundle.correlations.empty or "days_overdue" in bundle.correlations.columns
    assert bundle.note is None


def test_time_series_fallback_without_polars(monkeypatch):
    from overduetasks.analytics import timeseries

    prepared = prepare_task_dataframe(_sample_raw_frame())

    monkeypatch.setattr(timeseries, "pl", None)
    bundle = timeseries.build_time_series_bundle(prepared)
    assert bundle.note is not None
    assert bundle.weekly.empty or "overdue_rate" in bundle.weekly.columns


def test_generate_excel_report_contains_expected_sheets(monkeypatch):
    prepared = prepare_task_dataframe(_sample_raw_frame())

    # Avoid heavy SciPy dependency in test runtime
    monkeypatch.setattr("overduetasks.analytics.exports.identify_outliers", lambda df: pd.DataFrame())

    workbook = generate_excel_report(prepared, summarize_tasks(prepared))
    with pd.ExcelFile(io.BytesIO(workbook)) as xl:
        sheet_names = set(xl.sheet_names)
        assert {"Tasks", "Summary", "Overdue Tasks"}.issubset(sheet_names)
        assert "Fleet Overdue Share" in sheet_names

    # Also ensure utility helper works standalone
    snippet = dataframe_to_excel_bytes(prepared.head(1))
    with pd.ExcelFile(io.BytesIO(snippet)) as xl:
        assert xl.sheet_names == ["Breakdown"]


def test_create_visualizations_returns_bundles():
    prepared = prepare_task_dataframe(_sample_raw_frame())
    from overduetasks.analytics.visuals import plt, sns

    if plt is None or sns is None:
        pytest.skip("matplotlib/seaborn not available for visualization test")

    visuals = create_visualizations(prepared)
    assert "severity_overdue" in visuals
    assert "deferral_class_overdue" in visuals
    assert "fleet_overdue_share" in visuals

    severity_bundle = visuals["severity_overdue"]
    assert isinstance(severity_bundle, VisualizationBundle)

    severity_data = severity_bundle.datasets["Severity vs Overdue"]
    assert {"Severity", "Overdue Status", "Count"}.issubset(severity_data.columns)
    assert not severity_data.empty

    fleet_bundle = visuals["fleet_overdue_share"]
    fleet_data = fleet_bundle.datasets["Fleet Overdue Share"]
    assert {"Fleet", "Total Tasks", "Overdue Tasks"}.issubset(fleet_data.columns)
