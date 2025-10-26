"""Excel and PDF export helpers for overdue task analytics."""

from __future__ import annotations

import io
from typing import Dict

import pandas as pd

from .breakdowns import build_overdue_breakdowns, prepare_overdue_detail_frame
from .modeling import identify_outliers
from .summaries import TaskAnalyticsSummary, build_fleet_overdue_share, build_fleet_summary
from .timeseries import build_time_series_bundle

try:  # Optional dependency for PDF output
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False

def generate_pdf_report(df: pd.DataFrame, summary: TaskAnalyticsSummary) -> bytes:
    """Produce a professional PDF report using ReportLab."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is required for PDF export. Install it with `pip install reportlab`." )

    buffer = io.BytesIO()
    try:
        bundle = build_time_series_bundle(df)
    except ImportError as exc:
        bundle = None
        bundle_error = str(exc)
    else:
        bundle_error = None
    fleet_summary = build_fleet_summary(df)
    fleet_overdue_share = build_fleet_overdue_share(df)
    breakdowns = build_overdue_breakdowns(df)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = [Paragraph("Non-Routine Overdue Tasks Analytics", styles["Title"])]
    story.append(Paragraph("Reporting snapshot", styles["Heading2"]))

    summary_rows = [
        ["Metric", "Value"],
        ["Total Tasks", f"{summary.total_tasks:,}"],
        ["Overdue Tasks", f"{summary.overdue_tasks:,}"],
        ["Completion Rate", f"{summary.completion_rate * 100:.1f}%"],
        ["Mean Days Until Due", f"{summary.mean_days_until_due:.1f}"],
        ["Mean Days Overdue", f"{summary.mean_days_overdue:.1f}"],
    ]
    if summary.report_date:
        summary_rows.insert(1, ["Reporting Date", summary.report_date.strftime("%d %b %Y")])

    story.append(_build_table(summary_rows))
    story.append(Spacer(1, 12))

    if not fleet_summary.empty:
        story.append(Paragraph("Fleet Health Snapshot", styles["Heading2"]))
        fleet_head = [fleet_summary.columns.tolist()] + fleet_summary.round(1).astype(str).values.tolist()
        story.append(_build_table(fleet_head))
        story.append(Spacer(1, 12))

    if not fleet_overdue_share.empty:
        story.append(Paragraph("Fleet Overdue Share", styles["Heading2"]))
        share_head = [fleet_overdue_share.columns.tolist()] + fleet_overdue_share.astype(str).values.tolist()
        story.append(_build_table(share_head))
        story.append(Spacer(1, 12))

    if bundle_error:
        story.append(Paragraph(f"Time-Series Analytics: {bundle_error}", styles["Heading3"]))
        story.append(Spacer(1, 12))
    elif bundle is not None:
        for label, table in bundle.to_excel_tables().items():
            if table.empty or label == "Correlations":
                continue
            headline = Paragraph(f"{label} Overview", styles["Heading2"])
            story.append(headline)
            trimmed = table.tail(8).round(2)
            story.append(_build_table([trimmed.columns.tolist()] + trimmed.astype(str).values.tolist()))
            story.append(Spacer(1, 12))

        if not bundle.correlations.empty:
            story.append(Paragraph("Key Metric Correlations", styles["Heading2"]))
            corr_df = bundle.correlations.round(3)
            story.append(_build_table([corr_df.columns.tolist()] + corr_df.astype(str).values.tolist()))
            story.append(Spacer(1, 12))

    if breakdowns:
        story.append(Paragraph("Overdue Breakdown Highlights", styles["Heading2"]))
        for idx, (section, table) in enumerate(breakdowns.items(), start=1):
            if table.empty:
                continue
            story.append(Paragraph(f"{idx}. {section}", styles["Heading3"]))
            trimmed = table.head(5).round(2)
            story.append(_build_table([trimmed.columns.tolist()] + trimmed.astype(str).values.tolist()))
            story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_excel_report(df: pd.DataFrame, summary: TaskAnalyticsSummary) -> bytes:
    """Create a multi-sheet Excel workbook with raw data and analytic views."""
    buffer = io.BytesIO()
    try:
        outliers = identify_outliers(df)
    except ImportError:
        outliers = pd.DataFrame()

    fleet_summary = build_fleet_summary(df)
    fleet_overdue_share = build_fleet_overdue_share(df)
    breakdowns = build_overdue_breakdowns(df)
    matrices = _build_analysis_tables(df)
    timeline_table = _build_timeline_table(df)
    try:
        ts_bundle = build_time_series_bundle(df)
    except ImportError:
        ts_bundle = None

    if "is_overdue" in df:
        overdue_mask = df["is_overdue"].fillna(False).astype(bool)
        overdue_only = df.loc[overdue_mask].copy()
    else:
        overdue_only = pd.DataFrame()

    overdue_detail = prepare_overdue_detail_frame(overdue_only) if not overdue_only.empty else pd.DataFrame()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:  # type: ignore[arg-type]
        df.to_excel(writer, sheet_name="Tasks", index=False)
        _summary_to_dataframe(summary).to_excel(writer, sheet_name="Summary", index=False)

        if not overdue_detail.empty:
            overdue_detail.to_excel(writer, sheet_name="Overdue Tasks", index=False)

        if not fleet_summary.empty:
            fleet_summary.to_excel(writer, sheet_name="Fleet Snapshot", index=False)

        if not fleet_overdue_share.empty:
            fleet_overdue_share.to_excel(writer, sheet_name="Fleet Overdue Share", index=False)

        for name, table in matrices.items():
            if not table.empty:
                table.to_excel(writer, sheet_name=name[:31], index=True)

        if not timeline_table.empty:
            timeline_table.to_excel(writer, sheet_name="Timeline", index=False)

        if not outliers.empty:
            outliers.to_excel(writer, sheet_name="Outliers", index=False)

        for section, table in breakdowns.items():
            if not table.empty:
                sheet = f"Overdue - {section}"
                table.to_excel(writer, sheet_name=sheet[:31], index=False)

        if ts_bundle is not None:
            for label, table in ts_bundle.to_excel_tables().items():
                if table.empty:
                    continue
                sheet = f"TS - {label}"
                table.to_excel(writer, sheet_name=sheet[:31], index=False)

    buffer.seek(0)
    return buffer.getvalue()


def dataframe_to_excel_bytes(df: pd.DataFrame, *, sheet_name: str = "Breakdown") -> bytes:
    """Serialize a single dataframe to an Excel worksheet for download controls."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:  # type: ignore[arg-type]
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buffer.seek(0)
    return buffer.getvalue()


def _summary_to_dataframe(summary: TaskAnalyticsSummary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Total Tasks", summary.total_tasks),
            ("Overdue Tasks", summary.overdue_tasks),
            ("Completion Rate (%)", round(summary.completion_rate * 100, 2)),
            ("Mean Days Until Due", round(summary.mean_days_until_due, 2)),
            ("Mean Days Overdue", round(summary.mean_days_overdue, 2)),
        ],
        columns=["Metric", "Value"],
    )


def _build_analysis_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    if {"fleet", "severity"}.issubset(df.columns):
        pivot = (
            df.groupby(["fleet", "severity"])
            .size()
            .unstack(fill_value=0)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        tables["Fleet vs Severity"] = pivot

    if {"status", "severity"}.issubset(df.columns):
        pivot = (
            df.groupby(["status", "severity"])
            .size()
            .unstack(fill_value=0)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        tables["Status vs Severity"] = pivot

    if {"fleet", "status"}.issubset(df.columns):
        pivot = (
            df.groupby(["fleet", "status"])
            .size()
            .unstack(fill_value=0)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        tables["Fleet vs Status"] = pivot

    return tables


def _build_timeline_table(df: pd.DataFrame) -> pd.DataFrame:
    if "due_date" not in df or df["due_date"].dropna().empty:
        return pd.DataFrame()
    timeline = (
        df.dropna(subset=["due_date"])
        .assign(due_day=lambda frame: frame["due_date"].dt.to_period("D").dt.to_timestamp())
        .groupby("due_day")
        .size()
        .rename("Task Count")
        .reset_index()
    )
    return timeline


def _build_table(rows: list[list[str]]) -> Table:
    table = Table(rows)
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c7ed6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]
    )
    table.setStyle(style)
    return table


__all__ = [
    "dataframe_to_excel_bytes",
    "generate_excel_report",
    "generate_pdf_report",
]
