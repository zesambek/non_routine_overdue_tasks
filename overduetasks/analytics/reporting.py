"""Backward-compatible façade for legacy imports.

The analytics package has been split across several focused modules. Import
directly from ``overduetasks.analytics`` where possible; this file simply
re-exports the public surface so older call sites continue to work.
"""

from __future__ import annotations

from .breakdowns import build_overdue_breakdown_details, build_overdue_breakdowns
from .exports import dataframe_to_excel_bytes, generate_excel_report, generate_pdf_report
from .modeling import identify_outliers, train_due_date_model
from .preparation import (
    DATE_COLUMNS,
    KNOWN_DATETIME_FORMATS,
    KNOWN_TIMEZONE_SUFFIXES,
    SEVERITY_HIGH_KEYWORDS,
    STATUS_RESPONSE_KEYWORDS,
    prepare_task_dataframe,
)
from .summaries import (
    TaskAnalyticsSummary,
    build_fleet_overdue_share,
    build_fleet_summary,
    summarize_tasks,
)
from .visuals import (
    VisualizationBundle,
    create_summary_gauge_figures,
    create_summary_indicators,
    create_timeline_figure,
    create_visualizations,
    create_wordcloud_figure,
)

__all__ = [
    "DATE_COLUMNS",
    "KNOWN_DATETIME_FORMATS",
    "KNOWN_TIMEZONE_SUFFIXES",
    "SEVERITY_HIGH_KEYWORDS",
    "STATUS_RESPONSE_KEYWORDS",
    "TaskAnalyticsSummary",
    "build_fleet_overdue_share",
    "build_fleet_summary",
    "build_overdue_breakdown_details",
    "build_overdue_breakdowns",
    "create_summary_gauge_figures",
    "create_summary_indicators",
    "create_timeline_figure",
    "create_visualizations",
    "create_wordcloud_figure",
    "VisualizationBundle",
    "dataframe_to_excel_bytes",
    "generate_excel_report",
    "generate_pdf_report",
    "identify_outliers",
    "prepare_task_dataframe",
    "summarize_tasks",
    "train_due_date_model",
]
