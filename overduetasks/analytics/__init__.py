"""Analytics helpers for preparing reports and visualizations."""

from .breakdowns import build_overdue_breakdown_details, build_overdue_breakdowns
from .exports import dataframe_to_excel_bytes, generate_excel_report, generate_pdf_report
from .modeling import identify_outliers, train_due_date_model
from .preparation import prepare_task_dataframe
from .summaries import (
    TaskAnalyticsSummary,
    build_fleet_overdue_share,
    build_fleet_summary,
    summarize_tasks,
)
from .timeseries import TimeSeriesBundle, build_time_series_bundle
from .visuals import (
    VisualizationBundle,
    create_summary_gauge_figures,
    create_summary_indicators,
    create_timeline_figure,
    create_visualizations,
    create_wordcloud_figure,
)

__all__ = [
    "TaskAnalyticsSummary",
    "build_fleet_overdue_share",
    "build_fleet_summary",
    "build_overdue_breakdown_details",
    "build_overdue_breakdowns",
    "build_time_series_bundle",
    "TimeSeriesBundle",
    "VisualizationBundle",
    "create_summary_gauge_figures",
    "create_summary_indicators",
    "create_timeline_figure",
    "create_visualizations",
    "create_wordcloud_figure",
    "dataframe_to_excel_bytes",
    "generate_excel_report",
    "generate_pdf_report",
    "identify_outliers",
    "prepare_task_dataframe",
    "summarize_tasks",
    "train_due_date_model",
]
