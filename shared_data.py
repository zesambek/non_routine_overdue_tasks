# overduetasks/shared_data.py
# =============================================================================
# Shared data layer for both apps (Scraper & Analysis)
# - Saves scraped DataFrames to disk (CSV) in data/shared/
# - Maintains a small in-memory cache for fast reuse during one Python session
# - Loads the "latest" CSV for the analysis app
#
# Requires: pandas
# =============================================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Storage directory (relative to project root)
SHARED_DIR = Path("data/shared")
SHARED_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache (only valid for the current Python process)
_MEMORY_CACHE: dict[str, pd.DataFrame] = {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _latest_csv_path() -> Path:
    return SHARED_DIR / "open_tasks_latest.csv"


def save_dataframe(df: pd.DataFrame, base_name: str = "open_tasks") -> Path:
    """
    Persist the DataFrame to disk (timestamped CSV) and update the 'latest' pointer.
    Also store it in the in-memory cache for immediate reuse by the analysis app.
    Returns the path to the timestamped CSV.
    """
    if df is None or df.empty:
        raise ValueError("save_dataframe: DataFrame is empty or None.")

    ts_path = SHARED_DIR / f"{base_name}_{_timestamp()}.csv"
    df.to_csv(ts_path, index=False)

    latest = _latest_csv_path()
    df.to_csv(latest, index=False)

    _MEMORY_CACHE["latest"] = df.copy()
    return ts_path


def has_memory_cache() -> bool:
    return "latest" in _MEMORY_CACHE and isinstance(_MEMORY_CACHE["latest"], pd.DataFrame)


def get_cached_dataframe() -> Optional[pd.DataFrame]:
    return _MEMORY_CACHE["latest"].copy() if has_memory_cache() else None


def load_latest_from_disk() -> Optional[pd.DataFrame]:
    p = _latest_csv_path()
    if not p.exists():
        return None
    df = pd.read_csv(p)
    _MEMORY_CACHE["latest"] = df.copy()
    return df


def load_dataframe(prefer_memory: bool = True) -> Optional[pd.DataFrame]:
    """
    Primary accessor for the analysis app:
      - If prefer_memory and cache present → return it
      - Otherwise, attempt to read the 'latest' CSV from disk
    """
    if prefer_memory and has_memory_cache():
        return get_cached_dataframe()
    return load_latest_from_disk()

