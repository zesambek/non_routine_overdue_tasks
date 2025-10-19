"""Mapping utilities shared between the Streamlit UI and scraper service."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from overduetasks.config import REQUIRED_MAP_COLS


def default_mapping() -> pd.DataFrame:
    """Provide a starter mapping to avoid empty UI tables."""
    return pd.DataFrame(
        {
            "Aircraft Registration": ["ET-ATH", "ET-AOU", "ET-ALM", "ET-AZA", "ET-ATQ", "ET-ARL", "ET-AYB", "ET-AVT", "ET-ASK"],
            "Fleet Type": ["B787", "B787", "B737NG", "B737MAX", "A350", "Q400", "A350", "B777F", "B777"],
            "Assigned Engineer ID": ["23423", "34181", "34180", "26988", "29504", "20144", "36461", "36465", "31345"],
        },
        dtype="string",
    )


def normalize_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist with string dtype and drop blank registrations."""
    df = df.rename(columns={c: c.strip() for c in df.columns}).copy()
    aliases = {
        "aircraft registration": "Aircraft Registration",
        "registration": "Aircraft Registration",
        "reg": "Aircraft Registration",
        "fleet": "Fleet Type",
        "assigned engineer id": "Assigned Engineer ID",
        "engineer id": "Assigned Engineer ID",
        "assigned engineer": "Assigned Engineer ID",
    }
    for col in list(df.columns):
        key = col.lower().strip()
        if key in aliases and aliases[key] not in df.columns:
            df = df.rename(columns={col: aliases[key]})

    for col in REQUIRED_MAP_COLS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="string")

    out = df[REQUIRED_MAP_COLS].astype("string")
    out["Aircraft Registration"] = out["Aircraft Registration"].astype("string").str.strip()
    out["Fleet Type"] = out["Fleet Type"].astype("string").str.strip()
    out["Assigned Engineer ID"] = out["Assigned Engineer ID"].astype("string").str.strip()

    out = out[out["Aircraft Registration"] != ""].drop_duplicates().reset_index(drop=True)
    return out


def enrich_with_mapping(df: pd.DataFrame, reg: str, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Attach mapping columns and run date to extracted task rows."""
    base = normalize_mapping(mapping_df)
    reg = str(reg)
    fleet: Optional[pd.Series] = base.loc[base["Aircraft Registration"] == reg, "Fleet Type"]
    eng: Optional[pd.Series] = base.loc[base["Aircraft Registration"] == reg, "Assigned Engineer ID"]

    out = df.copy()
    out["Aircraft Registration"] = reg
    out["Fleet Type"] = fleet.iloc[0] if isinstance(fleet, pd.Series) and not fleet.empty else ""
    out["Assigned Engineer ID"] = eng.iloc[0] if isinstance(eng, pd.Series) and not eng.empty else ""
    out["Run Date"] = pd.Timestamp.today().date().isoformat()

    preferred = [c for c in ["Aircraft Registration", "Fleet Type", "Assigned Engineer ID", "Run Date"] if c in out.columns]
    rest = [c for c in out.columns if c not in preferred]
    return out[preferred + rest] if preferred else out
