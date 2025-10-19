"""Configuration models and constants shared across the app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

REQUIRED_MAP_COLS = ["Aircraft Registration", "Fleet Type", "Assigned Engineer ID"]


@dataclass
class DriverConfig:
    # Common
    browser: str = "chrome"  # "chrome" | "firefox"
    headless: bool = True
    page_load_timeout_s: int = 60
    implicit_wait_s: int = 0
    script_timeout_s: int = 60
    window_w: int = 1440
    window_h: int = 900

    # Chrome advanced (exposed in UI)
    ch_page_load_strategy: str = "normal"  # normal | eager | none
    ch_timeouts_page: Optional[int] = None  # ms
    ch_timeouts_script: Optional[int] = None  # ms
    ch_timeouts_implicit: Optional[int] = None  # ms
    ch_unhandled_prompt_behavior: Optional[str] = None
    ch_strict_file_interactability: bool = False
    ch_accept_insecure_certs: bool = False
    ch_http_proxy: Optional[str] = None  # host:port
    ch_browser_version: Optional[str] = None  # e.g., "stable"
    ch_platform_name: Optional[str] = None  # e.g., "any"
    chrome_binary_override: Optional[str] = None

    # Firefox advanced
    ff_binary_override: Optional[str] = None


@dataclass
class Credentials:
    username: str
    password: str
