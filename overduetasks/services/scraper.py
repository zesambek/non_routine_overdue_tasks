"""Service object that orchestrates driver, navigation, and data enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import pandas as pd
from selenium.webdriver.remote.webdriver import WebDriver

from overduetasks.config import Credentials, DriverConfig
from overduetasks.data.mapping import enrich_with_mapping, normalize_mapping
from overduetasks.selenium.driver import cleanup_temp_profile, create_driver
from overduetasks.selenium.workflow import extract_open_tasks, go_to_open_tasks, login

IterationHook = Callable[[int, int, str, bool, Exception | None], None]


@dataclass
class ScrapeResult:
    dataframe: pd.DataFrame
    errors: List[tuple[str, Exception]]


class OpenTasksScraper:
    """Coordinate the end-to-end scraping flow with optional iteration callbacks."""

    def __init__(
        self,
        cfg: DriverConfig,
        credentials: Credentials,
        mapping_df: pd.DataFrame,
        *,
        driver_factory: Callable[..., WebDriver] = create_driver,
        firefox_retry_notifier: Callable[[Exception], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.credentials = credentials
        self.mapping_df = normalize_mapping(mapping_df)
        self._driver_factory = driver_factory
        self._firefox_retry_notifier = firefox_retry_notifier

    def scrape(
        self,
        registrations: Sequence[str],
        *,
        iteration_hook: IterationHook | None = None,
    ) -> ScrapeResult:
        total = len(registrations)
        frames: list[pd.DataFrame] = []
        errors: list[tuple[str, Exception]] = []

        driver = self._start_driver()
        try:
            login(driver, self.credentials)

            for idx, reg in enumerate(registrations, start=1):
                success = False
                error: Exception | None = None
                try:
                    go_to_open_tasks(driver, reg)
                    df = extract_open_tasks(driver)
                    df = enrich_with_mapping(df, reg, self.mapping_df)
                    frames.append(df)
                    success = True
                except Exception as exc:  # pragma: no cover - depends on live site
                    errors.append((reg, exc))
                    error = exc
                finally:
                    if iteration_hook:
                        iteration_hook(idx, total, reg, success, error)

            result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            return ScrapeResult(dataframe=result_df, errors=errors)
        finally:
            try:
                driver.quit()
            finally:
                cleanup_temp_profile(driver)

    def _start_driver(self) -> WebDriver:
        return self._driver_factory(self.cfg, firefox_retry_notifier=self._firefox_retry_notifier)
