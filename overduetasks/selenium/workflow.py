"""Maintenix-specific navigation and extraction steps."""

from __future__ import annotations

import time
from io import StringIO
from typing import Iterable, Tuple

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from overduetasks.config import Credentials
from overduetasks.exceptions import LoginError, OpenTasksError


_OPEN_TASK_HEADER_OVERRIDES: dict[Tuple[str, str], str] = {
    ("Task Name", "Name"): "Task Name",
    ("Task Name", "ID"): "Task ID",
    ("Fault Name", "Name"): "Fault Name",
    ("Fault Name", "ID"): "Fault ID",
    ("Driving Task", "Name"): "Driving Task Name",
    ("Driving Task", "ID"): "Driving Task ID",
    ("Work Package", "Name"): "Work Package Name",
    ("Work Package", "ID"): "Work Package ID",
    ("Work Package No", "Work Package No"): "Work Package No",
    ("Config Position", "Config Position"): "Config Position",
    ("Must Be Removed", "Must Be Removed"): "Must Be Removed",
    ("Due", "Due"): "Due",
    ("Next Shop Visit", "Next Shop Visit"): "Next Shop Visit",
    ("Inventory", "Inventory"): "Inventory",
    ("Found on Date", "Found on Date"): "Found on Date",
    ("Found On Flight", "Found On Flight"): "Found On Flight",
    ("Severity", "Severity"): "Severity",
    ("Status", "Status"): "Status",
    ("Failure Type", "Failure Type"): "Failure Type",
    ("Fault Priority", "Fault Priority"): "Fault Priority",
    ("Deferral Class", "Deferral Class"): "Deferral Class",
    ("Deferral Reference", "Deferral Reference"): "Deferral Reference",
    ("Work Type(s)", "Work Type(s)"): "Work Type(s)",
    ("Operational Restrictions", "Operational Restrictions"): "Operational Restrictions",
    ("ETOPS Significant", "ETOPS Significant"): "ETOPS Significant",
    ("Material Availability", "Material Availability"): "Material Availability",
}


def wait_dom_ready(driver: webdriver.Remote, timeout: int = 30) -> None:
    """Wait until the document reaches the 'complete' readyState."""
    WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return document.readyState") == "complete")


def login(driver: webdriver.Remote, creds: Credentials) -> None:
    """Log into Maintenix and verify navigation readiness."""
    if not creds.username or not creds.password:
        raise LoginError("Username and password are required.")

    driver.get("http://etmxi.ethiopianairlines.com/maintenix/common/security/login.jsp")
    wait_dom_ready(driver)

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "j_username")))
    user_input = driver.find_element(By.ID, "j_username")
    user_input.clear()
    user_input.send_keys(creds.username)

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "j_password")))
    pass_input = driver.find_element(By.ID, "j_password")
    pass_input.clear()
    pass_input.send_keys(creds.password)

    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "idButtonLogin")))
    driver.find_element(By.ID, "idButtonLogin").click()
    wait_dom_ready(driver)

    deadline = time.time() + 35
    while time.time() < deadline:
        if _visible(driver, By.CSS_SELECTOR, ".error, .loginError, #loginError, .alert.alert-danger"):
            raise LoginError("Login error banner is visible (credentials/access).")
        if "/maintenix/" in (driver.current_url or ""):
            return
        if _visible(driver, By.ID, "idMainMenu"):
            return
        if _visible(driver, By.PARTIAL_LINK_TEXT, "Logout") or _visible(driver, By.PARTIAL_LINK_TEXT, "Log Out"):
            return
        time.sleep(0.4)
    raise LoginError("Post-login readiness not detected.")


def go_to_open_tasks(driver: webdriver.Remote, aircraft_reg: str) -> None:
    """Navigate to the Open Tasks view for the specified aircraft."""
    if not aircraft_reg:
        raise OpenTasksError("Aircraft registration is required.")

    driver.get("http://etmxi.ethiopianairlines.com/maintenix/common/ToDoList.jsp")
    wait_dom_ready(driver)

    search = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "idBarcodeSearchInput")))
    search.clear()
    search.send_keys(aircraft_reg)
    search.submit()

    WebDriverWait(driver, 50).until(
        EC.any_of(
            EC.text_to_be_present_in_element((By.ID, "idGrpInventoryDetails"), aircraft_reg),
            EC.text_to_be_present_in_element((By.ID, "idMxSubTitle"), aircraft_reg),
        )
    )

    open_tab = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "Open_link")))
    try:
        open_tab.click()
    except Exception:
        driver.execute_script("arguments[0].click();", open_tab)

    WebDriverWait(driver, 50).until(EC.text_to_be_present_in_element((By.ID, "OpenFaults_link"), "Open Faults"))
    open_faults = driver.find_element(By.ID, "OpenFaults_link")
    try:
        open_faults.click()
    except Exception:
        driver.execute_script("arguments[0].click();", open_faults)
        print("samson")

    WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.ID, "idButtonRaiseFault")))


def extract_open_tasks(driver: webdriver.Remote) -> pd.DataFrame:
    """Extract and normalize the Open Faults grid into a tidy DataFrame."""
    table = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "idTableOpenFaults")))
    html = (table.get_attribute("outerHTML") or "").strip()

    if not html:
        raise OpenTasksError("Open faults table did not return any HTML.")

    df = _read_open_tasks_table(html)
    if df.empty:
        return df

    return _normalize_open_tasks_frame(df)


def _visible(driver: webdriver.Remote, by: By, value: str) -> bool:
    """Return True if any located elements are displayed."""
    try:
        return any(el.is_displayed() for el in driver.find_elements(by, value))
    except Exception:
        return False


def _read_open_tasks_table(html: str) -> pd.DataFrame:
    """Parse the Open Faults table HTML using pandas with sensible fallbacks."""
    parsing_strategies = ([0, 1], 0)

    for header in parsing_strategies:
        try:
            frames = pd.read_html(
                StringIO(html),
                header=header,
                index_col=None,
                keep_default_na=True,
                thousands=",",
                decimal=".",
                displayed_only=True,
            )
        except ValueError:
            continue

        if frames:
            return frames[0]

    return pd.DataFrame()


def _normalize_open_tasks_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten column headers and drop unusable columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df = _flatten_multiindex_columns(df)
    else:
        df = _normalize_single_level_columns(df)

    return df.reset_index(drop=True)


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    keepers: list[tuple[Tuple[str, ...], str]] = []

    for column in df.columns:
        raw = tuple("" if part is None else str(part).strip() for part in column)
        override = _OPEN_TASK_HEADER_OVERRIDES.get(raw)

        if override:
            keepers.append((column, override))
            continue

        cleaned = [_clean_header_text(label) for label in raw]
        cleaned = [label for label in cleaned if label]

        if not cleaned:
            continue

        # Drop duplicated words within the same column name while preserving order.
        deduped_seen: set[str] = set()
        ordered: list[str] = []
        for label in cleaned:
            if label in deduped_seen:
                continue
            deduped_seen.add(label)
            ordered.append(label)
        deduped_name = " ".join(ordered)
        keepers.append((column, deduped_name))

    if not keepers:
        return df.iloc[:, 0:0]

    columns, names = zip(*keepers)
    normalized = df.loc[:, list(columns)]
    normalized.columns = _dedupe(names)
    return normalized


def _normalize_single_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    keepers: list[tuple[str, str]] = []

    for column in df.columns:
        label = _clean_header_text("" if column is None else str(column).strip())
        if not label:
            continue
        keepers.append((column, label))

    if not keepers:
        return df.iloc[:, 0:0]

    columns, names = zip(*keepers)
    normalized = df.loc[:, list(columns)]
    normalized.columns = _dedupe(names)
    return normalized


def _clean_header_text(value: str) -> str:
    text = value.strip()
    if not text or text.lower() in {"nan", "none"} or text.lower().startswith("unnamed"):
        return ""
    return text


def _dedupe(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    resolved: list[str] = []

    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        resolved.append(name if count == 1 else f"{name}.{count}")

    return resolved
