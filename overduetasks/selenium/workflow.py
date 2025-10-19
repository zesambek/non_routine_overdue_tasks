"""Maintenix-specific navigation and extraction steps."""

from __future__ import annotations

import time
from io import StringIO

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from overduetasks.config import Credentials
from overduetasks.exceptions import LoginError, OpenTasksError


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

    open_tab = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "OpenFaults_link")))
    try:
        open_tab.click()
    except Exception:
        driver.execute_script("arguments[0].click();", open_tab)

    WebDriverWait(driver, 50).until(EC.text_to_be_present_in_element((By.ID, "idTableOpenFaults"), "Open Faults"))
    open_faults = driver.find_element(By.ID, "idTableOpenFaults")
    try:
        open_faults.click()
    except Exception:
        driver.execute_script("arguments[0].click();", open_faults)

    WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.ID, "idTableOpenFaultsCol_0_")))


def extract_open_tasks(driver: webdriver.Remote) -> pd.DataFrame:
    """Extract the open tasks table into a DataFrame with flattened headers."""
    table = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "idTableOpenFaultsCol_0_")))
    html = table.get_attribute("outerHTML") or ""

    try:
        dfs = pd.read_html(
            StringIO(html),
            match=".+",
            header=[0, 1],
            index_col=None,
            skiprows=None,
            attrs=None,
            parse_dates=False,
            thousands=",",
            encoding=None,
            decimal=".",
            converters=None,
            na_values=None,
            keep_default_na=True,
            displayed_only=True,
            extract_links=None,
        )
        if dfs and not dfs[0].empty:
            df = dfs[0]

            if hasattr(df.columns, "levels") and len(df.columns.levels) == 2:
                explicit_map = {
                    ("Task Name", "Name"): "Task Name",
                    ("Task Name", "ID"): "Task ID",
                    ("Config Position", "Config Position"): "Config Position",
                    ("Must Be Removed", "Must Be Removed"): "Must Be Removed",
                    ("Due", "Due"): "Due",
                    ("Next Shop Visit", "Next Shop Visit"): "Next Shop Visit",
                    ("Inventory", "Inventory"): "Inventory",
                    ("Task Status", "Task Status"): "Task Status",
                    ("Task Type", "Task Type"): "Task Type",
                    ("Work Type(s)", "Work Type(s)"): "Work Type(s)",
                    ("Originator", "Originator"): "Originator",
                    ("Task Priority", "Task Priority"): "Task Priority",
                    ("Schedule Priority", "Schedule Priority"): "Schedule Priority",
                    ("ETOPS Significant", "ETOPS Significant"): "ETOPS Significant",
                    ("Work Package", "Name"): "Work Package Name",
                    ("Work Package", "ID"): "Work Package ID",
                    ("Work Package No", "Work Package No"): "Work Package No",
                }

                new_cols: list[str] = []
                keep_mask: list[bool] = []

                for top, sub in df.columns:
                    a = str(top).strip()
                    b = str(sub).strip()

                    if a.startswith("Unnamed: 0_level_0") and b.startswith("Unnamed: 0_level_1"):
                        keep_mask.append(False)
                        new_cols.append("")
                        continue

                    key = (a, b)
                    if key in explicit_map:
                        keep_mask.append(True)
                        new_cols.append(explicit_map[key])
                    else:
                        if a == b:
                            keep_mask.append(True)
                            new_cols.append(a)
                        elif b in ("", "nan", "None"):
                            keep_mask.append(True)
                            new_cols.append(a)
                        else:
                            keep_mask.append(True)
                            new_cols.append(f"{a} {b}")

                df = df.loc[:, keep_mask]
                df.columns = new_cols

                seen = {}
                uniq = []
                for name in df.columns:
                    if name not in seen:
                        seen[name] = 1
                        uniq.append(name)
                    else:
                        seen[name] += 1
                        uniq.append(f"{name}.{seen[name]}")
                df.columns = uniq

            return df
    except Exception:
        pass

    dfs = pd.read_html(
        StringIO(html),
        match=".+",
        header=0,
        index_col=None,
        skiprows=None,
        attrs=None,
        parse_dates=False,
        thousands=",",
        encoding=None,
        decimal=".",
        converters=None,
        na_values=None,
        keep_default_na=True,
        displayed_only=True,
        extract_links=None,
    )
    return dfs[0] if dfs else pd.DataFrame()


def _visible(driver: webdriver.Remote, by: By, value: str) -> bool:
    """Return True if any located elements are displayed."""
    try:
        return any(el.is_displayed() for el in driver.find_elements(by, value))
    except Exception:
        return False
