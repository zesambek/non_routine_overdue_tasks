# overduetasks/LoginPage.py
# ==================================================================================
# Maintenix → Open Tasks Scraper (Top-Down, Stepwise-Refined, No force-kill)
#
# GOAL
#   Log in to Maintenix, navigate to Open Tasks for selected aircraft, extract the
#   table to a Pandas DataFrame, enrich with engineer–aircraft mapping, and export.
#
# WHAT’S NEW IN THIS REVISION
#   • Robust two-row header parsing for the Open Tasks table:
#       - Prefer read_html(header=[0, 1]) with StringIO to avoid FutureWarnings.
#       - Drop spurious ("Unnamed: 0_level_0","Unnamed: 0_level_1") pair.
#       - Flatten to EXACT requested column names (no invented data/labels).
#       - Fall back to single-row header=0 if needed.
#   • All previous functionality preserved (drag & drop mapping, filters, etc.).
#
# RUN:
#   streamlit run LoginPage.py
#
# REQS:
#   pip install streamlit selenium webdriver-manager pandas lxml openpyxl
# ==================================================================================

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from shutil import which
from typing import List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

# ---- Selenium 4 ----
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Auto-install drivers
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

# Shared persistence (used by analysis app)
from shared_data import save_dataframe


# ========================= Errors =========================
class SetupError(RuntimeError): ...
class LoginError(RuntimeError): ...
class OpenTasksError(RuntimeError): ...


# ========================= Config Models =========================
@dataclass
class DriverConfig:
    # Common
    browser: str = "chrome"           # "chrome" | "firefox"
    headless: bool = True
    page_load_timeout_s: int = 60
    implicit_wait_s: int = 0
    script_timeout_s: int = 60
    window_w: int = 1440
    window_h: int = 900

    # Chrome advanced (exposed in UI)
    ch_page_load_strategy: str = "normal"       # normal | eager | none
    ch_timeouts_page: Optional[int] = None      # ms
    ch_timeouts_script: Optional[int] = None    # ms
    ch_timeouts_implicit: Optional[int] = None  # ms
    ch_unhandled_prompt_behavior: Optional[str] = None
    ch_strict_file_interactability: bool = False
    ch_accept_insecure_certs: bool = False
    ch_http_proxy: Optional[str] = None         # host:port
    ch_browser_version: Optional[str] = None    # e.g., "stable"
    ch_platform_name: Optional[str] = None      # e.g., "any"
    chrome_binary_override: Optional[str] = None

    # Firefox advanced
    ff_binary_override: Optional[str] = None


@dataclass
class Credentials:
    username: str
    password: str


# ========================= Page Setup =========================
st.set_page_config(page_title="Maintenix → Open Tasks (Scraper)", page_icon="🛠️", layout="wide")
st.title("🔐 Maintenix Login → Open Tasks (Scraper)")


# ========================= Defaults & Helpers =========================
REQUIRED_MAP_COLS = ["Aircraft Registration", "Fleet Type", "Assigned Engineer ID"]

def _default_mapping() -> pd.DataFrame:
    # Strings to avoid Streamlit data_editor dtype conflicts
    return pd.DataFrame(
        {
            "Aircraft Registration": ["ET-ATH","ET-AOU","ET-ALM","ET-AZA","ET-ATQ","ET-ARL","ET-AYB","ET-AVT","ET-ASK"],
            "Fleet Type":            ["B787","B787","B737NG","B737MAX","A350","Q400","A350","B777F","B777"],
            "Assigned Engineer ID":  ["23423","34181","34180","26988","29504","20144","36461","36465","31345"],
        },
        dtype="string",
    )

def _normalize_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure mapping has required columns as string dtype and no empty registrations."""
    # Normalise/alias headers
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
    for c in list(df.columns):
        key = c.lower().strip()
        if key in aliases and aliases[key] not in df.columns:
            df = df.rename(columns={c: aliases[key]})

    for col in REQUIRED_MAP_COLS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="string")

    out = df[REQUIRED_MAP_COLS].astype("string")
    out["Aircraft Registration"] = out["Aircraft Registration"].astype("string").str.strip()
    out["Fleet Type"]            = out["Fleet Type"].astype("string").str.strip()
    out["Assigned Engineer ID"]  = out["Assigned Engineer ID"].astype("string").str.strip()

    # Drop blank registrations
    out = out[out["Aircraft Registration"] != ""].drop_duplicates().reset_index(drop=True)
    return out


# ========================= UI (LEVEL 1) =========================
def collect_inputs_ui() -> tuple[DriverConfig, Credentials, pd.DataFrame, List[str]]:
    with st.sidebar:
        st.header("Driver")
        browser  = st.selectbox("Browser", ["chrome", "firefox"], index=0, key="drv_browser")
        headless = st.toggle("Headless", value=True, key="drv_headless")

        # Advanced Chrome options
        ch_page_load_strategy = "normal"
        ch_timeouts_page = None
        ch_timeouts_script = None
        ch_timeouts_implicit = None
        ch_unhandled = None
        ch_sfi = False
        ch_insec = False
        ch_proxy = None
        ch_bver = None
        ch_plat = None
        ch_bin = ""
        ff_bin = ""

        if browser == "chrome":
            st.markdown("### ⚙️ Advanced (Chrome)")
            ch_page_load_strategy = st.selectbox("page_load_strategy", ["normal", "eager", "none"], index=0, key="drv_pls")
            col1, col2, col3 = st.columns(3)
            with col1:
                ch_timeouts_page = st.number_input("timeouts.pageLoad (ms)", min_value=0, value=0, step=500,
                                                   help="0 = unset; driver.set_page_load_timeout still applies.",
                                                   key="drv_t_pageload")
                ch_timeouts_page = None if ch_timeouts_page == 0 else int(ch_timeouts_page)
            with col2:
                ch_timeouts_script = st.number_input("timeouts.script (ms)", min_value=0, value=0, step=500, key="drv_t_script")
                ch_timeouts_script = None if ch_timeouts_script == 0 else int(ch_timeouts_script)
            with col3:
                ch_timeouts_implicit = st.number_input("timeouts.implicit (ms)", min_value=0, value=0, step=500, key="drv_t_impl")
                ch_timeouts_implicit = None if ch_timeouts_implicit == 0 else int(ch_timeouts_implicit)

            ch_unhandled = st.selectbox(
                "unhandled_prompt_behavior",
                ["(default)", "accept", "dismiss", "dismiss and notify", "ignore"],
                index=0, key="drv_unhandled"
            )
            ch_unhandled = None if ch_unhandled == "(default)" else ch_unhandled

            r2 = st.columns(3)
            with r2[0]:
                ch_sfi = st.toggle("strict_file_interactability", value=False, key="drv_sfi")
            with r2[1]:
                ch_insec = st.toggle("accept_insecure_certs", value=False, key="drv_insec")
            with r2[2]:
                use_proxy = st.toggle("Use HTTP proxy", value=False, key="drv_proxy_toggle")
                if use_proxy:
                    ch_proxy = st.text_input("httpProxy (host:port)", value="http.proxy:1234", key="drv_proxy_text")

            r3 = st.columns(3)
            with r3[0]:
                ch_bver = st.text_input("browser_version (optional)", value="", key="drv_bver") or None
            with r3[1]:
                ch_plat = st.text_input("platform_name (optional)", value="", key="drv_plat") or None
            with r3[2]:
                ch_bin = st.text_input("Chrome/Chromium binary (optional)",
                                       value=_auto_find_chrome_binary() or "",
                                       help="Leave empty to auto-detect.",
                                       key="drv_chrome_bin")
        else:
            st.markdown("### ⚙️ Advanced (Firefox)")
            ff_bin = st.text_input("Firefox binary (optional)",
                                   value=_auto_find_firefox_binary() or "",
                                   help="Leave empty to auto-detect (e.g. /usr/bin/firefox or /snap/bin/firefox)",
                                   key="drv_ff_bin")

        cfg = DriverConfig(
            browser=browser, headless=headless,
            # chrome
            ch_page_load_strategy=ch_page_load_strategy,
            ch_timeouts_page=ch_timeouts_page,
            ch_timeouts_script=ch_timeouts_script,
            ch_timeouts_implicit=ch_timeouts_implicit,
            ch_unhandled_prompt_behavior=ch_unhandled,
            ch_strict_file_interactability=ch_sfi,
            ch_accept_insecure_certs=ch_insec,
            ch_http_proxy=ch_proxy,
            ch_browser_version=ch_bver,
            ch_platform_name=ch_plat,
            chrome_binary_override=ch_bin.strip() or None,
            # firefox
            ff_binary_override=ff_bin.strip() or None,
        )

    st.subheader("Credentials")
    c1, c2 = st.columns(2)
    with c1:
        username = st.text_input("Username", key="cred_user")
    with c2:
        password = st.text_input("Password", type="password", key="cred_pass")
    creds = Credentials(username=username, password=password)

    # ---- Engineer–Aircraft Assignment (Drag & Drop) ----
    st.subheader("Engineer–Aircraft Assignment")
    st.caption("Drag & drop a CSV with columns: **Aircraft Registration, Fleet Type, Assigned Engineer ID** "
               "(a sample file from your system is supported). You can also edit the table below.")
    uploaded_map = st.file_uploader("Drop CSV here", type=["csv"], accept_multiple_files=False,
                                    label_visibility="collapsed", key="map_uploader")

    if "mapping_df" not in st.session_state:
        st.session_state.mapping_df = _default_mapping()

    if uploaded_map is not None:
        try:
            incoming = pd.read_csv(uploaded_map)
            st.session_state.mapping_df = _normalize_mapping(incoming)
            st.success("Assignment mapping loaded from uploaded CSV.")
        except Exception as e:
            st.error(f"Failed to read uploaded mapping CSV: {e}")

    st.markdown("**Current Assignment Mapping (editable)**")
    mapping_df = st.data_editor(
        _normalize_mapping(st.session_state.mapping_df).astype("string"),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="map_editor"
    )
    # Validate/normalize after edits
    mapping_df = _normalize_mapping(mapping_df)
    st.session_state.mapping_df = mapping_df

    with st.expander("📎 Download template CSV"):
        tmpl = _default_mapping().to_csv(index=False).encode("utf-8")
        st.download_button("Download Template", tmpl, "engineer_aircraft_assignment_template.csv", "text/csv",
                           key="map_template_btn")

    # ---- Selection by Fleet Type / Engineer ID / Registration ----
    st.subheader("Selection Filters")
    left, mid, right = st.columns(3)

    all_fleets = sorted(mapping_df["Fleet Type"].dropna().unique().tolist())
    all_engs   = sorted(mapping_df["Assigned Engineer ID"].dropna().unique().tolist())
    all_regs   = sorted(mapping_df["Aircraft Registration"].dropna().unique().tolist())

    with left:
        sel_fleets = st.multiselect("Fleet Type", options=all_fleets, default=all_fleets, key="flt_sel")
    with mid:
        sel_engs   = st.multiselect("Assigned Engineer ID", options=all_engs, default=all_engs, key="eng_sel")
    with right:
        sel_regs   = st.multiselect("Aircraft Registration", options=all_regs, default=all_regs, key="reg_sel")

    # Compose final selected registrations by intersecting three filters
    filtered = mapping_df[
        mapping_df["Fleet Type"].isin(sel_fleets) &
        mapping_df["Assigned Engineer ID"].isin(sel_engs) &
        mapping_df["Aircraft Registration"].isin(sel_regs)
    ]
    selected_regs = filtered["Aircraft Registration"].tolist()

    st.caption(f"Selected aircraft: **{len(selected_regs)}** "
               f"(from fleets {len(sel_fleets)}, engineers {len(sel_engs)}, regs {len(sel_regs)})")

    return cfg, creds, mapping_df, selected_regs


# ========================= Utilities =========================
def _auto_find_firefox_binary() -> Optional[str]:
    for p in (which("firefox"), "/usr/bin/firefox", "/snap/bin/firefox", "/usr/local/bin/firefox"):
        if p and Path(p).exists():
            return p
    return None

def _auto_find_chrome_binary() -> Optional[str]:
    for p in (which("google-chrome"), which("chromium-browser"), which("chromium"),
              "/usr/bin/google-chrome", "/usr/bin/chromium", "/snap/bin/chromium", "/usr/local/bin/google-chrome"):
        if p and Path(p).exists():
            return p
    return None

def wait_dom_ready(driver: webdriver.Remote, timeout: int = 30) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

def cleanup_temp_profile(driver: webdriver.Remote) -> None:
    temp_dir = getattr(driver, "_temp_profile_dir", None)
    if temp_dir:
        try:
            p = Path(temp_dir)
            if p.exists():
                import shutil
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


# ========================= WebDriver (LEVEL 1) =========================
def _home_profile(folder_name: str) -> Path:
    """
    Create a unique, empty profile directory under ~/.selenium/<folder_name>/profile-<uuid>.
    Avoids "user data dir is in use" and is Snap-safe.
    """
    base = Path.home() / ".selenium" / folder_name
    base.mkdir(parents=True, exist_ok=True)
    prof = base / f"profile-{uuid4().hex}"
    prof.mkdir(parents=True, exist_ok=True)
    return prof

def _build_chrome_options(cfg: DriverConfig) -> webdriver.ChromeOptions:
    """
    Apply advanced options (page_load_strategy, timeouts, proxy, etc.)
    and robust isolation (unique user-data-dir + remote-debugging-pipe).
    """
    opts = webdriver.ChromeOptions()

    # Headless
    if cfg.headless:
        opts.add_argument("--headless=new")
    # Stability flags
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    # Unique HOME-based profile to avoid "in use" conflicts
    prof = _home_profile("chrome-profiles")
    opts.add_argument(f"--user-data-dir={prof}")
    setattr(opts, "_profile_path", str(prof))  # keep for cleanup
    # Avoid debug port collisions
    opts.add_argument("--remote-debugging-pipe")

    # Optional binary override
    if cfg.chrome_binary_override:
        opts.binary_location = cfg.chrome_binary_override

    # page_load_strategy
    if cfg.ch_page_load_strategy in ("normal", "eager", "none"):
        opts.page_load_strategy = cfg.ch_page_load_strategy

    # timeouts
    timeouts = {}
    if cfg.ch_timeouts_script is not None:
        timeouts["script"] = int(cfg.ch_timeouts_script)
    if cfg.ch_timeouts_page is not None:
        timeouts["pageLoad"] = int(cfg.ch_timeouts_page)
    if cfg.ch_timeouts_implicit is not None:
        timeouts["implicit"] = int(cfg.ch_timeouts_implicit)
    if timeouts:
        opts.timeouts = timeouts

    # unhandled prompt
    if cfg.ch_unhandled_prompt_behavior:
        opts.unhandled_prompt_behavior = cfg.ch_unhandled_prompt_behavior

    # strict file interactability
    if cfg.ch_strict_file_interactability:
        opts.strict_file_interactability = True

    # proxy
    if cfg.ch_http_proxy:
        prox = Proxy({
            "proxyType": ProxyType.MANUAL,
            "httpProxy": cfg.ch_http_proxy
        })
        opts.proxy = prox

    # insecure certs
    opts.accept_insecure_certs = bool(cfg.ch_accept_insecure_certs)

    # capabilities
    if cfg.ch_browser_version:
        opts.browser_version = cfg.ch_browser_version
    if cfg.ch_platform_name:
        opts.platform_name = cfg.ch_platform_name

    return opts

def create_driver(cfg: DriverConfig) -> webdriver.Remote:
    """
    Start Selenium WebDriver using either Chrome or Firefox.
    NO force-kill of processes. We rely on unique, per-run profiles for isolation.
    """
    # CHROME
    def _start_chrome() -> webdriver.Remote:
        opts = _build_chrome_options(cfg)
        service = ChromeService(ChromeDriverManager().install())
        drv = webdriver.Chrome(service=service, options=opts)
        setattr(drv, "_temp_profile_dir", getattr(opts, "_profile_path", None))
        drv.set_page_load_timeout(cfg.page_load_timeout_s)
        drv.implicitly_wait(cfg.implicit_wait_s)
        drv.set_script_timeout(cfg.script_timeout_s)
        drv.set_window_size(cfg.window_w, cfg.window_h)
        return drv

    # FIREFOX
    def _start_firefox(headless_flag: bool) -> webdriver.Remote:
        from selenium.webdriver.firefox.options import Options as FirefoxOptions

        # Environment flags to avoid profile/safe-mode prompts
        os.environ.setdefault("MOZ_LEGACY_PROFILES", "1")
        os.environ.setdefault("MOZ_DISABLE_AUTO_SAFE_MODE_KEY", "1")
        os.environ.setdefault("MOZ_HEADLESS", "1" if headless_flag else "0")

        prof = _home_profile("firefox-profiles")

        opts = FirefoxOptions()
        if headless_flag:
            opts.add_argument("--headless")
        # Robust isolation from any running instance (no killing)
        opts.add_argument("--no-remote")
        opts.add_argument("--new-instance")
        opts.add_argument("-profile"); opts.add_argument(str(prof))

        if cfg.ff_binary_override:
            opts.binary_location = cfg.ff_binary_override

        # QoL prefs
        opts.set_preference("browser.shell.checkDefaultBrowser", False)
        opts.set_preference("browser.startup.homepage_override.mstone", "ignore")
        opts.set_preference("startup.homepage_welcome_url", "about:blank")
        opts.set_preference("toolkit.telemetry.reportingpolicy.firstRun", False)

        gecko_log = Path("logs") / "geckodriver.log"
        gecko_log.parent.mkdir(parents=True, exist_ok=True)

        service = FirefoxService(
            GeckoDriverManager().install(),
            log_output=str(gecko_log)
        )

        drv = webdriver.Firefox(service=service, options=opts)
        setattr(drv, "_temp_profile_dir", str(prof))
        drv.set_page_load_timeout(cfg.page_load_timeout_s)
        drv.implicitly_wait(cfg.implicit_wait_s)
        drv.set_script_timeout(cfg.script_timeout_s)
        drv.set_window_size(cfg.window_w, cfg.window_h)
        return drv

    try:
        if cfg.browser.lower() == "firefox":
            try:
                return _start_firefox(headless_flag=cfg.headless)
            except Exception:
                # Retry once WITHOUT headless (Wayland/X quirk)
                st.warning("Firefox headless failed; retrying with a visible window.", icon="⚠️")
                return _start_firefox(headless_flag=False)
        else:
            return _start_chrome()

    except Exception as e:
        raise SetupError(f"Failed to initialize WebDriver: {e}") from e


# ========================= Maintenix Steps (LEVEL 1) =========================
def login(driver: webdriver.Remote, creds: Credentials) -> None:
    """Log in and verify readiness via explicit waits and simple heuristics."""
    if not creds.username or not creds.password:
        raise LoginError("Username and password are required.")

    driver.get("http://etmxi.ethiopianairlines.com/maintenix/common/security/login.jsp")
    wait_dom_ready(driver)

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "j_username")))
    u = driver.find_element(By.ID, "j_username"); u.clear(); u.send_keys(creds.username)

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "j_password")))
    p = driver.find_element(By.ID, "j_password"); p.clear(); p.send_keys(creds.password)

    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "idButtonLogin")))
    driver.find_element(By.ID, "idButtonLogin").click()
    wait_dom_ready(driver)

    # Post-login readiness gate
    end = time.time() + 35
    while time.time() < end:
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

def _visible(driver: webdriver.Remote, by: By, value: str) -> bool:
    try:
        els = driver.find_elements(by, value)
        return any(el.is_displayed() for el in els)
    except Exception:
        return False

def go_to_open_tasks(driver: webdriver.Remote, aircraft_reg: str) -> None:
    """
    ToDoList → search reg → 'Open' tab → 'Open Tasks' sub-tab → wait for table region.
    """
    if not aircraft_reg:
        raise OpenTasksError("Aircraft registration is required.")

    driver.get("http://etmxi.ethiopianairlines.com/maintenix/common/ToDoList.jsp")
    wait_dom_ready(driver)

    search = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "idBarcodeSearchInput")))
    search.clear(); search.send_keys(aircraft_reg); search.submit()

    # Accept either of the common registration anchors
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

    WebDriverWait(driver, 50).until(EC.text_to_be_present_in_element((By.ID, "OpenTasks_link"), "Open Tasks"))
    ot = driver.find_element(By.ID, "OpenTasks_link")
    try:
        ot.click()
    except Exception:
        driver.execute_script("arguments[0].click();", ot)

    # Wait for any cell in the table grid
    WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.ID, "idTableOpenTasksCol_0_")))


# ========================= Extraction & Enrichment =========================
def extract_open_tasks(driver: webdriver.Remote) -> pd.DataFrame:
    """
    Read '#idTableOpenTasks' into a DataFrame using pandas.read_html with robust
    handling for two-row (MultiIndex) headers. We:
      • Prefer parsing with header=[0, 1] (two header rows) via StringIO.
      • Drop the spurious ('Unnamed: 0_level_0','Unnamed: 0_level_1') pair.
      • Flatten to the EXACT requested final names (no invented data/labels):
            ('Task Name','Name')                          → 'Task Name'
            ('Task Name','ID')                            → 'Task ID'
            ('Config Position','Config Position')         → 'Config Position'
            ('Must Be Removed','Must Be Removed')         → 'Must Be Removed'
            ('Due','Due')                                  → 'Due'
            ('Next Shop Visit','Next Shop Visit')         → 'Next Shop Visit'
            ('Inventory','Inventory')                      → 'Inventory'
            ('Task Status','Task Status')                  → 'Task Status'
            ('Task Type','Task Type')                      → 'Task Type'
            ('Work Type(s)','Work Type(s)')               → 'Work Type(s)'
            ('Originator','Originator')                    → 'Originator'
            ('Task Priority','Task Priority')              → 'Task Priority'
            ('Schedule Priority','Schedule Priority')      → 'Schedule Priority'
            ('ETOPS Significant','ETOPS Significant')      → 'ETOPS Significant'
            ('Work Package','Name')                        → 'Work Package Name'
            ('Work Package','ID')                          → 'Work Package ID'
            ('Work Package No','Work Package No')          → 'Work Package No'
      • If the two-row parse yields nothing, fall back to header=0.
    We DO NOT fabricate new data or extra columns; we only flatten/rename.
    """
    table = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "idTableOpenTasks")))
    html = table.get_attribute("outerHTML") or ""

    # ---- Preferred path: two-row header (MultiIndex) -------------------------
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
                # Mapping for exact requested final names
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

                    # Drop the spurious unnamed pair entirely
                    if a.startswith("Unnamed: 0_level_0") and b.startswith("Unnamed: 0_level_1"):
                        keep_mask.append(False)
                        new_cols.append("")  # placeholder
                        continue

                    key = (a, b)
                    if key in explicit_map:
                        keep_mask.append(True)
                        new_cols.append(explicit_map[key])
                    else:
                        # Generic conservative rules that do not invent new labels:
                        if a == b:
                            keep_mask.append(True)
                            new_cols.append(a)
                        elif b in ("", "nan", "None"):
                            keep_mask.append(True)
                            new_cols.append(a)
                        else:
                            keep_mask.append(True)
                            new_cols.append(f"{a} {b}")

                # Apply keep mask & assign flattened names
                df = df.loc[:, keep_mask]
                df.columns = new_cols

                # De-duplicate column names while preserving order/content
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
        # Fall through to single-row header parsing
        pass

    # ---- Fallback: single header row -----------------------------------------
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


def enrich_with_mapping(df: pd.DataFrame, reg: str, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Attach mapping columns + Run Date in a consistent, string-safe way."""
    base = _normalize_mapping(mapping_df)
    reg = str(reg)
    fleet = base.loc[base["Aircraft Registration"] == reg, "Fleet Type"]
    eng   = base.loc[base["Aircraft Registration"] == reg, "Assigned Engineer ID"]

    out = df.copy()
    out["Aircraft Registration"] = reg
    out["Fleet Type"]            = (fleet.iloc[0] if not fleet.empty else "")
    out["Assigned Engineer ID"]  = (eng.iloc[0] if not eng.empty else "")
    out["Run Date"]              = pd.Timestamp.today().date().isoformat()

    preferred = [c for c in ["Aircraft Registration", "Fleet Type", "Assigned Engineer ID", "Run Date"] if c in out.columns]
    rest = [c for c in out.columns if c not in preferred]
    return out[preferred + rest] if preferred else out


# ========================= Orchestrator (LEVEL 0) =========================
def run_app() -> None:
    cfg, creds, mapping_df, regs = collect_inputs_ui()

    if st.button("🚀 Log in & Fetch Open Tasks", type="primary", use_container_width=True, key="btn_fetch"):
        if not creds.username or not creds.password:
            st.error("Please enter username and password.")
            return
        if not regs:
            st.error("Please select at least one aircraft registration (via Fleet, Engineer, or Reg filter).")
            return

        driver: Optional[webdriver.Remote] = None
        frames: list[pd.DataFrame] = []

        try:
            with st.spinner("Starting browser…"):
                driver = create_driver(cfg)

            with st.spinner("Logging in…"):
                login(driver, creds)

            progress = st.progress(0.0, text="Fetching Open Tasks…")
            for idx, reg in enumerate(regs, start=1):
                try:
                    with st.spinner(f"[{idx}/{len(regs)}] {reg}"):
                        go_to_open_tasks(driver, reg)
                        df = extract_open_tasks(driver)
                        df = enrich_with_mapping(df, reg, mapping_df)
                        frames.append(df)
                    st.toast(f"✔ {reg}", icon="✅")
                except Exception as e:
                    st.toast(f"⚠️ Skipped {reg}: {e}", icon="⚠️")
                finally:
                    progress.progress(idx / len(regs))

            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if result.empty:
                st.warning("No data extracted.")
                return

            st.success(f"Fetched {len(result)} rows.")
            st.dataframe(result.head(200), use_container_width=True, hide_index=True)

            saved_path = save_dataframe(result)
            st.info(f"Saved to: {saved_path}")

            buf = BytesIO(); result.to_excel(buf, index=False); buf.seek(0)
            st.download_button("⬇️ Download (Excel)", buf.getvalue(),
                               file_name="open_tasks_latest.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True, key="dl_excel")

        except (SetupError, LoginError, OpenTasksError) as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
        finally:
            if driver is not None:
                try:
                    driver.quit()
                finally:
                    cleanup_temp_profile(driver)


# ========================= Entry =========================
if __name__ == "__main__":
    run_app()

