"""WebDriver setup and teardown utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService

from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

from overduetasks.config import DriverConfig
from overduetasks.exceptions import SetupError

logger = logging.getLogger(__name__)


def _home_profile(folder_name: str) -> Path:
    """Create an isolated profile directory under ~/.selenium/."""
    base = Path.home() / ".selenium" / folder_name
    base.mkdir(parents=True, exist_ok=True)
    prof = base / f"profile-{uuid4().hex}"
    prof.mkdir(parents=True, exist_ok=True)
    return prof


def _build_chrome_options(cfg: DriverConfig) -> webdriver.ChromeOptions:
    """Apply advanced options and profile isolation for Chrome/Chromium."""
    opts = webdriver.ChromeOptions()

    if cfg.headless:
        opts.add_argument("--headless=new")

    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")

    prof = _home_profile("chrome-profiles")
    opts.add_argument(f"--user-data-dir={prof}")
    setattr(opts, "_profile_path", str(prof))
    opts.add_argument("--remote-debugging-pipe")

    if cfg.chrome_binary_override:
        opts.binary_location = cfg.chrome_binary_override

    if cfg.ch_page_load_strategy in ("normal", "eager", "none"):
        opts.page_load_strategy = cfg.ch_page_load_strategy

    timeouts = {}
    if cfg.ch_timeouts_script is not None:
        timeouts["script"] = int(cfg.ch_timeouts_script)
    if cfg.ch_timeouts_page is not None:
        timeouts["pageLoad"] = int(cfg.ch_timeouts_page)
    if cfg.ch_timeouts_implicit is not None:
        timeouts["implicit"] = int(cfg.ch_timeouts_implicit)
    if timeouts:
        opts.timeouts = timeouts

    if cfg.ch_unhandled_prompt_behavior:
        opts.unhandled_prompt_behavior = cfg.ch_unhandled_prompt_behavior

    if cfg.ch_strict_file_interactability:
        opts.strict_file_interactability = True

    if cfg.ch_http_proxy:
        prox = Proxy({"proxyType": ProxyType.MANUAL, "httpProxy": cfg.ch_http_proxy})
        opts.proxy = prox

    opts.accept_insecure_certs = bool(cfg.ch_accept_insecure_certs)

    if cfg.ch_browser_version:
        opts.browser_version = cfg.ch_browser_version
    if cfg.ch_platform_name:
        opts.platform_name = cfg.ch_platform_name

    return opts


def _start_firefox(cfg: DriverConfig, headless_flag: bool) -> webdriver.Remote:
    """Start a Firefox driver instance with an isolated profile."""
    from selenium.webdriver.firefox.options import Options as FirefoxOptions

    os.environ.setdefault("MOZ_LEGACY_PROFILES", "1")
    os.environ.setdefault("MOZ_DISABLE_AUTO_SAFE_MODE_KEY", "1")
    os.environ.setdefault("MOZ_HEADLESS", "1" if headless_flag else "0")

    prof = _home_profile("firefox-profiles")

    opts = FirefoxOptions()
    if headless_flag:
        opts.add_argument("--headless")
    opts.add_argument("--no-remote")
    opts.add_argument("--new-instance")
    opts.add_argument("-profile")
    opts.add_argument(str(prof))

    if cfg.ff_binary_override:
        opts.binary_location = cfg.ff_binary_override

    opts.set_preference("browser.shell.checkDefaultBrowser", False)
    opts.set_preference("browser.startup.homepage_override.mstone", "ignore")
    opts.set_preference("startup.homepage_welcome_url", "about:blank")
    opts.set_preference("toolkit.telemetry.reportingpolicy.firstRun", False)

    gecko_log = Path("logs") / "geckodriver.log"
    gecko_log.parent.mkdir(parents=True, exist_ok=True)

    service = FirefoxService(GeckoDriverManager().install(), log_output=str(gecko_log))

    drv = webdriver.Firefox(service=service, options=opts)
    setattr(drv, "_temp_profile_dir", str(prof))
    drv.set_page_load_timeout(cfg.page_load_timeout_s)
    drv.implicitly_wait(cfg.implicit_wait_s)
    drv.set_script_timeout(cfg.script_timeout_s)
    drv.set_window_size(cfg.window_w, cfg.window_h)
    return drv


def create_driver(cfg: DriverConfig, *, firefox_retry_notifier: Callable[[Exception], None] | None = None) -> webdriver.Remote:
    """
    Start Selenium WebDriver using either Chrome or Firefox.

    When Firefox headless fails, we retry once with a visible window and emit the
    optional notifier so UIs can surface a message.
    """

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

    try:
        if cfg.browser.lower() == "firefox":
            try:
                return _start_firefox(cfg, headless_flag=cfg.headless)
            except Exception as exc:
                if firefox_retry_notifier:
                    firefox_retry_notifier(exc)
                else:
                    logger.warning("Firefox headless failed; retrying with visible window: %s", exc)
                return _start_firefox(cfg, headless_flag=False)
        return _start_chrome()
    except Exception as exc:  # pragma: no cover - wraps webdriver init errors
        raise SetupError(f"Failed to initialize WebDriver: {exc}") from exc


def cleanup_temp_profile(driver: webdriver.Remote) -> None:
    """Remove temporary profile directories created during driver startup."""
    temp_dir = getattr(driver, "_temp_profile_dir", None)
    if not temp_dir:
        return
    try:
        profile_path = Path(temp_dir)
        if profile_path.exists():
            import shutil

            shutil.rmtree(profile_path, ignore_errors=True)
    except Exception:  # pragma: no cover - best effort cleanup
        pass
