"""Sidebar form collection for Streamlit front end."""

from __future__ import annotations

from pathlib import Path
from shutil import which
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from overduetasks.config import Credentials, DriverConfig
from overduetasks.data import default_mapping, normalize_mapping


def collect_inputs_ui() -> Tuple[DriverConfig, Credentials, pd.DataFrame, List[str]]:
    """Render sidebar inputs and return driver config, credentials, mapping, and selected regs."""
    with st.sidebar:
        st.header("Driver")
        browser = st.selectbox("Browser", ["chrome", "firefox"], index=0, key="drv_browser")
        headless = st.toggle("Headless", value=True, key="drv_headless")

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
                ch_timeouts_page = st.number_input(
                    "timeouts.pageLoad (ms)",
                    min_value=0,
                    value=0,
                    step=500,
                    help="0 = unset; driver.set_page_load_timeout still applies.",
                    key="drv_t_pageload",
                )
                ch_timeouts_page = None if ch_timeouts_page == 0 else int(ch_timeouts_page)
            with col2:
                ch_timeouts_script = st.number_input("timeouts.script (ms)", min_value=0, value=0, step=500, key="drv_t_script")
                ch_timeouts_script = None if ch_timeouts_script == 0 else int(ch_timeouts_script)
            with col3:
                ch_timeouts_implicit = st.number_input(
                    "timeouts.implicit (ms)", min_value=0, value=0, step=500, key="drv_t_impl"
                )
                ch_timeouts_implicit = None if ch_timeouts_implicit == 0 else int(ch_timeouts_implicit)

            ch_unhandled = st.selectbox(
                "unhandled_prompt_behavior",
                ["(default)", "accept", "dismiss", "dismiss and notify", "ignore"],
                index=0,
                key="drv_unhandled",
            )
            ch_unhandled = None if ch_unhandled == "(default)" else ch_unhandled

            row_flags = st.columns(3)
            with row_flags[0]:
                ch_sfi = st.toggle("strict_file_interactability", value=False, key="drv_sfi")
            with row_flags[1]:
                ch_insec = st.toggle("accept_insecure_certs", value=False, key="drv_insec")
            with row_flags[2]:
                use_proxy = st.toggle("Use HTTP proxy", value=False, key="drv_proxy_toggle")
                if use_proxy:
                    ch_proxy = st.text_input("httpProxy (host:port)", value="http.proxy:1234", key="drv_proxy_text")

            row_meta = st.columns(3)
            with row_meta[0]:
                ch_bver = st.text_input("browser_version (optional)", value="", key="drv_bver") or None
            with row_meta[1]:
                ch_plat = st.text_input("platform_name (optional)", value="", key="drv_plat") or None
            with row_meta[2]:
                ch_bin = st.text_input(
                    "Chrome/Chromium binary (optional)",
                    value=_auto_find_chrome_binary() or "",
                    help="Leave empty to auto-detect.",
                    key="drv_chrome_bin",
                )
        else:
            st.markdown("### ⚙️ Advanced (Firefox)")
            ff_bin = st.text_input(
                "Firefox binary (optional)",
                value=_auto_find_firefox_binary() or "",
                help="Leave empty to auto-detect (e.g. /usr/bin/firefox or /snap/bin/firefox)",
                key="drv_ff_bin",
            )

        cfg = DriverConfig(
            browser=browser,
            headless=headless,
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
            ff_binary_override=ff_bin.strip() or None,
        )

    st.subheader("Credentials")
    col_user, col_pass = st.columns(2)
    with col_user:
        username = st.text_input("Username", key="cred_user")
    with col_pass:
        password = st.text_input("Password", type="password", key="cred_pass")
    creds = Credentials(username=username, password=password)

    st.subheader("Engineer–Aircraft Assignment")
    st.caption(
        "Drag & drop a CSV with columns: **Aircraft Registration, Fleet Type, Assigned Engineer ID** "
        "(a sample file from your system is supported). You can also edit the table below."
    )
    uploaded_map = st.file_uploader(
        "Drop CSV here",
        type=["csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="map_uploader",
    )

    if "mapping_df" not in st.session_state:
        st.session_state.mapping_df = default_mapping()

    if uploaded_map is not None:
        try:
            incoming = pd.read_csv(uploaded_map)
            st.session_state.mapping_df = normalize_mapping(incoming)
            st.success("Assignment mapping loaded from uploaded CSV.")
        except Exception as exc:
            st.error(f"Failed to read uploaded mapping CSV: {exc}")

    st.markdown("**Current Assignment Mapping (editable)**")
    mapping_df = st.data_editor(
        normalize_mapping(st.session_state.mapping_df).astype("string"),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="map_editor",
    )
    mapping_df = normalize_mapping(mapping_df)
    st.session_state.mapping_df = mapping_df

    with st.expander("📎 Download template CSV"):
        tmpl = default_mapping().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Template",
            tmpl,
            "engineer_aircraft_assignment_template.csv",
            "text/csv",
            key="map_template_btn",
        )

    st.subheader("Selection Filters")
    left, mid, right = st.columns(3)

    all_fleets = sorted(mapping_df["Fleet Type"].dropna().unique().tolist())
    all_engs = sorted(mapping_df["Assigned Engineer ID"].dropna().unique().tolist())
    all_regs = sorted(mapping_df["Aircraft Registration"].dropna().unique().tolist())

    with left:
        sel_fleets = st.multiselect("Fleet Type", options=all_fleets, default=all_fleets, key="flt_sel")
    with mid:
        sel_engs = st.multiselect("Assigned Engineer ID", options=all_engs, default=all_engs, key="eng_sel")
    with right:
        sel_regs = st.multiselect("Aircraft Registration", options=all_regs, default=all_regs, key="reg_sel")

    filtered = mapping_df[
        mapping_df["Fleet Type"].isin(sel_fleets)
        & mapping_df["Assigned Engineer ID"].isin(sel_engs)
        & mapping_df["Aircraft Registration"].isin(sel_regs)
    ]
    selected_regs = filtered["Aircraft Registration"].tolist()

    st.caption(
        f"Selected aircraft: **{len(selected_regs)}** "
        f"(from fleets {len(sel_fleets)}, engineers {len(sel_engs)}, regs {len(sel_regs)})"
    )

    return cfg, creds, mapping_df, selected_regs


def _auto_find_firefox_binary() -> Optional[str]:
    for path in (which("firefox"), "/usr/bin/firefox", "/snap/bin/firefox", "/usr/local/bin/firefox"):
        if path and Path(path).exists():
            return path
    return None


def _auto_find_chrome_binary() -> Optional[str]:
    for path in (
        which("google-chrome"),
        which("chromium-browser"),
        which("chromium"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
        "/usr/local/bin/google-chrome",
    ):
        if path and Path(path).exists():
            return path
    return None
