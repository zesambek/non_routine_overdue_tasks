"""Streamlit entrypoint for the Maintenix Open Tasks scraper."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from overduetasks import LoginError, OpenTasksError, SetupError
from overduetasks.services import OpenTasksScraper
from overduetasks.ui import collect_inputs_ui, render_analysis_dashboard
from shared_data import save_dataframe

st.set_page_config(page_title="Maintenix → Open Tasks (Scraper)", page_icon="🛠️", layout="wide")
st.title("🔐 Maintenix Login → Open Tasks (Scraper)")


def _firefox_retry_warning(_: Exception) -> None:
    st.warning("Firefox headless failed; retrying with a visible window.", icon="⚠️")


def run_app() -> None:
    cfg, creds, mapping_df, regs = collect_inputs_ui()

    if st.button("🚀 Log in & Fetch Open Tasks", type="primary", use_container_width=True, key="btn_fetch"):
        if not creds.username or not creds.password:
            st.error("Please enter username and password.")
            return
        if not regs:
            st.error("Please select at least one aircraft registration (via Fleet, Engineer, or Reg filter).")
            return

        scraper = OpenTasksScraper(
            cfg,
            creds,
            mapping_df,
            firefox_retry_notifier=_firefox_retry_warning,
        )

        total = len(regs)
        progress = st.progress(0.0, text="Fetching Open Tasks…")

        def iteration_hook(idx: int, total_regs: int, reg: str, success: bool, error: Exception | None) -> None:
            ratio = idx / total_regs if total_regs else 1.0
            progress_text = f"Fetching Open Tasks… ({idx}/{total_regs})" if total_regs else "Fetching Open Tasks…"
            progress.progress(ratio, text=progress_text)
            if success:
                st.toast(f"✔ {reg}", icon="✅")
            else:
                st.toast(f"⚠️ Skipped {reg}: {error}", icon="⚠️")

        try:
            with st.spinner("Starting browser, logging in, and fetching data…"):
                result = scraper.scrape(regs, iteration_hook=iteration_hook)
        except (SetupError, LoginError, OpenTasksError) as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected errors
            st.error(f"Unexpected error: {exc}")
            return
        finally:
            progress.empty()

        if result.dataframe.empty:
            st.warning("No data extracted.")
            return

        st.success(f"Fetched {len(result.dataframe)} rows.")

        raw_tab, analytics_tab = st.tabs(["Raw Data", "Analytics Dashboard"])

        with raw_tab:
            st.dataframe(result.dataframe.head(200), use_container_width=True, hide_index=True)

            saved_path = save_dataframe(result.dataframe)
            st.info(f"Saved to: {saved_path}")

            buf = BytesIO()
            result.dataframe.to_excel(buf, index=False)
            buf.seek(0)
            st.download_button(
                "⬇️ Download (Excel)",
                buf.getvalue(),
                file_name="open_tasks_latest.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_excel",
            )

        with analytics_tab:
            render_analysis_dashboard(result.dataframe)

        if result.errors:
            with st.expander("⚠️ Skipped registrations"):
                for reg, error in result.errors:
                    st.write(f"{reg}: {error}")


if __name__ == "__main__":
    run_app()
