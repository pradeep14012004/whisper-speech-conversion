import time
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def render_performance_monitoring(result):
    if not result:
        st.info("Execution latency and resource monitoring appear after a complete run.")
        return
    performance = result.get("performance") or {}
    stage_times = performance.get("stage_times") or performance
    if not stage_times:
        st.info("No stage timing data available yet.")
        return

    stage_df = pd.DataFrame({"Stage": list(stage_times.keys()), "Time (s)": [float(v) for v in stage_times.values()]})
    fig_bar = px.bar(stage_df, x="Stage", y="Time (s)", color="Stage", template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

    if psutil is not None:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        st.metric("CPU usage", f"{cpu_pct}%")
        st.metric("RAM usage", f"{mem.percent}%")
    else:
        st.caption("psutil not installed; resource metrics are unavailable.")

    st.caption(f"Total pipeline execution time: {sum(float(v) for v in stage_times.values()):.2f}s")
