"""Run comparison page for dashboard."""

import streamlit as st

st.set_page_config(page_title="Compare - LLM-EvalLab", page_icon="🔍", layout="wide")


def main():
    st.title("🔍 Compare Runs")

    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    runs = registry.list_runs(limit=100)

    if len(runs) < 2:
        st.warning("Need at least 2 runs to compare.")
        return

    run_options = {f"{r.name} ({r.created_at.strftime('%m/%d')})": r.id for r in runs}

    col1, col2 = st.columns(2)
    with col1:
        run_a_label = st.selectbox("Baseline (Run A)", list(run_options.keys()))
    with col2:
        run_b_label = st.selectbox("Experiment (Run B)", list(run_options.keys()), index=min(1, len(run_options) - 1))

    run_a = run_options[run_a_label]
    run_b = run_options[run_b_label]

    if run_a == run_b:
        st.error("Select different runs.")
        return

    comparison = registry.compare_runs(run_a, run_b)

    # Metrics table
    import pandas as pd

    data = []
    for name, vals in comparison["metrics"].items():
        a_mean = vals.get("run_a", {}).get("mean")
        b_mean = vals.get("run_b", {}).get("mean")
        delta = vals.get("delta", {}).get("mean", 0)

        data.append({
            "Metric": name,
            "Baseline": f"{a_mean:.4f}" if a_mean else "-",
            "Experiment": f"{b_mean:.4f}" if b_mean else "-",
            "Delta": delta,
            "Change": "📈" if delta > 0.001 else ("📉" if delta < -0.001 else "➡️"),
        })

    df = pd.DataFrame(data)
    df = df.sort_values("Delta", key=abs, ascending=False)

    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
