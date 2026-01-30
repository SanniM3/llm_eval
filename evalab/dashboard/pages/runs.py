"""Runs list page for dashboard."""

import streamlit as st

st.set_page_config(page_title="Runs - LLM-EvalLab", page_icon="📋", layout="wide")


def main():
    st.title("📋 Evaluation Runs")

    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "completed", "running", "pending", "failed"],
        )
    with col2:
        search = st.text_input("Search")
    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50)

    status = None if status_filter == "All" else status_filter
    runs = registry.list_runs(
        status=status,
        name_contains=search if search else None,
        limit=limit,
    )

    st.markdown(f"**{len(runs)} runs found**")

    if not runs:
        st.info("No runs found.")
        return

    import pandas as pd

    data = []
    for run in runs:
        data.append({
            "Name": run.name,
            "Status": run.status,
            "Model": run.config_json.get("generation", {}).get("model", "N/A"),
            "Created": run.created_at.strftime("%Y-%m-%d %H:%M"),
            "ID": run.id,
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
