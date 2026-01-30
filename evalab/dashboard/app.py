"""Main Streamlit dashboard for LLM-EvalLab."""

import streamlit as st

st.set_page_config(
    page_title="LLM-EvalLab Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .stMetric {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 8px;
    }
    .run-card {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #e94560;
    }
    h1 {
        color: #e94560;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard entry point."""
    st.title("📊 LLM-EvalLab Dashboard")
    st.markdown("*Evaluation & Reliability Platform for LLM Applications*")

    # Initialize database
    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📋 Runs List", "🔍 Compare Runs", "📝 Example Viewer"],
    )

    if page == "📋 Runs List":
        show_runs_list(registry)
    elif page == "🔍 Compare Runs":
        show_compare_page(registry)
    elif page == "📝 Example Viewer":
        show_example_viewer(registry)


def show_runs_list(registry):
    """Display list of evaluation runs."""
    st.header("Evaluation Runs")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "completed", "running", "pending", "failed"],
        )
    with col2:
        search = st.text_input("Search by name")

    # Get runs
    status = None if status_filter == "All" else status_filter
    runs = registry.list_runs(status=status, name_contains=search if search else None, limit=50)

    if not runs:
        st.info("No runs found. Create a new run with `evalab run --config your_config.yaml`")
        return

    # Display runs
    for run in runs:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.markdown(f"**{run.name}**")
                st.caption(f"ID: {run.id[:40]}...")

            with col2:
                model = run.config_json.get("generation", {}).get("model", "N/A")
                st.text(f"Model: {model}")

            with col3:
                status_emoji = {
                    "completed": "✅",
                    "running": "🔄",
                    "pending": "⏳",
                    "failed": "❌",
                }.get(run.status, "❓")
                st.text(f"{status_emoji} {run.status}")
                st.caption(run.created_at.strftime("%Y-%m-%d %H:%M"))

            with col4:
                if st.button("View", key=f"view_{run.id}"):
                    st.session_state["selected_run"] = run.id
                    st.session_state["page"] = "detail"

        st.divider()

    # Run detail view
    if st.session_state.get("selected_run"):
        show_run_detail(registry, st.session_state["selected_run"])


def show_run_detail(registry, run_id):
    """Display detailed view of a run."""
    st.subheader("Run Details")

    run = registry.get_run(run_id)
    if not run:
        st.error(f"Run not found: {run_id}")
        return

    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", run.status)
    with col2:
        st.metric("Created", run.created_at.strftime("%Y-%m-%d %H:%M"))
    with col3:
        model = run.config_json.get("generation", {}).get("model", "N/A")
        st.metric("Model", model)

    # Metrics
    st.subheader("Metrics Summary")
    aggregates = registry.get_aggregates(run_id)

    if aggregates:
        # Create metrics grid
        global_aggs = [a for a in aggregates if a.slice_key is None]

        cols = st.columns(4)
        for i, agg in enumerate(global_aggs[:12]):
            with cols[i % 4]:
                st.metric(
                    agg.metric_name,
                    f"{agg.agg_json.get('mean', 0):.4f}",
                    delta=None,
                )

        # Full table
        with st.expander("All Metrics"):
            import pandas as pd

            data = []
            for agg in global_aggs:
                data.append({
                    "Metric": agg.metric_name,
                    "Mean": agg.agg_json.get("mean", 0),
                    "Std": agg.agg_json.get("std", 0),
                    "Min": agg.agg_json.get("min", 0),
                    "Max": agg.agg_json.get("max", 0),
                    "P50": agg.agg_json.get("p50", 0),
                    "P95": agg.agg_json.get("p95", 0),
                })
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No metrics available for this run.")

    # Config
    with st.expander("Configuration"):
        st.json(run.config_json)


def show_compare_page(registry):
    """Display run comparison page."""
    st.header("Compare Runs")

    runs = registry.list_runs(limit=100)

    if len(runs) < 2:
        st.info("Need at least 2 runs to compare.")
        return

    run_options = {f"{r.name} ({r.id[:20]}...)": r.id for r in runs}

    col1, col2 = st.columns(2)
    with col1:
        run_a_label = st.selectbox("Run A", list(run_options.keys()), key="run_a")
    with col2:
        run_b_label = st.selectbox("Run B", list(run_options.keys()), index=1, key="run_b")

    run_a = run_options[run_a_label]
    run_b = run_options[run_b_label]

    if run_a == run_b:
        st.warning("Please select different runs to compare.")
        return

    if st.button("Compare", type="primary"):
        comparison = registry.compare_runs(run_a, run_b)

        st.subheader("Metric Comparison")

        import pandas as pd

        data = []
        for metric_name, metric_data in comparison["metrics"].items():
            a_mean = metric_data.get("run_a", {}).get("mean", 0)
            b_mean = metric_data.get("run_b", {}).get("mean", 0)
            delta = metric_data.get("delta", {}).get("mean", 0)

            data.append({
                "Metric": metric_name,
                "Run A": f"{a_mean:.4f}" if a_mean else "-",
                "Run B": f"{b_mean:.4f}" if b_mean else "-",
                "Delta": f"{delta:+.4f}",
            })

        df = pd.DataFrame(data)

        # Color-code delta column
        def highlight_delta(val):
            if val.startswith("+"):
                return "color: green"
            elif val.startswith("-") and val != "-":
                return "color: red"
            return ""

        styled_df = df.style.applymap(highlight_delta, subset=["Delta"])
        st.dataframe(styled_df, use_container_width=True)

        # Summary
        st.subheader("Summary")
        improvements = sum(1 for m in comparison["metrics"].values()
                          if m.get("delta", {}).get("mean", 0) > 0)
        regressions = sum(1 for m in comparison["metrics"].values()
                         if m.get("delta", {}).get("mean", 0) < 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Improvements", improvements)
        with col2:
            st.metric("Regressions", regressions)
        with col3:
            st.metric("Unchanged", len(comparison["metrics"]) - improvements - regressions)


def show_example_viewer(registry):
    """Display example-level results viewer."""
    st.header("Example Viewer")

    runs = registry.list_runs(status="completed", limit=50)

    if not runs:
        st.info("No completed runs available.")
        return

    run_options = {f"{r.name} ({r.id[:20]}...)": r.id for r in runs}
    selected_run_label = st.selectbox("Select Run", list(run_options.keys()))
    selected_run = run_options[selected_run_label]

    # Get predictions
    predictions = registry.get_predictions(selected_run)

    if not predictions:
        st.info("No predictions found for this run.")
        return

    # Example selector
    example_ids = [p.example_id for p in predictions]
    selected_example = st.selectbox("Select Example", example_ids)

    # Get details
    prediction = next(p for p in predictions if p.example_id == selected_example)
    metrics = registry.get_metrics(selected_run, example_id=selected_example)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Output")
        st.text_area("Model Output", prediction.output_text, height=200, disabled=True)

        st.subheader("Generation Stats")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Latency", f"{prediction.latency_ms:.0f}ms")
        with stat_col2:
            st.metric("Input Tokens", prediction.input_tokens)
        with stat_col3:
            st.metric("Output Tokens", prediction.output_tokens)

    with col2:
        st.subheader("Metrics")
        if metrics:
            for m in metrics:
                st.metric(m.metric_name, f"{m.metric_value:.4f}")
        else:
            st.info("No metrics available for this example.")

    # Retrieval traces
    traces = registry.artifacts.load_retrieval_traces(selected_run)
    if traces:
        with st.expander("Retrieval Traces"):
            for trace in traces[:5]:  # Show first 5
                st.json(trace)


if __name__ == "__main__":
    main()
