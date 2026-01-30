"""Example viewer page for dashboard."""

import streamlit as st

st.set_page_config(page_title="Examples - LLM-EvalLab", page_icon="📝", layout="wide")


def main():
    st.title("📝 Example Viewer")

    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    runs = registry.list_runs(status="completed", limit=50)

    if not runs:
        st.warning("No completed runs available.")
        return

    run_options = {f"{r.name}": r.id for r in runs}
    selected_run_label = st.selectbox("Run", list(run_options.keys()))
    selected_run = run_options[selected_run_label]

    predictions = registry.get_predictions(selected_run)

    if not predictions:
        st.info("No predictions.")
        return

    example_ids = [p.example_id for p in predictions]
    selected = st.selectbox("Example", example_ids)

    pred = next(p for p in predictions if p.example_id == selected)
    metrics = registry.get_metrics(selected_run, example_id=selected)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Output")
        st.code(pred.output_text, language=None)

        st.metric("Latency", f"{pred.latency_ms:.0f} ms")
        st.metric("Tokens", f"{pred.input_tokens} in / {pred.output_tokens} out")

    with col2:
        st.subheader("Metrics")
        for m in metrics:
            st.metric(m.metric_name, f"{m.metric_value:.4f}")


if __name__ == "__main__":
    main()
