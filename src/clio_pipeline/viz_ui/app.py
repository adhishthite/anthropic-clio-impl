"""Streamlit application for visualizing CLIO run outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from clio_pipeline.viz_ui.loader import discover_runs, load_run_artifacts


def _parse_app_args() -> argparse.Namespace:
    """Parse app arguments passed after `streamlit run ... --`."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--allow-raw-messages", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--refresh-seconds", type=int, default=4)
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def _cached_discover_runs(runs_root: str) -> list[dict]:
    return discover_runs(Path(runs_root))


@st.cache_data(show_spinner=False)
def _cached_load_run(run_root: str) -> dict:
    return load_run_artifacts(Path(run_root))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _checkpoint_progress_rows(checkpoints: dict[str, dict]) -> list[dict]:
    """Build normalized progress rows from phase checkpoint payloads."""

    rows: list[dict] = []
    for phase, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict) or not checkpoint:
            continue
        status = "completed" if bool(checkpoint.get("completed", False)) else "running"

        processed = None
        total = None
        if (
            "raw_total" in checkpoint
            and "facet_total" in checkpoint
            and "cluster_total" in checkpoint
        ):
            total = (
                _safe_int(checkpoint.get("raw_total"))
                + _safe_int(checkpoint.get("facet_total"))
                + _safe_int(checkpoint.get("cluster_total"))
            )
            processed = (
                _safe_int(checkpoint.get("raw_processed"))
                + _safe_int(checkpoint.get("facet_processed"))
                + _safe_int(checkpoint.get("cluster_processed"))
            )
        elif "conversation_count_total" in checkpoint:
            total = _safe_int(checkpoint.get("conversation_count_total"))
            processed = _safe_int(checkpoint.get("conversation_count_processed"))
        elif "cluster_total" in checkpoint:
            total = _safe_int(checkpoint.get("cluster_total"))
            processed = _safe_int(checkpoint.get("cluster_processed"))
        elif "label_checkpoint_count" in checkpoint:
            processed = _safe_int(checkpoint.get("label_checkpoint_count"))

        rows.append(
            {
                "phase": phase,
                "status": status,
                "processed": processed,
                "total": total,
                "current_concurrency": _safe_int(checkpoint.get("current_concurrency"), 0),
                "note": str(checkpoint.get("note", "")),
                "updated_at_utc": str(checkpoint.get("updated_at_utc", "")),
            }
        )
    return rows


def _augment_map_points(points: list[dict], clusters: list[dict]) -> list[dict]:
    """Attach cluster labels/privacy state to map point rows."""

    cluster_by_id: dict[int, dict] = {}
    for cluster in clusters:
        if "cluster_id" in cluster:
            cluster_by_id[_safe_int(cluster["cluster_id"], -1)] = cluster

    output: list[dict] = []
    for point in points:
        row = dict(point)
        cluster_id = _safe_int(row.get("cluster_id"), -1)
        cluster = cluster_by_id.get(cluster_id, {})
        row["cluster_name"] = str(cluster.get("name", f"cluster-{cluster_id}"))
        row["final_kept"] = bool(cluster.get("final_kept", row.get("kept_by_threshold", False)))
        if "privacy_rating" in cluster:
            row["privacy_rating"] = _safe_int(cluster["privacy_rating"], 0)
        output.append(row)
    return output


def _render_overview(data: dict) -> None:
    manifest = data["manifest"]
    clusters = data["labeled_clusters"]
    privacy = data["privacy_audit"].get("summary", {})
    eval_metrics = data["eval_metrics"]
    run_metrics = data.get("run_metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run ID", str(data["run_id"]))
    col2.metric("Last Phase", str(manifest.get("phase", "unknown")))
    col3.metric("Completed Phases", len(manifest.get("completed_phases", [])))
    col4.metric("Cluster Rows", len(clusters))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Input Conversations", _safe_int(manifest.get("conversation_count_input")))
    col6.metric(
        "Processed Conversations",
        _safe_int(manifest.get("conversation_count_processed")),
    )
    col7.metric("Map Points", len(data["map_points"]))
    col8.metric(
        "Eval Samples",
        _safe_int(eval_metrics.get("synthetic_count")),
    )
    if data.get("run_lock_active", False):
        lock_payload = data.get("run_lock_payload", {})
        st.warning(
            "Run lock is active (processing may still be running). "
            f"Owner pid: {lock_payload.get('pid', 'unknown')}."
        )

    st.markdown("### Completed Phases")
    completed = manifest.get("completed_phases", [])
    if isinstance(completed, list) and completed:
        st.write(", ".join(str(item) for item in completed))
    else:
        st.write("No completed phase list found.")

    if privacy:
        stage = privacy.get("cluster_summary", {})
        st.markdown("### Privacy Snapshot")
        st.write(
            f"Cluster pass rate: {_safe_float(stage.get('pass_rate')):.2%} | "
            f"Threshold: {_safe_int(stage.get('threshold'))}"
        )

    checkpoint_rows = _checkpoint_progress_rows(data.get("checkpoints", {}))
    if checkpoint_rows:
        st.markdown("### Live Progress (Checkpoints)")
        st.dataframe(checkpoint_rows, width="stretch")

    if isinstance(run_metrics, dict) and run_metrics:
        usage = run_metrics.get("aggregate_llm_usage", {})
        st.markdown("### Run Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration (s)", f"{_safe_float(run_metrics.get('duration_seconds')):.1f}")
        m2.metric("Warnings", _safe_int(run_metrics.get("warning_count")))
        m3.metric("LLM Requests", _safe_int(usage.get("request_count")))
        m4.metric("LLM Tokens", _safe_int(usage.get("total_tokens")))

    run_events = data.get("run_events", [])
    if isinstance(run_events, list) and run_events:
        st.markdown("### Recent Run Events")
        st.dataframe(run_events[-20:], width="stretch")


def _render_cluster_map(data: dict) -> None:
    points = _augment_map_points(data["map_points"], data["labeled_clusters"])
    if not points:
        st.info("No map points found. Run at least through Phase 3.")
        return

    kept_points = [item for item in points if bool(item.get("final_kept", False))]
    dropped_points = [item for item in points if not bool(item.get("final_kept", False))]
    fig = go.Figure()

    if kept_points:
        fig.add_trace(
            go.Scatter(
                x=[_safe_float(item.get("x")) for item in kept_points],
                y=[_safe_float(item.get("y")) for item in kept_points],
                mode="markers",
                marker={"size": 8, "color": "#2ca02c", "opacity": 0.8},
                name="final_kept=True",
                text=[
                    "<br>".join(
                        [
                            f"cluster={item.get('cluster_id')}",
                            f"name={item.get('cluster_name')}",
                            f"lang={item.get('language', '')}",
                            f"concern={item.get('concerning_score', '')}",
                        ]
                    )
                    for item in kept_points
                ],
                hoverinfo="text",
            )
        )

    if dropped_points:
        fig.add_trace(
            go.Scatter(
                x=[_safe_float(item.get("x")) for item in dropped_points],
                y=[_safe_float(item.get("y")) for item in dropped_points],
                mode="markers",
                marker={"size": 8, "color": "#d62728", "opacity": 0.75},
                name="final_kept=False",
                text=[
                    "<br>".join(
                        [
                            f"cluster={item.get('cluster_id')}",
                            f"name={item.get('cluster_name')}",
                            f"lang={item.get('language', '')}",
                            f"concern={item.get('concerning_score', '')}",
                        ]
                    )
                    for item in dropped_points
                ],
                hoverinfo="text",
            )
        )

    centroids = data["map_clusters"]
    if centroids:
        fig.add_trace(
            go.Scatter(
                x=[_safe_float(item.get("x")) for item in centroids],
                y=[_safe_float(item.get("y")) for item in centroids],
                mode="markers+text",
                marker={
                    "size": 16,
                    "color": "#1f77b4",
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "#0e2f5a"},
                },
                text=[f"c{_safe_int(item.get('cluster_id'), -1)}" for item in centroids],
                textposition="top center",
                name="cluster centroids",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Conversation Map",
        xaxis_title="Projection X",
        yaxis_title="Projection Y",
        height=600,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Cluster Table")
    cluster_rows = []
    for cluster in data["labeled_clusters"]:
        cluster_rows.append(
            {
                "cluster_id": _safe_int(cluster.get("cluster_id"), -1),
                "name": str(cluster.get("name", "")),
                "size": _safe_int(cluster.get("size")),
                "unique_users": _safe_int(cluster.get("unique_users")),
                "kept_by_threshold": bool(cluster.get("kept_by_threshold", False)),
                "kept_by_privacy": cluster.get("kept_by_privacy"),
                "final_kept": cluster.get("final_kept"),
            }
        )
    st.dataframe(cluster_rows, width="stretch")


def _render_hierarchy(data: dict) -> None:
    tree_view = data["tree_view"]
    hierarchy = data["hierarchy"]
    nodes = tree_view.get("nodes", [])
    edges = tree_view.get("edges", [])
    if not isinstance(nodes, list) or not nodes:
        nodes = hierarchy.get("nodes", [])
    if not isinstance(edges, list):
        edges = hierarchy.get("edges", [])

    if not isinstance(nodes, list) or not nodes:
        st.info("No hierarchy output found. Run with `--with-hierarchy`.")
        return

    parent_by_child: dict[str, str] = {}
    for edge in edges:
        parent_id = str(edge.get("parent_id", ""))
        child_id = str(edge.get("child_id", ""))
        if parent_id and child_id:
            parent_by_child[child_id] = parent_id

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue
        ids.append(node_id)
        labels.append(str(node.get("name", node_id)))
        parents.append(parent_by_child.get(node_id, ""))
        values.append(max(1, _safe_int(node.get("size"), 1)))

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        )
    )
    fig.update_layout(title="Hierarchy Sunburst", height=650)
    st.plotly_chart(fig, width="stretch")

    top_level = hierarchy.get("top_level_clusters", [])
    if isinstance(top_level, list) and top_level:
        st.markdown("### Top-Level Clusters")
        st.dataframe(top_level, width="stretch")


def _render_privacy(data: dict) -> None:
    privacy = data["privacy_audit"]
    summary = privacy.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        st.info("No privacy audit output found. Run with `--with-privacy`.")
        return

    stages = ["raw_conversation", "facet_summary", "cluster_summary"]
    rows = []
    for stage in stages:
        stage_summary = summary.get(stage, {})
        if not isinstance(stage_summary, dict):
            continue
        rows.append(
            {
                "stage": stage,
                "pass_rate": _safe_float(stage_summary.get("pass_rate")),
                "pass_count": _safe_int(stage_summary.get("pass_count")),
                "fail_count": _safe_int(stage_summary.get("fail_count")),
                "threshold": _safe_int(stage_summary.get("threshold")),
            }
        )

    if rows:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[row["stage"] for row in rows],
                    y=[row["pass_rate"] for row in rows],
                    text=[f"{row['pass_rate']:.2%}" for row in rows],
                    textposition="outside",
                    marker={"color": "#9467bd"},
                )
            ]
        )
        fig.update_layout(
            title="Privacy Pass Rate by Stage",
            yaxis={"title": "Pass rate", "range": [0, 1]},
            height=450,
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(rows, width="stretch")

    validation = privacy.get("validation")
    if isinstance(validation, dict) and validation:
        st.markdown("### Auditor Validation Set")
        col1, col2, col3 = st.columns(3)
        col1.metric("Validation Cases", _safe_int(validation.get("total_cases")))
        col2.metric("In-range Rate", f"{_safe_float(validation.get('in_range_rate')):.2%}")
        col3.metric(
            "Mean Absolute Error", f"{_safe_float(validation.get('mean_absolute_error')):.3f}"
        )
        records = validation.get("records", [])
        if isinstance(records, list):
            st.dataframe(records, width="stretch")


def _render_evaluation(data: dict) -> None:
    eval_metrics = data["eval_metrics"]
    ablations = eval_metrics.get("ablations", {})
    if not isinstance(ablations, dict) or not ablations:
        st.info("No evaluation artifacts found. Run with `--with-eval`.")
        return

    names = sorted(ablations.keys())
    accuracy = [_safe_float(ablations[name].get("accuracy")) for name in names]
    macro_f1 = [_safe_float(ablations[name].get("macro_f1")) for name in names]
    weighted_f1 = [_safe_float(ablations[name].get("weighted_f1")) for name in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="accuracy", x=names, y=accuracy))
    fig.add_trace(go.Bar(name="macro_f1", x=names, y=macro_f1))
    fig.add_trace(go.Bar(name="weighted_f1", x=names, y=weighted_f1))
    fig.update_layout(
        barmode="group",
        title="Phase 6 Ablation Metrics",
        yaxis={"range": [0, 1]},
        height=520,
    )
    st.plotly_chart(fig, width="stretch")

    selected = st.selectbox(
        "Per-language breakdown representation",
        options=names,
        index=0,
    )
    per_language = ablations[selected].get("per_language", {})
    if isinstance(per_language, dict):
        rows = []
        for language, metrics in sorted(per_language.items()):
            rows.append(
                {
                    "language": language,
                    "samples": _safe_int(metrics.get("samples")),
                    "accuracy": _safe_float(metrics.get("accuracy")),
                    "macro_f1": _safe_float(metrics.get("macro_f1")),
                    "weighted_f1": _safe_float(metrics.get("weighted_f1")),
                }
            )
        st.dataframe(rows, width="stretch")

    report_md = data["eval_report_md"]
    if report_md:
        with st.expander("Phase 6 markdown report"):
            st.markdown(report_md)


def _render_artifacts(data: dict) -> None:
    statuses = data["artifact_status"]
    st.dataframe(statuses, width="stretch")
    with st.expander("Run manifest"):
        st.json(data["manifest"])


def _render_conversations(data: dict, *, allow_raw_messages: bool) -> None:
    if not allow_raw_messages:
        st.info(
            "Raw message preview is disabled. Re-run UI with "
            "`--allow-raw-messages` to enable this tab."
        )
        return

    rows = data["conversation_updated_rows_preview"]
    if not rows:
        rows = data["conversation_rows_preview"]

    if not rows:
        st.info("No conversation rows were found.")
        return

    st.write(f"Previewing {len(rows)} rows (messages-only snapshot shape).")
    preview_count = st.slider("Rows to show", min_value=1, max_value=min(30, len(rows)), value=8)

    for idx, row in enumerate(rows[:preview_count], start=1):
        messages = row.get("messages", [])
        analysis = row.get("analysis", {})
        with st.expander(f"Row {idx} ({len(messages)} messages)"):
            if isinstance(messages, list):
                for message in messages:
                    role = str(message.get("role", "unknown"))
                    content = str(message.get("content", ""))
                    st.markdown(f"**{role}**: {content}")
            if analysis:
                st.markdown("**analysis**")
                st.json(analysis)


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="CLIO Run Explorer", layout="wide")
    args = _parse_app_args()

    st.title("CLIO Run Explorer")
    st.caption("Inspect one pipeline run across map, hierarchy, privacy, and evaluation outputs.")

    with st.sidebar:
        st.header("Live")
        live_mode = st.toggle("Live mode", value=bool(args.live))
        refresh_seconds = st.slider(
            "Refresh interval (seconds)",
            min_value=2,
            max_value=30,
            value=max(2, int(args.refresh_seconds)),
            disabled=not live_mode,
        )

    if live_mode:
        runs = discover_runs(Path(args.runs_root))
    else:
        runs = _cached_discover_runs(args.runs_root)
    if not runs:
        st.error(f"No run manifests found under `{args.runs_root}`.")
        return

    run_ids = [str(item["run_id"]) for item in runs]
    default_run_id = args.run_id if args.run_id in run_ids else run_ids[0]

    with st.sidebar:
        st.header("Run Selection")
        selected_run_id = st.selectbox(
            "Run ID",
            options=run_ids,
            index=run_ids.index(default_run_id),
        )
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Runs root: {args.runs_root}")
        st.caption(f"Raw messages enabled: {args.allow_raw_messages}")
        st.caption(f"Live mode: {live_mode}")
        if live_mode:
            st.caption(f"Auto-refresh every {refresh_seconds}s")

    if live_mode:
        st.markdown(
            f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
            unsafe_allow_html=True,
        )

    selected_run = next(item for item in runs if str(item["run_id"]) == selected_run_id)
    if live_mode:
        run_data = load_run_artifacts(Path(str(selected_run["run_root"])))
    else:
        run_data = _cached_load_run(str(selected_run["run_root"]))

    tabs = st.tabs(
        [
            "Overview",
            "Cluster Map",
            "Hierarchy",
            "Privacy",
            "Evaluation",
            "Artifacts",
            "Conversations",
        ]
    )
    with tabs[0]:
        _render_overview(run_data)
    with tabs[1]:
        _render_cluster_map(run_data)
    with tabs[2]:
        _render_hierarchy(run_data)
    with tabs[3]:
        _render_privacy(run_data)
    with tabs[4]:
        _render_evaluation(run_data)
    with tabs[5]:
        _render_artifacts(run_data)
    with tabs[6]:
        _render_conversations(run_data, allow_raw_messages=args.allow_raw_messages)


if __name__ == "__main__":
    main()
