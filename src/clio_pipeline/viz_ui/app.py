"""Streamlit application for visualizing CLIO run outputs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from clio_pipeline.io import validate_conversations_jsonl
from clio_pipeline.pipeline import generate_run_id
from clio_pipeline.viz_ui.loader import discover_runs, load_run_artifacts


def _parse_app_args() -> argparse.Namespace:
    """Parse app arguments passed after `streamlit run ... --`."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
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


_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_DEFAULT_UPLOAD_RETENTION_DAYS = 14


def _project_root() -> Path:
    """Return repository root path."""

    return Path(__file__).resolve().parents[3]


def _uploads_root(runs_root: Path) -> Path:
    """Return upload storage root for UI ingestion."""

    path = runs_root / "_uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jobs_root(runs_root: Path) -> Path:
    """Return background job metadata root for UI-triggered runs."""

    path = runs_root / "_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _is_pid_running(pid: int) -> bool:
    """Check whether one process ID appears alive."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _format_bytes(value: int) -> str:
    """Render byte counts using human-readable units."""

    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _persist_uploaded_file(uploaded_file: Any, runs_root: Path) -> tuple[Path, int]:
    """Persist uploaded JSONL file to local upload storage."""

    payload = uploaded_file.getvalue()
    payload_size = len(payload)
    if payload_size > _MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large: {_format_bytes(payload_size)} "
            f"(limit {_format_bytes(_MAX_UPLOAD_BYTES)})."
        )

    uploads_root = _uploads_root(runs_root)
    safe_name = Path(str(uploaded_file.name)).name.replace(" ", "_")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = uploads_root / f"{stamp}_{safe_name}"
    target.write_bytes(payload)
    return target, payload_size


def _build_ui_run_command(
    *,
    args: argparse.Namespace,
    run_id: str,
    input_path: Path,
    options: dict[str, Any],
) -> list[str]:
    """Build run command invoked from UI."""

    command = [
        sys.executable,
        "-m",
        "clio_pipeline.cli",
        "--config",
        args.config,
        "run",
        "--run-id",
        run_id,
        "--input",
        input_path.as_posix(),
    ]
    if options.get("with_facets"):
        command.append("--with-facets")
    if options.get("with_clustering"):
        command.append("--with-clustering")
    if options.get("with_labeling"):
        command.append("--with-labeling")
    if options.get("with_hierarchy"):
        command.append("--with-hierarchy")
    if options.get("with_privacy"):
        command.append("--with-privacy")
    if options.get("with_eval"):
        command.append("--with-eval")
    limit_value = options.get("limit")
    if isinstance(limit_value, int) and limit_value > 0:
        command.extend(["--limit", str(limit_value)])
    eval_count = options.get("eval_count")
    if isinstance(eval_count, int) and eval_count > 0:
        command.extend(["--eval-count", str(eval_count)])
    if options.get("streaming"):
        command.append("--streaming")
        chunk_size = max(1, int(options.get("stream_chunk_size", 32)))
        command.extend(["--stream-chunk-size", str(chunk_size)])
    if options.get("strict"):
        command.append("--strict")
    return command


def _start_background_run(
    *,
    args: argparse.Namespace,
    input_path: Path,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Start one CLI run in the background and persist job metadata."""

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id()
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    command = _build_ui_run_command(
        args=args,
        run_id=run_id,
        input_path=input_path,
        options=options,
    )
    log_path = run_root / "ui_run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=_project_root(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    payload = {
        "run_id": run_id,
        "run_root": run_root.as_posix(),
        "pid": process.pid,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "input_path": input_path.as_posix(),
        "command": command,
        "log_path": log_path.as_posix(),
    }
    _write_json_dict(_jobs_root(runs_root) / f"{run_id}.json", payload)
    return payload


def _collect_jobs(runs_root: Path) -> list[dict[str, Any]]:
    """Load UI-triggered run jobs and derive status fields."""

    jobs_root = _jobs_root(runs_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(jobs_root.glob("*.json"), reverse=True):
        payload = _read_json_dict(path)
        if not payload:
            continue
        run_id = str(payload.get("run_id", path.stem))
        run_root = Path(str(payload.get("run_root", runs_root / run_id)))
        pid = _safe_int(payload.get("pid"), 0)
        running = _is_pid_running(pid)
        locked = (run_root / ".run.lock").exists()
        has_metrics = (run_root / "run_metrics.json").exists()
        has_warnings = (run_root / "run_warnings.json").exists()
        status = "running" if (running or locked) else "finished"
        if has_metrics and has_warnings:
            status = "finished_with_warnings"
        elif has_metrics:
            status = "finished_ok"
        payload.update(
            {
                "run_id": run_id,
                "pid": pid,
                "running": running or locked,
                "status": status,
                "has_metrics": has_metrics,
                "has_warnings": has_warnings,
                "log_path": str(payload.get("log_path", (run_root / "ui_run.log").as_posix())),
            }
        )
        rows.append(payload)
    return rows


def _terminate_job(job: dict[str, Any]) -> tuple[bool, str]:
    """Terminate one background run process."""

    pid = _safe_int(job.get("pid"), 0)
    if pid <= 0:
        return False, "invalid_pid"
    if not _is_pid_running(pid):
        return False, "process_not_running"
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "terminated"


def _read_log_tail(log_path: Path, *, max_lines: int = 80) -> str:
    """Read trailing lines from one log file."""

    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _prune_uploads(runs_root: Path, *, retention_days: int) -> dict[str, int]:
    """Delete uploaded files older than retention threshold."""

    uploads_root = _uploads_root(runs_root)
    cutoff = datetime.now(UTC) - timedelta(days=max(0, retention_days))
    deleted_count = 0
    deleted_bytes = 0
    for path in uploads_root.glob("*"):
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if modified >= cutoff:
            continue
        deleted_bytes += int(path.stat().st_size)
        path.unlink(missing_ok=True)
        deleted_count += 1
    return {
        "deleted_count": deleted_count,
        "deleted_bytes": deleted_bytes,
    }


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


def _render_ingest_and_run(args: argparse.Namespace) -> None:
    """Render UI controls for upload, validation, and run launch."""

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    st.markdown("### Input Safety")
    st.warning(
        "Uploaded files may contain sensitive text. Use sanitized exports when possible, "
        "and prune old uploads regularly."
    )

    st.markdown("### Upload or Select Input")
    uploaded = st.file_uploader(
        "Upload conversation JSONL",
        type=["jsonl", "txt"],
        accept_multiple_files=False,
        key="ingest_upload",
    )
    if uploaded is not None:
        payload_size = len(uploaded.getvalue())
        st.write(f"File size: {_format_bytes(payload_size)}")
        if payload_size > _MAX_UPLOAD_BYTES:
            st.error(
                "File exceeds upload limit "
                f"({_format_bytes(_MAX_UPLOAD_BYTES)}). Use CLI for larger files."
            )
        elif st.button("Save upload", key="save_upload"):
            try:
                saved_path, saved_size = _persist_uploaded_file(uploaded, runs_root)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["ingest_input_path"] = saved_path.as_posix()
                st.success(
                    f"Saved: {saved_path.as_posix()} ({_format_bytes(saved_size)})"
                )

    default_path = str(st.session_state.get("ingest_input_path", ""))
    input_path_raw = st.text_input(
        "Input JSONL path",
        value=default_path,
        help="Use an uploaded file path or any local JSONL file path.",
        key="ingest_input_path_text",
    ).strip()
    if input_path_raw:
        st.session_state["ingest_input_path"] = input_path_raw

    validate_col, max_errors_col = st.columns([1, 1])
    max_errors = max_errors_col.number_input(
        "Validation max errors",
        min_value=1,
        max_value=500,
        value=100,
        step=1,
        key="ingest_max_errors",
    )
    if validate_col.button("Validate Input", key="validate_input"):
        path = Path(str(st.session_state.get("ingest_input_path", "")).strip()).expanduser()
        if not path.exists():
            st.error(f"Input file not found: {path}")
        else:
            try:
                report = validate_conversations_jsonl(path, max_errors=int(max_errors))
            except Exception as exc:
                st.error(f"Validation failed: {exc}")
            else:
                st.session_state["ingest_validation_report"] = report.to_dict()
                st.session_state["ingest_validated_path"] = path.as_posix()
                if report.is_valid:
                    st.success("Validation passed.")
                else:
                    st.error("Validation failed.")

    report_dict = st.session_state.get("ingest_validation_report")
    if isinstance(report_dict, dict):
        st.markdown("### Validation Result")
        st.json(report_dict)

    st.markdown("### Run Options")
    with st.form("ingest_run_form"):
        with_facets = st.checkbox("Run facets (phase2)", value=True)
        with_clustering = st.checkbox("Run clustering (phase3)", value=True)
        with_labeling = st.checkbox("Run labeling (phase4)", value=True)
        with_hierarchy = st.checkbox("Run hierarchy", value=True)
        with_privacy = st.checkbox("Run privacy audit", value=True)
        with_eval = st.checkbox("Run evaluation", value=True)

        limit_value = st.number_input(
            "Conversation limit (0 = no cap)",
            min_value=0,
            max_value=100000,
            value=20,
            step=1,
        )
        eval_count = st.number_input(
            "Eval sample count",
            min_value=1,
            max_value=10000,
            value=20,
            step=1,
        )
        streaming = st.checkbox("Enable streaming mode", value=True)
        stream_chunk_size = st.number_input(
            "Streaming chunk size",
            min_value=1,
            max_value=5000,
            value=32,
            step=1,
            disabled=not streaming,
        )
        strict_mode = st.checkbox("Strict mode (--strict)", value=False)
        start = st.form_submit_button("Start Run")

    if start:
        raw_input = str(st.session_state.get("ingest_input_path", "")).strip()
        path = Path(raw_input).expanduser()
        if not raw_input:
            st.error("Provide an input JSONL path before starting a run.")
        elif not path.exists():
            st.error(f"Input file not found: {path}")
        else:
            validated_path = str(st.session_state.get("ingest_validated_path", "")).strip()
            report_payload = st.session_state.get("ingest_validation_report", {})
            if (
                validated_path == path.as_posix()
                and isinstance(report_payload, dict)
                and not bool(report_payload.get("is_valid", False))
            ):
                st.error("Selected input failed validation. Fix errors before launching.")
                return
            if validated_path != path.as_posix():
                st.warning(
                    "Input was not validated in this session for the selected path. "
                    "Run validation first for best safety."
                )
            options = {
                "with_facets": with_facets,
                "with_clustering": with_clustering,
                "with_labeling": with_labeling,
                "with_hierarchy": with_hierarchy,
                "with_privacy": with_privacy,
                "with_eval": with_eval,
                "limit": int(limit_value) if int(limit_value) > 0 else None,
                "eval_count": int(eval_count),
                "streaming": streaming,
                "stream_chunk_size": int(stream_chunk_size),
                "strict": strict_mode,
            }
            try:
                launch = _start_background_run(
                    args=args,
                    input_path=path,
                    options=options,
                )
            except Exception as exc:
                st.error(f"Failed to start run: {exc}")
            else:
                st.success(f"Started run: {launch['run_id']}")
                st.caption(f"Log: {launch['log_path']}")

    st.markdown("### Active / Recent UI Runs")
    jobs = _collect_jobs(runs_root)
    if not jobs:
        st.info("No UI-triggered runs found yet.")
    else:
        display_rows = [
            {
                "run_id": item.get("run_id"),
                "status": item.get("status"),
                "pid": item.get("pid"),
                "started_at_utc": item.get("started_at_utc"),
                "input_path": item.get("input_path"),
                "log_path": item.get("log_path"),
            }
            for item in jobs
        ]
        st.dataframe(display_rows, width="stretch")

        running_jobs = [item for item in jobs if bool(item.get("running", False))]
        if running_jobs:
            selected_running = st.selectbox(
                "Terminate running job",
                options=[item["run_id"] for item in running_jobs],
                index=0,
                key="terminate_run_id",
            )
            if st.button("Terminate selected run"):
                target = next(item for item in running_jobs if item["run_id"] == selected_running)
                ok, message = _terminate_job(target)
                if ok:
                    st.success(f"Termination signal sent for {selected_running}.")
                else:
                    st.warning(f"Could not terminate {selected_running}: {message}")

        selected_log = st.selectbox(
            "View log tail",
            options=[item["run_id"] for item in jobs],
            index=0,
            key="log_run_id",
        )
        selected_job = next(item for item in jobs if item["run_id"] == selected_log)
        log_tail = _read_log_tail(Path(str(selected_job.get("log_path", ""))))
        if log_tail:
            st.code(log_tail)
        else:
            st.caption("No log output found yet.")

    st.markdown("### Upload Retention")
    retention_days = st.number_input(
        "Delete uploaded files older than N days",
        min_value=0,
        max_value=365,
        value=_DEFAULT_UPLOAD_RETENTION_DAYS,
        step=1,
        key="upload_retention_days",
    )
    if st.button("Prune old uploads"):
        result = _prune_uploads(runs_root, retention_days=int(retention_days))
        st.success(
            "Pruned uploads: "
            f"{result['deleted_count']} files, {_format_bytes(result['deleted_bytes'])} freed."
        )


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
    has_runs = bool(runs)

    selected_run_id: str | None = None
    if has_runs:
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
    else:
        with st.sidebar:
            st.header("Run Selection")
            st.caption("No existing runs yet.")
            st.caption(f"Runs root: {args.runs_root}")
            st.caption(f"Raw messages enabled: {args.allow_raw_messages}")
            st.caption(f"Live mode: {live_mode}")

    page_names = [
        "Ingest & Run",
        "Overview",
        "Cluster Map",
        "Hierarchy",
        "Privacy",
        "Evaluation",
        "Artifacts",
        "Conversations",
    ]
    with st.sidebar:
        selected_page = st.selectbox(
            "Page",
            options=page_names,
            index=0,
        )
        if live_mode and selected_page == "Ingest & Run":
            st.caption("Auto-refresh is paused on Ingest & Run.")

    should_auto_refresh = live_mode and selected_page != "Ingest & Run"
    if should_auto_refresh:
        st.markdown(
            f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
            unsafe_allow_html=True,
        )

    if selected_page == "Ingest & Run":
        _render_ingest_and_run(args)
        return

    if not has_runs or selected_run_id is None:
        st.info("No run manifests found yet. Use the `Ingest & Run` page to start one.")
        return

    selected_run = next(item for item in runs if str(item["run_id"]) == selected_run_id)
    if live_mode:
        run_data = load_run_artifacts(Path(str(selected_run["run_root"])))
    else:
        run_data = _cached_load_run(str(selected_run["run_root"]))

    if selected_page == "Overview":
        _render_overview(run_data)
    elif selected_page == "Cluster Map":
        _render_cluster_map(run_data)
    elif selected_page == "Hierarchy":
        _render_hierarchy(run_data)
    elif selected_page == "Privacy":
        _render_privacy(run_data)
    elif selected_page == "Evaluation":
        _render_evaluation(run_data)
    elif selected_page == "Artifacts":
        _render_artifacts(run_data)
    elif selected_page == "Conversations":
        _render_conversations(run_data, allow_raw_messages=args.allow_raw_messages)


if __name__ == "__main__":
    main()
