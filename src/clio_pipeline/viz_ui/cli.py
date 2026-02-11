"""CLI wrapper for launching CLIO run visualization UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from clio_pipeline.viz_ui.loader import build_check_only_summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="clio-viz",
        description="Launch local visualization dashboard for CLIO run artifacts.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier under runs/ (defaults to most recent run).",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory containing run folders (default: runs).",
    )
    parser.add_argument(
        "--allow-raw-messages",
        action="store_true",
        help="Enable raw message preview tab in the UI.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate run artifacts and print summary without launching Streamlit.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Launch UI with auto-refresh enabled by default.",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=4,
        help="Default live refresh interval in seconds (default: 4).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Server port (default: 8501).",
    )
    return parser


def _print_check_summary(summary: dict) -> None:
    """Render check-only summary to stdout."""

    print("clio-viz check")
    print(f"  Run ID:                 {summary['run_id']}")
    print(f"  Run root:               {summary['run_root']}")
    print(f"  Last phase:             {summary['phase']}")
    print(f"  Completed phases:       {summary['completed_phase_count']}")
    print(f"  Optional artifacts:     {summary['optional_present_count']}")
    print(f"  Required missing:       {summary['required_missing_count']}")
    print(f"  Cluster rows loaded:    {summary['cluster_count']}")
    print(f"  Map points loaded:      {summary['map_point_count']}")
    print(f"  Has privacy audit:      {summary['has_privacy_audit']}")
    print(f"  Has eval metrics:       {summary['has_eval_metrics']}")
    print(f"  Updated preview rows:   {summary['preview_row_count']}")
    if summary["required_missing"]:
        print("  Missing required artifacts:")
        for item in summary["required_missing"]:
            print(f"    - {item['relative_path']}")


def _build_streamlit_command(args: argparse.Namespace) -> list[str]:
    """Build subprocess command to start Streamlit app."""

    app_path = (Path(__file__).resolve().parent / "app.py").as_posix()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--",
        "--runs-root",
        args.runs_root,
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.allow_raw_messages:
        command.append("--allow-raw-messages")
    if args.live:
        command.append("--live")
    command.extend(["--refresh-seconds", str(args.refresh_seconds)])
    return command


def main() -> None:
    """Entry point for `clio-viz`."""

    parser = build_parser()
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if args.check_only:
        try:
            summary = build_check_only_summary(runs_root, run_id=args.run_id)
        except ValueError as exc:
            print(f"clio-viz check failed: {exc}")
            sys.exit(1)
        _print_check_summary(summary)
        sys.exit(0 if summary["required_missing_count"] == 0 else 2)

    command = _build_streamlit_command(args)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
