"""Tests for clio-viz CLI argument wiring."""

from clio_pipeline.viz_ui.cli import _build_streamlit_command, build_parser


def test_build_parser_accepts_live_flags():
    args = build_parser().parse_args(["--live", "--refresh-seconds", "7", "--config", "cfg.yaml"])
    assert args.live is True
    assert args.refresh_seconds == 7
    assert args.config == "cfg.yaml"


def test_build_streamlit_command_passes_live_args():
    args = build_parser().parse_args(
        [
            "--runs-root",
            "runs",
            "--run-id",
            "abc123",
            "--allow-raw-messages",
            "--live",
            "--refresh-seconds",
            "9",
        ]
    )
    command = _build_streamlit_command(args)
    assert "--run-id" in command
    assert "abc123" in command
    assert "--allow-raw-messages" in command
    assert "--live" in command
    assert "--refresh-seconds" in command
    assert "9" in command
    assert "--config" in command
