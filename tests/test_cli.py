"""Tests for CLI parser options."""

import json

from clio_pipeline.cli import _EtaProgressPrinter, _RunEventLogger, build_parser


def test_run_parser_accepts_fail_on_warning_flag():
    args = build_parser().parse_args(["run", "--with-facets", "--fail-on-warning"])
    assert args.command == "run"
    assert args.with_facets is True
    assert args.fail_on_warning is True
    assert args.strict is False


def test_run_parser_accepts_strict_flag():
    args = build_parser().parse_args(["run", "--strict"])
    assert args.command == "run"
    assert args.strict is True


def test_run_parser_accepts_streaming_flags():
    args = build_parser().parse_args(
        ["run", "--with-facets", "--streaming", "--stream-chunk-size", "64"]
    )
    assert args.command == "run"
    assert args.streaming is True
    assert args.stream_chunk_size == 64


def test_run_parser_accepts_hierarchy_levels_override():
    args = build_parser().parse_args(["run", "--with-hierarchy", "--hierarchy-levels", "9"])
    assert args.command == "run"
    assert args.with_hierarchy is True
    assert args.hierarchy_levels == 9


def test_run_parser_accepts_input_and_validation_flags():
    args = build_parser().parse_args(
        [
            "run",
            "--input",
            "data/external/sample.jsonl",
            "--input-validation-max-errors",
            "25",
            "--skip-input-validation",
        ]
    )
    assert args.command == "run"
    assert args.input == "data/external/sample.jsonl"
    assert args.input_validation_max_errors == 25
    assert args.skip_input_validation is True


def test_doctor_parser_accepts_network_check_flag():
    args = build_parser().parse_args(["doctor", "--network-check"])
    assert args.command == "doctor"
    assert args.network_check is True


def test_list_runs_parser_accepts_limit_and_json():
    args = build_parser().parse_args(["list-runs", "--limit", "5", "--json"])
    assert args.command == "list-runs"
    assert args.limit == 5
    assert args.json is True


def test_inspect_run_parser_requires_run_id():
    args = build_parser().parse_args(["inspect-run", "--run-id", "abc123", "--json"])
    assert args.command == "inspect-run"
    assert args.run_id == "abc123"
    assert args.json is True


def test_prune_runs_parser_accepts_retention_controls():
    args = build_parser().parse_args(
        ["prune-runs", "--keep-last", "10", "--max-age-days", "30", "--yes"]
    )
    assert args.command == "prune-runs"
    assert args.keep_last == 10
    assert args.max_age_days == 30
    assert args.yes is True


def test_eta_progress_printer_triggers_event_callback():
    captured: list[tuple[int, int, str]] = []
    printer = _EtaProgressPrinter(
        "phase-x",
        min_interval_seconds=0.0,
        event_callback=lambda done, total, detail: captured.append((done, total, detail)),
    )
    printer(1, 10, "first")
    printer(1, 10, "duplicate")
    printer(4, 10, "next")

    assert captured[0] == (1, 10, "first")
    assert captured[-1] == (4, 10, "next")
    assert len(captured) == 2


def test_run_event_logger_writes_structured_events(tmp_path):
    run_events_path = tmp_path / "run_events.jsonl"
    logger = _RunEventLogger(min_progress_interval_seconds=0.0)
    logger.bind(run_events_path)

    logger.emit(
        event_type="phase_started",
        phase="phase2_facet_extraction",
        status="running",
        message="started",
        details={"attempt": 1},
    )
    logger.emit_progress(
        phase="phase2_facet_extraction",
        done=5,
        total=20,
        detail="extract_facets_batch",
    )

    lines = run_events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event_type"] == "phase_started"
    assert first["phase"] == "phase2_facet_extraction"
    assert first["status"] == "running"
    assert second["event_type"] == "phase_progress"
    assert second["done"] == 5
    assert second["total"] == 20
