"""Tests for CLI parser options."""

from clio_pipeline.cli import build_parser


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
