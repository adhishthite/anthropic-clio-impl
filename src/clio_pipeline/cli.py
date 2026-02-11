"""CLI entrypoint for the CLIO pipeline."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime

from clio_pipeline import __version__
from clio_pipeline.config import Settings
from clio_pipeline.io import (
    ConversationDatasetError,
    RunLockError,
    append_jsonl,
    ensure_directory,
    load_conversations_jsonl,
    run_lock,
    save_json,
    validate_conversations_jsonl,
)
from clio_pipeline.observability import get_langsmith_status
from clio_pipeline.pipeline import (
    build_run_fingerprint,
    generate_run_id,
    initialize_run_artifacts,
    initialize_run_artifacts_streaming,
    load_phase2_facets,
    load_phase3_cluster_summaries,
    load_phase4_hierarchy,
    load_phase4_labeled_clusters,
    load_phase5_outputs,
    load_phase6_evaluation,
    run_phase1_dataset_load,
    run_phase2_facet_extraction,
    run_phase2_facet_extraction_streaming,
    run_phase3_base_clustering,
    run_phase4_cluster_labeling,
    run_phase4_hierarchy_scaffold,
    run_phase5_privacy_audit,
    run_phase6_evaluation,
)


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < float("inf")):
        return "--:--"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _EtaProgressPrinter:
    """Print throttled progress updates with elapsed time and ETA."""

    def __init__(self, label: str, *, min_interval_seconds: float = 2.0) -> None:
        self._label = label
        self._started_at = time.perf_counter()
        self._last_print_at = 0.0
        self._last_done = -1
        self._last_bucket = -1
        self._min_interval_seconds = min_interval_seconds

    def __call__(self, done: int, total: int, detail: str = "") -> None:
        capped_total = max(total, 1)
        capped_done = max(0, min(done, capped_total))
        now = time.perf_counter()
        elapsed = max(0.0, now - self._started_at)
        percent = capped_done / capped_total
        bucket = int(percent * 10)

        should_print = (
            capped_done == 1
            or capped_done >= capped_total
            or bucket > self._last_bucket
            or (now - self._last_print_at) >= self._min_interval_seconds
        )
        if not should_print or capped_done == self._last_done:
            return

        eta = float("inf")
        if capped_done > 0 and elapsed > 0:
            rate = capped_done / elapsed
            if rate > 0:
                eta = (capped_total - capped_done) / rate

        suffix = f" | {detail}" if detail else ""
        print(
            "    "
            f"{self._label}: {capped_done}/{capped_total} ({percent:.0%}) "
            f"| elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
            f"{suffix}"
        )
        self._last_print_at = now
        self._last_done = capped_done
        self._last_bucket = bucket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clio",
        description="CLIO-inspired pipeline for privacy-preserving AI conversation analysis",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show current configuration")
    validate_parser = sub.add_parser(
        "validate-input",
        help="Validate external conversation JSONL against the canonical input contract.",
    )
    validate_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to conversation JSONL. Defaults to configured input_conversations_path.",
    )
    validate_parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Maximum detailed line-level errors to retain in report output.",
    )
    validate_parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to write full validation report as JSON.",
    )

    run_parser = sub.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "--with-facets",
        action="store_true",
        help="Run Phase 2 facet extraction with OpenAI.",
    )
    run_parser.add_argument(
        "--with-clustering",
        action="store_true",
        help="Run Phase 3 embeddings and base k-means clustering (implies --with-facets).",
    )
    run_parser.add_argument(
        "--with-labeling",
        action="store_true",
        help="Run Phase 4 cluster labeling (implies --with-clustering).",
    )
    run_parser.add_argument(
        "--with-hierarchy",
        action="store_true",
        help="Run Phase 4 hierarchy scaffold (implies --with-labeling).",
    )
    run_parser.add_argument(
        "--with-privacy",
        action="store_true",
        help="Run Phase 5 privacy auditing and gating (implies --with-labeling).",
    )
    run_parser.add_argument(
        "--with-eval",
        action="store_true",
        help="Run Phase 6 synthetic evaluation harness and ablations.",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations processed for facets/clustering.",
    )
    run_parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable chunked streaming mode for input loading + phase2 facet extraction.",
    )
    run_parser.add_argument(
        "--stream-chunk-size",
        type=int,
        default=None,
        help="Chunk size used by --streaming (defaults to config stream_chunk_size).",
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier for runs/<run_id>/...",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run artifacts when available.",
    )
    run_parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero when run completes with warnings.",
    )
    run_parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict automation mode: implies --fail-on-warning.",
    )
    run_parser.add_argument(
        "--eval-count",
        type=int,
        default=None,
        help="Synthetic sample count for Phase 6 evaluation.",
    )

    return parser


def cmd_info(settings: Settings) -> None:
    langsmith = get_langsmith_status()

    print(f"ant-clio v{__version__}")
    print(f"  OpenAI model:     {settings.openai_model}")
    print(f"  Effective model:  {settings.resolved_openai_model()}")
    print(f"  OpenAI base URL:  {settings.resolved_openai_base_url() or '(default OpenAI)'}")
    print(f"  Key source:       {settings.resolved_openai_key_source()}")
    print(f"  OpenAI temp:      {settings.openai_temperature}")
    print(f"  OpenAI concurrency: {settings.openai_max_concurrency}")
    print(f"  Stream chunk size: {settings.stream_chunk_size}")
    print(f"  Client retries:   {settings.client_max_retries}")
    print(f"  Backoff seconds:  {settings.client_backoff_seconds}")
    print(f"  Embedding provider: {settings.embedding_provider}")
    print(f"  Embedding model:  {settings.embedding_model}")
    print(f"  Embed batch size: {settings.embedding_batch_size}")
    print(f"  Facet batch size: {settings.facet_batch_size}")
    print(f"  Facet concurrency: {settings.facet_max_concurrency}")
    print(f"  Label sample size: {settings.cluster_label_sample_size}")
    print(f"  Label concurrency: {settings.cluster_label_max_concurrency}")
    print(f"  Hierarchy top-k:  {settings.hierarchy_top_k}")
    print(f"  Hierarchy levels: {settings.hierarchy_levels}")
    print(f"  Target group size: {settings.hierarchy_target_group_size}")
    print(f"  Hierarchy label concurrency: {settings.hierarchy_label_max_concurrency}")
    print(f"  Viz projection:   {settings.viz_projection_method}")
    print(f"  Privacy threshold: {settings.privacy_threshold_min_rating}")
    print(f"  Privacy raw sample: {settings.privacy_audit_raw_sample_size}")
    print(f"  Privacy batch size: {settings.privacy_batch_size}")
    print(f"  Privacy concurrency: {settings.privacy_max_concurrency}")
    print(f"  Privacy validation: {settings.privacy_validation_enabled}")
    print(f"  Eval synthetic n: {settings.eval_synthetic_count}")
    print(f"  LangSmith tracing: {langsmith['enabled']}")
    print(f"  LangSmith endpoint: {langsmith['endpoint'] or '(not set)'}")
    print(f"  LangSmith project: {langsmith['project'] or '(not set)'}")
    print(f"  LangSmith key set: {langsmith['api_key_present']}")
    print(f"  Base clusters k:  {settings.k_base_clusters}")
    print(f"  Random seed:      {settings.random_seed}")
    print(f"  Input file:       {settings.input_conversations_path}")
    print(f"  Data dir:         {settings.data_dir}")
    print(f"  Output dir:       {settings.output_dir}")


def _cmd_run_unlocked(
    settings: Settings,
    args: argparse.Namespace,
    *,
    forced_run_id: str,
) -> None:
    fail_on_warning = bool(args.fail_on_warning or args.strict)
    resume_mode = bool(args.resume)
    streaming_mode = bool(args.streaming)
    stream_chunk_size = int(args.stream_chunk_size or settings.stream_chunk_size)
    if stream_chunk_size <= 0:
        print(f"Invalid stream chunk size: {stream_chunk_size}. Must be positive.")
        sys.exit(1)
    run_started_at_utc = datetime.now(UTC).isoformat()
    run_started_perf = time.perf_counter()
    phase_metrics: list[dict] = []

    def _record_phase_metric(
        *,
        phase: str,
        status: str,
        started_at: float | None = None,
        details: dict | None = None,
    ) -> None:
        metric = {
            "phase": phase,
            "status": status,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
        if started_at is not None:
            metric["duration_seconds"] = round(max(0.0, time.perf_counter() - started_at), 3)
        if details:
            metric.update(details)
        phase_metrics.append(metric)

    phase1_started = time.perf_counter()
    dataset_path = settings.input_conversations_path
    conversations = []

    run_fingerprint = build_run_fingerprint(
        settings,
        dataset_path=dataset_path,
        limit=args.limit,
    )

    try:
        if streaming_mode:
            run_id, run_root, summary = initialize_run_artifacts_streaming(
                settings=settings,
                dataset_path=dataset_path,
                chunk_size=stream_chunk_size,
                limit=args.limit,
                run_id=forced_run_id,
                run_fingerprint=run_fingerprint,
                enforce_resume_fingerprint=resume_mode,
            )
        else:
            conversations, summary, dataset_path = run_phase1_dataset_load(settings)
            run_id, run_root = initialize_run_artifacts(
                settings=settings,
                conversations=conversations,
                dataset_path=dataset_path,
                run_id=forced_run_id,
                run_fingerprint=run_fingerprint,
                enforce_resume_fingerprint=resume_mode,
            )
    except ConversationDatasetError as exc:
        print(f"Failed to load mock conversations: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"Run initialization failed: {exc}")
        sys.exit(1)

    print("Phase 1 complete: loaded mock conversation corpus.")
    print(f"  Run ID:             {run_id}")
    print(f"  Run directory:      {run_root}")
    print(f"  Dataset path:       {dataset_path}")
    print(f"  Conversations:      {summary.conversation_count}")
    print(f"  Unique users:       {summary.unique_user_count}")
    print(f"  Total messages:     {summary.message_count}")
    print(f"  Avg turns:          {summary.avg_turn_count:.2f}")
    print(f"  Min/Max turns:      {summary.min_turn_count}/{summary.max_turn_count}")
    if streaming_mode:
        print(f"  Streaming mode:     enabled (chunk_size={stream_chunk_size})")
    _record_phase_metric(
        phase="phase1_dataset_load",
        status="completed",
        started_at=phase1_started,
        details={
            "conversation_count": summary.conversation_count,
            "unique_user_count": summary.unique_user_count,
            "dataset_path": str(dataset_path),
            "streaming_mode": streaming_mode,
            "stream_chunk_size": stream_chunk_size if streaming_mode else None,
        },
    )

    should_run_hierarchy = bool(args.with_hierarchy)
    should_run_privacy = bool(args.with_privacy)
    should_run_labeling = bool(args.with_labeling or should_run_hierarchy or should_run_privacy)
    should_run_clustering = bool(args.with_clustering or should_run_labeling)
    should_run_facets = bool(args.with_facets or should_run_clustering)
    should_run_eval = bool(args.with_eval)
    run_warnings: list[str] = []

    def _warn(message: str) -> None:
        run_warnings.append(message)
        print("")
        print(f"WARNING: {message}")

    if should_run_privacy and not args.with_labeling:
        print("")
        print("Phase 5 privacy requested: enabling Phase 4 labeling automatically.")
    if should_run_hierarchy and not args.with_labeling:
        print("")
        print("Phase 4 hierarchy requested: enabling Phase 4 labeling automatically.")
    if should_run_labeling and not args.with_clustering:
        print("")
        print("Phase 4 labeling requested: enabling Phase 3 clustering automatically.")
    if args.with_clustering and not args.with_facets:
        print("")
        print("Phase 3 requested: enabling Phase 2 facet extraction automatically.")

    if not should_run_facets and not should_run_eval:
        print("")
        print("Use `clio run --with-facets` to execute Phase 2 facet extraction.")
        print("Use `clio run --with-facets --with-clustering` for Phase 3 clustering.")
        print("Use `clio run --with-labeling` for Phase 4 cluster labeling.")
        print("Use `clio run --with-hierarchy` for hierarchy scaffolding.")
        print("Use `clio run --with-privacy` for privacy auditing and gating.")
        print("Use `clio run --with-eval` for Phase 6 synthetic evaluation.")
        return

    facets = []
    phase2_started = time.perf_counter() if should_run_facets else None
    phase2_status = "skipped" if not should_run_facets else "pending"
    phase2_mode = "none"
    if should_run_facets:
        if resume_mode:
            facets = load_phase2_facets(run_root)
            if facets:
                print("")
                print("Phase 2 resumed from existing artifacts.")
                print(f"  Processed:          {len(facets)}")
                phase2_status = "resumed"
                phase2_mode = "resume"
        if not facets:
            try:
                print("")
                print("Phase 2 running: facet extraction in progress...")
                phase2_progress = _EtaProgressPrinter("phase2")
                if streaming_mode:
                    facets, run_root = run_phase2_facet_extraction_streaming(
                        settings=settings,
                        dataset_path=dataset_path,
                        run_id=run_id,
                        stream_chunk_size=stream_chunk_size,
                        limit=args.limit,
                        total_conversations=summary.conversation_count,
                        progress_callback=phase2_progress,
                    )
                else:
                    facets, run_root = run_phase2_facet_extraction(
                        settings=settings,
                        conversations=conversations,
                        run_id=run_id,
                        limit=args.limit,
                        progress_callback=phase2_progress,
                    )
                phase2_status = "completed" if facets else "failed"
                phase2_mode = "executed_streaming" if streaming_mode else "executed"
            except Exception as exc:
                _warn(f"Phase 2 failed: {exc}")
                facets = []
                phase2_status = "failed"
                phase2_mode = "executed_streaming" if streaming_mode else "executed"

            if facets:
                print("")
                print("Phase 2 complete: extracted conversation facets.")
                print(f"  Processed:          {len(facets)}")
                print(f"  Output directory:   {run_root}")
    _record_phase_metric(
        phase="phase2_facet_extraction",
        status=phase2_status,
        started_at=phase2_started,
        details={
            "mode": phase2_mode,
            "processed_count": len(facets),
        },
    )

    needs_conversations = should_run_clustering or should_run_labeling or should_run_privacy
    if needs_conversations and not conversations:
        try:
            conversations = load_conversations_jsonl(dataset_path)
            if args.limit is not None:
                conversations = conversations[: args.limit]
        except ConversationDatasetError as exc:
            _warn(f"Could not load conversations for downstream phases: {exc}")
            conversations = []

    phase3_started = time.perf_counter() if should_run_clustering else None
    phase3_status = "skipped" if not should_run_clustering else "pending"
    phase3_mode = "none"
    if not should_run_clustering:
        cluster_summaries = []
    else:
        cluster_summaries = []
        if resume_mode:
            cluster_summaries = load_phase3_cluster_summaries(run_root)
            if cluster_summaries:
                kept_count = sum(1 for item in cluster_summaries if item["kept_by_threshold"])
                print("")
                print("Phase 3 resumed from existing artifacts.")
                print(f"  Total clusters:     {len(cluster_summaries)}")
                print(f"  Kept clusters:      {kept_count}")
                phase3_status = "resumed"
                phase3_mode = "resume"
        if not cluster_summaries:
            if not facets:
                _warn("Phase 3 skipped: no facets available.")
                phase3_status = "skipped"
                phase3_mode = "missing_input"
            else:
                try:
                    print("")
                    print("Phase 3 running: embedding + clustering in progress...")
                    phase3_progress = _EtaProgressPrinter("phase3")
                    cluster_summaries, run_root = run_phase3_base_clustering(
                        settings=settings,
                        conversations=conversations,
                        facets=facets,
                        run_root=run_root,
                        progress_callback=phase3_progress,
                    )
                    phase3_status = "completed"
                    phase3_mode = "executed"
                except Exception as exc:
                    _warn(f"Phase 3 failed: {exc}")
                    cluster_summaries = []
                    phase3_status = "failed"
                    phase3_mode = "executed"

                if cluster_summaries:
                    kept_count = sum(1 for item in cluster_summaries if item["kept_by_threshold"])
                    print("")
                    print("Phase 3 complete: generated embeddings and base clusters.")
                    print(f"  Total clusters:     {len(cluster_summaries)}")
                    print(f"  Kept clusters:      {kept_count}")
                    print(f"  Output directory:   {run_root}")
    _record_phase_metric(
        phase="phase3_base_clustering",
        status=phase3_status,
        started_at=phase3_started,
        details={
            "mode": phase3_mode,
            "cluster_count": len(cluster_summaries) if should_run_clustering else 0,
        },
    )

    phase4_label_started = time.perf_counter() if should_run_labeling else None
    phase4_label_status = "skipped" if not should_run_labeling else "pending"
    phase4_label_mode = "none"
    if not should_run_labeling:
        labeled_clusters = []
    else:
        labeled_clusters = []
        if resume_mode:
            labeled_clusters = load_phase4_labeled_clusters(run_root)
            if labeled_clusters:
                print("")
                print("Phase 4 labeling resumed from existing artifacts.")
                print(f"  Labeled clusters:   {len(labeled_clusters)}")
                phase4_label_status = "resumed"
                phase4_label_mode = "resume"
        if not labeled_clusters:
            if not cluster_summaries:
                _warn("Phase 4 labeling skipped: no cluster summaries available.")
                phase4_label_status = "skipped"
                phase4_label_mode = "missing_input"
            else:
                try:
                    print("")
                    print("Phase 4 running: cluster labeling in progress...")
                    phase4_label_progress = _EtaProgressPrinter("phase4-label")
                    labeled_clusters, run_root = run_phase4_cluster_labeling(
                        settings=settings,
                        facets=facets,
                        cluster_summaries=cluster_summaries,
                        run_root=run_root,
                        conversations=conversations,
                        progress_callback=phase4_label_progress,
                    )
                    phase4_label_status = "completed"
                    phase4_label_mode = "executed"
                except Exception as exc:
                    _warn(f"Phase 4 labeling failed: {exc}")
                    labeled_clusters = []
                    phase4_label_status = "failed"
                    phase4_label_mode = "executed"

                if labeled_clusters:
                    print("")
                    print("Phase 4 complete: generated cluster labels.")
                    print(f"  Labeled clusters:   {len(labeled_clusters)}")
                    print(f"  Output directory:   {run_root}")
    _record_phase_metric(
        phase="phase4_cluster_labeling",
        status=phase4_label_status,
        started_at=phase4_label_started,
        details={
            "mode": phase4_label_mode,
            "labeled_cluster_count": len(labeled_clusters) if should_run_labeling else 0,
        },
    )

    phase4_hierarchy_started = time.perf_counter() if should_run_hierarchy else None
    phase4_hierarchy_status = "skipped" if not should_run_hierarchy else "pending"
    phase4_hierarchy_mode = "none"
    hierarchy = {}
    if should_run_hierarchy:
        hierarchy = {}
        if resume_mode:
            hierarchy = load_phase4_hierarchy(run_root)
            if hierarchy:
                print("")
                print("Phase 4 hierarchy resumed from existing artifacts.")
                print(f"  Top-level clusters: {hierarchy['top_level_cluster_count']}")
                print(f"  Leaf clusters:      {hierarchy['leaf_cluster_count']}")
                phase4_hierarchy_status = "resumed"
                phase4_hierarchy_mode = "resume"
        if not hierarchy:
            if not labeled_clusters:
                _warn("Phase 4 hierarchy skipped: no labeled clusters available.")
                phase4_hierarchy_status = "skipped"
                phase4_hierarchy_mode = "missing_input"
            else:
                try:
                    print("")
                    print("Phase 4 running: hierarchy construction in progress...")
                    phase4_hierarchy_progress = _EtaProgressPrinter("phase4-hierarchy")
                    hierarchy, run_root = run_phase4_hierarchy_scaffold(
                        settings=settings,
                        labeled_clusters=labeled_clusters,
                        run_root=run_root,
                        progress_callback=phase4_hierarchy_progress,
                    )
                    phase4_hierarchy_status = "completed"
                    phase4_hierarchy_mode = "executed"
                except Exception as exc:
                    _warn(f"Phase 4 hierarchy failed: {exc}")
                    hierarchy = {}
                    phase4_hierarchy_status = "failed"
                    phase4_hierarchy_mode = "executed"

                if hierarchy:
                    print("")
                    print("Phase 4 hierarchy complete.")
                    print(f"  Top-level clusters: {hierarchy['top_level_cluster_count']}")
                    print(f"  Leaf clusters:      {hierarchy['leaf_cluster_count']}")
                    print(f"  Output directory:   {run_root}")
    _record_phase_metric(
        phase="phase4_hierarchy_scaffold",
        status=phase4_hierarchy_status,
        started_at=phase4_hierarchy_started,
        details={
            "mode": phase4_hierarchy_mode,
            "top_level_cluster_count": int(hierarchy.get("top_level_cluster_count", 0))
            if isinstance(hierarchy, dict)
            else 0,
            "leaf_cluster_count": int(hierarchy.get("leaf_cluster_count", 0))
            if isinstance(hierarchy, dict)
            else 0,
        },
    )

    phase5_started = time.perf_counter() if should_run_privacy else None
    phase5_status = "skipped" if not should_run_privacy else "pending"
    phase5_mode = "none"
    privacy_summary = {}
    gated_clusters = []
    if should_run_privacy:
        privacy_summary = {}
        gated_clusters = []
        if resume_mode:
            privacy_summary, gated_clusters = load_phase5_outputs(run_root)
            if privacy_summary and gated_clusters:
                print("")
                print("Phase 5 resumed from existing artifacts.")
                print(f"  Cluster pass rate: {privacy_summary['cluster_summary']['pass_rate']:.2%}")
                phase5_status = "resumed"
                phase5_mode = "resume"
        if not privacy_summary or not gated_clusters:
            if not facets or not labeled_clusters:
                _warn("Phase 5 privacy skipped: missing facets or labeled clusters.")
                phase5_status = "skipped"
                phase5_mode = "missing_input"
            else:
                try:
                    print("")
                    print("Phase 5 running: privacy auditing in progress...")
                    phase5_progress = _EtaProgressPrinter("phase5")
                    privacy_summary, gated_clusters, run_root = run_phase5_privacy_audit(
                        settings=settings,
                        conversations=conversations,
                        facets=facets,
                        labeled_clusters=labeled_clusters,
                        run_root=run_root,
                        progress_callback=phase5_progress,
                    )
                    phase5_status = "completed"
                    phase5_mode = "executed"
                except Exception as exc:
                    _warn(f"Phase 5 privacy failed: {exc}")
                    privacy_summary = {}
                    gated_clusters = []
                    phase5_status = "failed"
                    phase5_mode = "executed"

                if privacy_summary and gated_clusters:
                    print("")
                    print("Phase 5 complete: privacy auditing and gating.")
                    print(
                        "  Cluster pass rate: "
                        f"{privacy_summary['cluster_summary']['pass_rate']:.2%}"
                    )
                    print(
                        "  Clusters kept:     "
                        f"{sum(1 for item in gated_clusters if item['final_kept'])}/"
                        f"{len(gated_clusters)}"
                    )
                    print(f"  Output directory:   {run_root}")
    _record_phase_metric(
        phase="phase5_privacy_audit",
        status=phase5_status,
        started_at=phase5_started,
        details={
            "mode": phase5_mode,
            "cluster_count": len(gated_clusters),
            "cluster_pass_rate": (
                float(privacy_summary.get("cluster_summary", {}).get("pass_rate", 0.0))
                if isinstance(privacy_summary, dict)
                else 0.0
            ),
        },
    )

    phase6_started = time.perf_counter() if should_run_eval else None
    phase6_status = "skipped" if not should_run_eval else "pending"
    phase6_mode = "none"
    eval_results = {}
    if should_run_eval:
        eval_results = {}
        if resume_mode:
            eval_results = load_phase6_evaluation(run_root)
            if eval_results:
                print("")
                print("Phase 6 resumed from existing artifacts.")
                privacy_metrics = eval_results["ablations"]["privacy_summary"]
                print(f"  Eval sample count:  {eval_results['synthetic_count']}")
                print(f"  Privacy-summary F1: {privacy_metrics['macro_f1']:.3f}")
                phase6_status = "resumed"
                phase6_mode = "resume"
        if not eval_results:
            if args.eval_count is not None:
                eval_count = args.eval_count
            else:
                eval_count = settings.eval_synthetic_count
            try:
                print("")
                print("Phase 6 running: synthetic evaluation in progress...")
                phase6_progress = _EtaProgressPrinter("phase6")
                eval_results, run_root = run_phase6_evaluation(
                    settings=settings,
                    run_root=run_root,
                    count=eval_count,
                    progress_callback=phase6_progress,
                )
                phase6_status = "completed"
                phase6_mode = "executed"
            except Exception as exc:
                _warn(f"Phase 6 evaluation failed: {exc}")
                eval_results = {}
                phase6_status = "failed"
                phase6_mode = "executed"

            if eval_results:
                print("")
                print("Phase 6 complete: synthetic evaluation and ablations.")
                print(f"  Eval sample count:  {eval_results['synthetic_count']}")
                print(
                    "  Privacy-summary acc: "
                    f"{eval_results['ablations']['privacy_summary']['accuracy']:.3f}"
                )
                print(
                    "  Raw-user-text acc:   "
                    f"{eval_results['ablations']['raw_user_text']['accuracy']:.3f}"
                )
                print(f"  Output directory:    {run_root}")
    _record_phase_metric(
        phase="phase6_evaluation",
        status=phase6_status,
        started_at=phase6_started,
        details={
            "mode": phase6_mode,
            "synthetic_count": int(eval_results.get("synthetic_count", 0))
            if isinstance(eval_results, dict)
            else 0,
        },
    )

    manifest: dict = {}
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            loaded_manifest = {}
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest

    phase_metric_events = [
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event_type": "phase_metric",
            **metric,
        }
        for metric in phase_metrics
    ]
    run_events_path = run_root / "run_events.jsonl"
    if phase_metric_events:
        append_jsonl(run_events_path, phase_metric_events)

    phase_llm_metrics = {
        "phase2_facet_extraction": manifest.get("phase2_openai_metrics", {}),
        "phase4_cluster_labeling": manifest.get("phase4_label_openai_metrics", {}),
        "phase4_hierarchy_scaffold": manifest.get("phase4_hierarchy_openai_metrics", {}),
        "phase5_privacy_audit": manifest.get("phase5_openai_metrics", {}),
    }
    aggregate_llm_usage = {
        "request_count": 0,
        "retry_count": 0,
        "schema_fallback_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": None,
    }
    for item in phase_llm_metrics.values():
        if not isinstance(item, dict):
            continue
        aggregate_llm_usage["request_count"] += int(item.get("request_count", 0) or 0)
        aggregate_llm_usage["retry_count"] += int(item.get("retry_count", 0) or 0)
        aggregate_llm_usage["schema_fallback_count"] += int(
            item.get("schema_fallback_count", 0) or 0
        )
        aggregate_llm_usage["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
        aggregate_llm_usage["completion_tokens"] += int(item.get("completion_tokens", 0) or 0)
        aggregate_llm_usage["total_tokens"] += int(item.get("total_tokens", 0) or 0)

    run_metrics = {
        "run_id": run_id,
        "started_at_utc": run_started_at_utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": round(max(0.0, time.perf_counter() - run_started_perf), 3),
        "warning_count": len(run_warnings),
        "warnings": run_warnings,
        "phase_metrics": phase_metrics,
        "phase_llm_metrics": phase_llm_metrics,
        "aggregate_llm_usage": aggregate_llm_usage,
        "checkpoint_resume_counts": {
            "phase2_facet_extraction": int(manifest.get("facet_resume_processed_count", 0) or 0),
            "phase4_cluster_labeling": int(
                manifest.get("cluster_label_resume_processed_count", 0) or 0
            ),
            "phase4_hierarchy_scaffold": int(
                manifest.get("hierarchy_label_resume_processed_count", 0) or 0
            ),
            "phase5_privacy_audit": int(manifest.get("privacy_resume_processed_count", 0) or 0),
        },
        "fallback_counts": {
            "phase2_facet_extraction_errors": int(
                manifest.get("facet_extraction_error_count", 0) or 0
            ),
            "phase4_cluster_labeling_fallbacks": int(
                manifest.get("cluster_label_fallback_count", 0) or 0
            ),
            "phase4_hierarchy_fallbacks": int(
                manifest.get("hierarchy_label_fallback_count", 0) or 0
            ),
            "phase5_privacy_audit": manifest.get("privacy_audit_fallback_counts", {}),
        },
    }
    metrics_path = save_json(run_root / "run_metrics.json", run_metrics)
    print(f"Run metrics saved:   {metrics_path}")

    if manifest:
        output_files = dict(manifest.get("output_files", {}))
        output_files["run_events_jsonl"] = str(run_events_path.as_posix())
        output_files["run_metrics_json"] = str(metrics_path.as_posix())
        manifest.update(
            {
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "output_files": output_files,
            }
        )
        save_json(manifest_path, manifest)

    if run_warnings:
        print("")
        print("Run finished with warnings:")
        for message in run_warnings:
            print(f"  - {message}")
        warnings_path = save_json(
            run_root / "run_warnings.json",
            {
                "run_id": run_id,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "warnings": run_warnings,
            },
        )
        print(f"  Warnings saved to:  {warnings_path}")
        if fail_on_warning:
            print("Strict failure mode active: exiting non-zero due to warnings.")
            sys.exit(2)


def cmd_run(settings: Settings, args: argparse.Namespace) -> None:
    """Run pipeline under an exclusive run-directory lock."""

    forced_run_id = args.run_id or generate_run_id()
    run_root = ensure_directory(settings.output_dir / forced_run_id)
    try:
        with run_lock(run_root):
            _cmd_run_unlocked(
                settings=settings,
                args=args,
                forced_run_id=forced_run_id,
            )
    except RunLockError as exc:
        print(f"Could not acquire run lock for run '{forced_run_id}': {exc}")
        sys.exit(3)


def cmd_validate_input(settings: Settings, args: argparse.Namespace) -> None:
    """Validate conversation input format and print a report."""

    input_path = args.input or str(settings.input_conversations_path)
    try:
        report = validate_conversations_jsonl(input_path, max_errors=args.max_errors)
    except ConversationDatasetError as exc:
        print(f"Input validation failed: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"Input validation configuration error: {exc}")
        sys.exit(1)

    print("Input validation complete.")
    print(f"  Schema version:     {report.schema_version}")
    print(f"  Input path:          {report.input_path}")
    print(f"  Total lines:         {report.total_lines}")
    print(f"  Non-empty lines:     {report.non_empty_lines}")
    print(f"  Valid conversations: {report.valid_conversation_count}")
    print(f"  Invalid lines:       {report.invalid_line_count}")
    print(f"  Duplicate IDs:       {report.duplicate_conversation_id_count}")
    print(f"  Unique users:        {report.summary.unique_user_count}")
    print(f"  Total messages:      {report.summary.message_count}")
    print(f"  Avg turns:           {report.summary.avg_turn_count:.2f}")

    if args.report_json:
        report_path = save_json(args.report_json, report.to_dict())
        print(f"  Report JSON:         {report_path}")

    if report.is_valid:
        print("Validation passed: input JSONL matches the pipeline contract.")
        return

    print("Validation failed: fix input errors before running the pipeline.")
    if report.errors:
        print("  Sample errors:")
        for item in report.errors[:5]:
            print(f"    - line {item.line_number} [{item.code}] {item.message}")
        if report.dropped_error_count > 0:
            print(
                "    - "
                f"... {report.dropped_error_count} additional errors omitted "
                f"(max-errors={args.max_errors})."
            )
    sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.from_yaml(args.config)

    if args.command == "info":
        cmd_info(settings)
    elif args.command == "validate-input":
        cmd_validate_input(settings, args)
    elif args.command == "run":
        cmd_run(settings, args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
