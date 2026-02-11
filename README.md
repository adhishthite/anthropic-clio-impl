# ant-clio

CLIO-inspired Python pipeline for privacy-preserving analysis of AI conversation usage patterns.

## Status

Scaffolded project with config, core schemas, CLI entrypoints, tests, and an implementation plan in `docs/clio_implementation_plan.md`.

## Quick start

1. Install dependencies:
   - `uv sync`
2. Set environment variables in `.env`:
   - `OPENAI_API_KEY=...`
   - `JINA_API_KEY=...`
   - Azure optional:
     - `AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com`
     - `AZURE_OPENAI_API_KEY=...`
     - `AZURE_OPENAI_DEPLOYMENT=<deployment-name>` (optional, falls back to `openai_model`)
   - LangSmith optional:
     - `LANGSMITH_TRACING=true`
     - `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`
     - `LANGSMITH_API_KEY=...`
     - `LANGSMITH_PROJECT=<project-name>`
3. View current config:
   - `uv run clio info`
4. View CLI help:
   - `uv run clio --help`
5. Validate external input JSONL against contract:
   - `uv run clio validate-input --input /path/to/conversations.jsonl --report-json /tmp/validation_report.json`
6. Run doctor checks before first execution:
   - `uv run clio doctor`
7. Run with explicit external input (auto-validates before run):
   - `uv run clio run --input /path/to/conversations.jsonl --with-hierarchy --with-privacy --with-eval --limit 20 --eval-count 20`
8. Run Phase 1 mock-data load:
   - `uv run clio run`
9. Run Phase 2 facet extraction (requires `OPENAI_API_KEY`):
   - `uv run clio run --with-facets --limit 10`
10. Run Phase 3 embeddings + base clustering (requires `OPENAI_API_KEY` and `JINA_API_KEY`):
   - `uv run clio run --with-facets --with-clustering --limit 10`
11. Run Phase 4 cluster labeling:
   - `uv run clio run --with-labeling --limit 20`
12. Run Phase 4 hierarchy scaffold:

- `uv run clio run --with-hierarchy --limit 20`

13. Run Phase 5 privacy auditing and cluster gating:

- `uv run clio run --with-privacy --limit 20`

14. Run Phase 6 synthetic evaluation harness (recommended for quick checks with 20 records):

- `uv run clio run --with-eval --eval-count 20`

15. Resume from an existing run ID and skip completed phases:

- `uv run clio run --run-id <run_id> --with-privacy --with-eval --limit 20 --eval-count 20 --resume`

16. Full smoke run (all major phases) with 20-conversation cap:

- `uv run clio run --with-hierarchy --with-privacy --with-eval --limit 20 --eval-count 20`

17. Full smoke run in streaming mode (chunked Phase 1 + Phase 2):

- `uv run clio run --streaming --stream-chunk-size 32 --with-hierarchy --with-privacy --with-eval --limit 20 --eval-count 20`

18. List/inspect/prune run directories:

- `uv run clio list-runs --limit 30`
- `uv run clio inspect-run --run-id <run_id>`
- `uv run clio prune-runs --keep-last 30` (dry-run)
- `uv run clio prune-runs --keep-last 30 --yes` (apply)

19. Regenerate larger mock corpus (200+ records):

- `uv run clio-generate-mock-data --count 240 --output data/mock/conversations_llm_200.jsonl`

20. Generate mock corpus with OpenAI `gpt-5-nano`:

- `uv run clio-generate-mock-data --use-llm --llm-model gpt-5-nano --count 240 --output data/mock/conversations_llm_200.jsonl`

21. Validate artifacts for latest run without launching UI:

- `uv run clio-viz --check-only`

22. Launch visualization UI for a run:

- `uv run clio-viz --run-id <run_id>`

23. Launch UI with raw message preview enabled:

- `uv run clio-viz --run-id <run_id> --allow-raw-messages`

24. Launch UI with live auto-refresh enabled:

- `uv run clio-viz --run-id <run_id> --live --refresh-seconds 4`

25. Upload + validate + start runs from UI:

- Launch `clio-viz`, open the `Ingest & Run` tab, upload/select JSONL, validate, then start.

## Defaults

- OpenAI model: `gpt-4.1-mini`
- Embedding provider: `jina`
- Embedding model: `jina-embeddings-v3`
- Default input file: `data/mock/conversations_llm_200.jsonl`
- Run output root: `runs/`

## Notes

- `clio run` executes Phase 1 (load + validate mock conversations and print dataset stats).
- `clio run --input <path>` lets users run external datasets without editing config files.
- `clio run` now runs input validation automatically before Phase 1 (unless explicitly skipped).
- `clio run` prints rough preflight estimates (LLM calls/tokens/runtime/cost when pricing is configured).
- Every run creates `runs/<nanoid>/` and stores:
  - `conversation.jsonl` (messages-only snapshot; no user/timestamp/metadata fields)
  - `conversation.updated.jsonl` (messages + analysis enrichment as stages run)
  - `run_manifest.json`
  - `run_events.jsonl` (machine-readable phase event stream)
  - `run_metrics.json` (machine-readable phase metrics + usage rollup)
  - `run_warnings.json` (present when recoverable phase-level warnings occur)
  - `.run.lock` (ephemeral lock while a run is active)
- `clio run --fail-on-warning` exits non-zero when recoverable warnings occurred.
- `clio run --strict` implies `--fail-on-warning` for CI/automation safety.
- `clio run --with-facets` executes Phase 2 and writes facets to `runs/<run_id>/facets/facets.jsonl`.
- `clio run --streaming` enables chunked input loading and streaming facet extraction.
- `clio run --stream-chunk-size <N>` controls chunk size used by streaming mode.
- `clio doctor` validates local setup (keys, paths, endpoint config, optional network reachability).
- `clio list-runs`, `clio inspect-run`, and `clio prune-runs` provide run lifecycle management.
- Phase 2 facet extraction supports async batching with:
  - `facet_batch_size` (default `8`)
  - `facet_max_concurrency` (default `8`)
  - Adaptive wave scaling (ramps down on batch issues, ramps up on healthy waves)
  - Mid-phase checkpoints in `runs/<run_id>/facets/`:
    - `facets.partial.jsonl`
    - `facets_errors.partial.jsonl`
    - `facet_checkpoint.json`
- Phase 4 labeling/hierarchy supports parallel label generation with:
  - `cluster_label_max_concurrency` (default `8`)
  - `hierarchy_label_max_concurrency` (default `8`)
  - Adaptive concurrency for both cluster and hierarchy labeling
  - Mid-phase checkpoints in `runs/<run_id>/clusters/`:
    - `cluster_label_checkpoint.json`
    - `labeled_clusters.partial.jsonl`
    - `hierarchy_checkpoint.json`
    - `hierarchy_label_groups.partial.jsonl`
- Phase 5 privacy auditing supports async batching with:
  - `privacy_batch_size` (default `12`)
  - `privacy_max_concurrency` (default `8`)
  - Adaptive wave scaling (same ramp-down/ramp-up behavior as Phase 2)
  - Mid-phase checkpoints in `runs/<run_id>/privacy/`:
    - `raw_conversation.partial.jsonl`
    - `facet_summary.partial.jsonl`
    - `cluster_summary.partial.jsonl`
    - `batch_errors.partial.jsonl`
    - `privacy_checkpoint.json`
- `clio run --with-facets --with-clustering` executes Phase 3 and writes:
  - `runs/<run_id>/embeddings/summary_embeddings.npy`
  - `runs/<run_id>/clusters/base_assignments.jsonl`
  - `runs/<run_id>/clusters/base_clusters.json`
- `clio run --with-labeling` executes Phase 4 cluster labeling and writes:
  - `runs/<run_id>/clusters/labeled_clusters.json`
- `clio run --with-hierarchy` executes Phase 4 hierarchy scaffold and writes:
  - `runs/<run_id>/clusters/hierarchy.json`
- `clio run --with-privacy` executes Phase 5 privacy audits and writes:
  - `runs/<run_id>/privacy/privacy_audit.json`
  - `runs/<run_id>/clusters/labeled_clusters_privacy_filtered.json`
- `clio run --with-eval` executes Phase 6 synthetic evaluation and writes:
  - `runs/<run_id>/eval/phase6_metrics.json`
  - `runs/<run_id>/eval/synthetic_conversations.jsonl`
  - `runs/<run_id>/eval/report.md`
- Phase 3 now exports map-ready artifacts:
  - `runs/<run_id>/viz/map_points.jsonl`
  - `runs/<run_id>/viz/map_clusters.json`
- Phase 4 hierarchy now also exports:
  - `runs/<run_id>/viz/tree_view.json`
- `clio-viz` provides local UI pages for overview, map, hierarchy, privacy, evaluation, and artifacts.
- `clio-viz` overview supports live checkpoint progress, recent run events, and run-lock status.
- `clio-viz` includes an `Ingest & Run` tab for upload, contract validation, background run launch,
  job monitoring, and log tail viewing.
- UI live mode can be toggled from the sidebar, or defaulted via `--live`.
- `clio-viz --check-only` validates run availability/artifacts in non-interactive mode.
- Input format contract for external data sources is documented in `docs/input_jsonl_contract.md`.
- Source extraction/normalization guidance for Mongo/Elastic/Postgres is in
  `docs/source_adapter_recipes.md`.
- `clio validate-input` checks JSONL schema/integrity and can emit a machine-readable report JSON.
- Input validation now reports `schema_version=1.0.0`; top-level conversation objects reject
  unknown fields outside the canonical contract.
- Core LLM phases enforce OpenAI Structured Outputs (`json_schema` + `strict=true`) and still
  run with JSON-mode fallback if an endpoint rejects schema response_format.
- `clio-generate-mock-data --use-llm` uses OpenAI JSON generation with schema validation and template fallback.
- API clients include retry/backoff and runs can be resumed with `--resume`.
- Resume safety uses a manifest fingerprint (dataset hash + key config knobs) and blocks unsafe
  `--resume` when input/model/concurrency drift is detected.
- Azure safeguard: when an Azure endpoint is configured, the pipeline requires `AZURE_OPENAI_API_KEY`
  and will not silently fall back to `OPENAI_API_KEY`.
- LangSmith safeguard: `.env` LangSmith keys are auto-hydrated into process env and OpenAI clients are
  LangSmith-wrapped when tracing is enabled.
- Graceful failure mode: per-item model failures are captured with fallback outputs and the run continues
  instead of halting the full pipeline.
- Troubleshooting playbook is available at `docs/troubleshooting.md`.
- Docker packaging is included via `Dockerfile` for containerized CLI usage.
- Basic CI is included at `.github/workflows/ci.yml` (ruff + pytest + phase1 smoke run).
- Full execution roadmap is in `docs/clio_implementation_plan.md`.
