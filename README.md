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
6. Run Phase 1 mock-data load:
   - `uv run clio run`
7. Run Phase 2 facet extraction (requires `OPENAI_API_KEY`):
   - `uv run clio run --with-facets --limit 10`
8. Run Phase 3 embeddings + base clustering (requires `OPENAI_API_KEY` and `JINA_API_KEY`):
   - `uv run clio run --with-facets --with-clustering --limit 10`
9. Run Phase 4 cluster labeling:
   - `uv run clio run --with-labeling --limit 20`
10. Run Phase 4 hierarchy scaffold:

- `uv run clio run --with-hierarchy --limit 20`

11. Run Phase 5 privacy auditing and cluster gating:

- `uv run clio run --with-privacy --limit 20`

12. Run Phase 6 synthetic evaluation harness (recommended for quick checks with 20 records):

- `uv run clio run --with-eval --eval-count 20`

13. Resume from an existing run ID and skip completed phases:

- `uv run clio run --run-id <run_id> --with-privacy --with-eval --limit 20 --eval-count 20 --resume`

14. Full smoke run (all major phases) with 20-conversation cap:

- `uv run clio run --with-hierarchy --with-privacy --with-eval --limit 20 --eval-count 20`

15. Regenerate larger mock corpus (200+ records):

- `uv run clio-generate-mock-data --count 240 --output data/mock/conversations_llm_200.jsonl`

16. Generate mock corpus with OpenAI `gpt-5-nano`:

- `uv run clio-generate-mock-data --use-llm --llm-model gpt-5-nano --count 240 --output data/mock/conversations_llm_200.jsonl`

17. Validate artifacts for latest run without launching UI:

- `uv run clio-viz --check-only`

18. Launch visualization UI for a run:

- `uv run clio-viz --run-id <run_id>`

19. Launch UI with raw message preview enabled:

- `uv run clio-viz --run-id <run_id> --allow-raw-messages`

## Defaults

- OpenAI model: `gpt-4.1-mini`
- Embedding provider: `jina`
- Embedding model: `jina-embeddings-v3`
- Default input file: `data/mock/conversations_llm_200.jsonl`
- Run output root: `runs/`

## Notes

- `clio run` executes Phase 1 (load + validate mock conversations and print dataset stats).
- Every run creates `runs/<nanoid>/` and stores:
  - `conversation.jsonl` (messages-only snapshot; no user/timestamp/metadata fields)
  - `conversation.updated.jsonl` (messages + analysis enrichment as stages run)
  - `run_manifest.json`
  - `run_warnings.json` (present when recoverable phase-level warnings occur)
- `clio run --with-facets` executes Phase 2 and writes facets to `runs/<run_id>/facets/facets.jsonl`.
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
- `clio-viz --check-only` validates run availability/artifacts in non-interactive mode.
- Input format contract for external data sources is documented in `docs/input_jsonl_contract.md`.
- `clio validate-input` checks JSONL schema/integrity and can emit a machine-readable report JSON.
- Core LLM phases enforce OpenAI Structured Outputs (`json_schema` + `strict=true`) and still
  run with JSON-mode fallback if an endpoint rejects schema response_format.
- `clio-generate-mock-data --use-llm` uses OpenAI JSON generation with schema validation and template fallback.
- API clients include retry/backoff and runs can be resumed with `--resume`.
- Azure safeguard: when an Azure endpoint is configured, the pipeline requires `AZURE_OPENAI_API_KEY`
  and will not silently fall back to `OPENAI_API_KEY`.
- LangSmith safeguard: `.env` LangSmith keys are auto-hydrated into process env and OpenAI clients are
  LangSmith-wrapped when tracing is enabled.
- Graceful failure mode: per-item model failures are captured with fallback outputs and the run continues
  instead of halting the full pipeline.
- Full execution roadmap is in `docs/clio_implementation_plan.md`.
