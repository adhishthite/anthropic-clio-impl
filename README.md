# ant-clio

CLIO-inspired Python pipeline for privacy-preserving analysis of AI conversation usage patterns.

`ant-clio` gives you two ways to work:

- Terminal-first workflow (`clio`) for automation and CI
- Visual workflow (`clio-viz`) for upload, validation, run launch, and live monitoring

## Pipeline At A Glance

```text
input.jsonl
   |
   v
validate-input (schema + integrity)
   |
   v
phase1: load + snapshot
   |
   v
phase2: facet extraction (LLM, async, checkpoints)
   |
   v
phase3: embeddings + base clustering
   |
   v
phase4: cluster labels + multi-level hierarchy
   |
   v
phase5: privacy audit + gating
   |
   v
phase6: synthetic evaluation
   |
   v
runs/<run_id>/ artifacts + metrics + warnings + events
```

## Quick Start (5 Minutes)

1. Install dependencies

```bash
uv sync
```

1. Configure `.env`

- Required for full pipeline:
  - `OPENAI_API_KEY=...` or Azure settings
  - `JINA_API_KEY=...`
- Optional:
  - LangSmith (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, etc.)

1. Check setup health

```bash
uv run clio doctor
```

1. Run with your own JSONL

```bash
uv run clio run \
  --input /path/to/conversations.jsonl \
  --with-hierarchy --with-privacy --with-eval \
  --limit 20 --eval-count 20
```

If you are on a fresh clone and want demo input first:

```bash
uv run clio-generate-mock-data \
  --count 240 \
  --output data/mock/conversations_llm_200.jsonl
```

## UI Workflow (Upload + Run)

Start UI:

```bash
uv run clio-viz --live --refresh-seconds 4
```

Open:

- `http://127.0.0.1:8501`

In UI:

1. Go to `Ingest & Run`
2. Upload JSONL (max upload size: 50 MB)
3. Save upload
4. Validate input
5. Start run
6. Watch logs and status in the same page

Note: auto-refresh is intentionally paused on `Ingest & Run` to avoid interrupting uploads.

## CLI Recipes

| Goal                        | Command                                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Show resolved config        | `uv run clio info`                                                                                                          |
| Validate input              | `uv run clio validate-input --input /path/to/file.jsonl --report-json /tmp/validation.json`                                 |
| Full run (all major phases) | `uv run clio run --with-hierarchy --with-privacy --with-eval --limit 20 --eval-count 20`                                    |
| Streaming mode              | `uv run clio run --streaming --stream-chunk-size 32 --with-hierarchy --with-privacy --with-eval --limit 20 --eval-count 20` |
| Strict CI mode              | `uv run clio run ... --strict`                                                                                              |
| Resume run                  | `uv run clio run --run-id <run_id> --resume --with-privacy --with-eval`                                                     |
| List runs                   | `uv run clio list-runs --limit 30`                                                                                          |
| Inspect run                 | `uv run clio inspect-run --run-id <run_id>`                                                                                 |
| Prune old runs (dry-run)    | `uv run clio prune-runs --keep-last 30`                                                                                     |
| Prune old runs (apply)      | `uv run clio prune-runs --keep-last 30 --yes`                                                                               |

## What Gets Produced

Each run writes to `runs/<nanoid>/`:

- `conversation.jsonl` (messages-only snapshot)
- `conversation.updated.jsonl` (messages + analysis enrichment)
- `run_manifest.json`
- `run_events.jsonl` (phase event stream)
- `run_metrics.json` (phase metrics + usage rollup)
- `run_warnings.json` (only when warnings exist)
- `.run.lock` (ephemeral while run is active)

Phase-specific outputs are under:

- `facets/`
- `embeddings/`
- `clusters/`
- `privacy/`
- `eval/`
- `viz/`

## Defaults

- OpenAI model: `gpt-4.1-mini`
- Embedding provider: `jina`
- Embedding model: `jina-embeddings-v3`
- Default input path: `data/mock/conversations_llm_200.jsonl`
- Run output root: `runs/`

## Important Behavior

- `clio run --input <path>` lets you run external datasets without editing config files.
- Input validation is automatic before runs (unless `--skip-input-validation` is set).
- Run preflight prints rough estimates (LLM calls/tokens/runtime/cost if pricing is configured).
- `clio run --fail-on-warning` and `--strict` support safe automation semantics.
- Resume is guarded by input/config fingerprint checks to prevent unsafe drift.
- Structured Outputs are enforced (`json_schema` + `strict=true`) with JSON fallback when needed.

## Data, Tests, And Fresh Clones

- `data/` is not committed by default.
- If `data/mock/conversations_llm_200.jsonl` is missing, generate it:
  - `uv run clio-generate-mock-data --count 240 --output data/mock/conversations_llm_200.jsonl`
- Tests are self-contained and synthesize temporary mock datasets, so they do not require checked-in
  data files.

Run quality checks:

```bash
uv run ruff check
uv run pytest -q
```

## Docker

Build:

```bash
docker build -t ant-clio .
```

Run CLI help:

```bash
docker run --rm ant-clio --help
```

## More Docs

- Input contract: `docs/input_jsonl_contract.md`
- Source adapters (Mongo/Elastic/Postgres): `docs/source_adapter_recipes.md`
- Troubleshooting: `docs/troubleshooting.md`
- Full implementation roadmap: `docs/clio_implementation_plan.md`
