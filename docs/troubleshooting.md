# Troubleshooting Guide

## Quick Diagnostic

Run:

```bash
uv run clio doctor
```

Optional network probe:

```bash
uv run clio doctor --network-check
```

## Common Issues

### Input Validation Fails

- Symptom: `clio run` exits before Phase 1 with schema errors.
- Fix:
  - Validate directly: `uv run clio validate-input --input /path/to/file.jsonl`
  - Check contract: `docs/input_jsonl_contract.md`
  - Normalize external data first: `scripts/normalize_external_jsonl.py`

### Default Mock Dataset Missing

- Symptom: run or tests fail with `Conversation file does not exist: data/mock/conversations_llm_200.jsonl`.
- Cause: `data/` is ignored in git, so fresh clones might not include generated mock files.
- Fix:
  - Generate dataset locally:
    - `uv run clio-generate-mock-data --count 240 --output data/mock/conversations_llm_200.jsonl`
  - Or run with explicit external input:
    - `uv run clio run --input /path/to/conversations.jsonl ...`
  - Tests in this repo synthesize temporary datasets and should not depend on checked-in `data/`.

### Azure Endpoint / Key Mismatch

- Symptom: run fails with auth or model/deployment errors.
- Fix:
  - Verify `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`.
  - Verify `AZURE_OPENAI_DEPLOYMENT` if using Azure deployment naming.
  - Check `uv run clio info` output for resolved endpoint/model/key source.

### No Jina Key for Clustering

- Symptom: phase 3 fails while embedding.
- Fix:
  - Set `JINA_API_KEY` in `.env`.
  - Re-run from phase 3 using `--resume` after key is fixed.

### Resume Blocked by Fingerprint Drift

- Symptom: resume fails with fingerprint mismatch.
- Cause: dataset path/hash or key settings changed since original run.
- Fix:
  - Start a new run ID for changed input/config.
  - Or revert to prior dataset/config to resume safely.

### Long Runs / Unclear Progress

- Use CLI ETA logs per phase.
- Use `clio-ui-v2` to inspect run lifecycle, checkpoints, and events.
- For large data, use `--streaming --stream-chunk-size <N>`.

### Recoverable Warnings in Automation

- Use strict mode:
  - `uv run clio run ... --strict`
- This exits non-zero when warnings are present.

## Run Management

- List runs: `uv run clio list-runs --limit 30`
- Inspect run: `uv run clio inspect-run --run-id <run_id>`
- Dry-run prune: `uv run clio prune-runs --keep-last 30`
- Apply prune: `uv run clio prune-runs --keep-last 30 --yes`
