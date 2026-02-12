# CLIO-Style Prototype Plan (Python + Jina Embeddings + GPT-5.2)

## Project intent

Implement a Python prototype inspired by Anthropic's CLIO paper:

- preserve privacy while analyzing AI conversation usage patterns
- discover bottom-up clusters of use and misuse
- provide aggregated, explorable outputs (not raw conversation review)

This plan targets a hybrid of:

- **Core prototype (a):** end-to-end pipeline on mocked data
- **Near-paper elements (b):** synthetic reconstruction + privacy evaluations
- **Product elements (c):** clean package structure, CLI, tests, docs

## Model and platform choices

- **LLM:** OpenAI GPT-5.2 (for facet extraction, labeling, hierarchy, auditing)
- **Embeddings:** Jina embeddings API
- **Language:** Python

## Architecture overview

```mermaid
flowchart LR
  inputData[InputConversationsJSONL] --> preprocess[Preprocessor]
  preprocess --> facets[FacetExtractionGPT52]
  facets --> summaryGuard[SummaryPrivacyGuard]
  summaryGuard --> embeddings[JinaEmbeddings]
  embeddings --> baseClusters[KMeansBaseClustering]
  baseClusters --> thresholds[AggregationThresholds]
  thresholds --> labels[ClusterLabelingGPT52]
  labels --> hierarchy[Hierarchizer]
  hierarchy --> privacyAudit[PrivacyAuditorGPT52]
  privacyAudit --> reports[JSONCSVMarkdownReports]
  reports --> explorer[MapAndTreeExplorer]
```

## CLIO-inspired pipeline we will build

1. **Facet extraction**
   - Extract conversation-level facets:
     - request/topic summary
     - task
     - language
     - turn count (computed)
     - optional concerning score (1-5)
   - Use prompts that explicitly suppress PII/proper nouns in summary facets.

2. **Semantic clustering**
   - Embed selected text facets with Jina embeddings.
   - Run k-means for base clusters (configurable `k`, reproducible seed).

3. **Cluster description**
   - Use GPT-5.2 to generate concise cluster names + short descriptions.

4. **Hierarchization**
   - Build higher-level cluster hierarchy iteratively:
     - embed cluster names/descriptions
     - cluster parents
     - assign children to best parent
     - relabel parent clusters

5. **Privacy layers (defense-in-depth)**
   - summary-level suppression of PII
   - cluster minimum thresholds (`min_unique_users`, `min_conversations`)
   - post-cluster privacy auditor scoring (1-5)
   - filtering/hiding cluster outputs below policy threshold

6. **Reporting and exploration**
   - export metrics and artifacts
   - simple map projection + hierarchy tree output

## Repository plan

```text
ant-clio/
  src/clio_pipeline/
    __init__.py
    config.py
    schemas.py
    io/
      load.py
      save.py
    prompts/
      facet_prompts.py
      label_prompts.py
      hierarchy_prompts.py
      privacy_prompts.py
    models/
      openai_client.py
      jina_client.py
    pipeline/
      preprocess.py
      facet_extraction.py
      embedding.py
      clustering.py
      labeling.py
      hierarchy.py
      privacy_audit.py
      evaluate.py
      run_pipeline.py
    viz/
      projection.py
      tree_export.py
    cli.py
  data/
    mock/
    synth/
  outputs/
  configs/
    default.yaml
    eval.yaml
  tests/
    test_schemas.py
    test_clustering.py
    test_privacy_policy.py
    test_pipeline_smoke.py
  docs/
    clio_implementation_plan.md
```

## Phase-by-phase implementation plan

### Phase 0 - Bootstrap and standards

- Initialize package and dependency baseline.
- Add `.env`-driven configuration and typed settings.
- Define run IDs and deterministic output paths (`outputs/<run_id>/...`).

Deliverables:

- runnable CLI entrypoint (`python -m clio_pipeline ...`)
- configuration schema + sample config files

### Phase 1 - Data schema + mocked corpora

- Define canonical conversation schema:
  - `conversation_id`, `user_id`, `timestamp`, `messages[]`, optional metadata
- Create mocked conversations (30-60):
  - coding, writing, education, business, multilingual, roleplay
  - benign and concerning examples
  - controlled fake PII examples for privacy testing
- Add synthetic data generator for larger eval datasets.

Deliverables:

- `data/mock/conversations.jsonl`
- `data/synth/` generator scripts and generated fixtures

### Phase 2 - Facet extraction with GPT-5.2

- Implement extraction jobs for each facet.
- Parse and validate model outputs into strict schema.
- Save both raw model outputs and normalized facet tables.

Deliverables:

- facet outputs in `outputs/<run_id>/facets/`
- extraction logs + basic retry/error handling

### Phase 3 - Embeddings + base clustering

- Implement Jina embedding client wrapper and batching.
- Compute embeddings for selected facets.
- Run k-means clustering with reproducibility and metrics.
- Enforce minimum aggregation thresholds.

Deliverables:

- base cluster assignments + centroids
- dropped/kept cluster accounting after thresholds

### Phase 4 - Cluster labeling + hierarchy

- Generate cluster titles/descriptions from representative examples.
- Build iterative hierarchy from base clusters to top clusters.
- Re-name parent clusters after assignments.

Deliverables:

- `clusters_base.json`
- `clusters_hierarchy.json`
- hierarchy quality diagnostics (size distribution, depth)

### Phase 5 - Privacy auditing

- Implement a 1-5 privacy rubric auditor prompt (model-based).
- Audit outputs at staged points:
  - raw conversation snippets (sampled benchmark only)
  - post-summary facets
  - final cluster descriptions
- Gate final outputs by policy threshold.

Deliverables:

- privacy score tables and stage comparison plots/data
- filtered final cluster outputs

### Phase 6 - Evaluation (near-paper style)

- Synthetic reconstruction benchmark:
  - known ground-truth topic distribution
  - accuracy + macro/weighted F1
- Multilingual breakdown by language.
- Ablations:
  - privacy prompt on/off
  - direct raw-text clustering vs facet clustering
  - parameter sweep for `k` and thresholds

Deliverables:

- `eval_metrics.json`
- `eval_report.md`

### Phase 7 - Exploration outputs + docs

- Export map coordinates (UMAP) for cluster visualization.
- Export hierarchy tree for drilldown.
- Produce a concise run report in markdown with:
  - top clusters
  - language differences
  - concerning-cluster signals
  - privacy benchmark

Deliverables:

- `report.md` per run
- map/tree data files

### Phase 8 - Hardening and tests

- Add unit tests for schema, parsing, clustering determinism, policy gates.
- Add integration smoke test for a tiny pipeline run.
- Finalize README with setup and usage.

Deliverables:

- stable CI-local test pass
- reproducible one-command demo run

## Outputs we will generate

Per run (`outputs/<run_id>/`):

- `facets/*.jsonl`
- `embeddings/*.npy` or `.parquet`
- `clusters_base.json`
- `clusters_hierarchy.json`
- `privacy_audit.json`
- `metrics.json`
- `report.md`

## Evaluation and success criteria

Success for V1:

- End-to-end execution from mocked input to final reports.
- Clear privacy improvement across stages.
- Synthetic reconstruction significantly above naive/random baseline.
- Deterministic outputs under fixed seed.
- Secrets managed only via environment variables.

## Configuration baseline

Key configuration fields:

- `openai_model`: `gpt-4.1-mini`
- `embedding_provider`: `jina`
- `embedding_model`: configurable Jina model ID
- `k_base_clusters`
- `min_unique_users`
- `min_conversations_per_cluster`
- `privacy_threshold_min_rating`
- `random_seed`

## Security and privacy guardrails

- Never commit secrets (`.env` ignored).
- Keep only aggregated outputs by default.
- Preserve optional secure mode to disable any trace exports.
- Separate "research insights mode" and "safety investigation mode" in config.

## Known limitations (to document in README/report)

- Cluster quality depends on extraction prompt quality.
- K-means can be suboptimal for non-spherical semantic clusters.
- Privacy auditor is model-based (high utility, not formal guarantee).
- Rare behaviors may not form detectable clusters.
- Conclusions are dataset- and model-specific.

## Prerequisites for implementation

- `OPENAI_API_KEY`
- `JINA_API_KEY`
- Confirm exact deployable model names in your account:
  - GPT-5.2 identifier
  - Jina embeddings model identifier

## Suggested execution order

1. Build schema + mock data.
2. Implement facet extraction.
3. Add embedding + base clustering.
4. Add labeling + hierarchy.
5. Add privacy auditor and gating.
6. Add evaluation harness.
7. Add reports + docs + tests.

This order yields usable intermediate artifacts quickly while keeping risk low.

## Implementation status update (Feb 2026)

Completed in codebase:

- Phase 1-5 complete with run-folder artifacts under `runs/<run_id>/`.
- Phase 6 evaluation harness implemented:
  - synthetic ground-truth conversation generation
  - reconstruction metrics (accuracy, macro/weighted F1)
  - per-language metrics
  - ablations for privacy-summary vs non-private-summary vs raw-user-text
  - outputs in `runs/<run_id>/eval/phase6_metrics.json` and `report.md`
- Medium priority delivered:
  - hierarchy upgraded to iterative multi-level scaffold
  - map and tree visualization artifacts exported (`viz/map_points.jsonl`, `viz/map_clusters.json`, `viz/tree_view.json`)
  - facet expansion with language confidence and turn/message statistics
- Hardening/Ops delivered:
  - client retries/backoff for OpenAI and Jina calls
  - checkpoint/resume support via CLI `--resume` and phase output loaders
  - partial-failure recovery for facet extraction with `facets_errors.jsonl`
  - privacy auditor validation set and consistency scoring

Testing notes:

- Automated tests pass (`40 passed`).
- End-to-end smoke run validated with `--limit 20 --eval-count 20`.

Visualization module:

- Streamlit `clio-viz` UI retired. Use `clio-ui-v2` for run exploration UI.
- UI can target a run ID and display overview, map, hierarchy, privacy, evaluation, and artifact status pages.
- Added non-interactive `--check-only` mode for CI-friendly run validation.
