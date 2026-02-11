# CLIO UI v2

`clio-ui-v2` is a Next.js + ShadCN run workspace for CLIO outputs.

It reads run artifacts from disk and follows a two-level information flow:

- home page as a lightweight blank slate
- one dedicated dashboard per run (`/runs/[runId]`)
- plain-language stage cards with tooltips for each pipeline step
- per-phase progress timeline from checkpoints
- live event feed and stream health
- user-friendly artifact names with inline help (instead of internal keys)
- map, hierarchy, privacy, and evaluation visuals
- artifact readiness and run lifecycle summaries
- launch-time hierarchy depth control (`2..20`, clamped)

## Local Development

Run from this directory:

```bash
bun run dev
```

Open `http://localhost:3000`.

## Data Source

The UI server reads run folders from:

- `CLIO_RUNS_ROOT` env var (if set)
- otherwise defaults to `../runs` relative to `clio-ui-v2`

This matches the current CLI output convention (`output_dir: runs`).

## Endpoints

- `GET /api/runs` - run list with summarized state
- `GET /api/runs/[runId]` - full detail payload for one run
- `GET /api/runs/[runId]/visuals` - chart-ready visual payload for one run
- `POST /api/runs/launch` - launch a background CLI run from uploaded JSONL input
  - accepts hierarchy depth override via `options.hierarchyLevels` (`2..20`)
- `GET /api/runs/jobs` - list UI-launched background jobs
- `POST /api/runs/jobs/[runId]/terminate` - send termination signal
- `GET /api/runs/jobs/[runId]/logs` - fetch run log tail
- `GET /api/stream/runs` - SSE stream for run list snapshots
- `GET /api/stream/runs/[runId]` - SSE stream for one run detail
- `GET /api/stream/jobs` - SSE stream for jobs + selected run log tail

All endpoints are `no-store` and designed for live refresh/streaming use.

## Quality Checks

```bash
bun run lint
bun run build
```

## Near-Term UX Roadmap

1. Add drill-down tabs and cross-linking between timeline events and artifacts.
2. Add run-to-run comparison for phase duration, privacy pass rates, and ablations.
