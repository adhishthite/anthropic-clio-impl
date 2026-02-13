#!/usr/bin/env python3
"""Promote partial Phase 2 facet results to final, enabling Phase 3+ to proceed.

Usage:
    uv run python scripts/promote_partial_phase2.py <run_id> [--runs-root runs]

What it does:
    1. Copies facets.partial.jsonl -> facets.jsonl
    2. Copies facets_errors.partial.jsonl -> facets_errors.jsonl (if present)
    3. Marks facet_checkpoint.json as completed
    4. Adds "phase2_facet_extraction" to completed_phases in run_manifest.json
    5. Updates conversation.updated.jsonl with facet data
"""

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote partial Phase 2 to final.")
    parser.add_argument("run_id", help="Run ID to promote")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    args = parser.parse_args()

    run_root = Path(args.runs_root) / args.run_id
    facets_dir = run_root / "facets"
    partial_path = facets_dir / "facets.partial.jsonl"
    final_path = facets_dir / "facets.jsonl"
    errors_partial = facets_dir / "facets_errors.partial.jsonl"
    errors_final = facets_dir / "facets_errors.jsonl"
    checkpoint_path = facets_dir / "facet_checkpoint.json"
    manifest_path = run_root / "run_manifest.json"

    # --- Validations ---
    if not run_root.exists():
        print(f"ERROR: Run directory not found: {run_root}", file=sys.stderr)
        sys.exit(1)

    if not partial_path.exists():
        print(f"ERROR: No partial facets file: {partial_path}", file=sys.stderr)
        sys.exit(1)

    if final_path.exists():
        print(
            "ERROR: Final facets.jsonl already exists - phase2 may already be complete.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Count records in partial
    with open(partial_path, encoding="utf-8") as f:
        facet_count = sum(1 for line in f if line.strip())

    if facet_count == 0:
        print("ERROR: Partial file is empty - nothing to promote.", file=sys.stderr)
        sys.exit(1)

    print(f"Run:            {args.run_id}")
    print(f"Partial facets: {facet_count} records")
    print()

    # --- 1. Copy partial to final ---
    shutil.copy2(partial_path, final_path)
    print(f"[1/4] Copied facets.partial.jsonl -> facets.jsonl ({facet_count} records)")

    # --- 2. Copy error partial if present ---
    error_count = 0
    if errors_partial.exists():
        with open(errors_partial, encoding="utf-8") as f:
            error_count = sum(1 for line in f if line.strip())
        if error_count > 0:
            shutil.copy2(errors_partial, errors_final)
            print(
                f"[2/4] Copied facets_errors.partial.jsonl -> facets_errors.jsonl"
                f" ({error_count} errors)"
            )
        else:
            print("[2/4] Error partial is empty - skipped")
    else:
        print("[2/4] No error partial file - skipped")

    # --- 3. Update checkpoint ---
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        checkpoint = {"phase": "phase2_facet_extraction", "run_id": args.run_id}

    checkpoint["completed"] = True
    checkpoint["note"] = "promoted_from_partial"
    checkpoint["facet_count_success"] = facet_count
    checkpoint["error_count_recorded"] = error_count
    checkpoint["updated_at_utc"] = datetime.now(UTC).isoformat()
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    print("[3/4] Updated checkpoint: completed=true")

    # --- 4. Update manifest ---
    if not manifest_path.exists():
        print(
            f"WARNING: No manifest found at {manifest_path} - creating minimal one",
            file=sys.stderr,
        )
        manifest: dict = {"run_id": args.run_id}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase2_facet_extraction")
    output_files = dict(manifest.get("output_files", {}))
    output_files["facets_jsonl"] = str(final_path.as_posix())
    output_files["facet_checkpoint_json"] = str(checkpoint_path.as_posix())
    output_files["facets_partial_jsonl"] = str(partial_path.as_posix())
    if error_count > 0:
        output_files["facets_errors_jsonl"] = str(errors_final.as_posix())

    manifest.update(
        {
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase2_facet_extraction",
            "completed_phases": sorted(completed_phases),
            "conversation_count_processed": facet_count,
            "facet_extraction_error_count": error_count,
            "phase2_promoted_from_partial": True,
            "output_files": output_files,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] Updated manifest: completed_phases={sorted(completed_phases)}")

    print()
    print(f"Done. Phase 2 promoted with {facet_count} facets.")
    print(
        f"You can now resume with:"
        f" uv run clio run --resume --run-id {args.run_id} --with-clustering"
    )


if __name__ == "__main__":
    main()
