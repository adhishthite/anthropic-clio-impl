import { promises as fs } from "node:fs";
import path from "node:path";

import type {
  ArtifactStatus,
  PhaseStatus,
  PhaseTimelineItem,
  RunDetailResponse,
  RunEventItem,
  RunListItem,
  RunState,
  RunVisualEvalAblation,
  RunVisualHierarchyNode,
  RunVisualMapCluster,
  RunVisualMapPoint,
  RunVisualPrivacyStage,
  RunVisualsResponse,
  RunVisualTopCluster,
} from "@/lib/clio-types";

const PHASE_ORDER = [
  "phase1_dataset_load",
  "phase2_facet_extraction",
  "phase3_base_clustering",
  "phase4_cluster_labeling",
  "phase4_hierarchy_scaffold",
  "phase5_privacy_audit",
  "phase6_evaluation",
] as const;

const PHASE_LABELS: Record<string, string> = {
  phase1_dataset_load: "Load dataset snapshot",
  phase2_facet_extraction: "Extract conversation facets",
  phase3_base_clustering: "Cluster similar conversations",
  phase4_cluster_labeling: "Label cluster themes",
  phase4_hierarchy_scaffold: "Build topic hierarchy",
  phase5_privacy_audit: "Apply privacy audit and filtering",
  phase6_evaluation: "Run synthetic quality evaluation",
};

const CHECKPOINT_PATHS: Record<string, string> = {
  phase2_facet_extraction: "facets/facet_checkpoint.json",
  phase4_cluster_labeling: "clusters/cluster_label_checkpoint.json",
  phase4_hierarchy_scaffold: "clusters/hierarchy_checkpoint.json",
  phase5_privacy_audit: "privacy/privacy_checkpoint.json",
};

const ARTIFACT_SPECS: ReadonlyArray<
  readonly [artifactKey: string, relativePath: string, required: boolean]
> = [
  ["run_manifest_json", "run_manifest.json", true],
  ["run_events_jsonl", "run_events.jsonl", false],
  ["run_metrics_json", "run_metrics.json", false],
  ["conversation_jsonl", "conversation.jsonl", true],
  ["conversation_updated_jsonl", "conversation.updated.jsonl", true],
  ["facets_jsonl", "facets/facets.jsonl", false],
  ["facets_errors_jsonl", "facets/facets_errors.jsonl", false],
  ["summary_embeddings_npy", "embeddings/summary_embeddings.npy", false],
  ["base_centroids_npy", "clusters/base_centroids.npy", false],
  ["base_assignments_jsonl", "clusters/base_assignments.jsonl", false],
  ["base_clusters_json", "clusters/base_clusters.json", false],
  ["labeled_clusters_json", "clusters/labeled_clusters.json", false],
  [
    "labeled_clusters_privacy_filtered_json",
    "clusters/labeled_clusters_privacy_filtered.json",
    false,
  ],
  ["hierarchy_json", "clusters/hierarchy.json", false],
  ["privacy_audit_json", "privacy/privacy_audit.json", false],
  ["phase6_metrics_json", "eval/phase6_metrics.json", false],
  ["phase6_report_md", "eval/report.md", false],
  ["viz_map_points_jsonl", "viz/map_points.jsonl", false],
  ["viz_map_clusters_json", "viz/map_clusters.json", false],
  ["tree_view_json", "viz/tree_view.json", false],
];

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as JsonRecord;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function asNodeIdArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const output: string[] = [];
  for (const item of value) {
    if (typeof item === "string" && item.trim()) {
      output.push(item);
      continue;
    }
    if (typeof item === "number" && Number.isFinite(item)) {
      output.push(String(Math.trunc(item)));
    }
  }
  return output;
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function toNullableNumber(value: unknown): number | null {
  const parsed = asNumber(value, Number.NaN);
  return Number.isNaN(parsed) ? null : parsed;
}

function toRoundedPercent(processed: number, total: number): number {
  if (!Number.isFinite(processed) || !Number.isFinite(total) || total <= 0) {
    return 0;
  }
  const raw = (processed / total) * 100;
  return Math.max(0, Math.min(100, Math.round(raw * 10) / 10));
}

function safeDateSortValue(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function normalizeRunsRoot(): string {
  const envRunsRoot = process.env.CLIO_RUNS_ROOT;
  if (typeof envRunsRoot === "string" && envRunsRoot.trim().length > 0) {
    return path.resolve(process.cwd(), envRunsRoot.trim());
  }
  return path.resolve(process.cwd(), "..", "runs");
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFile(filePath: string): Promise<JsonRecord | null> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return asRecord(JSON.parse(raw));
  } catch {
    return null;
  }
}

function isPidRunning(pid: number): boolean {
  if (!Number.isFinite(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function resolveRunLockState(runRoot: string): Promise<{
  active: boolean;
  payload: JsonRecord | null;
}> {
  const lockPath = path.join(runRoot, ".run.lock");
  const payload = await readJsonFile(lockPath);
  const lockFilePresent = payload !== null || (await fileExists(lockPath));
  if (!lockFilePresent) {
    return {
      active: false,
      payload: null,
    };
  }

  const lockOwnerPid = toNullableNumber(payload?.pid);
  if (lockOwnerPid && lockOwnerPid > 0) {
    return {
      active: isPidRunning(lockOwnerPid),
      payload,
    };
  }

  return {
    active: true,
    payload,
  };
}

async function readJsonlFile(
  filePath: string,
  limit = 400,
): Promise<JsonRecord[]> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const lines = raw.split(/\r?\n/);
    const output: JsonRecord[] = [];
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      const parsed = asRecord(JSON.parse(line));
      if (parsed) {
        output.push(parsed);
      }
      if (output.length >= limit) {
        break;
      }
    }
    return output;
  } catch {
    return [];
  }
}

async function buildArtifactStatus(runRoot: string): Promise<ArtifactStatus[]> {
  const checks = await Promise.all(
    ARTIFACT_SPECS.map(async ([artifactKey, relativePath, required]) => ({
      artifactKey,
      relativePath,
      required,
      exists: await fileExists(path.join(runRoot, relativePath)),
    })),
  );
  return checks;
}

function deriveRunState(params: {
  runLockActive: boolean;
  runMetrics: JsonRecord | null;
}): RunState {
  const { runLockActive, runMetrics } = params;
  if (runLockActive) {
    return "running";
  }

  const warningCount = asNumber(runMetrics?.warning_count, 0);
  const phaseMetrics = asArray(runMetrics?.phase_metrics)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));
  const hasFailedPhase = phaseMetrics.some(
    (metric) => asString(metric.status) === "failed",
  );

  if (hasFailedPhase) {
    return "failed";
  }
  if (warningCount > 0) {
    return "completed_with_warnings";
  }
  if (runMetrics) {
    return "completed";
  }
  return "partial";
}

function checkpointProgressForPhase(
  phase: string,
  checkpoint: JsonRecord | null,
): Pick<
  PhaseTimelineItem,
  | "processed"
  | "total"
  | "percent"
  | "note"
  | "updatedAtUtc"
  | "currentConcurrency"
> {
  if (!checkpoint) {
    return {
      processed: null,
      total: null,
      percent: null,
      note: "",
      updatedAtUtc: "",
      currentConcurrency: null,
    };
  }

  const note = asString(checkpoint.note);
  const updatedAtUtc = asString(checkpoint.updated_at_utc);
  const currentConcurrency = toNullableNumber(checkpoint.current_concurrency);

  if (phase === "phase2_facet_extraction") {
    const total = asNumber(checkpoint.conversation_count_total, 0);
    const processed = asNumber(checkpoint.conversation_count_processed, 0);
    return {
      processed,
      total,
      percent: total > 0 ? toRoundedPercent(processed, total) : null,
      note,
      updatedAtUtc,
      currentConcurrency,
    };
  }

  if (phase === "phase4_cluster_labeling") {
    const total = asNumber(checkpoint.cluster_total, 0);
    const processed = asNumber(checkpoint.cluster_processed, 0);
    return {
      processed,
      total,
      percent: total > 0 ? toRoundedPercent(processed, total) : null,
      note,
      updatedAtUtc,
      currentConcurrency,
    };
  }

  if (phase === "phase4_hierarchy_scaffold") {
    const processed = asNumber(checkpoint.label_checkpoint_count, 0);
    return {
      processed,
      total: null,
      percent: null,
      note,
      updatedAtUtc,
      currentConcurrency,
    };
  }

  if (phase === "phase5_privacy_audit") {
    const rawTotal = asNumber(checkpoint.raw_total, 0);
    const facetTotal = asNumber(checkpoint.facet_total, 0);
    const clusterTotal = asNumber(checkpoint.cluster_total, 0);
    const rawProcessed = asNumber(checkpoint.raw_processed, 0);
    const facetProcessed = asNumber(checkpoint.facet_processed, 0);
    const clusterProcessed = asNumber(checkpoint.cluster_processed, 0);
    const total = rawTotal + facetTotal + clusterTotal;
    const processed = rawProcessed + facetProcessed + clusterProcessed;
    return {
      processed,
      total,
      percent: total > 0 ? toRoundedPercent(processed, total) : null,
      note,
      updatedAtUtc,
      currentConcurrency,
    };
  }

  return {
    processed: null,
    total: null,
    percent: null,
    note,
    updatedAtUtc,
    currentConcurrency,
  };
}

function phaseStatusFromMetric(statusValue: string): PhaseStatus | null {
  if (
    statusValue === "pending" ||
    statusValue === "running" ||
    statusValue === "completed" ||
    statusValue === "resumed" ||
    statusValue === "failed" ||
    statusValue === "skipped"
  ) {
    return statusValue;
  }
  return null;
}

function derivePhaseTimeline(params: {
  manifest: JsonRecord;
  runLockActive: boolean;
  runMetrics: JsonRecord | null;
  checkpoints: Record<string, JsonRecord | null>;
}): PhaseTimelineItem[] {
  const { manifest, runLockActive, runMetrics, checkpoints } = params;
  const currentManifestPhase = asString(manifest.phase);
  const completedSet = new Set(asStringArray(manifest.completed_phases));
  const phaseMetrics = asArray(runMetrics?.phase_metrics)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));
  const metricStatusByPhase = new Map<string, PhaseStatus>();

  for (const metric of phaseMetrics) {
    const phase = asString(metric.phase);
    const parsedStatus = phaseStatusFromMetric(asString(metric.status));
    if (phase && parsedStatus) {
      metricStatusByPhase.set(phase, parsedStatus);
    }
  }

  return PHASE_ORDER.map((phase): PhaseTimelineItem => {
    const metricStatus = metricStatusByPhase.get(phase);
    const checkpoint = checkpoints[phase] ?? null;
    const progress = checkpointProgressForPhase(phase, checkpoint);
    let status: PhaseStatus = "pending";

    if (completedSet.has(phase)) {
      status = "completed";
    }
    if (metricStatus) {
      status = metricStatus;
    }
    if (
      runLockActive &&
      currentManifestPhase === phase &&
      !completedSet.has(phase)
    ) {
      status = "running";
    }

    return {
      phase,
      label: PHASE_LABELS[phase] ?? phase,
      status,
      processed: progress.processed,
      total: progress.total,
      percent: progress.percent,
      note: progress.note,
      updatedAtUtc: progress.updatedAtUtc,
      currentConcurrency: progress.currentConcurrency,
    };
  });
}

function deriveOverallProgressPercent(
  phaseTimeline: PhaseTimelineItem[],
): number {
  if (phaseTimeline.length === 0) {
    return 0;
  }

  const totalPhases = phaseTimeline.length;
  const completedCount = phaseTimeline.filter(
    (item) => item.status === "completed" || item.status === "resumed",
  ).length;
  const runningPhase = phaseTimeline.find((item) => item.status === "running");

  let progress = (completedCount / totalPhases) * 100;
  if (runningPhase?.percent !== null && runningPhase?.percent !== undefined) {
    progress += runningPhase.percent / totalPhases;
  }

  return Math.max(0, Math.min(100, Math.round(progress * 10) / 10));
}

function buildRunListItem(params: {
  runRoot: string;
  manifest: JsonRecord;
  runMetrics: JsonRecord | null;
  runLockActive: boolean;
}): RunListItem {
  const { runRoot, manifest, runMetrics, runLockActive } = params;
  const phaseTimeline = derivePhaseTimeline({
    manifest,
    runLockActive,
    runMetrics,
    checkpoints: {},
  });

  return {
    runId: asString(manifest.run_id) || path.basename(runRoot),
    runRoot,
    phase: asString(manifest.phase),
    createdAtUtc: asString(manifest.created_at_utc),
    updatedAtUtc: asString(manifest.updated_at_utc),
    completedPhases: asStringArray(manifest.completed_phases),
    conversationCountInput: asNumber(manifest.conversation_count_input, 0),
    conversationCountProcessed: asNumber(
      manifest.conversation_count_processed,
      0,
    ),
    clusterCountTotal: asNumber(manifest.cluster_count_total, 0),
    runLockActive,
    state: deriveRunState({ runLockActive, runMetrics }),
    overallProgressPercent: deriveOverallProgressPercent(phaseTimeline),
  };
}

async function listRunDirectories(runsRoot: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(runsRoot, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(runsRoot, entry.name));
  } catch {
    return [];
  }
}

function runEventToEvent(metric: JsonRecord, index: number): RunEventItem {
  const phase = asString(metric.phase);
  const status = asString(metric.status) || null;
  const eventType = asString(metric.event_type) || "event";
  const explicitMessage = asString(metric.message);
  const timestampUtc =
    asString(metric.timestamp_utc) || asString(metric.recorded_at_utc);

  let message = explicitMessage;
  if (!message && phase) {
    message = `${PHASE_LABELS[phase] ?? phase}${status ? ` • ${status}` : ""}`;
  }
  if (!message) {
    message = eventType.replace(/_/g, " ");
  }

  return {
    id: `run-event-${phase || eventType}-${index}-${timestampUtc || "none"}`,
    timestampUtc,
    source: "run_events",
    type: eventType,
    message,
    status,
  };
}

function checkpointToEvent(
  phase: string,
  checkpoint: JsonRecord,
): RunEventItem {
  const label = PHASE_LABELS[phase] ?? phase;
  const note = asString(checkpoint.note) || "Checkpoint updated";
  const status = checkpoint.completed ? "completed" : "running";
  const timestampUtc = asString(checkpoint.updated_at_utc);

  return {
    id: `checkpoint-${phase}-${timestampUtc || note}`,
    timestampUtc,
    source: "checkpoint",
    type: "checkpoint",
    message: `${label} • ${note}`,
    status,
  };
}

function lockToEvent(
  runLockActive: boolean,
  lockPayload: JsonRecord | null,
): RunEventItem | null {
  if (!runLockActive) {
    return null;
  }
  const acquiredAtUtc = asString(lockPayload?.acquired_at_utc);
  const ownerPid = asNumber(lockPayload?.pid, 0);
  return {
    id: `run-lock-${acquiredAtUtc || ownerPid}`,
    timestampUtc: acquiredAtUtc,
    source: "system",
    type: "run_lock",
    message:
      ownerPid > 0 ? `Run lock active (pid ${ownerPid})` : "Run lock active",
    status: "running",
  };
}

function buildLatestEvents(params: {
  runEvents: JsonRecord[];
  checkpoints: Record<string, JsonRecord | null>;
  runLockActive: boolean;
  runLockPayload: JsonRecord | null;
}): RunEventItem[] {
  const { runEvents, checkpoints, runLockActive, runLockPayload } = params;

  const phaseEvents = runEvents.map((item, index) =>
    runEventToEvent(item, index),
  );

  const checkpointEvents = Object.entries(checkpoints)
    .filter(([, checkpoint]) => Boolean(checkpoint))
    .map(([phase, checkpoint]) =>
      checkpointToEvent(phase, checkpoint as JsonRecord),
    );

  const lockEvent = lockToEvent(runLockActive, runLockPayload);
  const merged = lockEvent
    ? [...phaseEvents, ...checkpointEvents, lockEvent]
    : [...phaseEvents, ...checkpointEvents];

  merged.sort(
    (a, b) =>
      safeDateSortValue(b.timestampUtc) - safeDateSortValue(a.timestampUtc),
  );
  return merged.slice(0, 40);
}

export function getRunsRootPath(): string {
  return normalizeRunsRoot();
}

export async function discoverRuns(limit = 100): Promise<RunListItem[]> {
  const runsRoot = normalizeRunsRoot();
  const runDirs = await listRunDirectories(runsRoot);
  const rows = await Promise.all(
    runDirs.map(async (runRoot) => {
      const manifest = await readJsonFile(
        path.join(runRoot, "run_manifest.json"),
      );
      if (!manifest) {
        return null;
      }

      const runMetrics = await readJsonFile(
        path.join(runRoot, "run_metrics.json"),
      );
      const runLockState = await resolveRunLockState(runRoot);
      return buildRunListItem({
        runRoot,
        manifest,
        runMetrics,
        runLockActive: runLockState.active,
      });
    }),
  );

  const filtered = rows.filter((item): item is RunListItem => Boolean(item));
  filtered.sort((a, b) => {
    const updated =
      safeDateSortValue(b.updatedAtUtc) - safeDateSortValue(a.updatedAtUtc);
    if (updated !== 0) {
      return updated;
    }
    const created =
      safeDateSortValue(b.createdAtUtc) - safeDateSortValue(a.createdAtUtc);
    if (created !== 0) {
      return created;
    }
    return b.runId.localeCompare(a.runId);
  });

  if (limit <= 0) {
    return filtered;
  }
  return filtered.slice(0, limit);
}

export async function loadRunDetail(
  runId: string,
): Promise<RunDetailResponse | null> {
  const trimmedRunId = runId.trim();
  if (!trimmedRunId) {
    return null;
  }

  const runsRoot = normalizeRunsRoot();
  const runs = await discoverRuns(0);
  const selectedRun = runs.find((item) => item.runId === trimmedRunId);
  if (!selectedRun) {
    return null;
  }

  const manifest = await readJsonFile(
    path.join(selectedRun.runRoot, "run_manifest.json"),
  );
  if (!manifest) {
    return null;
  }

  const runMetrics = await readJsonFile(
    path.join(selectedRun.runRoot, "run_metrics.json"),
  );
  const runEvents = await readJsonlFile(
    path.join(selectedRun.runRoot, "run_events.jsonl"),
    500,
  );
  const runLockState = await resolveRunLockState(selectedRun.runRoot);
  const runLockPayload = runLockState.payload;
  const checkpointsEntries = await Promise.all(
    Object.entries(CHECKPOINT_PATHS).map(async ([phase, relativePath]) => [
      phase,
      await readJsonFile(path.join(selectedRun.runRoot, relativePath)),
    ]),
  );
  const checkpoints = Object.fromEntries(checkpointsEntries) as Record<
    string,
    JsonRecord | null
  >;
  const phaseTimeline = derivePhaseTimeline({
    manifest,
    runLockActive: runLockState.active,
    runMetrics,
    checkpoints,
  });
  const run: RunListItem = {
    ...selectedRun,
    runLockActive: runLockState.active,
    state: deriveRunState({
      runLockActive: runLockState.active,
      runMetrics,
    }),
    overallProgressPercent: deriveOverallProgressPercent(phaseTimeline),
  };

  const artifactStatus = await buildArtifactStatus(selectedRun.runRoot);
  const latestEvents = buildLatestEvents({
    runEvents,
    checkpoints,
    runLockActive: run.runLockActive,
    runLockPayload,
  });
  const warningCount = asNumber(runMetrics?.warning_count, 0);
  const requiredArtifactsMissing = artifactStatus.filter(
    (item) => item.required && !item.exists,
  ).length;
  const optionalArtifactsPresent = artifactStatus.filter(
    (item) => !item.required && item.exists,
  ).length;
  const lockOwnerPid = toNullableNumber(runLockPayload?.pid);
  const failedPhases = phaseTimeline.filter(
    (item) => item.status === "failed",
  ).length;
  const completedPhases = phaseTimeline.filter(
    (item) => item.status === "completed" || item.status === "resumed",
  ).length;

  return {
    generatedAtUtc: new Date().toISOString(),
    runsRoot,
    run,
    manifest,
    runMetrics,
    checkpoints,
    artifactStatus,
    phaseTimeline,
    latestEvents,
    summary: {
      totalPhases: phaseTimeline.length,
      completedPhases,
      failedPhases,
      requiredArtifactsMissing,
      optionalArtifactsPresent,
      warningCount,
      lockOwnerPid,
    },
  };
}

function boolValue(value: unknown): boolean {
  return value === true;
}

function mapPointFromRecord(params: {
  row: JsonRecord;
  clusterNames: Map<number, string>;
  clusterKept: Map<number, boolean>;
}): RunVisualMapPoint | null {
  const { row, clusterNames, clusterKept } = params;
  const clusterId = asNumber(row.cluster_id, Number.NaN);
  if (!Number.isFinite(clusterId)) {
    return null;
  }

  const clusterIdInt = Math.trunc(clusterId);
  const x = asNumber(row.x, Number.NaN);
  const y = asNumber(row.y, Number.NaN);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  const concerningScoreRaw = asNumber(row.concerning_score, Number.NaN);
  return {
    x,
    y,
    clusterId: clusterIdInt,
    clusterName: clusterNames.get(clusterIdInt) ?? `cluster-${clusterIdInt}`,
    kept:
      clusterKept.get(clusterIdInt) ??
      (boolValue(row.final_kept) || boolValue(row.kept_by_threshold)),
    language: asString(row.language),
    concerningScore: Number.isFinite(concerningScoreRaw)
      ? concerningScoreRaw
      : null,
  };
}

function mapClusterFromRecord(row: JsonRecord): RunVisualMapCluster | null {
  const clusterId = asNumber(row.cluster_id, Number.NaN);
  const x = asNumber(row.x, Number.NaN);
  const y = asNumber(row.y, Number.NaN);
  if (
    !Number.isFinite(clusterId) ||
    !Number.isFinite(x) ||
    !Number.isFinite(y)
  ) {
    return null;
  }

  return {
    clusterId: Math.trunc(clusterId),
    x,
    y,
    size: Math.max(0, Math.trunc(asNumber(row.size, 0))),
    kept: boolValue(row.final_kept) || boolValue(row.kept_by_threshold),
  };
}

function topClusterFromRecord(row: JsonRecord): RunVisualTopCluster | null {
  const id = asString(row.top_cluster_id);
  if (!id) {
    return null;
  }
  return {
    id,
    name: asString(row.name) || id,
    childCount: Math.max(0, Math.trunc(asNumber(row.child_count, 0))),
    description: asString(row.description),
  };
}

function hierarchyNodeFromRecord(params: {
  row: JsonRecord;
  parentIdByNode: Map<string, string>;
}): RunVisualHierarchyNode | null {
  const { row, parentIdByNode } = params;
  const id = asString(row.node_id);
  if (!id) {
    return null;
  }

  const level = Math.max(0, Math.trunc(asNumber(row.level, 0)));
  const size = Math.max(0, Math.trunc(asNumber(row.size, 0)));
  const sourceClusterIdRaw = asNumber(row.source_cluster_id, Number.NaN);
  const childIdsRaw =
    asNodeIdArray(row.child_ids).length > 0
      ? asNodeIdArray(row.child_ids)
      : asNodeIdArray(row.children);

  return {
    id,
    parentId: parentIdByNode.get(id) ?? null,
    level,
    name: asString(row.name) || id,
    description: asString(row.description),
    size,
    sourceClusterId: Number.isFinite(sourceClusterIdRaw)
      ? Math.trunc(sourceClusterIdRaw)
      : null,
    childIds: childIdsRaw,
  };
}

function privacyStageFromRecord(
  stageName: string,
  row: JsonRecord | null,
): RunVisualPrivacyStage | null {
  if (!row) {
    return null;
  }
  const total = Math.max(0, Math.trunc(asNumber(row.total, 0)));
  const threshold = Math.max(0, Math.trunc(asNumber(row.threshold, 0)));
  const passCount = Math.max(0, Math.trunc(asNumber(row.pass_count, 0)));
  const failCount = Math.max(0, Math.trunc(asNumber(row.fail_count, 0)));
  const passRate = Math.max(0, Math.min(1, asNumber(row.pass_rate, 0)));

  return {
    stage: stageName,
    total,
    threshold,
    passCount,
    failCount,
    passRate,
  };
}

function evalAblationFromRecord(
  name: string,
  row: JsonRecord | null,
): RunVisualEvalAblation | null {
  if (!row) {
    return null;
  }

  return {
    name,
    accuracy: Math.max(0, Math.min(1, asNumber(row.accuracy, 0))),
    macroF1: Math.max(0, Math.min(1, asNumber(row.macro_f1, 0))),
    weightedF1: Math.max(0, Math.min(1, asNumber(row.weighted_f1, 0))),
  };
}

export async function loadRunVisuals(
  runId: string,
): Promise<RunVisualsResponse | null> {
  const trimmedRunId = runId.trim();
  if (!trimmedRunId) {
    return null;
  }

  const runsRoot = normalizeRunsRoot();
  const runs = await discoverRuns(0);
  const selectedRun = runs.find((item) => item.runId === trimmedRunId);
  if (!selectedRun) {
    return null;
  }

  const runRoot = selectedRun.runRoot;
  const manifestPayload = await readJsonFile(
    path.join(runRoot, "run_manifest.json"),
  );
  const mapPointsRaw = await readJsonlFile(
    path.join(runRoot, "viz", "map_points.jsonl"),
    1600,
  );
  const mapClustersPayload = await readJsonFile(
    path.join(runRoot, "viz", "map_clusters.json"),
  );
  const mapClustersRaw = asArray(mapClustersPayload?.clusters)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));

  const privacyFiltered = await readJsonFile(
    path.join(runRoot, "clusters", "labeled_clusters_privacy_filtered.json"),
  );
  const labeledClusters = await readJsonFile(
    path.join(runRoot, "clusters", "labeled_clusters.json"),
  );
  const clusterRowsRaw =
    asArray(privacyFiltered?.clusters).length > 0
      ? asArray(privacyFiltered?.clusters)
      : asArray(labeledClusters?.clusters);
  const clusterRows = clusterRowsRaw
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));
  const clusterNames = new Map<number, string>();
  const clusterKept = new Map<number, boolean>();
  for (const row of clusterRows) {
    const clusterId = asNumber(row.cluster_id, Number.NaN);
    if (!Number.isFinite(clusterId)) {
      continue;
    }
    const normalizedId = Math.trunc(clusterId);
    const name = asString(row.name);
    if (name) {
      clusterNames.set(normalizedId, name);
    }
    const kept = boolValue(row.final_kept) || boolValue(row.kept_by_threshold);
    clusterKept.set(normalizedId, kept);
  }

  const mapPoints = mapPointsRaw
    .map((item) => mapPointFromRecord({ row: item, clusterNames, clusterKept }))
    .filter((item): item is RunVisualMapPoint => Boolean(item));
  const mapClusters = mapClustersRaw
    .map((item) => mapClusterFromRecord(item))
    .filter((item): item is RunVisualMapCluster => Boolean(item));

  const map =
    mapPoints.length > 0 || mapClusters.length > 0
      ? {
          projectionMethod:
            asString(mapClustersPayload?.projection_method_used) ||
            asString(mapClustersPayload?.projection_method_requested) ||
            null,
          totalPoints: mapPoints.length,
          sampled: false,
          points: mapPoints,
          clusters: mapClusters,
        }
      : null;

  const hierarchyPayload = await readJsonFile(
    path.join(runRoot, "clusters", "hierarchy.json"),
  );
  const treeViewPayload = await readJsonFile(
    path.join(runRoot, "viz", "tree_view.json"),
  );
  const hierarchyTopRaw = asArray(hierarchyPayload?.top_level_clusters)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));
  const topLevelClusters = hierarchyTopRaw
    .map((item) => topClusterFromRecord(item))
    .filter((item): item is RunVisualTopCluster => Boolean(item));
  const treeNodesRaw = asArray(treeViewPayload?.nodes)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));
  const treeEdgesRaw = asArray(treeViewPayload?.edges)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item));
  const parentIdByNode = new Map<string, string>();
  for (const edge of treeEdgesRaw) {
    const parentId = asString(edge.parent_id);
    const childId = asString(edge.child_id);
    if (!parentId || !childId) {
      continue;
    }
    parentIdByNode.set(childId, parentId);
  }
  const hierarchyNodes = treeNodesRaw
    .map((row) => hierarchyNodeFromRecord({ row, parentIdByNode }))
    .filter((item): item is RunVisualHierarchyNode => Boolean(item));
  const rootNodeIdsFromTree = asArray(treeViewPayload?.roots)
    .map((item) => asRecord(item))
    .filter((item): item is JsonRecord => Boolean(item))
    .map((item) => asString(item.node_id))
    .filter((item) => item.length > 0);
  const rootNodeIds =
    rootNodeIdsFromTree.length > 0
      ? rootNodeIdsFromTree
      : hierarchyNodes
          .filter((node) => node.parentId === null)
          .sort((a, b) => b.level - a.level)
          .map((node) => node.id);
  const generatedMaxLevel = Math.max(
    Math.trunc(asNumber(hierarchyPayload?.max_level, Number.NaN)),
    hierarchyNodes.reduce((max, node) => Math.max(max, node.level), -1),
  );
  const runFingerprint = asRecord(manifestPayload?.run_fingerprint);
  const requestedLevelsRaw = asNumber(
    manifestPayload?.hierarchy_levels ?? runFingerprint?.hierarchy_levels,
    Number.NaN,
  );
  const hierarchy =
    topLevelClusters.length > 0 || hierarchyNodes.length > 0
      ? {
          topLevelCount: Math.max(
            topLevelClusters.length,
            Math.trunc(asNumber(hierarchyPayload?.top_level_cluster_count, 0)),
          ),
          leafCount: Math.max(
            Math.trunc(asNumber(hierarchyPayload?.leaf_cluster_count, 0)),
            hierarchyNodes.filter((node) => node.level === 0).length,
          ),
          maxLevel:
            Number.isFinite(generatedMaxLevel) && generatedMaxLevel >= 0
              ? generatedMaxLevel
              : null,
          requestedLevels:
            Number.isFinite(requestedLevelsRaw) && requestedLevelsRaw >= 0
              ? Math.trunc(requestedLevelsRaw)
              : null,
          topLevelClusters,
          rootNodeIds,
          nodes: hierarchyNodes,
        }
      : null;

  const privacyPayload = await readJsonFile(
    path.join(runRoot, "privacy", "privacy_audit.json"),
  );
  const privacySummary = asRecord(privacyPayload?.summary);
  const privacyStages = [
    privacyStageFromRecord(
      "raw_conversation",
      asRecord(privacySummary?.raw_conversation),
    ),
    privacyStageFromRecord(
      "facet_summary",
      asRecord(privacySummary?.facet_summary),
    ),
    privacyStageFromRecord(
      "cluster_summary",
      asRecord(privacySummary?.cluster_summary),
    ),
  ].filter((item): item is RunVisualPrivacyStage => Boolean(item));
  const validation = asRecord(privacyPayload?.validation);
  const privacy =
    privacyStages.length > 0 || validation
      ? {
          stages: privacyStages,
          validation: validation
            ? {
                totalCases: Math.max(
                  0,
                  Math.trunc(asNumber(validation.total_cases, 0)),
                ),
                inRangeRate: Math.max(
                  0,
                  Math.min(1, asNumber(validation.in_range_rate, 0)),
                ),
                meanAbsoluteError: Math.max(
                  0,
                  asNumber(validation.mean_absolute_error, 0),
                ),
              }
            : null,
        }
      : null;

  const evalPayload = await readJsonFile(
    path.join(runRoot, "eval", "phase6_metrics.json"),
  );
  const ablationsRaw = asRecord(evalPayload?.ablations);
  const ablations = Object.entries(ablationsRaw ?? {})
    .map(([name, row]) => evalAblationFromRecord(name, asRecord(row)))
    .filter((item): item is RunVisualEvalAblation => Boolean(item));
  const evaluation =
    ablations.length > 0
      ? {
          syntheticCount: Math.max(
            0,
            Math.trunc(asNumber(evalPayload?.synthetic_count, 0)),
          ),
          topicCount: Math.max(
            0,
            Math.trunc(asNumber(evalPayload?.topic_count, 0)),
          ),
          languageCount: Math.max(
            0,
            Math.trunc(asNumber(evalPayload?.language_count, 0)),
          ),
          ablations,
        }
      : null;

  return {
    generatedAtUtc: new Date().toISOString(),
    runsRoot,
    runId: selectedRun.runId,
    map,
    hierarchy,
    privacy,
    evaluation,
  };
}
