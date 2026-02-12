import { NextResponse } from "next/server";

import { persistUploadedInput, startBackgroundRun } from "@/lib/clio-run-jobs";
import { getRunsRootPath } from "@/lib/clio-runs";
import type { RunLaunchOptions, RunLaunchResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_OPTIONS: RunLaunchOptions = {
  withFacets: true,
  withClustering: true,
  withLabeling: true,
  withHierarchy: true,
  withPrivacy: true,
  withEval: true,
  streaming: false,
  streamChunkSize: 32,
  hierarchyLevels: 5,
  hierarchyDepthPolicy: "adaptive",
  clusterStrategy: "hybrid",
  clusterLeafMode: "auto",
  clusterTargetLeafSize: 25,
  clusterMinLeafClusters: 20,
  clusterMaxLeafClusters: 600,
  clusterHdbscanMinClusterSize: 12,
  clusterHdbscanMinSamples: 6,
  clusterNoisePolicy: "nearest",
  strict: false,
  limit: null,
  evalCount: null,
};

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as JsonRecord;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true" || normalized === "1" || normalized === "yes") {
      return true;
    }
    if (normalized === "false" || normalized === "0" || normalized === "no") {
      return false;
    }
  }
  return fallback;
}

function asNullableInt(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  const normalized = Math.floor(parsed);
  return normalized > 0 ? normalized : null;
}

function asIntInRange(
  value: unknown,
  bounds: { fallback: number; min: number; max: number },
): number {
  const { fallback, min, max } = bounds;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  const normalized = Math.floor(parsed);
  if (normalized < min) {
    return min;
  }
  if (normalized > max) {
    return max;
  }
  return normalized;
}

function asOneOf<T extends string>(
  value: unknown,
  allowed: readonly T[],
  fallback: T,
): T {
  if (typeof value !== "string") {
    return fallback;
  }
  const normalized = value.trim();
  return allowed.includes(normalized as T) ? (normalized as T) : fallback;
}

function parseOptions(
  raw: FormDataEntryValue | null,
): Partial<RunLaunchOptions> {
  if (typeof raw !== "string" || !raw.trim()) {
    return DEFAULT_OPTIONS;
  }
  try {
    const parsed = asRecord(JSON.parse(raw));
    if (!parsed) {
      return DEFAULT_OPTIONS;
    }
    return {
      withFacets: asBoolean(parsed.withFacets, DEFAULT_OPTIONS.withFacets),
      withClustering: asBoolean(
        parsed.withClustering,
        DEFAULT_OPTIONS.withClustering,
      ),
      withLabeling: asBoolean(
        parsed.withLabeling,
        DEFAULT_OPTIONS.withLabeling,
      ),
      withHierarchy: asBoolean(
        parsed.withHierarchy,
        DEFAULT_OPTIONS.withHierarchy,
      ),
      withPrivacy: asBoolean(parsed.withPrivacy, DEFAULT_OPTIONS.withPrivacy),
      withEval: asBoolean(parsed.withEval, DEFAULT_OPTIONS.withEval),
      streaming: asBoolean(parsed.streaming, DEFAULT_OPTIONS.streaming),
      streamChunkSize:
        asNullableInt(parsed.streamChunkSize) ??
        DEFAULT_OPTIONS.streamChunkSize,
      hierarchyLevels: asIntInRange(parsed.hierarchyLevels, {
        fallback: DEFAULT_OPTIONS.hierarchyLevels,
        min: 2,
        max: 20,
      }),
      hierarchyDepthPolicy: asOneOf(
        parsed.hierarchyDepthPolicy,
        ["adaptive", "strict_min"] as const,
        DEFAULT_OPTIONS.hierarchyDepthPolicy,
      ),
      clusterStrategy: asOneOf(
        parsed.clusterStrategy,
        ["kmeans", "hdbscan", "hybrid"] as const,
        DEFAULT_OPTIONS.clusterStrategy,
      ),
      clusterLeafMode: asOneOf(
        parsed.clusterLeafMode,
        ["fixed", "auto"] as const,
        DEFAULT_OPTIONS.clusterLeafMode,
      ),
      clusterTargetLeafSize: asIntInRange(parsed.clusterTargetLeafSize, {
        fallback: DEFAULT_OPTIONS.clusterTargetLeafSize,
        min: 1,
        max: 5000,
      }),
      clusterMinLeafClusters: asIntInRange(parsed.clusterMinLeafClusters, {
        fallback: DEFAULT_OPTIONS.clusterMinLeafClusters,
        min: 1,
        max: 5000,
      }),
      clusterMaxLeafClusters: asIntInRange(parsed.clusterMaxLeafClusters, {
        fallback: DEFAULT_OPTIONS.clusterMaxLeafClusters,
        min: 1,
        max: 10000,
      }),
      clusterHdbscanMinClusterSize: asIntInRange(
        parsed.clusterHdbscanMinClusterSize,
        {
          fallback: DEFAULT_OPTIONS.clusterHdbscanMinClusterSize,
          min: 2,
          max: 5000,
        },
      ),
      clusterHdbscanMinSamples: asIntInRange(parsed.clusterHdbscanMinSamples, {
        fallback: DEFAULT_OPTIONS.clusterHdbscanMinSamples,
        min: 1,
        max: 2000,
      }),
      clusterNoisePolicy: asOneOf(
        parsed.clusterNoisePolicy,
        ["nearest", "singleton", "drop"] as const,
        DEFAULT_OPTIONS.clusterNoisePolicy,
      ),
      strict: asBoolean(parsed.strict, DEFAULT_OPTIONS.strict),
      limit: asNullableInt(parsed.limit),
      evalCount: asNullableInt(parsed.evalCount),
    };
  } catch {
    return DEFAULT_OPTIONS;
  }
}

function withNoStoreHeaders(
  payload: RunLaunchResponse | { error: string },
  status = 200,
) {
  return NextResponse.json(payload, {
    status,
    headers: {
      "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
      Pragma: "no-cache",
      Expires: "0",
    },
  });
}

export async function POST(
  request: Request,
): Promise<NextResponse<RunLaunchResponse | { error: string }>> {
  try {
    const formData = await request.formData();
    const runsRoot = getRunsRootPath();
    const requestedRunIdValue = formData.get("runId");
    const requestedRunId =
      typeof requestedRunIdValue === "string" ? requestedRunIdValue : "";
    const options = parseOptions(formData.get("options"));

    const uploaded = formData.get("file");
    if (!(uploaded instanceof File) || uploaded.size <= 0) {
      throw new Error("Upload a JSONL file before starting a run.");
    }
    const persisted = await persistUploadedInput(uploaded, runsRoot);

    const job = await startBackgroundRun({
      inputPath: persisted.inputPath,
      requestedRunId,
      options,
      runsRoot,
    });

    return withNoStoreHeaders(
      {
        generatedAtUtc: new Date().toISOString(),
        runsRoot,
        job,
      },
      201,
    );
  } catch (error) {
    return withNoStoreHeaders(
      {
        error: error instanceof Error ? error.message : "Failed to launch run.",
      },
      400,
    );
  }
}
