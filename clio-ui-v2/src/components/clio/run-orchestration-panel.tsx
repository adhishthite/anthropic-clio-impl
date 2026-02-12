"use client";

import {
  AlertTriangle,
  FileUp,
  Loader2,
  Play,
  Square,
  TerminalSquare,
  Workflow,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { StreamTimelineDrawer } from "@/components/clio/stream-timeline-drawer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  type SseStreamHealth,
  type SseStreamState,
  useSseStream,
} from "@/hooks/use-sse-stream";
import type {
  RunJobRecord,
  RunJobsResponse,
  RunLaunchOptions,
  RunLaunchResponse,
  RunLogResponse,
  RunTerminateResponse,
} from "@/lib/clio-types";
import { cn } from "@/lib/utils";

type RunOrchestrationPanelProps = {
  autoRefresh: boolean;
  onRunStarted: (runId: string) => void;
};
const STREAM_REFRESH_INTERVAL_MS = 30000;

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

type JsonError = {
  error: string;
};

type JobsStreamPayload = {
  generatedAtUtc: string;
  runsRoot: string;
  jobs: RunJobRecord[];
  logData: RunLogResponse | null;
};

type BooleanOptionKey =
  | "withFacets"
  | "withClustering"
  | "withLabeling"
  | "withHierarchy"
  | "withPrivacy"
  | "withEval"
  | "streaming"
  | "strict";

const PIPELINE_OPTION_ROWS: Array<{
  key: Extract<
    BooleanOptionKey,
    | "withFacets"
    | "withClustering"
    | "withLabeling"
    | "withHierarchy"
    | "withPrivacy"
    | "withEval"
  >;
  label: string;
  detail: string;
}> = [
  {
    key: "withFacets",
    label: "Facet extraction (Phase 2)",
    detail:
      "Extract structured conversation attributes used for downstream grouping.",
  },
  {
    key: "withClustering",
    label: "Semantic clustering (Phase 3)",
    detail:
      "Embed conversations and cluster them into recurring usage patterns.",
  },
  {
    key: "withLabeling",
    label: "Cluster description (Phase 4a)",
    detail:
      "Generate concise cluster names and descriptions for interpretation.",
  },
  {
    key: "withHierarchy",
    label: "Hierarchy building (Phase 4b)",
    detail:
      "Organize related clusters into a multi-level exploration structure.",
  },
  {
    key: "withPrivacy",
    label: "Privacy audit and gating (Phase 5)",
    detail:
      "Audit raw, facet, and cluster outputs, then apply privacy thresholds.",
  },
  {
    key: "withEval",
    label: "Evaluation harness (Phase 6)",
    detail: "Run synthetic evaluation and ablations for quality tracking.",
  },
];

const EXECUTION_OPTION_ROWS: Array<{
  key: Extract<BooleanOptionKey, "streaming" | "strict">;
  label: string;
  detail: string;
}> = [
  {
    key: "streaming",
    label: "Streaming mode",
    detail: "Chunked ingest and phase2 extraction for larger datasets.",
  },
  {
    key: "strict",
    label: "Strict mode",
    detail: "Treat warnings as failures for safer automation behavior.",
  },
];

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, {
    method: "GET",
    cache: "no-store",
    signal,
  });
  const payload = (await response.json()) as T | JsonError;
  if (!response.ok) {
    if (typeof payload === "object" && payload && "error" in payload) {
      throw new Error(payload.error);
    }
    throw new Error(`Request failed (${response.status})`);
  }
  return payload as T;
}

function parseStreamPayload<T>(event: MessageEvent<string>): T | null {
  try {
    return JSON.parse(event.data) as T;
  } catch {
    return null;
  }
}

function streamStateLabel(state: SseStreamState): string {
  if (state === "connected") {
    return "Connected";
  }
  if (state === "connecting") {
    return "Connecting";
  }
  if (state === "reconnecting") {
    return "Reconnecting";
  }
  if (state === "disconnected") {
    return "Disconnected";
  }
  return "Idle";
}

function streamStateBadgeVariant(
  state: SseStreamState,
): "default" | "secondary" | "outline" | "destructive" {
  if (state === "connected") {
    return "secondary";
  }
  if (state === "connecting") {
    return "default";
  }
  if (state === "reconnecting") {
    return "outline";
  }
  if (state === "disconnected") {
    return "destructive";
  }
  return "outline";
}

function streamHint(health: SseStreamHealth): string {
  if (health.state === "reconnecting" && health.nextRetryMs !== null) {
    return `retry in ${Math.max(1, Math.ceil(health.nextRetryMs / 1000))}s`;
  }
  if (health.lastMessageAt) {
    return `last msg ${formatDateTime(health.lastMessageAt)}`;
  }
  return "waiting for stream data";
}

function formatDateTime(value: string): string {
  if (!value) {
    return "n/a";
  }
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

function statusBadgeVariant(status: RunJobRecord["status"]) {
  if (status === "running") {
    return "default" as const;
  }
  if (status === "finished_ok") {
    return "secondary" as const;
  }
  return "outline" as const;
}

function statusLabel(status: RunJobRecord["status"]): string {
  if (status === "running") {
    return "Running";
  }
  if (status === "finished_ok") {
    return "Completed";
  }
  if (status === "finished_with_warnings") {
    return "Completed with warnings";
  }
  return "Stopped early";
}

function statusHint(job: RunJobRecord): string {
  if (job.status === "running") {
    return "Process or run lock is still active.";
  }
  if (job.status === "finished_ok") {
    return "Run metrics were written successfully.";
  }
  if (job.status === "finished_with_warnings") {
    return "Run metrics exist and warnings were recorded.";
  }
  return "Process exited before final metrics were written.";
}

function parseNullableInt(value: string): number | null {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  const normalized = Math.floor(parsed);
  return normalized > 0 ? normalized : null;
}

function parseIntInRange(
  value: string,
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

export function RunOrchestrationPanel({
  autoRefresh,
  onRunStarted,
}: RunOrchestrationPanelProps) {
  const [requestedRunId, setRequestedRunId] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [limitInput, setLimitInput] = useState<string>("");
  const [evalCountInput, setEvalCountInput] = useState<string>("");
  const [streamChunkSizeInput, setStreamChunkSizeInput] =
    useState<string>("32");
  const [hierarchyLevelsInput, setHierarchyLevelsInput] = useState<string>("5");
  const [clusterTargetLeafSizeInput, setClusterTargetLeafSizeInput] =
    useState<string>("25");
  const [clusterMinLeafClustersInput, setClusterMinLeafClustersInput] =
    useState<string>("20");
  const [clusterMaxLeafClustersInput, setClusterMaxLeafClustersInput] =
    useState<string>("600");
  const [
    clusterHdbscanMinClusterSizeInput,
    setClusterHdbscanMinClusterSizeInput,
  ] = useState<string>("12");
  const [clusterHdbscanMinSamplesInput, setClusterHdbscanMinSamplesInput] =
    useState<string>("6");
  const [options, setOptions] = useState<RunLaunchOptions>(DEFAULT_OPTIONS);

  const [launching, setLaunching] = useState<boolean>(false);
  const [launchError, setLaunchError] = useState<string>("");

  const [jobsData, setJobsData] = useState<RunJobsResponse | null>(null);
  const [jobsError, setJobsError] = useState<string>("");
  const [jobsLoading, setJobsLoading] = useState<boolean>(true);

  const [selectedLogRunId, setSelectedLogRunId] = useState<string>("");
  const [logData, setLogData] = useState<RunLogResponse | null>(null);
  const [logError, setLogError] = useState<string>("");
  const [logLoading, setLogLoading] = useState<boolean>(false);

  const setBooleanOption = useCallback(
    (key: BooleanOptionKey, checked: boolean) => {
      setOptions((current) => ({ ...current, [key]: checked }));
    },
    [],
  );

  const enabledPipelineStages = useMemo(
    () => PIPELINE_OPTION_ROWS.filter((option) => options[option.key]).length,
    [options],
  );

  const inputSummary = useMemo(() => {
    if (selectedFile) {
      return `Uploaded file: ${selectedFile.name}`;
    }
    return "Upload a JSONL file to continue";
  }, [selectedFile]);

  const runModeSummary = useMemo(() => {
    const mode = options.streaming ? "Streaming" : "Batch";
    const strict = options.strict ? "strict" : "non-strict";
    return `${mode}, ${strict}`;
  }, [options.streaming, options.strict]);

  const loadJobs = useCallback(async (signal?: AbortSignal) => {
    setJobsLoading(true);
    try {
      const payload = await fetchJson<RunJobsResponse>(
        "/api/runs/jobs?limit=120",
        signal,
      );
      setJobsData(payload);
      setJobsError("");
      setSelectedLogRunId((current) => {
        if (current && payload.jobs.some((job) => job.runId === current)) {
          return current;
        }
        return payload.jobs[0]?.runId ?? "";
      });
    } catch (error) {
      if (signal?.aborted) {
        return;
      }
      setJobsError(
        error instanceof Error ? error.message : "Failed to load jobs.",
      );
    } finally {
      if (!signal?.aborted) {
        setJobsLoading(false);
      }
    }
  }, []);

  const loadLogTail = useCallback(
    async (signal?: AbortSignal) => {
      if (!selectedLogRunId) {
        setLogData(null);
        setLogError("");
        return;
      }

      setLogLoading(true);
      try {
        const payload = await fetchJson<RunLogResponse>(
          `/api/runs/jobs/${encodeURIComponent(selectedLogRunId)}/logs?lines=180`,
          signal,
        );
        setLogData(payload);
        setLogError("");
      } catch (error) {
        if (signal?.aborted) {
          return;
        }
        setLogError(
          error instanceof Error ? error.message : "Failed to load logs.",
        );
      } finally {
        if (!signal?.aborted) {
          setLogLoading(false);
        }
      }
    },
    [selectedLogRunId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadJobs(controller.signal);
    return () => {
      controller.abort();
    };
  }, [loadJobs]);

  useEffect(() => {
    const controller = new AbortController();
    void loadLogTail(controller.signal);
    return () => {
      controller.abort();
    };
  }, [loadLogTail]);

  const handleJobsSnapshot = useCallback((event: MessageEvent<string>) => {
    const payload = parseStreamPayload<JobsStreamPayload>(event);
    if (!payload) {
      return;
    }
    setJobsData({
      generatedAtUtc: payload.generatedAtUtc,
      runsRoot: payload.runsRoot,
      jobs: payload.jobs,
    });
    setJobsError("");
    setJobsLoading(false);
    setSelectedLogRunId((current) => {
      if (current && payload.jobs.some((job) => job.runId === current)) {
        return current;
      }
      return payload.jobs[0]?.runId ?? "";
    });
    setLogData(payload.logData);
    setLogError("");
    setLogLoading(false);
  }, []);

  const handleJobsStreamError = useCallback((event: MessageEvent<string>) => {
    const payload = parseStreamPayload<JsonError>(event);
    if (!payload?.error) {
      return;
    }
    setJobsError(payload.error);
    setJobsLoading(false);
    setLogLoading(false);
  }, []);

  const jobsStreamHandlers = useMemo(
    () => ({
      jobs_snapshot: handleJobsSnapshot,
      jobs_error: handleJobsStreamError,
    }),
    [handleJobsSnapshot, handleJobsStreamError],
  );

  const jobsStreamUrl = useMemo(() => {
    const params = new URLSearchParams({
      limit: "120",
      logLines: "180",
      intervalMs: String(STREAM_REFRESH_INTERVAL_MS),
    });
    if (selectedLogRunId) {
      params.set("logRunId", selectedLogRunId);
    }
    return `/api/stream/jobs?${params.toString()}`;
  }, [selectedLogRunId]);

  const jobsStreamHealth = useSseStream({
    url: jobsStreamUrl,
    enabled: autoRefresh,
    eventHandlers: jobsStreamHandlers,
    streamName: "jobs",
    baseRetryMs: 1200,
    maxRetryMs: 12000,
    jitterMs: 500,
  });

  const jobsTimelineStreams = useMemo(
    () => [{ label: "jobs", health: jobsStreamHealth }],
    [jobsStreamHealth],
  );

  const handleLaunchRun = useCallback(async () => {
    setLaunching(true);
    setLaunchError("");
    try {
      if (!selectedFile) {
        throw new Error("Upload a JSONL file before starting a run.");
      }

      const launchOptions: RunLaunchOptions = {
        ...options,
        limit: parseNullableInt(limitInput),
        evalCount: parseNullableInt(evalCountInput),
        streamChunkSize: parseNullableInt(streamChunkSizeInput) ?? 32,
        hierarchyLevels: parseIntInRange(hierarchyLevelsInput, {
          fallback: 5,
          min: 2,
          max: 20,
        }),
        clusterTargetLeafSize: parseIntInRange(clusterTargetLeafSizeInput, {
          fallback: 25,
          min: 1,
          max: 5000,
        }),
        clusterMinLeafClusters: parseIntInRange(clusterMinLeafClustersInput, {
          fallback: 20,
          min: 1,
          max: 5000,
        }),
        clusterMaxLeafClusters: parseIntInRange(clusterMaxLeafClustersInput, {
          fallback: 600,
          min: 1,
          max: 10000,
        }),
        clusterHdbscanMinClusterSize: parseIntInRange(
          clusterHdbscanMinClusterSizeInput,
          {
            fallback: 12,
            min: 2,
            max: 5000,
          },
        ),
        clusterHdbscanMinSamples: parseIntInRange(
          clusterHdbscanMinSamplesInput,
          {
            fallback: 6,
            min: 1,
            max: 2000,
          },
        ),
      };

      const formData = new FormData();
      if (requestedRunId.trim()) {
        formData.set("runId", requestedRunId.trim());
      }
      formData.set("file", selectedFile);
      formData.set("options", JSON.stringify(launchOptions));

      const response = await fetch("/api/runs/launch", {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as RunLaunchResponse | JsonError;
      if (!response.ok) {
        if (typeof payload === "object" && payload && "error" in payload) {
          throw new Error(payload.error);
        }
        throw new Error(`Run launch failed (${response.status})`);
      }

      const launchPayload = payload as RunLaunchResponse;
      toast.success("Run started", {
        description: launchPayload.job.runId,
      });
      onRunStarted(launchPayload.job.runId);
      setSelectedLogRunId(launchPayload.job.runId);
      setRequestedRunId("");
      setSelectedFile(null);
      await Promise.all([loadJobs(), loadLogTail()]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to launch run.";
      setLaunchError(message);
      toast.error(message);
    } finally {
      setLaunching(false);
    }
  }, [
    clusterHdbscanMinClusterSizeInput,
    clusterHdbscanMinSamplesInput,
    clusterMaxLeafClustersInput,
    clusterMinLeafClustersInput,
    clusterTargetLeafSizeInput,
    evalCountInput,
    hierarchyLevelsInput,
    limitInput,
    loadJobs,
    loadLogTail,
    onRunStarted,
    options,
    requestedRunId,
    selectedFile,
    streamChunkSizeInput,
  ]);

  const handleTerminate = useCallback(
    async (runId: string) => {
      try {
        const response = await fetch(
          `/api/runs/jobs/${encodeURIComponent(runId)}/terminate`,
          {
            method: "POST",
          },
        );
        const payload = (await response.json()) as
          | RunTerminateResponse
          | JsonError;
        if (!response.ok) {
          if (typeof payload === "object" && payload && "error" in payload) {
            throw new Error(payload.error);
          }
          const status =
            typeof payload === "object" && payload && "status" in payload
              ? payload.status
              : `HTTP ${response.status}`;
          throw new Error(`Could not terminate run ${runId}: ${status}`);
        }
        const terminatePayload = payload as RunTerminateResponse;
        if (terminatePayload.ok) {
          toast.success("Stop requested", {
            description: runId,
          });
        } else {
          toast.warning(
            `Could not terminate ${runId}: ${terminatePayload.status}.`,
          );
        }
        await Promise.all([loadJobs(), loadLogTail()]);
      } catch (error) {
        toast.error(
          error instanceof Error
            ? error.message
            : "Termination request failed.",
        );
      }
    },
    [loadJobs, loadLogTail],
  );

  return (
    <div className="space-y-4">
      <Card className="clio-panel border-border/70">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Workflow className="size-4 text-primary" />
            Ingest and run
          </CardTitle>
          <CardDescription>
            Validate a JSONL dataset, choose CLIO phases, and launch a
            background run from one operator workflow.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {launchError ? (
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>Run launch failed</AlertTitle>
              <AlertDescription>{launchError}</AlertDescription>
            </Alert>
          ) : null}

          <section className="clio-panel-subtle space-y-4 p-4">
            <div className="space-y-1">
              <p className="clio-kicker">Step 1</p>
              <h3 className="clio-display text-xl">Input source</h3>
              <p className="text-sm text-muted-foreground">
                Upload the input JSONL dataset for this run. Uploaded files may
                contain sensitive text, so prune old uploads regularly.
              </p>
            </div>

            <div className="grid gap-4">
              <div className="space-y-2">
                <Label htmlFor="launch-file-upload">Upload JSONL input</Label>
                <Input
                  id="launch-file-upload"
                  type="file"
                  accept=".jsonl,application/json,text/plain"
                  onChange={(event) => {
                    const nextFile = event.target.files?.[0] ?? null;
                    setSelectedFile(nextFile);
                  }}
                />
                <p className="text-xs text-muted-foreground">
                  Uploaded files are saved under <code>runs/_uploads</code>.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{inputSummary}</Badge>
              {selectedFile ? (
                <Badge variant="secondary" className="gap-1">
                  <FileUp className="size-3.5" />
                  {selectedFile.name}
                </Badge>
              ) : null}
            </div>
          </section>

          <section className="clio-panel-subtle space-y-4 p-4">
            <div className="space-y-1">
              <p className="clio-kicker">Step 2</p>
              <h3 className="clio-display text-xl">CLIO phase profile</h3>
              <p className="text-sm text-muted-foreground">
                Configure the pipeline to extract facets, cluster patterns, and
                privacy-gated summaries from conversation logs.
              </p>
            </div>

            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              {PIPELINE_OPTION_ROWS.map((row) => (
                <div
                  key={row.key}
                  className="rounded-lg border border-border/70 bg-card/70 p-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">{row.label}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {row.detail}
                      </p>
                    </div>
                    <Switch
                      checked={options[row.key]}
                      onCheckedChange={(checked) =>
                        setBooleanOption(row.key, checked)
                      }
                      aria-label={row.label}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="grid gap-2 md:grid-cols-2">
              {EXECUTION_OPTION_ROWS.map((row) => (
                <div
                  key={row.key}
                  className="rounded-lg border border-border/70 bg-card/70 p-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">{row.label}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {row.detail}
                      </p>
                    </div>
                    <Switch
                      checked={options[row.key]}
                      onCheckedChange={(checked) =>
                        setBooleanOption(row.key, checked)
                      }
                      aria-label={row.label}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Badge variant="outline">
                {enabledPipelineStages} CLIO stages enabled
              </Badge>
              <Badge variant="outline">{runModeSummary}</Badge>
            </div>
          </section>

          <section className="clio-panel-subtle space-y-4 p-4">
            <div className="space-y-1">
              <p className="clio-kicker">Step 3</p>
              <h3 className="clio-display text-xl">Launch parameters</h3>
              <p className="text-sm text-muted-foreground">
                Optional run ID plus processing bounds for scoped execution.
              </p>
            </div>

            <div className="grid gap-4 lg:grid-cols-5">
              <div className="space-y-2">
                <Label htmlFor="launch-run-id">Run ID (optional)</Label>
                <Input
                  id="launch-run-id"
                  placeholder="auto-generate"
                  value={requestedRunId}
                  onChange={(event) => setRequestedRunId(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-limit">Limit (optional)</Label>
                <Input
                  id="launch-limit"
                  inputMode="numeric"
                  placeholder="e.g. 200"
                  value={limitInput}
                  onChange={(event) => setLimitInput(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-eval-count">Eval count (optional)</Label>
                <Input
                  id="launch-eval-count"
                  inputMode="numeric"
                  placeholder="e.g. 120"
                  value={evalCountInput}
                  onChange={(event) => setEvalCountInput(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-hierarchy-levels">
                  Hierarchy levels
                </Label>
                <Input
                  id="launch-hierarchy-levels"
                  inputMode="numeric"
                  value={hierarchyLevelsInput}
                  onChange={(event) =>
                    setHierarchyLevelsInput(event.target.value)
                  }
                  disabled={!options.withHierarchy}
                />
                <p className="text-xs text-muted-foreground">Clamped to 2-20</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-stream-chunk-size">
                  Stream chunk size
                </Label>
                <Input
                  id="launch-stream-chunk-size"
                  inputMode="numeric"
                  value={streamChunkSizeInput}
                  onChange={(event) =>
                    setStreamChunkSizeInput(event.target.value)
                  }
                />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-strategy">
                  Cluster strategy
                </Label>
                <Select
                  value={options.clusterStrategy}
                  onValueChange={(value) =>
                    setOptions((current) => ({
                      ...current,
                      clusterStrategy:
                        value as RunLaunchOptions["clusterStrategy"],
                    }))
                  }
                  disabled={!options.withClustering}
                >
                  <SelectTrigger id="launch-cluster-strategy">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hybrid">Hybrid (recommended)</SelectItem>
                    <SelectItem value="hdbscan">HDBSCAN</SelectItem>
                    <SelectItem value="kmeans">K-means</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-leaf-mode">
                  Leaf sizing mode
                </Label>
                <Select
                  value={options.clusterLeafMode}
                  onValueChange={(value) =>
                    setOptions((current) => ({
                      ...current,
                      clusterLeafMode:
                        value as RunLaunchOptions["clusterLeafMode"],
                    }))
                  }
                  disabled={!options.withClustering}
                >
                  <SelectTrigger id="launch-cluster-leaf-mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto-size by dataset</SelectItem>
                    <SelectItem value="fixed">Fixed k</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-hierarchy-depth-policy">
                  Hierarchy depth policy
                </Label>
                <Select
                  value={options.hierarchyDepthPolicy}
                  onValueChange={(value) =>
                    setOptions((current) => ({
                      ...current,
                      hierarchyDepthPolicy:
                        value as RunLaunchOptions["hierarchyDepthPolicy"],
                    }))
                  }
                  disabled={!options.withHierarchy}
                >
                  <SelectTrigger id="launch-hierarchy-depth-policy">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="adaptive">Adaptive</SelectItem>
                    <SelectItem value="strict_min">
                      Strict minimum depth
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-noise-policy">Noise policy</Label>
                <Select
                  value={options.clusterNoisePolicy}
                  onValueChange={(value) =>
                    setOptions((current) => ({
                      ...current,
                      clusterNoisePolicy:
                        value as RunLaunchOptions["clusterNoisePolicy"],
                    }))
                  }
                  disabled={!options.withClustering}
                >
                  <SelectTrigger id="launch-noise-policy">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="nearest">Nearest centroid</SelectItem>
                    <SelectItem value="singleton">
                      Singleton clusters
                    </SelectItem>
                    <SelectItem value="drop">Drop bucket</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-target-leaf-size">
                  Target leaf size
                </Label>
                <Input
                  id="launch-cluster-target-leaf-size"
                  inputMode="numeric"
                  value={clusterTargetLeafSizeInput}
                  onChange={(event) =>
                    setClusterTargetLeafSizeInput(event.target.value)
                  }
                  disabled={!options.withClustering}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-min-leaf-clusters">
                  Min leaf clusters
                </Label>
                <Input
                  id="launch-cluster-min-leaf-clusters"
                  inputMode="numeric"
                  value={clusterMinLeafClustersInput}
                  onChange={(event) =>
                    setClusterMinLeafClustersInput(event.target.value)
                  }
                  disabled={!options.withClustering}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-max-leaf-clusters">
                  Max leaf clusters
                </Label>
                <Input
                  id="launch-cluster-max-leaf-clusters"
                  inputMode="numeric"
                  value={clusterMaxLeafClustersInput}
                  onChange={(event) =>
                    setClusterMaxLeafClustersInput(event.target.value)
                  }
                  disabled={!options.withClustering}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-hdbscan-min-cluster-size">
                  HDBSCAN min cluster size
                </Label>
                <Input
                  id="launch-cluster-hdbscan-min-cluster-size"
                  inputMode="numeric"
                  value={clusterHdbscanMinClusterSizeInput}
                  onChange={(event) =>
                    setClusterHdbscanMinClusterSizeInput(event.target.value)
                  }
                  disabled={!options.withClustering}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="launch-cluster-hdbscan-min-samples">
                  HDBSCAN min samples
                </Label>
                <Input
                  id="launch-cluster-hdbscan-min-samples"
                  inputMode="numeric"
                  value={clusterHdbscanMinSamplesInput}
                  onChange={(event) =>
                    setClusterHdbscanMinSamplesInput(event.target.value)
                  }
                  disabled={!options.withClustering}
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => void handleLaunchRun()}
                disabled={launching}
                className="gap-2"
              >
                {launching ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    Launching...
                  </>
                ) : (
                  <>
                    <Play className="size-4" />
                    Start run
                  </>
                )}
              </Button>
              <Badge variant="outline">
                chunk {streamChunkSizeInput || "32"}
              </Badge>
              <Badge variant="outline">
                hierarchy depth{" "}
                {parseIntInRange(hierarchyLevelsInput, {
                  fallback: 5,
                  min: 2,
                  max: 20,
                })}
              </Badge>
              <Badge variant="outline">
                clustering {options.clusterStrategy}/{options.clusterLeafMode}
              </Badge>
              <Badge variant="outline">
                hierarchy policy {options.hierarchyDepthPolicy}
              </Badge>
            </div>
          </section>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
        <Card className="clio-panel border-border/70">
          <CardHeader>
            <CardTitle className="text-lg">Background jobs</CardTitle>
            <CardDescription>
              Process-level state for UI-launched runs, with terminate controls
              and log targeting.
            </CardDescription>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Badge
                variant={streamStateBadgeVariant(jobsStreamHealth.state)}
                className="gap-1"
              >
                Jobs stream {streamStateLabel(jobsStreamHealth.state)}
              </Badge>
              <StreamTimelineDrawer
                title="Jobs Stream Timeline"
                description="Connection and reconnect history for background job and live-log streaming."
                triggerLabel="Timeline"
                streams={jobsTimelineStreams}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {streamHint(jobsStreamHealth)} · retries{" "}
              {jobsStreamHealth.reconnectCount}, errors{" "}
              {jobsStreamHealth.totalErrors}
            </p>
            {jobsData?.runsRoot ? (
              <p className="text-xs text-muted-foreground">
                runs root {jobsData.runsRoot}
              </p>
            ) : null}
          </CardHeader>
          <CardContent className="space-y-3">
            {jobsError ? (
              <Alert variant="destructive">
                <AlertTriangle />
                <AlertTitle>Could not load jobs</AlertTitle>
                <AlertDescription>{jobsError}</AlertDescription>
              </Alert>
            ) : null}

            {jobsLoading && !jobsData ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Loading jobs...
              </div>
            ) : null}

            {!jobsLoading && jobsData && jobsData.jobs.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No UI-launched jobs yet.
              </p>
            ) : null}

            <div className="space-y-2">
              {jobsData?.jobs.map((job) => (
                <div
                  key={job.runId}
                  className={cn(
                    "rounded-xl border border-border/70 bg-muted/20 p-3",
                    selectedLogRunId === job.runId &&
                      "border-primary/55 bg-primary/5",
                  )}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="font-medium">{job.runId}</p>
                      <p className="text-xs text-muted-foreground">
                        Started {formatDateTime(job.startedAtUtc)} · pid{" "}
                        {job.pid}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {statusHint(job)}
                      </p>
                    </div>
                    <Badge variant={statusBadgeVariant(job.status)}>
                      {statusLabel(job.status)}
                    </Badge>
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setSelectedLogRunId(job.runId)}
                      className="gap-1"
                    >
                      <TerminalSquare className="size-3.5" />
                      View logs
                    </Button>
                    {job.running ? (
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => void handleTerminate(job.runId)}
                        className="gap-1"
                      >
                        <Square className="size-3.5" />
                        Terminate
                      </Button>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="clio-panel border-border/70">
          <CardHeader>
            <CardTitle className="text-lg">Live log tail</CardTitle>
            <CardDescription>
              Streaming CLI output for the selected background run process.
            </CardDescription>
            <p className="text-xs text-muted-foreground">
              messages {jobsStreamHealth.totalMessages}, last error{" "}
              {jobsStreamHealth.lastErrorAt
                ? formatDateTime(jobsStreamHealth.lastErrorAt)
                : "none"}
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              <Label htmlFor="run-log-selector">Run</Label>
              <Select
                value={selectedLogRunId}
                onValueChange={setSelectedLogRunId}
              >
                <SelectTrigger id="run-log-selector">
                  <SelectValue placeholder="Select run log" />
                </SelectTrigger>
                <SelectContent>
                  {jobsData?.jobs.map((job) => (
                    <SelectItem key={job.runId} value={job.runId}>
                      {job.runId} · {statusLabel(job.status)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {logError ? (
              <Alert variant="destructive">
                <AlertTriangle />
                <AlertTitle>Could not load logs</AlertTitle>
                <AlertDescription>{logError}</AlertDescription>
              </Alert>
            ) : null}

            {logLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Loading logs...
              </div>
            ) : null}

            <div className="rounded-xl border border-border/70 bg-muted/20 p-2">
              <ScrollArea className="h-[350px]">
                <pre className="whitespace-pre-wrap break-words p-2 text-xs leading-5">
                  {logData?.logTail || "No logs yet for this run."}
                </pre>
              </ScrollArea>
            </div>

            {logData ? (
              <div className="text-xs text-muted-foreground">
                {logData.lineCount} lines · {logData.status} · {logData.logPath}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
