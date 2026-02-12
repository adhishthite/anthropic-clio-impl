"use client";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  CircleHelp,
  Clock3,
  FolderCheck,
  FolderX,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Square,
} from "lucide-react";
import Link from "next/link";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";

import { RunOrchestrationPanel } from "@/components/clio/run-orchestration-panel";
import { RunVisualSummary } from "@/components/clio/run-visual-summary";
import { StreamTimelineDrawer } from "@/components/clio/stream-timeline-drawer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type SseStreamHealth,
  type SseStreamState,
  useSseStream,
} from "@/hooks/use-sse-stream";
import type {
  PhaseStatus,
  PhaseTimelineItem,
  RunDetailResponse,
  RunListItem,
  RunListResponse,
  RunState,
  RunTerminateResponse,
  RunVisualsResponse,
} from "@/lib/clio-types";
import { cn } from "@/lib/utils";

type StreamErrorPayload = {
  error: string;
};
const STREAM_REFRESH_INTERVAL_MS = 30000;

const RUN_STATE_META: Record<
  RunState,
  {
    label: string;
    badgeVariant: "default" | "secondary" | "outline" | "destructive";
  }
> = {
  running: { label: "Running", badgeVariant: "default" },
  completed: { label: "Completed", badgeVariant: "secondary" },
  completed_with_warnings: {
    label: "Completed with warnings",
    badgeVariant: "outline",
  },
  failed: { label: "Failed", badgeVariant: "destructive" },
  partial: { label: "Partial", badgeVariant: "outline" },
};

const PHASE_STATUS_META: Record<
  PhaseStatus,
  {
    label: string;
    badgeVariant: "default" | "secondary" | "outline" | "destructive";
  }
> = {
  pending: { label: "Pending", badgeVariant: "outline" },
  running: { label: "Running", badgeVariant: "default" },
  completed: { label: "Completed", badgeVariant: "secondary" },
  resumed: { label: "Resumed", badgeVariant: "secondary" },
  failed: { label: "Failed", badgeVariant: "destructive" },
  skipped: { label: "Skipped", badgeVariant: "outline" },
};

const PHASE_EXPLAINER: Record<
  string,
  { shortLabel: string; description: string }
> = {
  phase1_dataset_load: {
    shortLabel: "Load data",
    description:
      "Read the input dataset and create the normalized conversation snapshot.",
  },
  phase2_facet_extraction: {
    shortLabel: "Extract facets",
    description:
      "Generate structured facets from each conversation for downstream grouping.",
  },
  phase3_base_clustering: {
    shortLabel: "Cluster patterns",
    description:
      "Embed summaries and group similar conversations into semantic clusters.",
  },
  phase4_cluster_labeling: {
    shortLabel: "Label clusters",
    description:
      "Generate plain-language names and descriptions for each conversation cluster.",
  },
  phase4_hierarchy_scaffold: {
    shortLabel: "Build hierarchy",
    description:
      "Organize related clusters into a multi-level hierarchy for exploration.",
  },
  phase5_privacy_audit: {
    shortLabel: "Privacy filtering",
    description:
      "Audit privacy risk and keep only summaries that pass the configured threshold.",
  },
  phase6_evaluation: {
    shortLabel: "Quality evaluation",
    description:
      "Run synthetic evaluations to measure classification quality across representations.",
  },
};

const ARTIFACT_META: Record<string, { label: string; description: string }> = {
  run_manifest_json: {
    label: "Run manifest",
    description: "Top-level metadata for run lifecycle and output paths.",
  },
  run_events_jsonl: {
    label: "Run event stream",
    description: "Structured event log emitted while the pipeline is running.",
  },
  run_metrics_json: {
    label: "Run metrics summary",
    description: "Timing, phase-level outcomes, token usage, and warnings.",
  },
  conversation_jsonl: {
    label: "Conversation snapshot",
    description: "Normalized input conversations used by the pipeline.",
  },
  conversation_updated_jsonl: {
    label: "Conversation with analysis",
    description:
      "Conversation rows enriched with analysis outputs across phases.",
  },
  facets_jsonl: {
    label: "Facet outputs",
    description: "Structured facet extraction results for each conversation.",
  },
  facets_errors_jsonl: {
    label: "Facet extraction errors",
    description: "Rows that failed facet extraction and their error details.",
  },
  summary_embeddings_npy: {
    label: "Summary embeddings",
    description: "Numeric embedding vectors used for clustering.",
  },
  base_centroids_npy: {
    label: "Cluster centroids",
    description: "Centroid vectors for the base clustering stage.",
  },
  base_assignments_jsonl: {
    label: "Cluster assignments",
    description: "Conversation-to-cluster assignments from base clustering.",
  },
  base_clusters_json: {
    label: "Base cluster stats",
    description:
      "Initial cluster-level statistics before labeling and filtering.",
  },
  labeled_clusters_json: {
    label: "Labeled clusters",
    description: "Cluster names and summaries before privacy filtering.",
  },
  labeled_clusters_privacy_filtered_json: {
    label: "Privacy-filtered clusters",
    description: "Cluster summaries retained after privacy threshold checks.",
  },
  hierarchy_json: {
    label: "Hierarchy structure",
    description: "Parent-child topology used for hierarchy exploration.",
  },
  privacy_audit_json: {
    label: "Privacy audit report",
    description:
      "Stage-level privacy ratings, pass rates, and validation metrics.",
  },
  phase6_metrics_json: {
    label: "Evaluation metrics",
    description:
      "Phase 6 synthetic evaluation metrics, including accuracy and F1 by representation.",
  },
  phase6_report_md: {
    label: "Evaluation report",
    description: "Human-readable markdown report from phase 6 evaluation.",
  },
  viz_map_points_jsonl: {
    label: "Map points",
    description: "2D projected points for cluster map visualization.",
  },
  viz_map_clusters_json: {
    label: "Map cluster overlay",
    description: "Cluster centroids and map metadata for visualization.",
  },
  tree_view_json: {
    label: "Hierarchy tree view",
    description: "UI-ready hierarchy nodes/edges for tree and sunburst views.",
  },
};

type StatCardTone = "neutral" | "success" | "alert" | "info";

function formatDateTime(
  value: string,
  options?: { localize?: boolean },
): string {
  if (!value) {
    return "n/a";
  }
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return value;
  }
  if (!options?.localize) {
    return new Date(timestamp)
      .toISOString()
      .replace("T", " ")
      .replace("Z", " UTC");
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

function formatRelativeTime(value: string, nowMs: number | null): string {
  if (!value) {
    return "n/a";
  }
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return "n/a";
  }
  if (nowMs === null) {
    return "recently";
  }
  const diffSeconds = Math.round((timestamp - nowMs) / 1000);
  const absDiff = Math.abs(diffSeconds);
  const formatter = new Intl.RelativeTimeFormat(undefined, {
    numeric: "auto",
  });

  if (absDiff < 60) {
    return formatter.format(diffSeconds, "second");
  }
  if (absDiff < 3_600) {
    return formatter.format(Math.round(diffSeconds / 60), "minute");
  }
  if (absDiff < 86_400) {
    return formatter.format(Math.round(diffSeconds / 3_600), "hour");
  }
  return formatter.format(Math.round(diffSeconds / 86_400), "day");
}

function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }
  return (await response.json()) as T;
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

function streamHint(health: SseStreamHealth, nowMs: number | null): string {
  if (health.state === "reconnecting" && health.nextRetryMs !== null) {
    return `retry in ${Math.max(1, Math.ceil(health.nextRetryMs / 1000))}s`;
  }
  if (health.lastMessageAt) {
    return `last msg ${formatRelativeTime(health.lastMessageAt, nowMs)}`;
  }
  return "waiting for stream data";
}

function phaseIcon(status: PhaseStatus) {
  if (status === "running") {
    return <Loader2 className="size-4 animate-spin text-primary" />;
  }
  if (status === "completed" || status === "resumed") {
    return <CheckCircle2 className="size-4 text-emerald-600" />;
  }
  if (status === "failed") {
    return <AlertTriangle className="size-4 text-destructive" />;
  }
  return <CircleDashed className="size-4 text-muted-foreground" />;
}

function stateIcon(state: RunState) {
  if (state === "running") {
    return <Activity className="size-4" />;
  }
  if (state === "completed") {
    return <CheckCircle2 className="size-4" />;
  }
  if (state === "completed_with_warnings") {
    return <AlertTriangle className="size-4" />;
  }
  if (state === "failed") {
    return <AlertTriangle className="size-4" />;
  }
  return <CircleDashed className="size-4" />;
}

function phaseLabelFromTimeline(
  detail: RunDetailResponse | null,
  phaseValue: string,
): string {
  if (!detail) {
    return phaseValue;
  }
  const found = detail.phaseTimeline.find((item) => item.phase === phaseValue);
  return found?.label ?? phaseValue;
}

function metricLabel(item: PhaseTimelineItem): string {
  if (item.processed === null) {
    if (item.status === "completed" || item.status === "resumed") {
      return item.note || "Completed - no live checkpoint for this phase";
    }
    return item.note || "No checkpoint yet";
  }
  if (item.total !== null) {
    return `${item.processed}/${item.total}`;
  }
  return `${item.processed} processed`;
}

function checkpointTimestampLabel(
  item: PhaseTimelineItem,
  nowMs: number | null,
): string {
  if (item.updatedAtUtc) {
    return `updated ${formatRelativeTime(item.updatedAtUtc, nowMs)}`;
  }
  if (item.status === "completed" || item.status === "resumed") {
    return "checkpoint not emitted";
  }
  return "no checkpoint timestamp";
}

function streamDotClass(state: SseStreamState): string {
  if (state === "connected") {
    return "bg-emerald-500";
  }
  if (state === "connecting") {
    return "bg-sky-500";
  }
  if (state === "reconnecting") {
    return "bg-amber-500";
  }
  if (state === "disconnected") {
    return "bg-destructive";
  }
  return "bg-muted-foreground";
}

function eventBorderClass(status: string | null): string {
  if (status === "failed") {
    return "border-l-destructive";
  }
  if (status === "completed") {
    return "border-l-emerald-500";
  }
  if (status === "running") {
    return "border-l-primary";
  }
  return "border-l-border";
}

function stageCardTone(status: PhaseStatus): string {
  if (status === "running") {
    return "border-primary/55 bg-primary/5";
  }
  if (status === "completed" || status === "resumed") {
    return "border-emerald-300/60 bg-emerald-100/40 dark:border-emerald-900 dark:bg-emerald-950/20";
  }
  if (status === "failed") {
    return "border-red-300/60 bg-red-100/40 dark:border-red-900 dark:bg-red-950/20";
  }
  return "border-border/70 bg-muted/20";
}

function HelpTooltip({ content }: { content: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex size-4 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground"
          aria-label="More information"
        >
          <CircleHelp className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-72">
        {content}
      </TooltipContent>
    </Tooltip>
  );
}

type DashboardStatCardProps = {
  title: string;
  value: string;
  subtitle: string;
  icon: ReactNode;
  tone?: StatCardTone;
};

function DashboardStatCard({
  title,
  value,
  subtitle,
  icon,
  tone = "neutral",
}: DashboardStatCardProps) {
  return (
    <Card
      className={cn(
        "clio-panel overflow-hidden border transition-transform duration-200 hover:-translate-y-0.5",
        tone === "success" && "border-emerald-300/60",
        tone === "alert" && "border-amber-300/60",
        tone === "info" && "border-sky-300/60",
      )}
    >
      <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-primary/30 via-primary to-accent/40" />
      <CardHeader className="gap-1 pb-2">
        <CardDescription className="text-[11px] tracking-[0.16em] uppercase">
          {title}
        </CardDescription>
        <div className="flex items-center justify-between gap-4">
          <CardTitle className="text-2xl font-semibold tracking-tight">
            {value}
          </CardTitle>
          <span className="rounded-full border border-border/70 bg-muted/40 p-2">
            {icon}
          </span>
        </div>
      </CardHeader>
      <CardContent className="pt-0 text-sm text-muted-foreground">
        {subtitle}
      </CardContent>
    </Card>
  );
}

type StreamHealthRowProps = {
  label: string;
  health: SseStreamHealth;
};

function StreamHealthRow({ label, health }: StreamHealthRowProps) {
  return (
    <div className="clio-panel-subtle flex items-center justify-between gap-3 px-3 py-2">
      <div className="min-w-0">
        <p className="clio-kicker">{label}</p>
        <p className="truncate text-sm font-medium">
          {streamStateLabel(health.state)}
        </p>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span
          className={cn("size-2 rounded-full", streamDotClass(health.state))}
        />
        <span>r {health.reconnectCount}</span>
        <span>e {health.totalErrors}</span>
      </div>
    </div>
  );
}

type LiveRunDashboardProps = {
  lockedRunId?: string | null;
  showOrchestration?: boolean;
};

export function LiveRunDashboard({
  lockedRunId = null,
  showOrchestration = true,
}: LiveRunDashboardProps = {}) {
  const singleRunMode = Boolean(lockedRunId);
  const [runsData, setRunsData] = useState<RunListResponse | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string>(lockedRunId ?? "");
  const [detailData, setDetailData] = useState<RunDetailResponse | null>(null);
  const [visualsData, setVisualsData] = useState<RunVisualsResponse | null>(
    null,
  );
  const [runsError, setRunsError] = useState<string>("");
  const [detailError, setDetailError] = useState<string>("");
  const [visualsError, setVisualsError] = useState<string>("");
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [loadingVisuals, setLoadingVisuals] = useState<boolean>(false);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [terminatingRun, setTerminatingRun] = useState<boolean>(false);
  const [terminateDialogOpen, setTerminateDialogOpen] =
    useState<boolean>(false);
  const [isHydrated, setIsHydrated] = useState(false);
  const [relativeNowMs, setRelativeNowMs] = useState<number | null>(null);

  const previousDetailRef = useRef<RunDetailResponse | null>(null);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    if (!isHydrated) {
      return;
    }
    const update = () => setRelativeNowMs(Date.now());
    update();
    const handle = window.setInterval(update, 30_000);
    return () => window.clearInterval(handle);
  }, [isHydrated]);

  useEffect(() => {
    if (lockedRunId) {
      setSelectedRunId(lockedRunId);
    }
  }, [lockedRunId]);

  const loadRuns = useCallback(
    async (signal?: AbortSignal) => {
      setLoadingRuns(true);
      try {
        const payload = await fetchJson<RunListResponse>(
          "/api/runs?limit=120",
          signal,
        );
        setRunsData(payload);
        setRunsError("");
        setSelectedRunId((current) => {
          if (lockedRunId) {
            return lockedRunId;
          }
          if (current && payload.runs.some((run) => run.runId === current)) {
            return current;
          }
          return payload.runs[0]?.runId ?? "";
        });
      } catch (error) {
        if (signal?.aborted) {
          return;
        }
        setRunsError(
          error instanceof Error ? error.message : "Failed to load runs.",
        );
      } finally {
        if (!signal?.aborted) {
          setLoadingRuns(false);
        }
      }
    },
    [lockedRunId],
  );

  const loadDetail = useCallback(
    async (signal?: AbortSignal) => {
      if (!selectedRunId) {
        setDetailData(null);
        return;
      }

      setLoadingDetail(true);
      try {
        const payload = await fetchJson<RunDetailResponse>(
          `/api/runs/${encodeURIComponent(selectedRunId)}`,
          signal,
        );
        setDetailData(payload);
        setDetailError("");
      } catch (error) {
        if (signal?.aborted) {
          return;
        }
        setDetailError(
          error instanceof Error
            ? error.message
            : "Failed to load run details.",
        );
      } finally {
        if (!signal?.aborted) {
          setLoadingDetail(false);
        }
      }
    },
    [selectedRunId],
  );

  const loadVisuals = useCallback(
    async (signal?: AbortSignal) => {
      if (!selectedRunId) {
        setVisualsData(null);
        setVisualsError("");
        return;
      }

      setLoadingVisuals(true);
      try {
        const url = `/api/runs/${encodeURIComponent(selectedRunId)}/visuals`;
        const response = await fetch(url, { cache: "no-store", signal });
        if (response.status === 404) {
          setVisualsData(null);
          setVisualsError("");
          return;
        }
        if (!response.ok) {
          throw new Error(`Request failed (${response.status}) for ${url}`);
        }
        const payload = (await response.json()) as RunVisualsResponse;
        setVisualsData(payload);
        setVisualsError("");
      } catch (error) {
        if (signal?.aborted) {
          return;
        }
        setVisualsError(
          error instanceof Error ? error.message : "Failed to load visuals.",
        );
      } finally {
        if (!signal?.aborted) {
          setLoadingVisuals(false);
        }
      }
    },
    [selectedRunId],
  );

  const handleManualRefresh = useCallback(async () => {
    await Promise.all([loadRuns(), loadDetail(), loadVisuals()]);
    toast("Dashboard updated", {
      description: selectedRunId
        ? `Refreshed run ${selectedRunId}`
        : "Latest run snapshot loaded",
      id: "dashboard-manual-refresh",
    });
  }, [loadDetail, loadRuns, loadVisuals, selectedRunId]);

  const handleTerminateRun = useCallback(async () => {
    if (!selectedRunId) {
      return;
    }

    setTerminatingRun(true);
    try {
      const response = await fetch(
        `/api/runs/jobs/${encodeURIComponent(selectedRunId)}/terminate`,
        {
          method: "POST",
        },
      );
      const payload = (await response.json()) as
        | RunTerminateResponse
        | { error?: string };

      if (!response.ok) {
        if (typeof payload === "object" && payload && "error" in payload) {
          throw new Error(payload.error || "Termination request failed.");
        }
        throw new Error(`Termination request failed (${response.status}).`);
      }

      const terminatePayload = payload as RunTerminateResponse;
      if (terminatePayload.ok) {
        toast.success("Stop requested", {
          description: `Run ${selectedRunId}`,
        });
      } else if (
        terminatePayload.status === "job_not_found" &&
        detailData?.summary.lockOwnerPid
      ) {
        toast.warning(
          `No UI job record for this run. In terminal: kill -TERM ${detailData.summary.lockOwnerPid}`,
        );
      } else {
        toast.warning(
          `Could not terminate ${selectedRunId}: ${terminatePayload.status}.`,
        );
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Termination request failed.",
      );
    } finally {
      setTerminatingRun(false);
      await Promise.all([loadRuns(), loadDetail(), loadVisuals()]);
    }
  }, [detailData, loadDetail, loadRuns, loadVisuals, selectedRunId]);

  const handleRunsSnapshot = useCallback(
    (event: MessageEvent<string>) => {
      const payload = parseStreamPayload<RunListResponse>(event);
      if (!payload) {
        return;
      }
      setRunsData(payload);
      setRunsError("");
      setLoadingRuns(false);
      setSelectedRunId((current) => {
        if (lockedRunId) {
          return lockedRunId;
        }
        if (current && payload.runs.some((run) => run.runId === current)) {
          return current;
        }
        return payload.runs[0]?.runId ?? "";
      });
    },
    [lockedRunId],
  );

  const handleRunsStreamError = useCallback((event: MessageEvent<string>) => {
    const payload = parseStreamPayload<StreamErrorPayload>(event);
    if (!payload?.error) {
      return;
    }
    setRunsError(payload.error);
    setLoadingRuns(false);
  }, []);

  const runsStreamHandlers = useMemo(
    () => ({
      runs_snapshot: handleRunsSnapshot,
      runs_error: handleRunsStreamError,
    }),
    [handleRunsSnapshot, handleRunsStreamError],
  );

  const runsStreamHealth = useSseStream({
    url: `/api/stream/runs?limit=120&intervalMs=${STREAM_REFRESH_INTERVAL_MS}`,
    enabled: autoRefresh,
    eventHandlers: runsStreamHandlers,
    streamName: "runs",
    baseRetryMs: 1200,
    maxRetryMs: 12000,
    jitterMs: 500,
  });

  const handleRunDetailSnapshot = useCallback((event: MessageEvent<string>) => {
    const payload = parseStreamPayload<RunDetailResponse>(event);
    if (!payload) {
      return;
    }
    setDetailData(payload);
    setDetailError("");
    setLoadingDetail(false);
  }, []);

  const handleRunDetailStreamError = useCallback(
    (event: MessageEvent<string>) => {
      const payload = parseStreamPayload<StreamErrorPayload>(event);
      if (!payload?.error) {
        return;
      }
      setDetailError(payload.error);
      setLoadingDetail(false);
    },
    [],
  );

  const runDetailStreamHandlers = useMemo(
    () => ({
      run_detail: handleRunDetailSnapshot,
      run_detail_error: handleRunDetailStreamError,
    }),
    [handleRunDetailSnapshot, handleRunDetailStreamError],
  );

  const runDetailStreamUrl = selectedRunId
    ? `/api/stream/runs/${encodeURIComponent(selectedRunId)}?intervalMs=${STREAM_REFRESH_INTERVAL_MS}`
    : null;
  const runDetailStreamHealth = useSseStream({
    url: runDetailStreamUrl,
    enabled: autoRefresh && Boolean(selectedRunId),
    eventHandlers: runDetailStreamHandlers,
    streamName: "run-detail",
    baseRetryMs: 1000,
    maxRetryMs: 10000,
    jitterMs: 500,
  });
  const dashboardTimelineStreams = useMemo(
    () => [
      { label: "runs", health: runsStreamHealth },
      { label: "run-detail", health: runDetailStreamHealth },
    ],
    [runDetailStreamHealth, runsStreamHealth],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadRuns(controller.signal);
    return () => {
      controller.abort();
    };
  }, [loadRuns]);

  useEffect(() => {
    const controller = new AbortController();
    void loadDetail(controller.signal);
    return () => {
      controller.abort();
    };
  }, [loadDetail]);

  useEffect(() => {
    const controller = new AbortController();
    void loadVisuals(controller.signal);
    return () => {
      controller.abort();
    };
  }, [loadVisuals]);

  useEffect(() => {
    if (!detailData || !selectedRunId) {
      return;
    }
    const shouldRefreshVisuals =
      detailData.run.state !== "running" ||
      detailData.artifactStatus.some(
        (artifact) =>
          artifact.exists &&
          (artifact.artifactKey === "viz_map_points_jsonl" ||
            artifact.artifactKey === "viz_map_clusters_json" ||
            artifact.artifactKey === "tree_view_json" ||
            artifact.artifactKey === "privacy_audit_json" ||
            artifact.artifactKey === "phase6_metrics_json"),
      );
    if (!shouldRefreshVisuals) {
      return;
    }
    const controller = new AbortController();
    void loadVisuals(controller.signal);
    return () => {
      controller.abort();
    };
  }, [detailData, loadVisuals, selectedRunId]);

  useEffect(() => {
    if (!detailData) {
      return;
    }
    const previous = previousDetailRef.current;
    if (!previous || previous.run.runId !== detailData.run.runId) {
      previousDetailRef.current = detailData;
      return;
    }

    const justFinished =
      previous.run.runLockActive && !detailData.run.runLockActive;
    if (justFinished) {
      if (detailData.summary.failedPhases > 0) {
        toast.error("Run finished with failures", {
          description: detailData.run.runId,
        });
      } else if (detailData.summary.warningCount > 0) {
        toast.warning("Run finished with warnings", {
          description: detailData.run.runId,
        });
      } else {
        toast.success("Run completed", {
          description: detailData.run.runId,
        });
      }
    } else {
      if (detailData.summary.failedPhases > previous.summary.failedPhases) {
        toast.error("A pipeline stage failed", {
          description: detailData.run.runId,
          id: `run-failed-phase-${detailData.run.runId}-${detailData.summary.failedPhases}`,
        });
      } else if (
        detailData.summary.warningCount > previous.summary.warningCount
      ) {
        toast.warning("New run warnings", {
          description: `${detailData.run.runId} now has ${detailData.summary.warningCount} warning${detailData.summary.warningCount === 1 ? "" : "s"}`,
          id: `run-warning-${detailData.run.runId}-${detailData.summary.warningCount}`,
        });
      }
    }

    previousDetailRef.current = detailData;
  }, [detailData]);

  const selectedRun = useMemo<RunListItem | null>(() => {
    if (!runsData || !selectedRunId) {
      return null;
    }
    return runsData.runs.find((item) => item.runId === selectedRunId) ?? null;
  }, [runsData, selectedRunId]);
  const prioritizeVisuals = useMemo(() => {
    if (!detailData) {
      return false;
    }
    return (
      detailData.run.state === "running" ||
      detailData.run.state === "completed" ||
      detailData.run.state === "completed_with_warnings"
    );
  }, [detailData]);

  const hasRuns = Boolean(runsData && runsData.runs.length > 0);
  const handleRunStarted = useCallback(
    (runId: string) => {
      if (singleRunMode) {
        return;
      }
      setSelectedRunId(runId);
      void loadRuns();
    },
    [loadRuns, singleRunMode],
  );

  return (
    <div className="relative min-h-screen pb-10">
      <div className="clio-orb top-0 -left-16 h-56 w-56 bg-primary/30" />
      <div className="clio-orb top-20 right-0 h-64 w-64 bg-accent/20 [animation-delay:2s]" />

      <main className="mx-auto flex w-full max-w-[2280px] flex-col gap-5 px-3 py-5 md:px-6 md:py-8 2xl:px-8">
        <section className="clio-shell relative px-5 py-5 md:px-7 md:py-6">
          <div className="clio-grid-pattern pointer-events-none absolute inset-0 opacity-35" />
          <div className="relative grid gap-5 xl:grid-cols-[1.45fr_0.8fr] 2xl:grid-cols-[1.7fr_0.8fr]">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary" className="gap-1">
                  <Sparkles className="size-3.5" />
                  CLIO run detail
                </Badge>
                <Badge variant="outline" className="gap-1">
                  <Clock3 className="size-3.5" />
                  {runsData?.generatedAtUtc
                    ? `Updated ${formatRelativeTime(runsData.generatedAtUtc, relativeNowMs)}`
                    : "Waiting for first sync"}
                </Badge>
              </div>

              <h1 className="clio-display text-3xl leading-tight md:text-4xl">
                Inspect one CLIO run from ingest to evaluation
              </h1>
              <p className="max-w-2xl text-sm text-muted-foreground md:text-base">
                CLIO surfaces usage patterns as aggregate outputs: facet
                extraction, semantic clustering, hierarchy grouping, privacy
                filtering, and synthetic evaluation metrics.
              </p>
            </div>

            <div className="clio-panel relative overflow-hidden p-4 md:p-5">
              <p className="clio-kicker">Run context</p>
              <div className="mt-3 space-y-3 text-sm">
                <div className="clio-panel-subtle px-3 py-2">
                  <p className="text-xs text-muted-foreground">Runs root</p>
                  <code className="mt-1 block truncate text-[12px]">
                    {runsData?.runsRoot ?? "(discovering...)"}
                  </code>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="clio-panel-subtle px-3 py-2">
                    <p className="text-xs text-muted-foreground">
                      Indexed runs
                    </p>
                    <p className="mt-1 text-xl font-semibold">
                      {runsData ? runsData.runs.length : "-"}
                    </p>
                  </div>
                  <div className="clio-panel-subtle px-3 py-2">
                    <p className="text-xs text-muted-foreground">
                      Auto refresh
                    </p>
                    <p className="mt-1 text-xl font-semibold">
                      {autoRefresh ? "ON" : "OFF"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.55fr_0.8fr] 2xl:grid-cols-[1.85fr_0.75fr]">
          <Card className="clio-panel border-border/70">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">
                {singleRunMode ? "Run dashboard" : "Run selection"}
              </CardTitle>
              <CardDescription>
                {singleRunMode
                  ? "This page is scoped to one run for lower cognitive load."
                  : "Select one run to review phase progression, events, and artifact completeness."}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {singleRunMode ? (
                <div className="grid gap-3 md:grid-cols-[1fr_auto_auto] md:items-end">
                  <div className="space-y-2">
                    <p className="clio-kicker">Run ID</p>
                    <div className="rounded-md border border-border/70 bg-muted/20 px-3 py-2 font-mono text-sm">
                      {selectedRunId || "(missing)"}
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-10 gap-2"
                    onClick={() => void handleManualRefresh()}
                  >
                    <RefreshCw className="size-4" />
                    Refresh
                  </Button>
                  <Button asChild variant="outline" size="sm" className="h-10">
                    <Link href="/">All runs</Link>
                  </Button>
                </div>
              ) : (
                <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
                  <div className="space-y-2">
                    <label className="clio-kicker" htmlFor="run-selector">
                      Select run
                    </label>
                    <Select
                      value={selectedRunId}
                      onValueChange={setSelectedRunId}
                    >
                      <SelectTrigger id="run-selector" className="w-full">
                        <SelectValue placeholder="Select a run" />
                      </SelectTrigger>
                      <SelectContent>
                        {runsData?.runs.map((run) => (
                          <SelectItem key={run.runId} value={run.runId}>
                            {run.runId} · {RUN_STATE_META[run.state].label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-10 gap-2"
                    onClick={() => void handleManualRefresh()}
                  >
                    <RefreshCw className="size-4" />
                    Refresh
                  </Button>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center gap-2 rounded-md border border-border/70 px-2.5 py-1.5 text-sm">
                  <Switch
                    checked={autoRefresh}
                    onCheckedChange={setAutoRefresh}
                    aria-label="Toggle auto refresh"
                  />
                  Auto refresh (30s)
                </div>
                {selectedRun ? (
                  <Badge
                    variant={RUN_STATE_META[selectedRun.state].badgeVariant}
                    className="gap-1"
                  >
                    {stateIcon(selectedRun.state)}
                    {RUN_STATE_META[selectedRun.state].label}
                  </Badge>
                ) : null}
                {selectedRun?.updatedAtUtc ? (
                  <Badge variant="outline">
                    touched{" "}
                    {formatRelativeTime(
                      selectedRun.updatedAtUtc,
                      relativeNowMs,
                    )}
                  </Badge>
                ) : null}
              </div>

              {singleRunMode && selectedRun?.state === "running" ? (
                <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="space-y-0.5">
                      <p className="text-xs font-semibold uppercase tracking-wide text-destructive">
                        Danger zone
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Stop this run process immediately.
                      </p>
                    </div>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="h-9 gap-2"
                      onClick={() => setTerminateDialogOpen(true)}
                      disabled={terminatingRun}
                    >
                      {terminatingRun ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <Square className="size-4" />
                      )}
                      Terminate run
                    </Button>
                  </div>
                </div>
              ) : null}

              <AlertDialog
                open={terminateDialogOpen}
                onOpenChange={setTerminateDialogOpen}
              >
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Terminate this run?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will send a termination signal to the process for{" "}
                      <span className="font-mono text-foreground">
                        {selectedRunId || "the selected run"}
                      </span>
                      . Partial outputs may remain and final metrics may not be
                      written.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel disabled={terminatingRun}>
                      Cancel
                    </AlertDialogCancel>
                    <AlertDialogAction
                      variant="destructive"
                      disabled={terminatingRun}
                      onClick={() => {
                        setTerminateDialogOpen(false);
                        void handleTerminateRun();
                      }}
                    >
                      {terminatingRun ? "Terminating..." : "Yes, terminate run"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </CardContent>
          </Card>

          <Card className="clio-panel border-border/70">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Stream health</CardTitle>
              <CardDescription>
                SSE channel state for live run snapshots and selected run detail
                updates.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <StreamHealthRow label="runs" health={runsStreamHealth} />
              <StreamHealthRow
                label="run-detail"
                health={runDetailStreamHealth}
              />
              <div className="clio-panel-subtle px-3 py-2 text-xs text-muted-foreground">
                <p>
                  runs: {streamHint(runsStreamHealth, relativeNowMs)} - messages{" "}
                  {runsStreamHealth.totalMessages}
                </p>
                <p>
                  detail: {streamHint(runDetailStreamHealth, relativeNowMs)} -
                  messages {runDetailStreamHealth.totalMessages}
                </p>
              </div>
              <StreamTimelineDrawer
                title="Dashboard Stream Timeline"
                description="Connection lifecycle, reconnect attempts, and health events for dashboard SSE channels."
                triggerLabel="Open timeline"
                streams={dashboardTimelineStreams}
              />
            </CardContent>
          </Card>
        </section>

        {runsError ? (
          <Alert variant="destructive">
            <AlertTriangle />
            <AlertTitle>Could not load runs</AlertTitle>
            <AlertDescription>{runsError}</AlertDescription>
          </Alert>
        ) : null}

        {!hasRuns && !loadingRuns ? (
          <Card className="clio-shell border-border/70">
            <CardContent className="py-6">
              <Empty className="border-border bg-card/85">
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <Activity className="size-5" />
                  </EmptyMedia>
                  <EmptyTitle>No runs discovered yet</EmptyTitle>
                  <EmptyDescription>
                    Start a pipeline run, then this dashboard will stream
                    progress, events, and artifact readiness.
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent>
                  <code className="w-full rounded-md border bg-muted/40 px-3 py-2 text-left text-xs">
                    uv run clio run --with-hierarchy --with-privacy --with-eval
                  </code>
                </EmptyContent>
              </Empty>
            </CardContent>
          </Card>
        ) : null}

        {loadingRuns && !runsData ? (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-6">
            {[1, 2, 3, 4].map((skeletonId) => (
              <Card
                key={`run-skeleton-${skeletonId}`}
                className="clio-panel border-border/70"
              >
                <CardHeader>
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-8 w-28" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-4 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : null}

        {selectedRun && detailData ? (
          <>
            {prioritizeVisuals ? (
              <RunVisualSummary
                visuals={visualsData}
                loading={loadingVisuals}
                error={visualsError}
              />
            ) : null}

            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <DashboardStatCard
                title="Run lifecycle"
                value={RUN_STATE_META[detailData.run.state].label}
                subtitle={`Run ${detailData.run.runId}`}
                icon={stateIcon(detailData.run.state)}
                tone="info"
              />
              <DashboardStatCard
                title="Overall progress"
                value={`${detailData.run.overallProgressPercent.toFixed(1)}%`}
                subtitle={
                  detailData.summary.skippedPhases > 0
                    ? `${detailData.summary.completedPhases}/${detailData.summary.totalPhases} active phases done - ${detailData.summary.skippedPhases} skipped`
                    : `${detailData.summary.completedPhases}/${detailData.summary.totalPhases} phases done`
                }
                icon={<ShieldCheck className="size-4 text-emerald-600" />}
                tone="success"
              />
              <DashboardStatCard
                title="Conversations processed"
                value={formatCompactNumber(
                  detailData.run.conversationCountProcessed,
                )}
                subtitle={`of ${formatCompactNumber(detailData.run.conversationCountInput)} input conversations`}
                icon={<Activity className="size-4 text-primary" />}
              />
              <DashboardStatCard
                title="Required artifacts missing"
                value={`${detailData.summary.requiredArtifactsMissing}`}
                subtitle={`${detailData.summary.optionalArtifactsPresent} optional artifacts present`}
                icon={
                  detailData.summary.requiredArtifactsMissing > 0 ? (
                    <FolderX className="size-4 text-destructive" />
                  ) : (
                    <FolderCheck className="size-4 text-emerald-600" />
                  )
                }
                tone={
                  detailData.summary.requiredArtifactsMissing > 0
                    ? "alert"
                    : "success"
                }
              />
            </section>

            <Card className="clio-shell border-border/70">
              <CardHeader className="pb-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="clio-kicker">Current phase</p>
                    <CardTitle className="mt-1 text-2xl">
                      {phaseLabelFromTimeline(detailData, detailData.run.phase)}
                    </CardTitle>
                    <CardDescription>
                      Last update{" "}
                      {formatRelativeTime(
                        detailData.run.updatedAtUtc,
                        relativeNowMs,
                      )}{" "}
                      (
                      {formatDateTime(detailData.run.updatedAtUtc, {
                        localize: isHydrated,
                      })}
                      )
                    </CardDescription>
                  </div>
                  <Badge
                    variant={RUN_STATE_META[detailData.run.state].badgeVariant}
                    className="gap-1"
                  >
                    {stateIcon(detailData.run.state)}
                    {RUN_STATE_META[detailData.run.state].label}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <Progress value={detailData.run.overallProgressPercent} />
                <div className="grid gap-2 text-xs text-muted-foreground md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5">
                  <span>
                    Created{" "}
                    {formatDateTime(detailData.run.createdAtUtc, {
                      localize: isHydrated,
                    })}
                  </span>
                  <span>
                    Lock {detailData.run.runLockActive ? "active" : "released"}
                  </span>
                  <span>
                    Warnings {detailData.summary.warningCount} · failures{" "}
                    {detailData.summary.failedPhases}
                  </span>
                  {detailData.summary.lockOwnerPid ? (
                    <span>Owner PID {detailData.summary.lockOwnerPid}</span>
                  ) : (
                    <span>Owner PID not reported</span>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="clio-shell border-border/70">
              <CardHeader>
                <CardTitle className="text-lg">Pipeline stages</CardTitle>
                <CardDescription>
                  Live stage view of where this run is in the CLIO pipeline.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-7">
                  {detailData.phaseTimeline.map((item, index) => {
                    const stageMeta = PHASE_EXPLAINER[item.phase] ?? {
                      shortLabel: item.label,
                      description: item.label,
                    };
                    const statusMeta = PHASE_STATUS_META[item.status];
                    return (
                      <div
                        key={item.phase}
                        className={cn(
                          "rounded-xl border px-3 py-2 text-sm",
                          stageCardTone(item.status),
                        )}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="text-xs font-semibold text-muted-foreground">
                                {index + 1}.
                              </span>
                              <p className="truncate font-medium">
                                {stageMeta.shortLabel}
                              </p>
                              <HelpTooltip content={stageMeta.description} />
                            </div>
                            <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                              {item.note || stageMeta.description}
                            </p>
                          </div>
                          <Badge variant={statusMeta.badgeVariant}>
                            {statusMeta.label}
                          </Badge>
                        </div>
                        {item.percent !== null ? (
                          <Progress className="mt-2" value={item.percent} />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
                <p className="text-xs text-muted-foreground">
                  Current stage:{" "}
                  <span className="font-medium text-foreground">
                    {phaseLabelFromTimeline(detailData, detailData.run.phase)}
                  </span>
                </p>
              </CardContent>
            </Card>

            {detailError ? (
              <Alert variant="destructive">
                <AlertTriangle />
                <AlertTitle>Could not refresh run details</AlertTitle>
                <AlertDescription>{detailError}</AlertDescription>
              </Alert>
            ) : null}

            {detailData.run.state === "partial" ? (
              <Alert>
                <AlertTriangle />
                <AlertTitle>Run ended before finalization</AlertTitle>
                <AlertDescription>
                  This run has no final metrics file. It was likely terminated
                  early or exited before writing completion metadata.
                </AlertDescription>
              </Alert>
            ) : null}

            <div className="grid gap-4 xl:grid-cols-[1.35fr_0.95fr] 2xl:grid-cols-[1.5fr_1fr]">
              <Card className="clio-shell border-border/70">
                <CardHeader>
                  <CardTitle className="text-lg">Phase timeline</CardTitle>
                  <CardDescription>
                    Checkpoint-backed progression through CLIO phases.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {detailData.phaseTimeline.map((item, index) => (
                    <div key={item.phase} className="relative pl-7">
                      {index < detailData.phaseTimeline.length - 1 ? (
                        <span className="absolute left-[0.53rem] top-5 h-[calc(100%-0.35rem)] w-px bg-border/90" />
                      ) : null}

                      <span className="absolute top-0 left-0">
                        {phaseIcon(item.status)}
                      </span>
                      <div className="space-y-2 pb-2">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="font-medium">{item.label}</p>
                          <Badge
                            variant={
                              PHASE_STATUS_META[item.status].badgeVariant
                            }
                          >
                            {PHASE_STATUS_META[item.status].label}
                          </Badge>
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                          <span>{metricLabel(item)}</span>
                          <span>
                            {checkpointTimestampLabel(item, relativeNowMs)}
                          </span>
                        </div>
                        {item.percent !== null ? (
                          <Progress value={item.percent} />
                        ) : null}
                        {item.note ? (
                          <p className="text-xs text-muted-foreground">
                            {item.note}
                          </p>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="clio-shell border-border/70">
                <CardHeader>
                  <CardTitle className="text-lg">Live events</CardTitle>
                  <CardDescription>
                    Structured run events from manifests and phase checkpoints.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {detailData.latestEvents.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No events recorded yet.
                    </p>
                  ) : (
                    <ScrollArea className="h-[430px] 2xl:h-[520px] pr-3">
                      <div className="space-y-3">
                        {detailData.latestEvents.map((event) => (
                          <div
                            key={event.id}
                            className={cn(
                              "rounded-xl border border-border/70 border-l-4 bg-muted/20 p-3",
                              eventBorderClass(event.status),
                            )}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <p className="text-sm font-medium">
                                {event.message}
                              </p>
                              <Badge
                                variant="outline"
                                className={cn(
                                  "shrink-0",
                                  event.status === "failed" &&
                                    "border-red-300 text-red-600 dark:border-red-900 dark:text-red-300",
                                  event.status === "completed" &&
                                    "border-emerald-300 text-emerald-600 dark:border-emerald-900 dark:text-emerald-300",
                                  event.status === "running" &&
                                    "border-blue-300 text-blue-600 dark:border-blue-900 dark:text-blue-300",
                                )}
                              >
                                {event.status ?? event.type}
                              </Badge>
                            </div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              {event.source} ·{" "}
                              {formatDateTime(event.timestampUtc, {
                                localize: isHydrated,
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </div>

            <Card className="clio-shell border-border/70">
              <CardHeader>
                <CardTitle className="text-lg">Artifact readiness</CardTitle>
                <CardDescription>
                  Expected CLIO outputs across conversations, clusters, privacy,
                  evaluation, and visualization.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                {detailData.artifactStatus.map((artifact) => {
                  const artifactMeta = ARTIFACT_META[artifact.artifactKey];
                  return (
                    <div
                      key={artifact.artifactKey}
                      className={cn(
                        "rounded-xl border px-3 py-2 text-sm",
                        artifact.exists
                          ? "border-emerald-300/60 bg-emerald-100/40 dark:border-emerald-900 dark:bg-emerald-950/20"
                          : artifact.required
                            ? "border-red-300/60 bg-red-100/40 dark:border-red-900 dark:bg-red-950/20"
                            : "border-border/70 bg-muted/20",
                      )}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-medium">
                              {artifactMeta?.label ?? artifact.artifactKey}
                            </span>
                            {artifactMeta ? (
                              <HelpTooltip content={artifactMeta.description} />
                            ) : null}
                          </div>
                          <p className="mt-1 truncate text-xs text-muted-foreground">
                            {artifact.relativePath}
                          </p>
                          {artifactMeta ? (
                            <p className="truncate text-[11px] text-muted-foreground">
                              {artifact.artifactKey}
                            </p>
                          ) : null}
                        </div>
                        <Badge
                          variant={artifact.exists ? "secondary" : "outline"}
                        >
                          {artifact.exists
                            ? "present"
                            : artifact.required
                              ? "required"
                              : "optional"}
                        </Badge>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {!prioritizeVisuals ? (
              <RunVisualSummary
                visuals={visualsData}
                loading={loadingVisuals}
                error={visualsError}
              />
            ) : null}
          </>
        ) : null}

        {loadingDetail && selectedRunId ? (
          <Card className="clio-panel border-border/70">
            <CardContent className="py-5">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Syncing live details for {selectedRunId}...
              </div>
            </CardContent>
          </Card>
        ) : null}

        {!singleRunMode && showOrchestration ? (
          <section className="clio-shell border-border/70 p-4 md:p-5">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="clio-kicker">Execution</p>
                <h2 className="clio-display text-2xl">
                  Launch and monitor CLIO runs
                </h2>
              </div>
              <Badge variant="outline" className="gap-1">
                <Activity className="size-3.5" />
                Ingest and run controls
              </Badge>
            </div>
            <RunOrchestrationPanel
              autoRefresh={autoRefresh}
              onRunStarted={handleRunStarted}
            />
          </section>
        ) : null}
      </main>
    </div>
  );
}
