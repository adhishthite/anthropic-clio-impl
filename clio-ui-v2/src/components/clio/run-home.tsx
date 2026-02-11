"use client";

import {
  ArrowRight,
  Clock3,
  FolderKanban,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { RunOrchestrationPanel } from "@/components/clio/run-orchestration-panel";
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
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import type { RunListResponse, RunState } from "@/lib/clio-types";
import { cn } from "@/lib/utils";

const RUN_LIST_REFRESH_INTERVAL_MS = 30000;

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

const RUN_STATE_CARD_CLASS: Record<RunState, string> = {
  running:
    "border-blue-300/70 bg-blue-100/45 dark:border-blue-900/60 dark:bg-blue-950/20",
  completed:
    "border-emerald-300/70 bg-emerald-100/45 dark:border-emerald-900/60 dark:bg-emerald-950/20",
  completed_with_warnings:
    "border-amber-300/70 bg-amber-100/45 dark:border-amber-900/60 dark:bg-amber-950/20",
  failed:
    "border-red-300/70 bg-red-100/45 dark:border-red-900/60 dark:bg-red-950/20",
  partial:
    "border-violet-300/70 bg-violet-100/45 dark:border-violet-900/60 dark:bg-violet-950/20",
};

const RUN_STATE_DOT_CLASS: Record<RunState, string> = {
  running: "bg-blue-500",
  completed: "bg-emerald-500",
  completed_with_warnings: "bg-amber-500",
  failed: "bg-red-500",
  partial: "bg-violet-500",
};

function formatRelativeTime(value: string): string {
  if (!value) {
    return "n/a";
  }
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return "n/a";
  }
  const diffSeconds = Math.round((timestamp - Date.now()) / 1000);
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

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }
  return (await response.json()) as T;
}

export function RunHome() {
  const router = useRouter();

  const [runsData, setRunsData] = useState<RunListResponse | null>(null);
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [runsError, setRunsError] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [selectedRunId, setSelectedRunId] = useState<string>("");

  const loadRuns = useCallback(
    async (signal?: AbortSignal, options?: { background?: boolean }) => {
      const background = options?.background ?? false;
      if (!background) {
        setLoadingRuns(true);
      }
      try {
        const payload = await fetchJson<RunListResponse>(
          "/api/runs?limit=80",
          signal,
        );
        setRunsData(payload);
        setRunsError("");
        setSelectedRunId((current) => {
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
        if (!signal?.aborted && !background) {
          setLoadingRuns(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadRuns(controller.signal);
    return () => {
      controller.abort();
    };
  }, [loadRuns]);

  useEffect(() => {
    if (!autoRefresh) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void loadRuns(undefined, { background: true });
    }, RUN_LIST_REFRESH_INTERVAL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [autoRefresh, loadRuns]);

  const hasRuns = Boolean(runsData && runsData.runs.length > 0);

  const selectedRun = useMemo(
    () => runsData?.runs.find((run) => run.runId === selectedRunId) ?? null,
    [runsData, selectedRunId],
  );
  const runSnapshot = useMemo(() => {
    if (!runsData) {
      return {
        total: 0,
        running: 0,
        completed: 0,
        failed: 0,
      };
    }
    return runsData.runs.reduce(
      (acc, run) => {
        acc.total += 1;
        if (run.state === "running") {
          acc.running += 1;
        } else if (
          run.state === "completed" ||
          run.state === "completed_with_warnings"
        ) {
          acc.completed += 1;
        } else if (run.state === "failed") {
          acc.failed += 1;
        }
        return acc;
      },
      {
        total: 0,
        running: 0,
        completed: 0,
        failed: 0,
      },
    );
  }, [runsData]);

  const openSelectedRun = useCallback(() => {
    if (!selectedRunId) {
      return;
    }
    router.push(`/runs/${encodeURIComponent(selectedRunId)}`);
  }, [router, selectedRunId]);

  const handleRunStarted = useCallback(
    (runId: string) => {
      router.push(`/runs/${encodeURIComponent(runId)}`);
    },
    [router],
  );

  return (
    <div className="relative min-h-screen pb-10">
      <div className="clio-orb top-0 -left-16 h-56 w-56 bg-primary/30" />
      <div className="clio-orb top-16 right-0 h-64 w-64 bg-accent/20 [animation-delay:2s]" />

      <main className="mx-auto flex w-full max-w-[2280px] flex-col gap-5 px-3 py-5 md:px-6 md:py-8 2xl:px-8">
        <section className="clio-shell relative px-5 py-5 md:px-7 md:py-6">
          <div className="clio-grid-pattern pointer-events-none absolute inset-0 opacity-35" />
          <div className="relative grid gap-4 xl:grid-cols-[1.45fr_0.8fr] 2xl:grid-cols-[1.7fr_0.8fr] xl:items-end">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary" className="gap-1">
                  <Sparkles className="size-3.5" />
                  Run workspace
                </Badge>
                <Badge variant="outline" className="gap-1">
                  <Clock3 className="size-3.5" />
                  {runsData?.generatedAtUtc
                    ? `Updated ${formatRelativeTime(runsData.generatedAtUtc)}`
                    : "Waiting for first sync"}
                </Badge>
              </div>

              <h1 className="clio-display text-3xl leading-tight md:text-4xl">
                Pick one run. Keep everything else out of the way.
              </h1>
              <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
                CLIO is easiest to inspect one run at a time. This home stays
                lightweight, while each run page shows timelines, map and
                hierarchy visuals, privacy audit outputs, and evaluation
                metrics.
              </p>
            </div>

            <div className="clio-panel p-4 md:p-5">
              <p className="clio-kicker">Run snapshot</p>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <div className="clio-panel-subtle px-3 py-2">
                  <p className="text-xs text-muted-foreground">Total runs</p>
                  <p className="mt-1 text-xl font-semibold">
                    {runSnapshot.total}
                  </p>
                </div>
                <div className="clio-panel-subtle px-3 py-2">
                  <p className="text-xs text-muted-foreground">Running</p>
                  <p className="mt-1 text-xl font-semibold">
                    {runSnapshot.running}
                  </p>
                </div>
                <div className="clio-panel-subtle px-3 py-2">
                  <p className="text-xs text-muted-foreground">Completed</p>
                  <p className="mt-1 text-xl font-semibold">
                    {runSnapshot.completed}
                  </p>
                </div>
                <div className="clio-panel-subtle px-3 py-2">
                  <p className="text-xs text-muted-foreground">Failed</p>
                  <p className="mt-1 text-xl font-semibold">
                    {runSnapshot.failed}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.45fr_0.8fr] 2xl:grid-cols-[1.75fr_0.85fr]">
          <Card className="clio-panel border-border/70">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Open run dashboard</CardTitle>
              <CardDescription>
                Choose one run and open its dedicated dashboard page.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
                <div className="space-y-2">
                  <label className="clio-kicker" htmlFor="run-select-home">
                    Run ID
                  </label>
                  <Select
                    value={selectedRunId}
                    onValueChange={setSelectedRunId}
                  >
                    <SelectTrigger id="run-select-home">
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
                  className="h-10 gap-2"
                  onClick={openSelectedRun}
                  disabled={!selectedRunId}
                >
                  Open dashboard
                  <ArrowRight className="size-4" />
                </Button>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center gap-2 rounded-md border border-border/70 px-2.5 py-1.5 text-sm">
                  <Switch
                    checked={autoRefresh}
                    onCheckedChange={setAutoRefresh}
                    aria-label="Toggle run list auto refresh"
                  />
                  Auto refresh run list (30s)
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => void loadRuns()}
                >
                  <RefreshCw className="size-4" />
                  Refresh list
                </Button>
              </div>

              {selectedRun ? (
                <div className="rounded-lg border border-border/70 bg-muted/20 p-3 text-sm">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="font-medium">{selectedRun.runId}</p>
                    <Badge
                      variant={RUN_STATE_META[selectedRun.state].badgeVariant}
                    >
                      {RUN_STATE_META[selectedRun.state].label}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    updated {formatRelativeTime(selectedRun.updatedAtUtc)} ·
                    progress {selectedRun.overallProgressPercent.toFixed(1)}%
                  </p>
                </div>
              ) : null}

              {runsError ? (
                <Alert variant="destructive">
                  <AlertTitle>Could not load runs</AlertTitle>
                  <AlertDescription>{runsError}</AlertDescription>
                </Alert>
              ) : null}
            </CardContent>
          </Card>

          <Card className="clio-panel border-border/70">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Recent runs</CardTitle>
              <CardDescription>
                Lightweight index view. Open a run to see full details.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {loadingRuns && !runsData ? (
                <div className="grid gap-2 2xl:grid-cols-3">
                  {[1, 2, 3].map((row) => (
                    <Skeleton key={row} className="h-12 w-full" />
                  ))}
                </div>
              ) : null}

              {!loadingRuns && !hasRuns ? (
                <Empty className="border-border bg-card/85">
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <FolderKanban className="size-5" />
                    </EmptyMedia>
                    <EmptyTitle>No runs yet</EmptyTitle>
                    <EmptyDescription>
                      Launch your first run from this page.
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : null}

              <div className="grid gap-2 2xl:grid-cols-3">
                {runsData?.runs.slice(0, 12).map((run) => (
                  <div
                    key={run.runId}
                    className={cn(
                      "flex items-center justify-between gap-3 rounded-lg border px-3 py-2",
                      RUN_STATE_CARD_CLASS[run.state],
                    )}
                  >
                    <div className="min-w-0 space-y-0.5">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={cn(
                            "size-2 rounded-full",
                            RUN_STATE_DOT_CLASS[run.state],
                          )}
                        />
                        <span className="text-[11px] font-medium text-muted-foreground">
                          {RUN_STATE_META[run.state].label}
                        </span>
                      </div>
                      <p className="truncate text-sm font-medium">
                        {run.runId}
                      </p>
                      <p className="truncate text-xs text-muted-foreground">
                        {formatRelativeTime(run.updatedAtUtc)} ·{" "}
                        {run.overallProgressPercent.toFixed(1)}%
                      </p>
                    </div>
                    <Button asChild size="sm" variant="outline">
                      <Link href={`/runs/${encodeURIComponent(run.runId)}`}>
                        Open
                      </Link>
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="clio-shell border-border/70 p-4 md:p-5">
          <RunOrchestrationPanel
            autoRefresh={autoRefresh}
            onRunStarted={handleRunStarted}
          />
        </section>
      </main>
    </div>
  );
}
