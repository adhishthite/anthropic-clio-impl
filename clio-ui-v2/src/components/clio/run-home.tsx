"use client";

import { ChevronRight, FolderKanban, RefreshCw } from "lucide-react";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import type { RunListResponse, RunState } from "@/lib/clio-types";
import { formatRelativeTime } from "@/lib/format-utils";
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
  const [showAllRuns, setShowAllRuns] = useState<boolean>(false);

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

  const runSnapshot = useMemo(() => {
    if (!runsData) {
      return { total: 0, running: 0, completed: 0, failed: 0 };
    }

    return runsData.runs.reduce(
      (acc, run) => {
        acc.total += 1;

        switch (run.state) {
          case "running":
            acc.running += 1;
            break;
          case "completed":
          case "completed_with_warnings":
            acc.completed += 1;
            break;
          case "failed":
            acc.failed += 1;
            break;
        }

        return acc;
      },
      { total: 0, running: 0, completed: 0, failed: 0 },
    );
  }, [runsData]);

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
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="relative flex size-2">
                  <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary/60" />
                  <span className="relative inline-flex size-2 rounded-full bg-primary" />
                </span>
                {runsData?.generatedAtUtc
                  ? `Updated ${formatRelativeTime(runsData.generatedAtUtc)}`
                  : "Waiting for first sync"}
              </div>

              <h1 className="clio-display text-3xl leading-tight md:text-4xl">
                Choose a run to inspect
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
                {[
                  { label: "Total runs", value: runSnapshot.total },
                  { label: "Running", value: runSnapshot.running },
                  { label: "Completed", value: runSnapshot.completed },
                  { label: "Failed", value: runSnapshot.failed },
                ].map((cell) => (
                  <div key={cell.label} className="clio-panel-subtle px-3 py-2">
                    <p className="text-xs text-muted-foreground">
                      {cell.label}
                    </p>
                    <p
                      className={cn(
                        "mt-1 text-xl",
                        cell.value > 0
                          ? "font-bold text-foreground"
                          : "font-medium text-muted-foreground",
                      )}
                    >
                      {cell.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section>
          <Card className="clio-panel border-border/70">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle className="text-lg">Runs</CardTitle>
                  <CardDescription>
                    Select a run to open its dashboard.
                  </CardDescription>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex items-center gap-2 rounded-md border border-border/70 px-2.5 py-1.5 text-sm">
                    <Switch
                      checked={autoRefresh}
                      onCheckedChange={setAutoRefresh}
                      aria-label="Toggle run list auto refresh"
                    />
                    Auto-refresh (30s)
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="gap-2"
                    onClick={() => void loadRuns()}
                  >
                    <RefreshCw className="size-4" />
                    Refresh
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {runsError ? (
                <Alert variant="destructive">
                  <AlertTitle>Could not load runs</AlertTitle>
                  <AlertDescription>{runsError}</AlertDescription>
                </Alert>
              ) : null}
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

              {(() => {
                const DEFAULT_VISIBLE = 12;
                const allRuns = runsData?.runs ?? [];
                const visibleRuns = showAllRuns
                  ? allRuns
                  : allRuns.slice(0, DEFAULT_VISIBLE);
                const hiddenCount = allRuns.length - DEFAULT_VISIBLE;
                return (
                  <>
                    <div className="grid gap-2 2xl:grid-cols-3">
                      {visibleRuns.map((run) => {
                        const isCompleted =
                          run.state === "completed" ||
                          run.state === "completed_with_warnings";
                        return (
                          <Link
                            key={run.runId}
                            href={`/runs/${encodeURIComponent(run.runId)}`}
                            className={cn(
                              "group flex items-center justify-between gap-3 rounded-lg border px-3 py-2 transition-shadow hover:shadow-sm",
                              RUN_STATE_CARD_CLASS[run.state],
                            )}
                          >
                            <div className="min-w-0 space-y-0.5">
                              <Badge
                                variant={RUN_STATE_META[run.state].badgeVariant}
                                className="text-[11px]"
                              >
                                {RUN_STATE_META[run.state].label}
                              </Badge>
                              <p className="truncate text-sm font-medium">
                                {run.runId}
                              </p>
                              {run.conversationCountInput > 0 ? (
                                <p className="truncate text-xs text-muted-foreground">
                                  {run.conversationCountInput} conversations
                                  {run.clusterCountTotal > 0
                                    ? ` · ${run.clusterCountTotal} clusters`
                                    : ""}
                                </p>
                              ) : null}
                              <p className="truncate text-xs text-muted-foreground">
                                {formatRelativeTime(run.updatedAtUtc)}
                                {!isCompleted
                                  ? ` · ${run.overallProgressPercent.toFixed(1)}%`
                                  : ""}
                              </p>
                            </div>
                            <ChevronRight className="size-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                          </Link>
                        );
                      })}
                    </div>
                    {hiddenCount > 0 && !showAllRuns ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-2 w-full"
                        onClick={() => setShowAllRuns(true)}
                      >
                        Show {hiddenCount} more run
                        {hiddenCount > 1 ? "s" : ""}
                      </Button>
                    ) : null}
                    {showAllRuns && allRuns.length > DEFAULT_VISIBLE ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-2 w-full"
                        onClick={() => setShowAllRuns(false)}
                      >
                        Show fewer
                      </Button>
                    ) : null}
                  </>
                );
              })()}
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
