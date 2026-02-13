"use client";

import {
  ArrowRight,
  BarChart3,
  Eye,
  RefreshCw,
  ShieldCheck,
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
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import type { RunListResponse, RunState } from "@/lib/clio-types";
import { formatRelativeTime } from "@/lib/format-utils";
import { cn } from "@/lib/utils";

const RUN_LIST_REFRESH_INTERVAL_MS = 30000;
const CLIO_LINKS = {
  paper: "https://arxiv.org/abs/2412.13678",
  research: "https://www.anthropic.com/research/clio",
  privacyCenter:
    "https://privacy.claude.com/en/articles/10807912-how-does-clio-analyze-usage-patterns-while-protecting-user-data",
};

const CLIO_RESOURCE_LINKS = [
  { label: "Read arXiv paper", href: CLIO_LINKS.paper },
  { label: "Anthropic overview", href: CLIO_LINKS.research },
  { label: "Privacy model", href: CLIO_LINKS.privacyCenter },
] as const;

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

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }
  return (await response.json()) as T;
}

const PRIVACY_POINTS = [
  "Automatic anonymization plus aggregation before outputs are reviewed.",
  "Minimum aggregation thresholds and automated checks block low-frequency topics.",
  "Raw conversations and customer-specific data are not surfaced in aggregate analysis.",
  "Human reviewers only see high-level clusters, summaries, and topic-level trends.",
] as const;

const ANALYSIS_POINTS = [
  "Bottom-up discovery from conversation data, not only pre-defined safety checks.",
  "Semantic clustering of conversation-level facets into interpretable topics.",
  "Multilevel hierarchy for exploration across language, task, and usage patterns.",
  "Designed to scale to large corpora and produce reproducible aggregate signals.",
] as const;

const CLIO_PIPELINE_STEPS = [
  {
    step: "Facet extraction",
    detail:
      "Convert each conversation into metadata attributes that preserve meaning, not content.",
  },
  {
    step: "Semantic clustering",
    detail:
      "Group similar conversations into usage clusters for coarse-grained pattern discovery.",
  },
  {
    step: "Cluster labeling",
    detail:
      "Generate concise labels and summaries with private details removed.",
  },
  {
    step: "Hierarchy + audit",
    detail:
      "Build exploration trees and run privacy gates and threshold checks before surfacing.",
  },
] as const;

export function ClioHome() {
  const router = useRouter();
  const [runsData, setRunsData] = useState<RunListResponse | null>(null);
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [runsError, setRunsError] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);

  const openResource = useCallback((href: string) => {
    window.open(href, "_blank", "noopener,noreferrer");
  }, []);

  const loadRuns = useCallback(
    async (signal?: AbortSignal, background = false) => {
      if (!background) {
        setLoadingRuns(true);
      }
      try {
        const payload = await fetchJson<RunListResponse>(
          "/api/runs?limit=20",
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
      void loadRuns(undefined, true);
    }, RUN_LIST_REFRESH_INTERVAL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [autoRefresh, loadRuns]);

  const runSnapshot = useMemo(() => {
    if (!runsData) {
      return { total: 0, running: 0, completed: 0, failed: 0 };
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
      { total: 0, running: 0, completed: 0, failed: 0 },
    );
  }, [runsData]);

  const hasRuns = Boolean(runsData && runsData.runs.length > 0);

  return (
    <div className="relative min-h-screen pb-10">
      <div className="clio-orb top-0 -left-16 h-56 w-56 bg-primary/25" />
      <div className="clio-orb top-12 right-0 h-56 w-56 bg-accent/15 [animation-delay:2s]" />

      <main className="mx-auto flex w-full max-w-[2280px] flex-col gap-5 px-3 py-5 md:px-6 md:py-8 2xl:px-8">
        <section className="clio-shell relative px-5 py-5 md:px-7 md:py-6">
          <div className="clio-grid-pattern pointer-events-none absolute inset-0 opacity-35" />
          <div className="relative grid gap-4 xl:grid-cols-[1.45fr_0.8fr] 2xl:grid-cols-[1.7fr_0.8fr] xl:items-start">
            <div className="space-y-3">
              <h1 className="clio-display text-3xl font-bold leading-tight md:text-4xl">
                CLIO: Privacy-first usage analysis for conversation systems
              </h1>
              <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
                CLIO is an abstraction layer for aggregated conversation
                analysis: turn raw usage data into trends and risk signals
                without exposing raw individual conversations. It is built to
                support both product understanding and safety observability at
                scale.
              </p>
              <div className="mt-6 md:mt-7 flex flex-wrap gap-2">
                {CLIO_RESOURCE_LINKS.map((link) => (
                  <Button
                    key={link.label}
                    variant="default"
                    size="sm"
                    onClick={() => openResource(link.href)}
                  >
                    {link.label}
                  </Button>
                ))}
              </div>
            </div>
            <Card className="clio-panel clio-accent-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Pipeline snapshot</CardTitle>
                <CardDescription>Latest local run states</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
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
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-2">
          <Card className="clio-panel clio-accent-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheck className="size-5" />
                Why this is stronger for privacy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                {PRIVACY_POINTS.map((point) => (
                  <li key={point} className="flex items-start gap-2">
                    <span className="mt-1 size-1.5 rounded-full bg-primary" />
                    <span>{point}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
          <Card className="clio-panel clio-accent-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="size-5" />
                Why this is stronger for analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                {ANALYSIS_POINTS.map((point) => (
                  <li key={point} className="flex items-start gap-2">
                    <span className="mt-1 size-1.5 rounded-full bg-accent-foreground" />
                    <span>{point}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-5 lg:grid-cols-[1.35fr_0.65fr]">
          <Card className="clio-panel clio-accent-card flex min-h-0 max-h-[30rem] flex-col">
            <CardHeader>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle>How CLIO-style analysis flows</CardTitle>
                  <CardDescription>
                    This repo mirrors the same four-stage pattern described in
                    the paper.
                  </CardDescription>
                </div>
                <Button
                  asChild
                  size="sm"
                  variant="outline"
                  className="shrink-0"
                >
                  <a href={CLIO_LINKS.paper} rel="noreferrer" target="_blank">
                    Source paper
                  </a>
                </Button>
              </div>
            </CardHeader>
            <CardContent className="grid min-h-0 flex-1 gap-3 overflow-y-auto">
              {CLIO_PIPELINE_STEPS.map((step, index) => (
                <div
                  key={step.step}
                  className="grid gap-1 rounded-lg border border-border/70 p-3"
                >
                  <p className="text-sm font-medium">
                    {index + 1}. {step.step}
                  </p>
                  <p className="text-xs text-muted-foreground">{step.detail}</p>
                </div>
              ))}
            </CardContent>
          </Card>
          <Card className="clio-panel clio-accent-card flex min-h-0 max-h-[30rem] flex-col">
            <CardHeader>
              <CardTitle className="text-lg">Run details</CardTitle>
              <CardDescription>
                Open a specific run to inspect cluster outputs, traces, and
                diagnostics.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 min-h-0 flex-1 overflow-y-auto">
              {runsError ? (
                <Alert variant="destructive">
                  <AlertTitle>Could not load runs</AlertTitle>
                  <AlertDescription>{runsError}</AlertDescription>
                </Alert>
              ) : null}

              {loadingRuns && !runsData ? (
                <div className="space-y-2">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : null}

              {!loadingRuns && hasRuns ? (
                <div className="space-y-2">
                  {runsData?.runs.slice(0, 6).map((run) => (
                    <Link
                      key={run.runId}
                      href={`/runs/${encodeURIComponent(run.runId)}`}
                      className="flex items-center justify-between gap-3 rounded-lg border border-border/70 px-3 py-2 text-sm transition-shadow hover:border-primary/50"
                    >
                      <div className="min-w-0">
                        <Badge
                          variant={RUN_STATE_META[run.state].badgeVariant}
                          className="text-[11px]"
                        >
                          {RUN_STATE_META[run.state].label}
                        </Badge>
                        <p className="mt-1 truncate font-medium">{run.runId}</p>
                        <p className="truncate text-xs text-muted-foreground">
                          {run.state === "running"
                            ? `${run.overallProgressPercent.toFixed(1)}% complete`
                            : `Updated ${formatRelativeTime(run.updatedAtUtc)}`}
                        </p>
                      </div>
                      <ArrowRight className="size-4 text-muted-foreground" />
                    </Link>
                  ))}
                </div>
              ) : null}

              {!loadingRuns && !hasRuns ? (
                <Empty className="border-border/70 bg-card/85">
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <Eye className="size-5" />
                    </EmptyMedia>
                    <EmptyTitle>No runs yet</EmptyTitle>
                    <EmptyDescription>
                      Launch a run below to generate clusters, hierarchy, and
                      privacy-audited outputs.
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : null}
            </CardContent>
          </Card>
        </section>

        <section className="clio-shell border-border/70 p-4 md:p-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="clio-kicker">Launch and configure CLIO run</p>
            <div className="flex items-center gap-2 rounded-md border border-border/70 px-2.5 py-1.5 text-sm">
              <Switch
                checked={autoRefresh}
                onCheckedChange={setAutoRefresh}
                aria-label="Toggle run list auto refresh"
              />
              <span className="text-xs text-muted-foreground">
                Auto-refresh runs (30s)
              </span>
              <Button
                onClick={() => void loadRuns()}
                variant="outline"
                size="sm"
                className="gap-2"
              >
                <RefreshCw className="size-4" />
                Refresh runs
              </Button>
            </div>
          </div>
          <RunOrchestrationPanel
            autoRefresh={autoRefresh}
            onRunStarted={(runId) => {
              router.push(`/runs/${encodeURIComponent(runId)}`);
              void loadRuns(undefined, true);
            }}
          />
        </section>
      </main>
    </div>
  );
}
