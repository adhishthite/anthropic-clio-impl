"use client";

import { AlertTriangle, Loader2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { LiveRunDashboard } from "@/components/clio/live-run-dashboard";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

type RunDetailContentProps = {
  runId: string;
};

const MAX_INIT_RETRIES = 5;
const RETRY_DELAY_MS = 2000;

type RunCheckStatus = "checking" | "initializing" | "ready" | "not_found";

export function RunDetailContent({ runId }: RunDetailContentProps) {
  const [status, setStatus] = useState<RunCheckStatus>("checking");

  const checkRunExists = useCallback(
    async (signal?: AbortSignal) => {
      for (let attempt = 0; attempt <= MAX_INIT_RETRIES; attempt++) {
        if (signal?.aborted) return;

        try {
          const response = await fetch(
            `/api/runs/${encodeURIComponent(runId)}`,
            { cache: "no-store", signal },
          );
          if (response.ok) {
            setStatus("ready");
            return;
          }
        } catch {
          if (signal?.aborted) return;
        }

        if (attempt === 0) {
          setStatus("initializing");
        }

        if (attempt < MAX_INIT_RETRIES) {
          await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
        }
      }

      setStatus("not_found");
    },
    [runId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void checkRunExists(controller.signal);
    return () => {
      controller.abort();
    };
  }, [checkRunExists]);

  if (status === "checking") {
    return null;
  }

  if (status === "initializing") {
    return (
      <div className="mx-auto max-w-xl px-4 py-16">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Loader2 className="size-5 animate-spin" />
          <p className="text-sm">
            Waiting for run{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
              {runId}
            </code>{" "}
            to initialize...
          </p>
        </div>
      </div>
    );
  }

  if (status === "not_found") {
    return (
      <div className="mx-auto max-w-xl px-4 py-16">
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>Run not found</AlertTitle>
          <AlertDescription className="space-y-3">
            <p>
              No run data exists for{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                {runId}
              </code>
              . It may have been deleted or the ID is incorrect.
            </p>
            <Button asChild variant="outline" size="sm">
              <Link href="/">Back to runs</Link>
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return <LiveRunDashboard lockedRunId={runId} showOrchestration={false} />;
}
