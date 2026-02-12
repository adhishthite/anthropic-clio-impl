"use client";

import { AlertTriangle } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { LiveRunDashboard } from "@/components/clio/live-run-dashboard";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

type RunDetailContentProps = {
  runId: string;
};

export function RunDetailContent({ runId }: RunDetailContentProps) {
  const [notFound, setNotFound] = useState(false);
  const [checking, setChecking] = useState(true);

  const checkRunExists = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {
          cache: "no-store",
          signal,
        });
        if (!response.ok) {
          setNotFound(true);
        } else {
          setNotFound(false);
        }
      } catch {
        if (signal?.aborted) {
          return;
        }
        setNotFound(true);
      } finally {
        if (!signal?.aborted) {
          setChecking(false);
        }
      }
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

  if (checking) {
    return null;
  }

  if (notFound) {
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
