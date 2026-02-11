import { collectRunJobs, readRunJobLogTail } from "@/lib/clio-run-jobs";
import { getRunsRootPath } from "@/lib/clio-runs";
import { createSseResponse, normalizeStreamInterval } from "@/lib/sse";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_INTERVAL_MS = 30000;
const HEARTBEAT_MS = 15000;

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const runsRoot = getRunsRootPath();
  const limitParam = Number(url.searchParams.get("limit") ?? "120");
  const intervalParam = Number(
    url.searchParams.get("intervalMs") ?? DEFAULT_INTERVAL_MS,
  );
  const logLinesParam = Number(url.searchParams.get("logLines") ?? "180");
  const logRunId = (url.searchParams.get("logRunId") ?? "").trim();

  const limit =
    Number.isFinite(limitParam) && limitParam > 0
      ? Math.floor(limitParam)
      : 120;
  const intervalMs = normalizeStreamInterval(
    intervalParam,
    DEFAULT_INTERVAL_MS,
  );
  const logLines =
    Number.isFinite(logLinesParam) && logLinesParam > 0
      ? Math.floor(logLinesParam)
      : 180;

  return createSseResponse((writer) => {
    let active = true;

    const emitSnapshot = async () => {
      if (!active) {
        return;
      }
      try {
        const jobs = await collectRunJobs(runsRoot, limit);
        const logData = logRunId
          ? await readRunJobLogTail({
              runId: logRunId,
              lines: logLines,
              runsRoot,
            })
          : null;
        writer.send("jobs_snapshot", {
          generatedAtUtc: new Date().toISOString(),
          runsRoot,
          jobs,
          logData,
        });
      } catch (error) {
        writer.send("jobs_error", {
          generatedAtUtc: new Date().toISOString(),
          runsRoot,
          error:
            error instanceof Error ? error.message : "Failed to stream jobs.",
        });
      }
    };

    void emitSnapshot();
    const snapshotInterval = setInterval(() => {
      void emitSnapshot();
    }, intervalMs);
    const heartbeatInterval = setInterval(() => {
      writer.comment("heartbeat");
    }, HEARTBEAT_MS);

    return () => {
      active = false;
      clearInterval(snapshotInterval);
      clearInterval(heartbeatInterval);
    };
  });
}
