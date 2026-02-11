import { getRunsRootPath, loadRunDetail } from "@/lib/clio-runs";
import { createSseResponse, normalizeStreamInterval } from "@/lib/sse";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_INTERVAL_MS = 30000;
const HEARTBEAT_MS = 15000;

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> },
): Promise<Response> {
  const { runId } = await context.params;
  const decodedRunId = decodeURIComponent(runId);

  const url = new URL(request.url);
  const intervalParam = Number(
    url.searchParams.get("intervalMs") ?? DEFAULT_INTERVAL_MS,
  );
  const intervalMs = normalizeStreamInterval(
    intervalParam,
    DEFAULT_INTERVAL_MS,
  );
  const runsRoot = getRunsRootPath();

  return createSseResponse((writer) => {
    let active = true;

    const emitSnapshot = async () => {
      if (!active) {
        return;
      }
      try {
        const detail = await loadRunDetail(decodedRunId);
        if (!detail) {
          writer.send("run_detail_error", {
            generatedAtUtc: new Date().toISOString(),
            runsRoot,
            runId: decodedRunId,
            error: `Run '${decodedRunId}' not found.`,
          });
          return;
        }
        writer.send("run_detail", detail);
      } catch (error) {
        writer.send("run_detail_error", {
          generatedAtUtc: new Date().toISOString(),
          runsRoot,
          runId: decodedRunId,
          error:
            error instanceof Error
              ? error.message
              : "Failed to load run detail.",
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
