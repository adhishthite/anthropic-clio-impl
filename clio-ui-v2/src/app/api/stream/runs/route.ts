import { discoverRuns, getRunsRootPath } from "@/lib/clio-runs";
import { createSseResponse, normalizeStreamInterval } from "@/lib/sse";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_INTERVAL_MS = 30000;
const HEARTBEAT_MS = 15000;

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const limitParam = Number(url.searchParams.get("limit") ?? "120");
  const intervalParam = Number(
    url.searchParams.get("intervalMs") ?? DEFAULT_INTERVAL_MS,
  );
  const intervalMs = normalizeStreamInterval(
    intervalParam,
    DEFAULT_INTERVAL_MS,
  );
  const limit =
    Number.isFinite(limitParam) && limitParam > 0
      ? Math.floor(limitParam)
      : 120;
  const runsRoot = getRunsRootPath();

  return createSseResponse((writer) => {
    let active = true;

    const emitSnapshot = async () => {
      if (!active) {
        return;
      }
      try {
        const runs = await discoverRuns(limit);
        writer.send("runs_snapshot", {
          generatedAtUtc: new Date().toISOString(),
          runsRoot,
          runs,
        });
      } catch (error) {
        writer.send("runs_error", {
          generatedAtUtc: new Date().toISOString(),
          runsRoot,
          error:
            error instanceof Error ? error.message : "Failed to discover runs.",
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
