import { NextResponse } from "next/server";

import { loadClusterConversations } from "@/lib/clio-runs";
import type { RunClusterConversationsResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunClusterConversationsResponse,
): NextResponse<RunClusterConversationsResponse> {
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
      Pragma: "no-cache",
      Expires: "0",
    },
  });
}

export async function GET(
  _: Request,
  context: { params: Promise<{ runId: string; clusterId: string }> },
): Promise<NextResponse<RunClusterConversationsResponse | { error: string }>> {
  const { runId, clusterId } = await context.params;
  const normalizedRunId = decodeURIComponent(runId);
  const normalizedClusterId = Number.parseInt(
    decodeURIComponent(clusterId),
    10,
  );
  if (!Number.isFinite(normalizedClusterId)) {
    return NextResponse.json(
      { error: `Cluster id '${clusterId}' is invalid.` },
      {
        status: 400,
        headers: {
          "Cache-Control":
            "no-store, no-cache, must-revalidate, proxy-revalidate",
        },
      },
    );
  }

  const payload = await loadClusterConversations(
    normalizedRunId,
    normalizedClusterId,
  );
  if (!payload) {
    return NextResponse.json(
      { error: `Run '${runId}' was not found.` },
      {
        status: 404,
        headers: {
          "Cache-Control":
            "no-store, no-cache, must-revalidate, proxy-revalidate",
        },
      },
    );
  }

  return withNoStoreHeaders(payload);
}
