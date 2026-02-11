import { NextResponse } from "next/server";

import { loadRunVisuals } from "@/lib/clio-runs";
import type { RunVisualsResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunVisualsResponse,
): NextResponse<RunVisualsResponse> {
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
  context: { params: Promise<{ runId: string }> },
): Promise<NextResponse<RunVisualsResponse | { error: string }>> {
  const { runId } = await context.params;
  const visuals = await loadRunVisuals(decodeURIComponent(runId));
  if (!visuals) {
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

  return withNoStoreHeaders(visuals);
}
