import { NextResponse } from "next/server";

import { loadRunDetail } from "@/lib/clio-runs";
import type { RunDetailResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunDetailResponse,
): NextResponse<RunDetailResponse> {
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
): Promise<NextResponse<RunDetailResponse | { error: string }>> {
  const { runId } = await context.params;
  const detail = await loadRunDetail(decodeURIComponent(runId));
  if (!detail) {
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

  return withNoStoreHeaders(detail);
}
