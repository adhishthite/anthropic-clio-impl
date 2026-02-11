import { NextResponse } from "next/server";

import { readRunJobLogTail } from "@/lib/clio-run-jobs";
import { getRunsRootPath } from "@/lib/clio-runs";
import type { RunLogResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunLogResponse,
): NextResponse<RunLogResponse> {
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
      Pragma: "no-cache",
      Expires: "0",
    },
  });
}

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> },
): Promise<NextResponse<RunLogResponse>> {
  const { runId } = await context.params;
  const url = new URL(request.url);
  const linesParam = Number(url.searchParams.get("lines") ?? "120");
  const lines =
    Number.isFinite(linesParam) && linesParam > 0
      ? Math.floor(linesParam)
      : 120;
  const runsRoot = getRunsRootPath();

  const payload = await readRunJobLogTail({
    runId: decodeURIComponent(runId),
    lines,
    runsRoot,
  });
  return withNoStoreHeaders({
    generatedAtUtc: new Date().toISOString(),
    runsRoot,
    ...payload,
  });
}
