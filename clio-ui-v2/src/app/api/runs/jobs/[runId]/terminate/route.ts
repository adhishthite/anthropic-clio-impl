import { NextResponse } from "next/server";

import { terminateRunJob } from "@/lib/clio-run-jobs";
import { getRunsRootPath } from "@/lib/clio-runs";
import type { RunTerminateResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunTerminateResponse,
  status = 200,
): NextResponse<RunTerminateResponse> {
  return NextResponse.json(payload, {
    status,
    headers: {
      "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
      Pragma: "no-cache",
      Expires: "0",
    },
  });
}

export async function POST(
  _: Request,
  context: { params: Promise<{ runId: string }> },
): Promise<NextResponse<RunTerminateResponse>> {
  const { runId } = await context.params;
  const runsRoot = getRunsRootPath();
  const result = await terminateRunJob(decodeURIComponent(runId), runsRoot);

  return withNoStoreHeaders(
    {
      generatedAtUtc: new Date().toISOString(),
      runsRoot,
      runId,
      ok: result.ok,
      status: result.status,
      job: result.job,
    },
    result.ok ? 200 : 400,
  );
}
