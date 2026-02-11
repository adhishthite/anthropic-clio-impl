import { NextResponse } from "next/server";

import { collectRunJobs } from "@/lib/clio-run-jobs";
import { getRunsRootPath } from "@/lib/clio-runs";
import type { RunJobsResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunJobsResponse,
): NextResponse<RunJobsResponse> {
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
): Promise<NextResponse<RunJobsResponse>> {
  const runsRoot = getRunsRootPath();
  const url = new URL(request.url);
  const limitParam = Number(url.searchParams.get("limit") ?? "120");
  const limit =
    Number.isFinite(limitParam) && limitParam > 0
      ? Math.floor(limitParam)
      : 120;

  const jobs = await collectRunJobs(runsRoot, limit);
  return withNoStoreHeaders({
    generatedAtUtc: new Date().toISOString(),
    runsRoot,
    jobs,
  });
}
