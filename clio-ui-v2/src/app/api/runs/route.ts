import { NextResponse } from "next/server";

import { discoverRuns, getRunsRootPath } from "@/lib/clio-runs";
import type { RunListResponse } from "@/lib/clio-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function withNoStoreHeaders(
  payload: RunListResponse,
): NextResponse<RunListResponse> {
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
): Promise<NextResponse<RunListResponse>> {
  const url = new URL(request.url);
  const limitParam = Number(url.searchParams.get("limit") ?? "100");
  const limit =
    Number.isFinite(limitParam) && limitParam > 0
      ? Math.floor(limitParam)
      : 100;

  const runs = await discoverRuns(limit);
  return withNoStoreHeaders({
    generatedAtUtc: new Date().toISOString(),
    runsRoot: getRunsRootPath(),
    runs,
  });
}
