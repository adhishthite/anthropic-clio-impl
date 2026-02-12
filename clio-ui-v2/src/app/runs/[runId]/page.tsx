import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { RunDetailContent } from "./run-detail-content";

type RunPageProps = {
  params: Promise<{
    runId: string;
  }>;
};

export default async function RunPage({ params }: RunPageProps) {
  const { runId } = await params;
  const decodedRunId = decodeURIComponent(runId);

  return (
    <div>
      <nav
        aria-label="Breadcrumb"
        className="mx-auto flex max-w-[2280px] items-center gap-1.5 px-3 pt-4 pb-0 text-sm text-muted-foreground md:px-6 2xl:px-8"
      >
        <Link href="/" className="hover:text-foreground transition-colors">
          CLIO
        </Link>
        <ChevronRight className="size-3.5" />
        <Link href="/" className="hover:text-foreground transition-colors">
          Runs
        </Link>
        <ChevronRight className="size-3.5" />
        <span className="truncate font-medium text-foreground">
          {decodedRunId}
        </span>
      </nav>
      <RunDetailContent runId={decodedRunId} />
    </div>
  );
}
