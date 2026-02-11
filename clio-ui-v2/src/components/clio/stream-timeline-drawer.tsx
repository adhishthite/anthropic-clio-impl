"use client";

import { Activity } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import type {
  SseConnectionEvent,
  SseStreamHealth,
} from "@/hooks/use-sse-stream";

type StreamTimelineDrawerProps = {
  title: string;
  description: string;
  triggerLabel?: string;
  streams: Array<{
    label: string;
    health: SseStreamHealth;
  }>;
};

function formatDateTime(
  value: string,
  options?: { localize?: boolean },
): string {
  if (!value) {
    return "n/a";
  }
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return value;
  }
  if (!options?.localize) {
    return new Date(timestamp)
      .toISOString()
      .replace("T", " ")
      .replace("Z", " UTC");
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(timestamp));
}

function eventBadgeVariant(eventType: SseConnectionEvent["type"]) {
  if (eventType === "connected") {
    return "secondary" as const;
  }
  if (
    eventType === "error" ||
    eventType === "retry_scheduled" ||
    eventType === "disconnected"
  ) {
    return "destructive" as const;
  }
  if (eventType === "reconnecting") {
    return "outline" as const;
  }
  return "default" as const;
}

export function StreamTimelineDrawer({
  title,
  description,
  triggerLabel = "Timeline",
  streams,
}: StreamTimelineDrawerProps) {
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  const mergedEvents = useMemo(() => {
    const rows = streams.flatMap(({ label, health }) =>
      health.timeline.map((event) => ({
        label,
        event,
      })),
    );
    rows.sort(
      (a, b) =>
        Date.parse(b.event.atUtc || "1970-01-01T00:00:00Z") -
        Date.parse(a.event.atUtc || "1970-01-01T00:00:00Z"),
    );
    return rows;
  }, [streams]);

  const totalReconnects = useMemo(
    () =>
      streams.reduce(
        (sum, stream) => sum + Math.max(0, stream.health.reconnectCount),
        0,
      ),
    [streams],
  );

  const totalErrors = useMemo(
    () =>
      streams.reduce(
        (sum, stream) => sum + Math.max(0, stream.health.totalErrors),
        0,
      ),
    [streams],
  );

  if (!isHydrated) {
    return (
      <Button variant="outline" size="sm" className="gap-2" type="button">
        <Activity className="size-4" />
        {triggerLabel}
      </Button>
    );
  }

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Activity className="size-4" />
          {triggerLabel}
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-full border-l-border/70 bg-background/95 sm:max-w-xl"
      >
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
          <SheetDescription>{description}</SheetDescription>
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
            <Badge variant="outline">reconnects {totalReconnects}</Badge>
            <Badge variant="outline">errors {totalErrors}</Badge>
            <Badge variant="outline">events {mergedEvents.length}</Badge>
          </div>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-170px)] px-4 pb-4">
          <div className="space-y-2">
            {mergedEvents.length === 0 ? (
              <div className="rounded-xl border border-dashed p-4 text-sm text-muted-foreground">
                No connection events captured yet.
              </div>
            ) : (
              mergedEvents.map(({ label, event }) => (
                <div
                  key={`${event.id}-${label}`}
                  className="rounded-xl border border-border/70 bg-muted/20 p-3"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline">{label}</Badge>
                    <Badge variant={eventBadgeVariant(event.type)}>
                      {event.type}
                    </Badge>
                  </div>
                  <p className="mt-2 text-sm">{event.detail}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {formatDateTime(event.atUtc, { localize: isHydrated })}
                  </p>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
