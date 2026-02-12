type SseWriter = {
  send: (event: string, payload: unknown) => void;
  comment: (text: string) => void;
};

type SseConnectOptions = {
  lastEventId?: string | null;
};

let _globalEventSeq = 0;

function nextEventId(): string {
  _globalEventSeq += 1;
  return `${Date.now()}-${_globalEventSeq}`;
}

function formatSseEvent(event: string, payload: unknown, id: string): string {
  return `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function formatSseComment(comment: string): string {
  return `: ${comment}\n\n`;
}

function streamHeaders(): HeadersInit {
  return {
    "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
    Connection: "keep-alive",
    "Content-Type": "text/event-stream; charset=utf-8",
    Pragma: "no-cache",
    Expires: "0",
    "X-Accel-Buffering": "no",
  };
}

export function parseLastEventId(request: Request): string | null {
  return request.headers.get("Last-Event-Id") || null;
}

export function createSseResponse(
  connect: (
    writer: SseWriter,
    options: SseConnectOptions,
  ) => (() => void) | undefined,
  options?: SseConnectOptions,
): Response {
  const encoder = new TextEncoder();
  let cleanup: (() => void) | undefined;
  let closed = false;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const enqueue = (chunk: string) => {
        if (closed) {
          return;
        }
        try {
          controller.enqueue(encoder.encode(chunk));
        } catch {
          closed = true;
        }
      };

      const writer: SseWriter = {
        send(event, payload) {
          enqueue(formatSseEvent(event, payload, nextEventId()));
        },
        comment(text) {
          enqueue(formatSseComment(text));
        },
      };

      enqueue("retry: 3000\n\n");
      cleanup = connect(writer, options ?? {});
    },
    cancel() {
      closed = true;
      cleanup?.();
    },
  });

  return new Response(stream, {
    headers: streamHeaders(),
  });
}

export function normalizeStreamInterval(
  value: number,
  fallback: number,
): number {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }
  const rounded = Math.floor(value);
  return Math.max(30000, Math.min(300000, rounded));
}
