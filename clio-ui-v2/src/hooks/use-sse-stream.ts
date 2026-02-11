"use client";

import { useEffect, useRef, useState } from "react";

export type SseStreamState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected";

export type SseConnectionEventType =
  | "disabled"
  | "connecting"
  | "reconnecting"
  | "connected"
  | "message"
  | "error"
  | "retry_scheduled"
  | "disconnected";

export type SseConnectionEvent = {
  id: string;
  atUtc: string;
  streamName: string;
  type: SseConnectionEventType;
  detail: string;
};

export type SseStreamHealth = {
  state: SseStreamState;
  reconnectCount: number;
  consecutiveFailures: number;
  totalMessages: number;
  totalErrors: number;
  lastConnectedAt: string;
  lastMessageAt: string;
  lastErrorAt: string;
  nextRetryMs: number | null;
  timeline: SseConnectionEvent[];
};

type EventHandlerMap = Record<string, (event: MessageEvent<string>) => void>;

type UseSseStreamOptions = {
  url: string | null;
  enabled: boolean;
  eventHandlers: EventHandlerMap;
  streamName?: string;
  baseRetryMs?: number;
  maxRetryMs?: number;
  jitterMs?: number;
  maxTimelineEvents?: number;
};

const INITIAL_HEALTH: SseStreamHealth = {
  state: "idle",
  reconnectCount: 0,
  consecutiveFailures: 0,
  totalMessages: 0,
  totalErrors: 0,
  lastConnectedAt: "",
  lastMessageAt: "",
  lastErrorAt: "",
  nextRetryMs: null,
  timeline: [],
};

export function useSseStream({
  url,
  enabled,
  eventHandlers,
  streamName = "stream",
  baseRetryMs = 1000,
  maxRetryMs = 15000,
  jitterMs = 250,
  maxTimelineEvents = 80,
}: UseSseStreamOptions): SseStreamHealth {
  const [health, setHealth] = useState<SseStreamHealth>(INITIAL_HEALTH);
  const failuresRef = useRef<number>(0);
  const reconnectCountRef = useRef<number>(0);
  const messageCountRef = useRef<number>(0);
  const errorCountRef = useRef<number>(0);
  const eventSeqRef = useRef<number>(0);

  useEffect(() => {
    const createTimelineEvent = (
      type: SseConnectionEventType,
      detail: string,
    ): SseConnectionEvent => {
      eventSeqRef.current += 1;
      return {
        id: `${Date.now()}-${eventSeqRef.current}`,
        atUtc: new Date().toISOString(),
        streamName,
        type,
        detail,
      };
    };

    const appendTimelineEvent = (
      type: SseConnectionEventType,
      detail: string,
    ) => {
      const event = createTimelineEvent(type, detail);
      setHealth((previous) => ({
        ...previous,
        timeline: [event, ...previous.timeline].slice(0, maxTimelineEvents),
      }));
    };

    if (!enabled || !url) {
      setHealth((previous) => ({
        ...previous,
        state: "idle",
        nextRetryMs: null,
      }));
      appendTimelineEvent("disabled", "Stream disabled.");
      return;
    }

    failuresRef.current = 0;
    reconnectCountRef.current = 0;
    messageCountRef.current = 0;
    errorCountRef.current = 0;
    setHealth({
      ...INITIAL_HEALTH,
      state: "connecting",
    });

    let active = true;
    let source: EventSource | null = null;
    let retryTimeout: number | null = null;

    const updateState = (
      nextState: SseStreamState,
      nextRetryMs: number | null,
    ) => {
      setHealth((previous) => ({
        ...previous,
        state: nextState,
        nextRetryMs,
      }));
    };

    const cleanupSource = () => {
      if (source) {
        source.close();
        source = null;
      }
    };

    const connect = () => {
      if (!active) {
        return;
      }

      const nextState = failuresRef.current > 0 ? "reconnecting" : "connecting";
      updateState(nextState, null);
      appendTimelineEvent(
        nextState,
        nextState === "reconnecting"
          ? `Reconnect attempt #${reconnectCountRef.current + 1}.`
          : "Opening SSE connection.",
      );
      const nextSource = new EventSource(url);
      source = nextSource;

      nextSource.addEventListener("open", () => {
        failuresRef.current = 0;
        setHealth((previous) => ({
          ...previous,
          state: "connected",
          consecutiveFailures: 0,
          nextRetryMs: null,
          lastConnectedAt: new Date().toISOString(),
        }));
        appendTimelineEvent("connected", "SSE connection established.");
      });

      for (const [eventName, handler] of Object.entries(eventHandlers)) {
        nextSource.addEventListener(eventName, (event) => {
          messageCountRef.current += 1;
          setHealth((previous) => ({
            ...previous,
            totalMessages: messageCountRef.current,
            lastMessageAt: new Date().toISOString(),
          }));
          if (
            messageCountRef.current === 1 ||
            messageCountRef.current % 25 === 0
          ) {
            appendTimelineEvent(
              "message",
              `${eventName} message #${messageCountRef.current}.`,
            );
          }
          handler(event as MessageEvent<string>);
        });
      }

      nextSource.addEventListener("error", () => {
        if (!active) {
          return;
        }

        errorCountRef.current += 1;
        failuresRef.current += 1;
        reconnectCountRef.current += 1;
        const backoffBase = Math.min(
          maxRetryMs,
          baseRetryMs * 2 ** Math.max(0, failuresRef.current - 1),
        );
        const jitter = Math.floor(Math.random() * Math.max(0, jitterMs));
        const retryMs = Math.min(maxRetryMs, backoffBase + jitter);

        setHealth((previous) => ({
          ...previous,
          state: "reconnecting",
          reconnectCount: reconnectCountRef.current,
          consecutiveFailures: failuresRef.current,
          totalErrors: errorCountRef.current,
          lastErrorAt: new Date().toISOString(),
          nextRetryMs: retryMs,
        }));
        appendTimelineEvent(
          "error",
          `Connection error detected (failure ${failuresRef.current}).`,
        );
        appendTimelineEvent(
          "retry_scheduled",
          `Retry scheduled in ${Math.max(1, Math.ceil(retryMs / 1000))}s.`,
        );

        cleanupSource();
        if (retryTimeout !== null) {
          window.clearTimeout(retryTimeout);
          retryTimeout = null;
        }
        retryTimeout = window.setTimeout(() => {
          retryTimeout = null;
          connect();
        }, retryMs);
      });
    };

    connect();

    return () => {
      active = false;
      if (retryTimeout !== null) {
        window.clearTimeout(retryTimeout);
      }
      cleanupSource();
      updateState("disconnected", null);
      appendTimelineEvent("disconnected", "SSE connection closed.");
    };
  }, [
    baseRetryMs,
    enabled,
    eventHandlers,
    jitterMs,
    maxRetryMs,
    maxTimelineEvents,
    streamName,
    url,
  ]);

  return health;
}
