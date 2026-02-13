"use client";

import { Agentation } from "agentation";

const AGENTATION_ENDPOINT = process.env.NEXT_PUBLIC_AGENTATION_ENDPOINT;

export function AgentationInit() {
  if (process.env.NODE_ENV !== "development") {
    return null;
  }

  return (
    <Agentation
      {...(AGENTATION_ENDPOINT ? { endpoint: AGENTATION_ENDPOINT } : {})}
    />
  );
}
