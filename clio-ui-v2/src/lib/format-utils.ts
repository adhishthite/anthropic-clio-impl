/**
 * Shared formatting utilities for UI components
 */

export function formatDateTime(
  value: string,
  options?: { showSeconds?: boolean; localize?: boolean },
): string {
  if (!value) return "n/a";

  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "n/a";

  const date = new Date(timestamp);

  // If not localized, return ISO format
  if (options?.localize === false) {
    return date.toISOString().replace("T", " ").replace("Z", " UTC");
  }

  const formatter = new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: options?.showSeconds ? "2-digit" : undefined,
    hour12: true,
  });

  return formatter.format(date);
}

export function formatRelativeTime(
  value: string,
  nowMs?: number | null,
): string {
  if (!value) return "n/a";

  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "n/a";

  const referenceTime = nowMs ?? Date.now();
  const diffSeconds = Math.round((timestamp - referenceTime) / 1000);
  const absDiff = Math.abs(diffSeconds);
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

  if (absDiff < 60) return formatter.format(diffSeconds, "second");
  if (absDiff < 3_600)
    return formatter.format(Math.round(diffSeconds / 60), "minute");
  if (absDiff < 86_400)
    return formatter.format(Math.round(diffSeconds / 3_600), "hour");

  return formatter.format(Math.round(diffSeconds / 86_400), "day");
}

export function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  if (value < 1000) return value.toString();

  const formatter = new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  });

  return formatter.format(value);
}

export function formatPercent(value: number): string {
  return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
}

export function formatShare(value: number): string {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return `${pct % 1 === 0 ? pct.toFixed(0) : pct.toFixed(1)}%`;
}
