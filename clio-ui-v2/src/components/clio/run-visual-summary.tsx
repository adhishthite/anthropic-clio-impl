"use client";

import {
  AlertTriangle,
  BarChart3,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  GitBranch,
  Layers3,
  Loader2,
  Map as MapIcon,
  Network,
  Search,
  Shield,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type {
  RunClusterConversationsResponse,
  RunVisualHierarchyNode,
  RunVisualMapPoint,
  RunVisualsResponse,
} from "@/lib/clio-types";

type RunVisualSummaryProps = {
  visuals: RunVisualsResponse | null;
  loading: boolean;
  error: string;
};

function formatPercent(value: number): string {
  return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
}

const PRIVACY_STAGE_META: Record<string, { label: string; help: string }> = {
  raw_conversation: {
    label: "Original conversation text",
    help: "Privacy score pass rate when auditing the original conversation text.",
  },
  facet_summary: {
    label: "Facet summary",
    help: "Pass rate after reducing conversations into structured facet summaries.",
  },
  cluster_summary: {
    label: "Cluster summary",
    help: "Pass rate for cluster-level summaries used for aggregate insights.",
  },
};

const EVAL_REPRESENTATION_META: Record<
  string,
  { label: string; help: string }
> = {
  privacy_summary: {
    label: "Privacy-safe summary",
    help: "Topic classification quality using privacy-filtered summaries.",
  },
  non_private_summary: {
    label: "Detailed summary (unfiltered)",
    help: "Topic classification quality using summaries before privacy filtering.",
  },
  raw_user_text: {
    label: "Raw user text",
    help: "Topic classification quality directly from original user text.",
  },
};

const EVAL_METRIC_META: Record<string, { label: string; help: string }> = {
  accuracy: {
    label: "Topic match accuracy",
    help: "Share of examples where predicted topic exactly matches ground truth.",
  },
  macroF1: {
    label: "Balanced F1",
    help: "F1 score averaged equally across topics, so rare topics count equally.",
  },
  weightedF1: {
    label: "Volume-weighted F1",
    help: "F1 score weighted by topic frequency, emphasizing common topics.",
  },
};

function stageLabel(stage: string): string {
  return PRIVACY_STAGE_META[stage]?.label ?? stage.replace(/_/g, " ");
}

function stageHelp(stage: string): string {
  return (
    PRIVACY_STAGE_META[stage]?.help || "Privacy pass-rate for this audit stage."
  );
}

function evalRepresentationLabel(name: string): string {
  return EVAL_REPRESENTATION_META[name]?.label ?? name.replace(/_/g, " ");
}

function evalRepresentationHelp(name: string): string {
  return (
    EVAL_REPRESENTATION_META[name]?.help ||
    "Evaluation representation used to classify synthetic conversations."
  );
}

function HelpTooltip({
  content,
  ariaLabel = "More information",
}: {
  content: string;
  ariaLabel?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          tabIndex={-1}
          className="inline-flex size-4 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground"
          aria-label={ariaLabel}
        >
          <CircleHelp className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-72">
        {content}
      </TooltipContent>
    </Tooltip>
  );
}

function levelPillClass(level: number): string {
  if (level <= 0) {
    return "border-emerald-300/70 bg-emerald-100/80 text-emerald-700";
  }
  if (level === 1) {
    return "border-sky-300/70 bg-sky-100/80 text-sky-700";
  }
  return "border-violet-300/70 bg-violet-100/80 text-violet-700";
}

const ROOT_ACCENT_PALETTE = [
  "#2563eb",
  "#0d9488",
  "#16a34a",
  "#d97706",
  "#9333ea",
  "#dc2626",
  "#0891b2",
  "#4f46e5",
  "#65a30d",
  "#b45309",
  "#db2777",
  "#475569",
];

function humanizeEnum(value: string): string {
  return value.replace(/_/g, " ");
}

function normalizeProjectionMethod(projectionMethod: string | null): string {
  if (!projectionMethod || !projectionMethod.trim()) {
    return "2D projection";
  }
  return projectionMethod.trim().toUpperCase();
}

function projectionAxisLabel(
  projectionMethod: string | null,
  axis: 1 | 2,
): string {
  const normalized = projectionMethod?.trim().toLowerCase() ?? "";
  if (normalized === "pca") {
    return `PCA component ${axis}`;
  }
  if (normalized.includes("umap")) {
    return `UMAP axis ${axis}`;
  }
  return `Projection axis ${axis}`;
}

function formatProjectionValue(value: number): string {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(3);
}

function projectionExplanation(projectionMethod: string | null): string {
  const method = normalizeProjectionMethod(projectionMethod);
  return `${method} compresses high-dimensional embeddings into 2D. Nearby points are semantically similar clusters.`;
}

type ClusterMapTooltipPayload = {
  payload?: RunVisualMapPoint;
};

function ClusterMapTooltip({
  active,
  payload,
  projectionMethod,
}: {
  active?: boolean;
  payload?: ClusterMapTooltipPayload[];
  projectionMethod: string | null;
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) {
    return null;
  }

  const xLabel = projectionAxisLabel(projectionMethod, 1);
  const yLabel = projectionAxisLabel(projectionMethod, 2);

  return (
    <div className="max-w-72 rounded-lg border border-border/80 bg-card/95 p-3 shadow-md backdrop-blur-sm">
      <p className="truncate text-sm font-semibold">{row.clusterName}</p>
      <p className="text-[11px] text-muted-foreground">
        cluster {row.clusterId}
      </p>

      <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        <span className="text-muted-foreground">Status</span>
        <span>{row.kept ? "Kept after privacy filter" : "Filtered out"}</span>
        <span className="text-muted-foreground">{xLabel}</span>
        <span>{formatProjectionValue(row.x)}</span>
        <span className="text-muted-foreground">{yLabel}</span>
        <span>{formatProjectionValue(row.y)}</span>
        {row.language ? (
          <>
            <span className="text-muted-foreground">Language</span>
            <span>{row.language}</span>
          </>
        ) : null}
        {row.concerningScore !== null ? (
          <>
            <span className="text-muted-foreground">Concern score</span>
            <span>{row.concerningScore.toFixed(2)}</span>
          </>
        ) : null}
      </div>

      <p className="mt-2 text-[11px] text-muted-foreground">
        {projectionExplanation(projectionMethod)}
      </p>
    </div>
  );
}

function formatShare(value: number): string {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return `${pct % 1 === 0 ? pct.toFixed(0) : pct.toFixed(1)}%`;
}

function metadataPreview(value: unknown): string {
  if (typeof value === "string") {
    return value.length > 120 ? `${value.slice(0, 117)}...` : value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value === null || value === undefined) {
    return "null";
  }
  try {
    const encoded = JSON.stringify(value);
    return encoded.length > 120 ? `${encoded.slice(0, 117)}...` : encoded;
  } catch {
    return "[unavailable]";
  }
}

type FlatTableRow = {
  node: RunVisualHierarchyNode;
  depth: number;
  accentColor: string;
  shareOfParent: number;
  shareOfRun: number;
  hasChildren: boolean;
  isExpanded: boolean;
};

export function RunVisualSummary({
  visuals,
  loading,
  error,
}: RunVisualSummaryProps) {
  const hasMap = Boolean(visuals?.map && visuals.map.points.length > 0);
  const hasHierarchy = Boolean(
    visuals?.hierarchy &&
      (visuals.hierarchy.topLevelClusters.length > 0 ||
        visuals.hierarchy.nodes.length > 0),
  );
  const hasPrivacy = Boolean(
    visuals?.privacy && visuals.privacy.stages.length > 0,
  );
  const hasEvaluation = Boolean(
    visuals?.evaluation && visuals.evaluation.ablations.length > 0,
  );
  const hasAny = hasMap || hasHierarchy || hasPrivacy || hasEvaluation;

  const keptPoints = visuals?.map?.points.filter((point) => point.kept) ?? [];
  const filteredPoints =
    visuals?.map?.points.filter((point) => !point.kept) ?? [];
  const privacyStages =
    visuals?.privacy?.stages.map((stage) => ({
      ...stage,
      label: stageLabel(stage.stage),
      help: stageHelp(stage.stage),
    })) ?? [];
  const evaluationRows =
    visuals?.evaluation?.ablations.map((ablation) => ({
      ...ablation,
      displayName: evalRepresentationLabel(ablation.name),
      help: evalRepresentationHelp(ablation.name),
    })) ?? [];
  const [isHierarchyDialogOpen, setIsHierarchyDialogOpen] = useState(false);
  const hierarchyNodes = visuals?.hierarchy?.nodes ?? [];
  const nodeById = useMemo(
    () => new Map(hierarchyNodes.map((node) => [node.id, node])),
    [hierarchyNodes],
  );
  const childIdsByParent = useMemo(() => {
    const grouped = new Map<string, string[]>();
    for (const node of hierarchyNodes) {
      if (!node.parentId) {
        continue;
      }
      const current = grouped.get(node.parentId) ?? [];
      current.push(node.id);
      grouped.set(node.parentId, current);
    }
    return grouped;
  }, [hierarchyNodes]);
  const rootNodeIds = useMemo(() => {
    const configuredRoots = visuals?.hierarchy?.rootNodeIds ?? [];
    if (configuredRoots.length > 0) {
      return configuredRoots.filter((nodeId) => nodeById.has(nodeId));
    }
    return hierarchyNodes
      .filter((node) => node.parentId === null)
      .sort((a, b) => b.level - a.level)
      .map((node) => node.id);
  }, [hierarchyNodes, nodeById, visuals?.hierarchy?.rootNodeIds]);
  const [selectedHierarchyNodeId, setSelectedHierarchyNodeId] =
    useState<string>("");
  const [hierarchySearchQuery, setHierarchySearchQuery] = useState("");
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(
    () => new Set<string>(),
  );
  const [leafRecordsNodeId, setLeafRecordsNodeId] = useState<string>("");
  const [leafRecordsClusterId, setLeafRecordsClusterId] = useState<
    number | null
  >(null);
  const [leafRecordsPayload, setLeafRecordsPayload] =
    useState<RunClusterConversationsResponse | null>(null);
  const [leafRecordsLoading, setLeafRecordsLoading] = useState(false);
  const [leafRecordsError, setLeafRecordsError] = useState("");
  const [leafRecordSearch, setLeafRecordSearch] = useState("");

  useEffect(() => {
    if (!isHierarchyDialogOpen) {
      return;
    }
    if (selectedHierarchyNodeId && nodeById.has(selectedHierarchyNodeId)) {
      return;
    }
    const fallbackId = rootNodeIds[0] ?? hierarchyNodes[0]?.id ?? "";
    setSelectedHierarchyNodeId(fallbackId);
  }, [
    hierarchyNodes,
    isHierarchyDialogOpen,
    nodeById,
    rootNodeIds,
    selectedHierarchyNodeId,
  ]);

  const leafNodeByClusterId = useMemo(() => {
    const byClusterId = new Map<number, RunVisualHierarchyNode>();
    for (const node of hierarchyNodes) {
      if (node.level !== 0 || node.sourceClusterId === null) {
        continue;
      }
      byClusterId.set(node.sourceClusterId, node);
    }
    return byClusterId;
  }, [hierarchyNodes]);
  const normalizedHierarchySearchQuery = hierarchySearchQuery
    .trim()
    .toLowerCase();
  const openLeafRecords = (nodeId: string, clusterId: number) => {
    setLeafRecordsNodeId(nodeId);
    setLeafRecordsClusterId(clusterId);
    setLeafRecordSearch("");
  };
  const rootIdByNodeId = useMemo(() => {
    const cache = new Map<string, string>();

    const resolveRoot = (nodeId: string): string => {
      const existing = cache.get(nodeId);
      if (existing) {
        return existing;
      }
      const node = nodeById.get(nodeId);
      if (!node) {
        return nodeId;
      }
      if (!node.parentId || !nodeById.has(node.parentId)) {
        cache.set(nodeId, node.id);
        return node.id;
      }
      const resolved = resolveRoot(node.parentId);
      cache.set(nodeId, resolved);
      return resolved;
    };

    for (const node of hierarchyNodes) {
      resolveRoot(node.id);
    }
    return cache;
  }, [hierarchyNodes, nodeById]);
  const rootAccentById = useMemo(() => {
    const map = new Map<string, string>();
    rootNodeIds.forEach((rootId, index) => {
      map.set(rootId, ROOT_ACCENT_PALETTE[index % ROOT_ACCENT_PALETTE.length]);
    });
    return map;
  }, [rootNodeIds]);
  const nodeAccentById = useMemo(() => {
    const map = new Map<string, string>();
    for (const node of hierarchyNodes) {
      const rootId = rootIdByNodeId.get(node.id) ?? node.id;
      map.set(node.id, rootAccentById.get(rootId) ?? ROOT_ACCENT_PALETTE[0]);
    }
    return map;
  }, [hierarchyNodes, rootAccentById, rootIdByNodeId]);
  const hierarchyTotalSize = useMemo(() => {
    return rootNodeIds.reduce((total, rootId) => {
      return total + Math.max(0, nodeById.get(rootId)?.size ?? 0);
    }, 0);
  }, [nodeById, rootNodeIds]);
  const hierarchyMatchNodeIds = useMemo(() => {
    if (!normalizedHierarchySearchQuery) {
      return new Set<string>();
    }
    const matches = new Set<string>();
    for (const node of hierarchyNodes) {
      const haystack =
        `${node.name} ${node.description} ${node.id}`.toLowerCase();
      if (haystack.includes(normalizedHierarchySearchQuery)) {
        matches.add(node.id);
      }
    }
    return matches;
  }, [hierarchyNodes, normalizedHierarchySearchQuery]);
  const hierarchyVisibleNodeIds = useMemo(() => {
    if (hierarchyMatchNodeIds.size === 0) {
      return null;
    }
    const visible = new Set<string>();
    const includeDescendants = (nodeId: string) => {
      const stack = [nodeId];
      while (stack.length > 0) {
        const currentId = stack.pop();
        if (!currentId || visible.has(currentId)) {
          continue;
        }
        visible.add(currentId);
        const childIds = childIdsByParent.get(currentId) ?? [];
        childIds.forEach((childId) => {
          stack.push(childId);
        });
      }
    };
    hierarchyMatchNodeIds.forEach((matchId) => {
      includeDescendants(matchId);
      let cursor = nodeById.get(matchId) ?? null;
      while (cursor?.parentId && nodeById.has(cursor.parentId)) {
        visible.add(cursor.parentId);
        cursor = nodeById.get(cursor.parentId) ?? null;
      }
    });
    return visible;
  }, [childIdsByParent, hierarchyMatchNodeIds, nodeById]);
  const flatTableRows = useMemo<FlatTableRow[]>(() => {
    const rows: FlatTableRow[] = [];
    const isSearching = hierarchyVisibleNodeIds !== null;

    const walk = (nodeId: string, depth: number) => {
      const node = nodeById.get(nodeId);
      if (!node) {
        return;
      }
      if (isSearching && !hierarchyVisibleNodeIds.has(nodeId)) {
        return;
      }
      const childIds = childIdsByParent.get(nodeId) ?? [];
      const visibleChildIds = isSearching
        ? childIds.filter((id) => hierarchyVisibleNodeIds.has(id))
        : childIds;
      const hasChildren = visibleChildIds.length > 0;
      const isExpanded = isSearching
        ? hasChildren
        : expandedNodeIds.has(nodeId);
      const parentSize =
        node.parentId && nodeById.has(node.parentId)
          ? Math.max(1, nodeById.get(node.parentId)?.size ?? 1)
          : Math.max(1, node.size);

      rows.push({
        node,
        depth,
        accentColor: nodeAccentById.get(node.id) ?? ROOT_ACCENT_PALETTE[0],
        shareOfParent: node.size / parentSize,
        shareOfRun: hierarchyTotalSize > 0 ? node.size / hierarchyTotalSize : 0,
        hasChildren,
        isExpanded,
      });

      if (isExpanded) {
        const sortedChildren = visibleChildIds
          .map((id) => nodeById.get(id))
          .filter((n): n is NonNullable<typeof n> => Boolean(n))
          .sort((a, b) => b.size - a.size);
        for (const child of sortedChildren) {
          walk(child.id, depth + 1);
        }
      }
    };

    for (const rootId of rootNodeIds) {
      walk(rootId, 0);
    }
    return rows;
  }, [
    childIdsByParent,
    expandedNodeIds,
    hierarchyTotalSize,
    hierarchyVisibleNodeIds,
    nodeAccentById,
    nodeById,
    rootNodeIds,
  ]);
  const generatedHierarchyDepth =
    visuals?.hierarchy?.generatedLevels !== null &&
    visuals?.hierarchy?.generatedLevels !== undefined &&
    visuals.hierarchy.generatedLevels > 0
      ? visuals.hierarchy.generatedLevels
      : visuals?.hierarchy?.maxLevel !== null &&
          visuals?.hierarchy?.maxLevel !== undefined &&
          visuals.hierarchy.maxLevel >= 0
        ? visuals.hierarchy.maxLevel + 1
        : null;
  const requestedHierarchyLevels =
    visuals?.hierarchy?.requestedLevels !== null &&
    visuals?.hierarchy?.requestedLevels !== undefined &&
    visuals.hierarchy.requestedLevels >= 0
      ? visuals.hierarchy.requestedLevels
      : null;
  const hierarchyDepthPolicy = visuals?.hierarchy?.depthPolicy ?? null;
  const hierarchyDepthStopReason = visuals?.hierarchy?.depthStopReason ?? null;
  const hierarchyWhyNotDeeper = visuals?.hierarchy?.whyNotDeeper ?? null;
  useEffect(() => {
    if (!isHierarchyDialogOpen || rootNodeIds.length === 0) {
      return;
    }
    setExpandedNodeIds((current) => {
      if (current.size > 0) {
        return current;
      }
      const initial = new Set<string>();
      for (const rootId of rootNodeIds) {
        initial.add(rootId);
      }
      return initial;
    });
  }, [isHierarchyDialogOpen, rootNodeIds]);

  useEffect(() => {
    if (leafRecordsClusterId === null || !visuals?.runId) {
      setLeafRecordsPayload(null);
      setLeafRecordsError("");
      setLeafRecordsLoading(false);
      return;
    }
    const controller = new AbortController();
    setLeafRecordsLoading(true);
    setLeafRecordsError("");

    void fetch(
      `/api/runs/${encodeURIComponent(visuals.runId)}/clusters/${leafRecordsClusterId}/conversations`,
      {
        cache: "no-store",
        signal: controller.signal,
      },
    )
      .then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as {
            error?: string;
          } | null;
          throw new Error(
            payload?.error ||
              `Failed to load cluster conversations (${response.status}).`,
          );
        }
        return (await response.json()) as RunClusterConversationsResponse;
      })
      .then((payload) => {
        if (!controller.signal.aborted) {
          setLeafRecordsPayload(payload);
        }
      })
      .catch((fetchError: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setLeafRecordsError(
          fetchError instanceof Error
            ? fetchError.message
            : "Failed to load cluster conversations.",
        );
        setLeafRecordsPayload(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLeafRecordsLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [leafRecordsClusterId, visuals?.runId]);

  const selectedLeafRecordsNode =
    leafRecordsNodeId && nodeById.has(leafRecordsNodeId)
      ? (nodeById.get(leafRecordsNodeId) ?? null)
      : leafRecordsClusterId !== null
        ? (leafNodeByClusterId.get(leafRecordsClusterId) ?? null)
        : null;
  const filteredLeafRecords = useMemo(() => {
    const rows = leafRecordsPayload?.records ?? [];
    const query = leafRecordSearch.trim().toLowerCase();
    if (!query) {
      return rows;
    }
    return rows.filter((row) => {
      const metadataHaystack = row.userMetadata
        ? Object.entries(row.userMetadata)
            .map(([key, value]) => `${key}:${metadataPreview(value)}`)
            .join(" ")
        : "";
      const facetSummary = row.facet?.summary ?? "";
      const facetTask = row.facet?.task ?? "";
      const facetLanguage = row.facet?.language ?? "";
      const haystack =
        `${row.conversationId} ${row.userId} ${facetSummary} ${facetTask} ${facetLanguage} ${metadataHaystack}`.toLowerCase();
      return haystack.includes(query);
    });
  }, [leafRecordSearch, leafRecordsPayload?.records]);

  return (
    <Card className="clio-shell border-border/70">
      <CardHeader>
        <CardTitle className="text-lg">Visual outputs</CardTitle>
        <CardDescription>
          Map, hierarchy, privacy, and evaluation views generated for this run.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading visual artifacts...
          </div>
        ) : null}

        {error ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>Could not load visual outputs</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {!loading && !error && !hasAny ? (
          <p className="text-sm text-muted-foreground">
            No visualization artifacts yet. Continue the run through clustering,
            hierarchy, privacy, or evaluation phases.
          </p>
        ) : null}

        {hasAny ? (
          <div className="space-y-4">
            <div className="grid gap-4 2xl:gap-5 xl:grid-cols-12">
              {hasHierarchy ? (
                <Card className="clio-panel border-border/70 xl:col-span-8">
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <CardTitle className="flex items-center gap-2 text-base">
                          <GitBranch className="size-4 text-primary" />
                          Hierarchy explorer
                        </CardTitle>
                        <CardDescription>
                          Drill down from top-level groups to leaf clusters.
                        </CardDescription>
                      </div>
                      <Button
                        size="sm"
                        className="gap-1"
                        onClick={() => setIsHierarchyDialogOpen(true)}
                      >
                        <Network className="size-4" />
                        Open full hierarchy
                      </Button>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                      <Badge variant="outline">
                        top-level {visuals?.hierarchy?.topLevelCount ?? 0}
                      </Badge>
                      <Badge variant="outline">
                        leaf {visuals?.hierarchy?.leafCount ?? 0}
                      </Badge>
                      <Badge variant="outline">
                        nodes {visuals?.hierarchy?.nodes.length ?? 0}
                      </Badge>
                      {generatedHierarchyDepth !== null &&
                      requestedHierarchyLevels !== null ? (
                        <>
                          <span className="mx-0.5 h-4 w-px bg-border" />
                          <Badge variant="outline" className="gap-1">
                            depth {generatedHierarchyDepth}/
                            {requestedHierarchyLevels}
                            <HelpTooltip
                              content={`Generated hierarchy depth vs requested hierarchy levels. Policy: ${hierarchyDepthPolicy ?? "adaptive"}. Requested levels are an upper bound.`}
                              ariaLabel="About hierarchy depth"
                            />
                          </Badge>
                        </>
                      ) : null}
                      {generatedHierarchyDepth !== null &&
                      requestedHierarchyLevels !== null &&
                      generatedHierarchyDepth < requestedHierarchyLevels &&
                      hierarchyDepthStopReason ? (
                        <Badge variant="outline" className="gap-1">
                          reason {humanizeEnum(hierarchyDepthStopReason)}
                          {hierarchyWhyNotDeeper ? (
                            <HelpTooltip
                              content={hierarchyWhyNotDeeper}
                              ariaLabel="About hierarchy stop reason"
                            />
                          ) : null}
                        </Badge>
                      ) : null}
                    </div>
                  </CardHeader>
                  <CardContent className="p-4">
                    {(() => {
                      const topLevel = (
                        visuals?.hierarchy?.topLevelClusters ?? []
                      ).toSorted((a, b) => {
                        const sizeA = nodeById.get(a.id)?.size ?? 0;
                        const sizeB = nodeById.get(b.id)?.size ?? 0;
                        return sizeB - sizeA;
                      });

                      // When only 1 top-level cluster, auto-expand to show its children
                      const displayItems =
                        topLevel.length === 1
                          ? (() => {
                              const singleRoot = topLevel[0];
                              const childIds =
                                childIdsByParent.get(singleRoot.id) ?? [];
                              const children = childIds
                                .map((id) => nodeById.get(id))
                                .filter((n): n is NonNullable<typeof n> => !!n)
                                .sort((a, b) => b.size - a.size)
                                .slice(0, 6)
                                .map((child) => ({
                                  id: child.id,
                                  name: child.name,
                                  description: child.description,
                                  childCount:
                                    childIdsByParent.get(child.id)?.length ?? 0,
                                }));
                              return children.length > 0
                                ? children
                                : topLevel.slice(0, 4);
                            })()
                          : topLevel.slice(0, 4);

                      return (
                        <div
                          className={`grid gap-2 ${displayItems.length >= 2 ? "md:grid-cols-2" : ""}`}
                        >
                          {displayItems.map((item) => {
                            const nodeSize = nodeById.get(item.id)?.size ?? 0;
                            const accent =
                              nodeAccentById.get(item.id) ??
                              ROOT_ACCENT_PALETTE[0];
                            const share =
                              hierarchyTotalSize > 0
                                ? nodeSize / hierarchyTotalSize
                                : 0;
                            return (
                              <button
                                type="button"
                                key={item.id}
                                className="cursor-pointer rounded-lg border border-border/70 border-l-[3px] bg-muted/20 p-3 text-left transition-all hover:bg-muted/40 hover:shadow-sm"
                                style={{ borderLeftColor: accent }}
                                onClick={() => {
                                  setSelectedHierarchyNodeId(item.id);
                                  setIsHierarchyDialogOpen(true);
                                }}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <p className="text-sm font-medium text-foreground/90">
                                    {item.name}
                                  </p>
                                  {item.childCount > 0 ? (
                                    <Badge
                                      variant="outline"
                                      className="text-xs"
                                    >
                                      children {item.childCount}
                                    </Badge>
                                  ) : null}
                                </div>
                                {item.description ? (
                                  <p className="mt-1 line-clamp-2 text-xs text-muted-foreground/90">
                                    {item.description}
                                  </p>
                                ) : null}
                                <div className="mt-2 flex items-center gap-2">
                                  <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted/60">
                                    <div
                                      className="absolute inset-y-0 left-0 rounded-full"
                                      style={{
                                        width: `${Math.max(2, share * 100)}%`,
                                        backgroundColor: accent,
                                      }}
                                    />
                                  </div>
                                  <span className="shrink-0 text-xs text-muted-foreground">
                                    {nodeSize} convs ({formatShare(share)})
                                  </span>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      );
                    })()}
                  </CardContent>
                </Card>
              ) : null}

              {hasMap ? (
                <Card className="clio-panel border-border/70 xl:col-span-4">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <MapIcon className="size-4 text-primary" />
                      Cluster map
                    </CardTitle>
                    <CardDescription>
                      Spatial view of conversation clusters.
                    </CardDescription>
                    <div className="flex flex-wrap gap-2 text-xs">
                      <Badge variant="outline" className="gap-1">
                        points {visuals?.map?.totalPoints ?? 0}
                        <HelpTooltip
                          content="Number of clustered conversation summaries plotted on this map."
                          ariaLabel="About points count"
                        />
                      </Badge>
                      <Badge variant="outline" className="gap-1">
                        projection{" "}
                        {normalizeProjectionMethod(
                          visuals?.map?.projectionMethod ?? null,
                        )}
                        <HelpTooltip
                          content="Each point has 2 reduced coordinates. They are not raw business metrics - they are map coordinates derived from embedding similarity."
                          ariaLabel="About projection method"
                        />
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[360px] 2xl:h-[420px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart
                          margin={{
                            top: 10,
                            right: 10,
                            bottom: 20,
                            left: 10,
                          }}
                        >
                          <XAxis
                            type="number"
                            dataKey="x"
                            name={projectionAxisLabel(
                              visuals?.map?.projectionMethod ?? null,
                              1,
                            )}
                            tick={false}
                            axisLine={{ stroke: "var(--border)" }}
                            label={{
                              value: projectionAxisLabel(
                                visuals?.map?.projectionMethod ?? null,
                                1,
                              ),
                              position: "insideBottom",
                              offset: -10,
                              fontSize: 11,
                              fill: "var(--muted-foreground)",
                            }}
                          />
                          <YAxis
                            type="number"
                            dataKey="y"
                            name={projectionAxisLabel(
                              visuals?.map?.projectionMethod ?? null,
                              2,
                            )}
                            tick={false}
                            axisLine={{ stroke: "var(--border)" }}
                            label={{
                              value: projectionAxisLabel(
                                visuals?.map?.projectionMethod ?? null,
                                2,
                              ),
                              angle: -90,
                              position: "insideLeft",
                              offset: 0,
                              fontSize: 11,
                              fill: "var(--muted-foreground)",
                            }}
                          />
                          <ChartTooltip
                            cursor={{ strokeDasharray: "3 3" }}
                            content={({ active, payload }) => (
                              <ClusterMapTooltip
                                active={active}
                                payload={
                                  payload as
                                    | ClusterMapTooltipPayload[]
                                    | undefined
                                }
                                projectionMethod={
                                  visuals?.map?.projectionMethod ?? null
                                }
                              />
                            )}
                          />
                          <Legend />
                          {keptPoints.length > 0 ? (
                            <Scatter
                              name="kept"
                              data={keptPoints}
                              fill="var(--color-chart-2)"
                              opacity={0.7}
                            />
                          ) : null}
                          {filteredPoints.length > 0 ? (
                            <Scatter
                              name="filtered"
                              data={filteredPoints}
                              fill="var(--color-chart-4)"
                              opacity={0.7}
                            />
                          ) : null}
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              ) : null}
            </div>

            {(hasPrivacy || hasEvaluation) && (
              <Accordion type="single" collapsible>
                <AccordionItem
                  value="secondary-analysis"
                  className="rounded-xl border border-border/70 px-4"
                >
                  <AccordionTrigger className="hover:no-underline">
                    <div className="flex items-center gap-2 text-left">
                      <Layers3 className="size-4 text-muted-foreground" />
                      <div>
                        <p className="font-medium">Secondary analysis</p>
                        <p className="text-xs text-muted-foreground">
                          Privacy and evaluation diagnostics
                        </p>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="grid gap-4 xl:grid-cols-2 2xl:gap-5">
                      {hasPrivacy ? (
                        <Card className="clio-panel border-border/70">
                          <CardHeader className="pb-2">
                            <CardTitle className="flex items-center gap-2 text-base">
                              <Shield className="size-4 text-primary" />
                              Privacy stage pass rates
                            </CardTitle>
                            <CardDescription>
                              Pass rate by raw, facet, and cluster audit stages.
                            </CardDescription>
                            {visuals?.privacy?.validation ? (
                              <div className="flex flex-wrap gap-2 text-xs">
                                <Badge variant="outline">
                                  validation cases{" "}
                                  {visuals.privacy.validation.totalCases}
                                </Badge>
                                <Badge variant="outline">
                                  within expected range{" "}
                                  {formatPercent(
                                    visuals.privacy.validation.inRangeRate,
                                  )}
                                </Badge>
                                <Badge variant="outline">
                                  average rating error{" "}
                                  {visuals.privacy.validation.meanAbsoluteError.toFixed(
                                    2,
                                  )}
                                </Badge>
                              </div>
                            ) : null}
                          </CardHeader>
                          <CardContent>
                            <div className="h-[300px] 2xl:h-[360px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={privacyStages}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="label" />
                                  <YAxis
                                    domain={[0, 1]}
                                    tickFormatter={formatPercent}
                                  />
                                  <ChartTooltip
                                    formatter={(value) => [
                                      formatPercent(Number(value)),
                                      "pass rate",
                                    ]}
                                  />
                                  <Legend />
                                  <Bar
                                    dataKey="passRate"
                                    name="pass rate"
                                    fill="var(--color-chart-1)"
                                  />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                            <div className="mt-3 space-y-1">
                              {privacyStages.map((stage) => (
                                <div
                                  key={stage.stage}
                                  className="flex items-center justify-between gap-2 text-xs text-muted-foreground"
                                >
                                  <div className="flex items-center gap-1.5">
                                    <span>{stage.label}</span>
                                    <HelpTooltip
                                      content={stage.help}
                                      ariaLabel={`About ${stage.label}`}
                                    />
                                  </div>
                                  <span>
                                    {stage.passCount}/{stage.total} passed
                                  </span>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      ) : null}

                      {hasEvaluation ? (
                        <Card className="clio-panel border-border/70">
                          <CardHeader className="pb-2">
                            <CardTitle className="flex items-center gap-2 text-base">
                              <BarChart3 className="size-4 text-primary" />
                              Evaluation ablations
                            </CardTitle>
                            <CardDescription>
                              Topic classification quality by representation
                              type.
                            </CardDescription>
                            <div className="flex flex-wrap gap-2 text-xs">
                              <Badge variant="outline">
                                synthetic{" "}
                                {visuals?.evaluation?.syntheticCount ?? 0}
                              </Badge>
                              <Badge variant="outline">
                                topics {visuals?.evaluation?.topicCount ?? 0}
                              </Badge>
                              <Badge variant="outline">
                                languages{" "}
                                {visuals?.evaluation?.languageCount ?? 0}
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent>
                            <div className="h-[300px] 2xl:h-[360px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={evaluationRows}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="displayName" />
                                  <YAxis domain={[0, 1]} />
                                  <ChartTooltip />
                                  <Legend />
                                  <Bar
                                    dataKey="accuracy"
                                    name={EVAL_METRIC_META.accuracy.label}
                                    fill="var(--color-chart-1)"
                                  />
                                  <Bar
                                    dataKey="macroF1"
                                    name={EVAL_METRIC_META.macroF1.label}
                                    fill="var(--color-chart-2)"
                                  />
                                  <Bar
                                    dataKey="weightedF1"
                                    name={EVAL_METRIC_META.weightedF1.label}
                                    fill="var(--color-chart-3)"
                                  />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </CardContent>
                        </Card>
                      ) : null}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}

            <Dialog
              open={isHierarchyDialogOpen}
              onOpenChange={setIsHierarchyDialogOpen}
            >
              <DialogContent className="h-[95dvh] w-[98vw] max-w-[1600px] sm:max-w-[1600px] overflow-hidden p-0">
                <DialogHeader className="border-b px-6 py-4">
                  <DialogTitle className="flex items-center gap-2">
                    <GitBranch className="size-4 text-primary" />
                    Hierarchy explorer
                  </DialogTitle>
                  <DialogDescription className="flex flex-wrap items-center gap-2">
                    <span>
                      Navigate hierarchy groups and drill into leaf clusters.
                    </span>
                    <Badge variant="outline" className="text-xs">
                      top-level {visuals?.hierarchy?.topLevelCount ?? 0}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      leaf {visuals?.hierarchy?.leafCount ?? 0}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      nodes {hierarchyNodes.length}
                    </Badge>
                    {generatedHierarchyDepth !== null &&
                    requestedHierarchyLevels !== null ? (
                      <Badge variant="outline" className="gap-1 text-xs">
                        depth {generatedHierarchyDepth}/
                        {requestedHierarchyLevels}
                        <HelpTooltip
                          content={`Generated hierarchy depth vs requested levels. Policy: ${hierarchyDepthPolicy ?? "adaptive"}.`}
                          ariaLabel="About hierarchy depth"
                        />
                      </Badge>
                    ) : null}
                  </DialogDescription>
                </DialogHeader>

                <div className="flex h-[calc(95dvh-92px)] min-h-0">
                  {/* Table side */}
                  <div
                    className="flex min-h-0 flex-col border-r transition-all duration-200"
                    style={{
                      flex:
                        leafRecordsClusterId !== null ? "0 0 55%" : "1 1 100%",
                    }}
                  >
                    {/* Controls bar */}
                    <div className="flex flex-wrap items-center gap-3 border-b px-6 py-3.5">
                      {(() => {
                        const allNonLeafIds = hierarchyNodes
                          .filter((n) => n.level > 0)
                          .map((n) => n.id);
                        const allExpanded =
                          allNonLeafIds.length > 0 &&
                          allNonLeafIds.every((id) => expandedNodeIds.has(id));
                        const noneExpandedBeyondRoots =
                          expandedNodeIds.size === 0 ||
                          [...expandedNodeIds].every((id) =>
                            rootNodeIds.includes(id),
                          );
                        return (
                          <>
                            <Button
                              size="default"
                              variant={allExpanded ? "default" : "outline"}
                              className="h-9 px-4 text-sm"
                              disabled={allExpanded}
                              onClick={() => {
                                setExpandedNodeIds(new Set(allNonLeafIds));
                              }}
                            >
                              Expand all
                            </Button>
                            <Button
                              size="default"
                              variant="outline"
                              className="h-9 px-4 text-sm"
                              disabled={noneExpandedBeyondRoots}
                              onClick={() =>
                                setExpandedNodeIds(new Set(rootNodeIds))
                              }
                            >
                              Collapse
                            </Button>
                          </>
                        );
                      })()}
                      <div className="relative ml-auto min-w-56 max-w-md flex-1">
                        <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
                        <Input
                          value={hierarchySearchQuery}
                          onChange={(event) =>
                            setHierarchySearchQuery(event.target.value)
                          }
                          placeholder="Search clusters..."
                          className="h-9 pl-9 text-sm"
                        />
                      </div>
                      {normalizedHierarchySearchQuery ? (
                        <span className="text-sm text-muted-foreground">
                          {hierarchyMatchNodeIds.size} matches
                        </span>
                      ) : null}
                    </div>

                    {/* Column headers */}
                    <div className="grid grid-cols-[1fr_84px_84px_160px] gap-3 border-b px-6 py-2.5 text-sm font-medium text-muted-foreground">
                      <span>Name</span>
                      <span className="text-right">Size</span>
                      <span className="text-center">Level</span>
                      <span>Distribution</span>
                    </div>

                    {/* Scrollable rows */}
                    <ScrollArea className="min-h-0 flex-1">
                      <div className="divide-y divide-border/40">
                        {flatTableRows.map((row) => {
                          const isLeaf = row.node.level === 0;
                          const isSelected =
                            isLeaf &&
                            row.node.id === selectedHierarchyNodeId &&
                            leafRecordsClusterId !== null;
                          const isMatch = hierarchyMatchNodeIds.has(
                            row.node.id,
                          );
                          const maxBarShare = Math.max(
                            0,
                            Math.min(1, row.shareOfRun),
                          );
                          return (
                            <button
                              type="button"
                              key={row.node.id}
                              onClick={() => {
                                if (isLeaf) {
                                  setSelectedHierarchyNodeId(row.node.id);
                                  if (row.node.sourceClusterId !== null) {
                                    openLeafRecords(
                                      row.node.id,
                                      row.node.sourceClusterId,
                                    );
                                  }
                                } else {
                                  setExpandedNodeIds((current) => {
                                    const next = new Set(current);
                                    if (next.has(row.node.id)) {
                                      next.delete(row.node.id);
                                    } else {
                                      next.add(row.node.id);
                                    }
                                    return next;
                                  });
                                }
                              }}
                              className={`grid w-full grid-cols-[1fr_84px_84px_160px] items-center gap-3 px-6 py-3 text-left text-[15px] transition-colors hover:bg-muted/40 ${
                                isSelected
                                  ? "border-l-2 border-l-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/20"
                                  : isMatch
                                    ? "bg-primary/5"
                                    : ""
                              }`}
                              style={{
                                paddingLeft: `${1.5 + row.depth * 1.75}rem`,
                              }}
                            >
                              {/* Name cell */}
                              <div className="flex min-w-0 items-center gap-2.5">
                                {row.hasChildren ? (
                                  row.isExpanded ? (
                                    <ChevronDown className="size-[18px] shrink-0 text-muted-foreground" />
                                  ) : (
                                    <ChevronRight className="size-[18px] shrink-0 text-muted-foreground" />
                                  )
                                ) : (
                                  <span className="size-[18px] shrink-0" />
                                )}
                                <span
                                  className="mr-1 size-3 shrink-0 rounded-full"
                                  style={{
                                    backgroundColor: row.accentColor,
                                  }}
                                />
                                <span className="truncate font-medium">
                                  {row.node.name}
                                </span>
                              </div>
                              {/* Size */}
                              <span className="text-right font-mono text-muted-foreground">
                                {row.node.size}
                              </span>
                              {/* Level badge */}
                              <div className="text-center">
                                <span
                                  className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-medium ${levelPillClass(row.node.level)}`}
                                >
                                  {row.node.level <= 0
                                    ? "Leaf"
                                    : row.node.level === 1
                                      ? "Group"
                                      : "Root"}
                                </span>
                              </div>
                              {/* Distribution bar with share % */}
                              <div className="flex items-center gap-2">
                                <div className="h-3.5 flex-1 rounded-full bg-muted/40">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${Math.max(2, maxBarShare * 100)}%`,
                                      backgroundColor: row.accentColor,
                                      opacity: 0.7,
                                    }}
                                  />
                                </div>
                                <span className="shrink-0 text-xs font-mono text-muted-foreground">
                                  {formatShare(row.shareOfRun)}
                                </span>
                              </div>
                            </button>
                          );
                        })}
                        {flatTableRows.length === 0 ? (
                          <p className="px-6 py-8 text-center text-sm text-muted-foreground">
                            No hierarchy nodes match this search.
                          </p>
                        ) : null}
                      </div>
                    </ScrollArea>
                  </div>

                  {/* Drawer */}
                  <div
                    className="flex min-h-0 flex-col overflow-hidden transition-all duration-200"
                    style={{
                      flex:
                        leafRecordsClusterId !== null ? "0 0 45%" : "0 0 0%",
                    }}
                  >
                    {leafRecordsClusterId !== null ? (
                      <>
                        {/* Drawer header */}
                        <div className="border-b px-5 py-4">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <h3 className="truncate text-base font-semibold">
                                {selectedLeafRecordsNode?.name ??
                                  "Cluster conversations"}
                              </h3>
                              <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                                {selectedLeafRecordsNode?.description ??
                                  "Loading cluster details..."}
                              </p>
                            </div>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="size-8 shrink-0 p-0"
                              onClick={() => {
                                setLeafRecordsClusterId(null);
                                setLeafRecordsNodeId("");
                                setSelectedHierarchyNodeId("");
                              }}
                            >
                              <X className="size-4" />
                              <span className="sr-only">Close drawer</span>
                            </Button>
                          </div>
                          <div className="mt-2.5 flex flex-wrap gap-2">
                            {(() => {
                              const total =
                                leafRecordsPayload?.totalConversations ?? null;
                              const shown = filteredLeafRecords.length;
                              const isFiltered =
                                total !== null && shown < total;
                              return isFiltered ? (
                                <>
                                  <Badge variant="outline" className="text-xs">
                                    {total} conversations
                                  </Badge>
                                  <Badge variant="outline" className="text-xs">
                                    shown {shown}
                                  </Badge>
                                </>
                              ) : (
                                <Badge variant="outline" className="text-xs">
                                  {total ?? "..."} conversations
                                </Badge>
                              );
                            })()}
                            <Badge variant="outline" className="text-xs">
                              metadata{" "}
                              {leafRecordsPayload?.metadataAvailable
                                ? "available"
                                : "limited"}
                            </Badge>
                          </div>
                        </div>

                        {/* Search */}
                        <div className="border-b px-5 py-2.5">
                          <Input
                            value={leafRecordSearch}
                            onChange={(event) =>
                              setLeafRecordSearch(event.target.value)
                            }
                            placeholder="Filter by ID, task, language, metadata..."
                            className="h-8 text-sm"
                          />
                        </div>

                        {/* Loading / error / content */}
                        {leafRecordsLoading ? (
                          <div className="flex items-center gap-2 px-5 py-8 text-sm text-muted-foreground">
                            <Loader2 className="size-4 animate-spin" />
                            Loading conversations...
                          </div>
                        ) : leafRecordsError ? (
                          <div className="px-5 py-4">
                            <Alert variant="destructive">
                              <AlertTriangle className="size-4" />
                              <AlertTitle>
                                Could not load conversations
                              </AlertTitle>
                              <AlertDescription>
                                {leafRecordsError}
                              </AlertDescription>
                            </Alert>
                          </div>
                        ) : (
                          <ScrollArea className="min-h-0 flex-1 px-4 py-3">
                            <div className="space-y-2.5">
                              {filteredLeafRecords.map((record) => (
                                <div
                                  key={`${record.clusterId}-${record.conversationId}`}
                                  className="rounded-lg border border-border/70 bg-card/70 p-4"
                                >
                                  {record.facet ? (
                                    <div>
                                      <div className="flex items-start justify-between gap-2">
                                        <p className="text-sm font-semibold">
                                          {record.facet.task || "Untitled task"}
                                        </p>
                                        {record.facet.concerningScore !==
                                          null &&
                                        record.facet.concerningScore >= 3 ? (
                                          <span
                                            className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium ${
                                              record.facet.concerningScore >= 4
                                                ? "border-red-300/70 bg-red-100/80 text-red-700"
                                                : "border-amber-300/70 bg-amber-100/80 text-amber-700"
                                            }`}
                                          >
                                            concern{" "}
                                            {record.facet.concerningScore}
                                          </span>
                                        ) : null}
                                      </div>
                                      <p className="mt-1.5 text-sm text-muted-foreground">
                                        {record.facet.summary || "No summary."}
                                      </p>
                                      <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
                                        <span>
                                          turns{" "}
                                          {record.facet.turnCount ?? "n/a"}
                                        </span>
                                        <span>
                                          msgs{" "}
                                          {record.facet.messageCount ?? "n/a"}
                                        </span>
                                        {record.facet.language ? (
                                          <span>{record.facet.language}</span>
                                        ) : null}
                                      </div>
                                    </div>
                                  ) : (
                                    <p className="text-sm text-muted-foreground">
                                      No facet data for this conversation.
                                    </p>
                                  )}
                                  <div className="mt-2.5 flex flex-wrap gap-2 text-xs text-muted-foreground">
                                    <span
                                      className="font-mono"
                                      title={record.conversationId}
                                    >
                                      {record.conversationId.length > 8
                                        ? `${record.conversationId.slice(0, 8)}...`
                                        : record.conversationId}
                                    </span>
                                    <span className="font-mono opacity-60">
                                      user {record.userId || "n/a"}
                                    </span>
                                  </div>
                                  {record.userMetadata &&
                                  Object.keys(record.userMetadata).length >
                                    0 ? (
                                    <div className="mt-2.5 rounded-md border border-border/60 bg-muted/20 p-2.5">
                                      <div className="grid gap-1 text-xs text-muted-foreground md:grid-cols-2">
                                        {Object.entries(record.userMetadata)
                                          .slice(0, 6)
                                          .map(([key, value]) => (
                                            <p
                                              key={`${record.conversationId}-${key}`}
                                            >
                                              <span className="font-medium text-foreground">
                                                {key}
                                              </span>
                                              : {metadataPreview(value)}
                                            </p>
                                          ))}
                                      </div>
                                    </div>
                                  ) : null}
                                </div>
                              ))}
                              {filteredLeafRecords.length === 0 ? (
                                <p className="py-8 text-center text-sm text-muted-foreground">
                                  No conversations match this filter.
                                </p>
                              ) : null}
                            </div>
                          </ScrollArea>
                        )}
                      </>
                    ) : null}
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
