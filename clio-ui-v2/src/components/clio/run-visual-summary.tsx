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

function HelpTooltip({ content }: { content: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex size-4 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground"
          aria-label="More information"
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

function levelLabel(level: number): string {
  if (level <= 0) {
    return "Leaf cluster";
  }
  if (level === 1) {
    return "Sub-group";
  }
  return "Top-level group";
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

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "").trim();
  const full =
    normalized.length === 3
      ? normalized
          .split("")
          .map((char) => `${char}${char}`)
          .join("")
      : normalized;
  const parsed = Number.parseInt(full, 16);
  if (!Number.isFinite(parsed)) {
    return `rgb(59 130 246 / ${alpha})`;
  }
  const r = (parsed >> 16) & 255;
  const g = (parsed >> 8) & 255;
  const b = parsed & 255;
  return `rgb(${r} ${g} ${b} / ${alpha})`;
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

type HierarchyGraphNode = {
  id: string;
  name: string;
  description: string;
  size: number;
  value: number;
  level: number;
  branchId: string;
  fill: string;
  stroke: string;
  parentId: string | null;
  sourceClusterId: number | null;
  children?: HierarchyGraphNode[];
};

type HierarchySunburstSegment = {
  id: string;
  name: string;
  size: number;
  level: number;
  parentId: string | null;
  startAngle: number;
  endAngle: number;
  innerRadius: number;
  outerRadius: number;
  fill: string;
  stroke: string;
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
  return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
}

function formatTimestamp(value: string | null): string {
  if (!value) {
    return "n/a";
  }
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(parsed));
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

function hashString(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function polarToCartesian(
  cx: number,
  cy: number,
  radius: number,
  angleRadians: number,
): { x: number; y: number } {
  return {
    x: cx + radius * Math.cos(angleRadians),
    y: cy + radius * Math.sin(angleRadians),
  };
}

function describeSunburstArc(
  cx: number,
  cy: number,
  innerRadius: number,
  outerRadius: number,
  startAngle: number,
  endAngle: number,
): string {
  const startOuter = polarToCartesian(cx, cy, outerRadius, startAngle);
  const endOuter = polarToCartesian(cx, cy, outerRadius, endAngle);
  const startInner = polarToCartesian(cx, cy, innerRadius, startAngle);
  const endInner = polarToCartesian(cx, cy, innerRadius, endAngle);
  const largeArcFlag = endAngle - startAngle > Math.PI ? 1 : 0;

  return [
    `M ${startOuter.x} ${startOuter.y}`,
    `A ${outerRadius} ${outerRadius} 0 ${largeArcFlag} 1 ${endOuter.x} ${endOuter.y}`,
    `L ${endInner.x} ${endInner.y}`,
    `A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${startInner.x} ${startInner.y}`,
    "Z",
  ].join(" ");
}

function truncateSunburstLabel(label: string, maxLength = 24): string {
  const normalized = label.trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(5, maxLength - 1)).trimEnd()}…`;
}

type HierarchyNavigationRow = {
  node: RunVisualHierarchyNode;
  depth: number;
  accentColor: string;
  guideOffsets: number[];
  shareOfParent: number;
  shareOfRun: number;
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
  const [collapsedRootIds, setCollapsedRootIds] = useState<Set<string>>(
    () => new Set<string>(),
  );
  const [hierarchySearchQuery, setHierarchySearchQuery] = useState("");
  const [hierarchyGraphMode, setHierarchyGraphMode] = useState<
    "sunburst" | "icicle"
  >("sunburst");
  const [activeSunburstNodeId, setActiveSunburstNodeId] = useState("");
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

  const selectedHierarchyNode = selectedHierarchyNodeId
    ? (nodeById.get(selectedHierarchyNodeId) ?? null)
    : null;
  const selectedChildren = selectedHierarchyNode
    ? (childIdsByParent.get(selectedHierarchyNode.id) ?? [])
        .map((id) => nodeById.get(id))
        .filter((node): node is NonNullable<typeof node> => Boolean(node))
    : [];
  const selectedLeaves = selectedChildren.filter((node) => node.level === 0);
  const selectedGroups = selectedChildren.filter((node) => node.level > 0);
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
  const selectedParentNode =
    selectedHierarchyNode?.parentId &&
    nodeById.has(selectedHierarchyNode.parentId)
      ? (nodeById.get(selectedHierarchyNode.parentId) ?? null)
      : null;
  const normalizedHierarchySearchQuery = hierarchySearchQuery
    .trim()
    .toLowerCase();
  const leafSiblingLeaves = useMemo(() => {
    if (!selectedHierarchyNode || selectedHierarchyNode.level !== 0) {
      return [];
    }
    if (!selectedParentNode) {
      return [];
    }
    return (childIdsByParent.get(selectedParentNode.id) ?? [])
      .map((id) => nodeById.get(id))
      .filter((node): node is NonNullable<typeof node> => {
        if (!node) {
          return false;
        }
        return node.level === 0;
      })
      .sort((a, b) => b.size - a.size);
  }, [childIdsByParent, nodeById, selectedHierarchyNode, selectedParentNode]);
  const openLeafRecords = (nodeId: string, clusterId: number) => {
    setLeafRecordsNodeId(nodeId);
    setLeafRecordsClusterId(clusterId);
    setLeafRecordSearch("");
  };
  const hierarchyBreadcrumb = useMemo(() => {
    if (!selectedHierarchyNode) {
      return [];
    }
    const chain = [selectedHierarchyNode];
    let cursor = selectedHierarchyNode;
    while (cursor.parentId && nodeById.has(cursor.parentId)) {
      const parent = nodeById.get(cursor.parentId);
      if (!parent) {
        break;
      }
      chain.push(parent);
      cursor = parent;
    }
    return chain.reverse();
  }, [nodeById, selectedHierarchyNode]);

  const flattenedHierarchy = useMemo(() => {
    const rows: Array<{ nodeId: string; depth: number }> = [];
    const visited = new Set<string>();

    const walk = (nodeId: string, depth: number) => {
      if (visited.has(nodeId)) {
        return;
      }
      visited.add(nodeId);
      rows.push({ nodeId, depth });
      const childIds = childIdsByParent.get(nodeId) ?? [];
      childIds
        .map((id) => nodeById.get(id))
        .filter((node): node is NonNullable<typeof node> => Boolean(node))
        .sort((a, b) => b.level - a.level || b.size - a.size)
        .forEach((child) => {
          walk(child.id, depth + 1);
        });
    };

    rootNodeIds.forEach((rootId) => {
      walk(rootId, 0);
    });
    return rows;
  }, [childIdsByParent, nodeById, rootNodeIds]);
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
  const hierarchyNavigationRows = useMemo<HierarchyNavigationRow[]>(() => {
    const rows: HierarchyNavigationRow[] = [];
    for (const row of flattenedHierarchy) {
      const node = nodeById.get(row.nodeId);
      if (!node) {
        continue;
      }
      const parentSize =
        node.parentId && nodeById.has(node.parentId)
          ? Math.max(1, nodeById.get(node.parentId)?.size ?? 1)
          : Math.max(1, node.size);
      rows.push({
        node,
        depth: row.depth,
        accentColor: nodeAccentById.get(node.id) ?? ROOT_ACCENT_PALETTE[0],
        guideOffsets: Array.from({ length: row.depth }, (_, index) => index),
        shareOfParent: node.size / parentSize,
        shareOfRun: hierarchyTotalSize > 0 ? node.size / hierarchyTotalSize : 0,
      });
    }
    return rows;
  }, [flattenedHierarchy, hierarchyTotalSize, nodeAccentById, nodeById]);
  const selectedHierarchyAccentColor = selectedHierarchyNode
    ? (nodeAccentById.get(selectedHierarchyNode.id) ?? ROOT_ACCENT_PALETTE[0])
    : ROOT_ACCENT_PALETTE[0];
  const branchIdByNodeId = useMemo(() => {
    const cache = new Map<string, string>();

    const resolveBranch = (nodeId: string): string => {
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
      const parent = nodeById.get(node.parentId);
      if (!parent) {
        cache.set(nodeId, node.id);
        return node.id;
      }
      if (!parent.parentId) {
        cache.set(nodeId, node.id);
        return node.id;
      }
      const resolved = resolveBranch(parent.id);
      cache.set(nodeId, resolved);
      return resolved;
    };

    for (const node of hierarchyNodes) {
      resolveBranch(node.id);
    }
    return cache;
  }, [hierarchyNodes, nodeById]);
  const branchAccentById = useMemo(() => {
    const orderedBranchIds = Array.from(
      new Set(
        hierarchyNodes
          .map((node) => branchIdByNodeId.get(node.id) ?? node.id)
          .filter((branchId) => branchId.length > 0),
      ),
    );
    const map = new Map<string, string>();
    orderedBranchIds.forEach((branchId, index) => {
      map.set(
        branchId,
        ROOT_ACCENT_PALETTE[index % ROOT_ACCENT_PALETTE.length],
      );
    });
    return map;
  }, [branchIdByNodeId, hierarchyNodes]);
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
  const hierarchyNavigationSections = useMemo(() => {
    const byRoot = new Map<string, HierarchyNavigationRow[]>();
    for (const row of hierarchyNavigationRows) {
      const rootId = rootIdByNodeId.get(row.node.id) ?? row.node.id;
      const current = byRoot.get(rootId) ?? [];
      current.push(row);
      byRoot.set(rootId, current);
    }
    return rootNodeIds
      .map((rootId) => ({
        rootId,
        rootNode: nodeById.get(rootId) ?? null,
        rows: (byRoot.get(rootId) ?? []).filter((row) =>
          hierarchyVisibleNodeIds
            ? hierarchyVisibleNodeIds.has(row.node.id)
            : true,
        ),
      }))
      .filter((section) => section.rootNode && section.rows.length > 0);
  }, [
    hierarchyNavigationRows,
    hierarchyVisibleNodeIds,
    nodeById,
    rootIdByNodeId,
    rootNodeIds,
  ]);
  const selectedRootId = selectedHierarchyNode
    ? (rootIdByNodeId.get(selectedHierarchyNode.id) ?? selectedHierarchyNode.id)
    : (rootNodeIds[0] ?? null);
  const hierarchyFocusNodeId = useMemo(() => {
    if (!selectedHierarchyNode) {
      return selectedRootId;
    }
    if (
      selectedHierarchyNode.level === 0 &&
      selectedHierarchyNode.parentId &&
      nodeById.has(selectedHierarchyNode.parentId)
    ) {
      return selectedHierarchyNode.parentId;
    }
    return selectedHierarchyNode.id;
  }, [nodeById, selectedHierarchyNode, selectedRootId]);
  const hierarchyGraphData = useMemo<HierarchyGraphNode | null>(() => {
    if (!hierarchyFocusNodeId) {
      return null;
    }

    const buildNode = (
      nodeId: string,
      parentId: string | null,
    ): HierarchyGraphNode | null => {
      const node = nodeById.get(nodeId);
      if (!node) {
        return null;
      }
      const childNodes = (childIdsByParent.get(node.id) ?? [])
        .map((childId) => buildNode(childId, node.id))
        .filter((child): child is HierarchyGraphNode => Boolean(child));
      const branchId = branchIdByNodeId.get(node.id) ?? node.id;
      const accent = branchAccentById.get(branchId) ?? ROOT_ACCENT_PALETTE[0];
      const alphaBase = node.level >= 2 ? 0.22 : node.level === 1 ? 0.3 : 0.4;
      const alphaJitter = (hashString(node.id) % 4) * 0.03 - 0.045;
      const alpha = clamp(alphaBase + alphaJitter, 0.2, 0.72);
      return {
        id: node.id,
        name: node.name,
        description: node.description,
        size: Math.max(1, node.size),
        value: Math.max(1, node.size),
        level: node.level,
        branchId,
        fill: hexToRgba(accent, alpha),
        stroke: hexToRgba(accent, 0.9),
        parentId,
        sourceClusterId: node.sourceClusterId,
        children: childNodes.length > 0 ? childNodes : undefined,
      };
    };
    return buildNode(hierarchyFocusNodeId, null);
  }, [
    branchAccentById,
    branchIdByNodeId,
    childIdsByParent,
    hierarchyFocusNodeId,
    nodeById,
  ]);
  const hierarchyGraphLegend = useMemo(() => {
    if (
      !hierarchyGraphData?.children ||
      hierarchyGraphData.children.length === 0
    ) {
      return [];
    }
    const focusSize = Math.max(1, hierarchyGraphData.size);
    return hierarchyGraphData.children
      .slice()
      .sort((a, b) => b.size - a.size)
      .slice(0, 8)
      .map((child) => ({
        id: child.id,
        name: child.name,
        color: child.fill,
        share: child.size / focusSize,
      }));
  }, [hierarchyGraphData]);
  const hierarchySunburstSegments = useMemo(() => {
    if (
      !hierarchyGraphData?.children ||
      hierarchyGraphData.children.length === 0
    ) {
      return [] as HierarchySunburstSegment[];
    }

    const maxDepthRef = { current: 1 };
    const depthOf = (node: HierarchyGraphNode, depth: number): number => {
      maxDepthRef.current = Math.max(maxDepthRef.current, depth);
      if (!node.children || node.children.length === 0) {
        return depth;
      }
      return node.children.reduce(
        (maxDepth, child) => Math.max(maxDepth, depthOf(child, depth + 1)),
        depth,
      );
    };

    hierarchyGraphData.children.forEach((child) => {
      depthOf(child, 1);
    });

    const maxDepth = Math.max(1, maxDepthRef.current);
    const baseInnerRadius = 36;
    const baseOuterRadius = 156;
    const ringPadding = 2;
    const ringThickness = (baseOuterRadius - baseInnerRadius) / maxDepth;

    const segments: HierarchySunburstSegment[] = [];
    const walk = (
      nodes: HierarchyGraphNode[],
      startAngle: number,
      endAngle: number,
      depth: number,
    ) => {
      const boundedSpan = Math.max(0, endAngle - startAngle);
      if (boundedSpan <= 0) {
        return;
      }
      const total = nodes.reduce(
        (sum, node) => sum + Math.max(1, node.size),
        0,
      );
      if (total <= 0) {
        return;
      }
      let cursor = startAngle;
      for (const node of nodes) {
        const nodeSize = Math.max(1, node.size);
        const slice = boundedSpan * (nodeSize / total);
        const nodeStart = cursor;
        const nodeEnd = cursor + slice;
        const innerRadius =
          baseInnerRadius + (depth - 1) * ringThickness + ringPadding / 2;
        const outerRadius = innerRadius + ringThickness - ringPadding;
        segments.push({
          id: node.id,
          name: node.name,
          size: node.size,
          level: node.level,
          parentId: node.parentId,
          startAngle: nodeStart,
          endAngle: nodeEnd,
          innerRadius,
          outerRadius,
          fill: node.fill,
          stroke: node.stroke,
        });
        if (node.children && node.children.length > 0) {
          walk(node.children, nodeStart, nodeEnd, depth + 1);
        }
        cursor = nodeEnd;
      }
    };

    walk(hierarchyGraphData.children, -Math.PI / 2, Math.PI * 1.5, 1);
    return segments;
  }, [hierarchyGraphData]);
  const hierarchyIcicleLanes = useMemo(() => {
    if (
      !hierarchyGraphData?.children ||
      hierarchyGraphData.children.length === 0
    ) {
      return [] as Array<{
        depth: number;
        nodes: HierarchyGraphNode[];
      }>;
    }
    const byDepth = new Map<number, HierarchyGraphNode[]>();
    const walk = (node: HierarchyGraphNode, depth: number) => {
      const current = byDepth.get(depth) ?? [];
      current.push(node);
      byDepth.set(depth, current);
      (node.children ?? []).forEach((child) => {
        walk(child, depth + 1);
      });
    };
    hierarchyGraphData.children.forEach((child) => {
      walk(child, 1);
    });
    return Array.from(byDepth.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([depth, nodes]) => ({
        depth,
        nodes: nodes.slice().sort((a, b) => b.size - a.size),
      }));
  }, [hierarchyGraphData]);
  const hierarchyFocusLabel = hierarchyFocusNodeId
    ? (nodeById.get(hierarchyFocusNodeId)?.name ?? "selected context")
    : "selected context";
  const hierarchyFocusNodeSize = Math.max(1, hierarchyGraphData?.size ?? 1);
  const activeSunburstNode =
    activeSunburstNodeId && nodeById.has(activeSunburstNodeId)
      ? (nodeById.get(activeSunburstNodeId) ?? null)
      : null;
  const hoveredSunburstNodeShare =
    activeSunburstNode && hierarchyFocusNodeSize > 0
      ? activeSunburstNode.size / hierarchyFocusNodeSize
      : null;
  const selectedShareOfRun =
    hierarchyTotalSize > 0 && selectedHierarchyNode
      ? selectedHierarchyNode.size / hierarchyTotalSize
      : 0;
  const selectedShareOfParent =
    selectedHierarchyNode && selectedParentNode
      ? selectedHierarchyNode.size / Math.max(1, selectedParentNode.size)
      : 1;
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
  const resolveSegmentIdFromEventTarget = (
    target: EventTarget | null,
  ): string | null => {
    if (!(target instanceof Element)) {
      return null;
    }
    const node = target.closest("[data-segment-id]");
    if (!node) {
      return null;
    }
    const segmentId = node.getAttribute("data-segment-id");
    return segmentId && segmentId.length > 0 ? segmentId : null;
  };

  useEffect(() => {
    if (!isHierarchyDialogOpen || rootNodeIds.length === 0) {
      return;
    }
    setCollapsedRootIds((current) => {
      const next = new Set(
        Array.from(current).filter((rootId) => rootNodeIds.includes(rootId)),
      );
      if (next.size > 0) {
        return next;
      }
      return new Set(rootNodeIds.slice(1));
    });
  }, [isHierarchyDialogOpen, rootNodeIds]);

  useEffect(() => {
    if (!selectedHierarchyNodeId) {
      return;
    }
    const rootId = rootIdByNodeId.get(selectedHierarchyNodeId);
    if (!rootId) {
      return;
    }
    setCollapsedRootIds((current) => {
      if (!current.has(rootId)) {
        return current;
      }
      const next = new Set(current);
      next.delete(rootId);
      return next;
    });
  }, [rootIdByNodeId, selectedHierarchyNodeId]);
  useEffect(() => {
    if (!isHierarchyDialogOpen) {
      return;
    }
    if (!normalizedHierarchySearchQuery) {
      return;
    }
    setCollapsedRootIds(new Set());
  }, [isHierarchyDialogOpen, normalizedHierarchySearchQuery]);

  useEffect(() => {
    if (!hierarchyFocusNodeId) {
      setActiveSunburstNodeId("");
      return;
    }
    if (activeSunburstNodeId && nodeById.has(activeSunburstNodeId)) {
      return;
    }
    setActiveSunburstNodeId(hierarchyFocusNodeId);
  }, [activeSunburstNodeId, hierarchyFocusNodeId, nodeById]);

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

  const leafRecordsDialogOpen = leafRecordsClusterId !== null;
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
                    <div className="flex flex-wrap gap-2 text-xs">
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
                        <Badge variant="outline" className="gap-1">
                          depth {generatedHierarchyDepth}/
                          {requestedHierarchyLevels}
                          <HelpTooltip
                            content={`Generated hierarchy depth vs requested hierarchy levels. Policy: ${hierarchyDepthPolicy ?? "adaptive"}. Requested levels are an upper bound.`}
                          />
                        </Badge>
                      ) : null}
                      {generatedHierarchyDepth !== null &&
                      requestedHierarchyLevels !== null &&
                      generatedHierarchyDepth < requestedHierarchyLevels &&
                      hierarchyDepthStopReason ? (
                        <Badge variant="outline" className="gap-1">
                          reason {hierarchyDepthStopReason}
                          {hierarchyWhyNotDeeper ? (
                            <HelpTooltip content={hierarchyWhyNotDeeper} />
                          ) : null}
                        </Badge>
                      ) : null}
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid gap-2 md:grid-cols-2">
                      {(visuals?.hierarchy?.topLevelClusters ?? [])
                        .slice(0, 4)
                        .map((item) => (
                          <div
                            key={item.id}
                            className="rounded-lg border border-border/70 bg-muted/20 p-3"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <p className="text-sm font-medium">{item.name}</p>
                              <Badge variant="outline">
                                children {item.childCount}
                              </Badge>
                            </div>
                            {item.description ? (
                              <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                                {item.description}
                              </p>
                            ) : null}
                            <Button
                              size="sm"
                              variant="outline"
                              className="mt-2"
                              onClick={() => {
                                setSelectedHierarchyNodeId(item.id);
                                setIsHierarchyDialogOpen(true);
                              }}
                            >
                              Explore group
                            </Button>
                          </div>
                        ))}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Insight focus: cluster semantics and hierarchy structure.
                      Privacy and evaluation metrics are available under
                      secondary analysis.
                    </p>
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
                        <HelpTooltip content="Number of clustered conversation summaries plotted on this map." />
                      </Badge>
                      <Badge variant="outline" className="gap-1">
                        projection{" "}
                        {normalizeProjectionMethod(
                          visuals?.map?.projectionMethod ?? null,
                        )}
                        <HelpTooltip content="Each point has 2 reduced coordinates. They are not raw business metrics - they are map coordinates derived from embedding similarity." />
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[360px] 2xl:h-[420px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart
                          margin={{ top: 10, right: 10, bottom: 10, left: 0 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            type="number"
                            dataKey="x"
                            name={projectionAxisLabel(
                              visuals?.map?.projectionMethod ?? null,
                              1,
                            )}
                            tickFormatter={(value) =>
                              formatProjectionValue(Number(value))
                            }
                          />
                          <YAxis
                            type="number"
                            dataKey="y"
                            name={projectionAxisLabel(
                              visuals?.map?.projectionMethod ?? null,
                              2,
                            )}
                            tickFormatter={(value) =>
                              formatProjectionValue(Number(value))
                            }
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
                          <Scatter
                            name="kept"
                            data={keptPoints}
                            fill="var(--color-chart-2)"
                          />
                          <Scatter
                            name="filtered"
                            data={filteredPoints}
                            fill="var(--color-chart-4)"
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground">
                      Hover any point to see cluster identity and coordinate
                      meaning. Distances on this map represent semantic
                      similarity, not volume or score magnitude.
                    </p>
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
                                    <HelpTooltip content={stage.help} />
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
              <DialogContent className="h-[95dvh] w-[98vw] max-w-[98vw] overflow-hidden p-0 sm:max-w-[98vw]">
                <DialogHeader className="border-b px-6 py-4">
                  <DialogTitle className="flex items-center gap-2">
                    <GitBranch className="size-4 text-primary" />
                    Hierarchy drill-down
                  </DialogTitle>
                  <DialogDescription>
                    Navigate hierarchy groups and inspect leaf clusters in one
                    large view.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid h-[calc(95dvh-92px)] grid-cols-1 xl:grid-cols-[420px_1fr]">
                  <div className="border-b xl:border-r xl:border-b-0">
                    <div className="border-b px-4 py-3">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-medium">
                          Hierarchy navigation
                        </p>
                        <div className="flex items-center gap-1">
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 px-2 text-[11px]"
                            onClick={() => setCollapsedRootIds(new Set())}
                          >
                            Expand all
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 px-2 text-[11px]"
                            onClick={() =>
                              setCollapsedRootIds(new Set(rootNodeIds))
                            }
                          >
                            Collapse all
                          </Button>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Select any group to drill down to leaf clusters.
                      </p>
                      <div className="mt-2 space-y-1.5">
                        <label
                          htmlFor="hierarchy-search"
                          className="text-[11px] font-medium text-muted-foreground"
                        >
                          Search groups and leaf clusters
                        </label>
                        <div className="relative">
                          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                          <Input
                            id="hierarchy-search"
                            value={hierarchySearchQuery}
                            onChange={(event) =>
                              setHierarchySearchQuery(event.target.value)
                            }
                            placeholder="Type keywords, cluster names, or ids"
                            className="h-8 pl-8 text-xs"
                          />
                        </div>
                        {normalizedHierarchySearchQuery ? (
                          <p className="text-[11px] text-muted-foreground">
                            {hierarchyMatchNodeIds.size} matching nodes
                          </p>
                        ) : null}
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
                        <span
                          className={`rounded-full border px-1.5 py-0.5 ${levelPillClass(2)}`}
                        >
                          Top-level
                        </span>
                        <span
                          className={`rounded-full border px-1.5 py-0.5 ${levelPillClass(1)}`}
                        >
                          Sub-group
                        </span>
                        <span
                          className={`rounded-full border px-1.5 py-0.5 ${levelPillClass(0)}`}
                        >
                          Leaf
                        </span>
                      </div>
                    </div>
                    <ScrollArea className="h-[38dvh] xl:h-[calc(95dvh-146px)] px-2 py-2">
                      <div className="space-y-2">
                        {hierarchyNavigationSections.map((section) => {
                          const isCollapsed = collapsedRootIds.has(
                            section.rootId,
                          );
                          const rootAccent =
                            nodeAccentById.get(section.rootId) ??
                            ROOT_ACCENT_PALETTE[0];
                          return (
                            <div
                              key={section.rootId}
                              className="rounded-lg border border-border/65 bg-muted/15 p-1"
                            >
                              <button
                                type="button"
                                onClick={() =>
                                  setCollapsedRootIds((current) => {
                                    const next = new Set(current);
                                    if (next.has(section.rootId)) {
                                      next.delete(section.rootId);
                                    } else {
                                      next.add(section.rootId);
                                    }
                                    return next;
                                  })
                                }
                                className="flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted/30"
                                style={{
                                  borderLeft: `3px solid ${rootAccent}`,
                                }}
                              >
                                <div className="min-w-0">
                                  <p className="truncate text-xs font-semibold">
                                    {section.rootNode?.name}
                                  </p>
                                  <p className="text-[10px] text-muted-foreground">
                                    {section.rows.length} nodes
                                  </p>
                                </div>
                                {isCollapsed ? (
                                  <ChevronRight className="size-3.5 text-muted-foreground" />
                                ) : (
                                  <ChevronDown className="size-3.5 text-muted-foreground" />
                                )}
                              </button>

                              {!isCollapsed ? (
                                <div className="mt-1 space-y-1">
                                  {section.rows.map((row) => {
                                    const isSelected =
                                      row.node.id === selectedHierarchyNodeId;
                                    const isMatch = hierarchyMatchNodeIds.has(
                                      row.node.id,
                                    );
                                    return (
                                      <button
                                        type="button"
                                        key={row.node.id}
                                        onClick={() =>
                                          setSelectedHierarchyNodeId(
                                            row.node.id,
                                          )
                                        }
                                        className={`[content-visibility:auto] relative w-full rounded-md border px-2 py-2 text-left transition-colors ${
                                          isSelected
                                            ? "border-border/80 ring-1 ring-border/40"
                                            : isMatch
                                              ? "border-border/75 bg-muted/28 ring-1 ring-primary/25 hover:bg-muted/35"
                                              : "border-border/60 bg-muted/20 hover:bg-muted/35"
                                        }`}
                                        style={{
                                          paddingLeft: `${0.9 + row.depth * 0.9}rem`,
                                          borderLeftWidth: "3px",
                                          borderLeftColor: row.accentColor,
                                          backgroundColor: isSelected
                                            ? hexToRgba(row.accentColor, 0.19)
                                            : hexToRgba(row.accentColor, 0.08),
                                        }}
                                      >
                                        {row.guideOffsets.length > 0 ? (
                                          <div className="pointer-events-none absolute top-1.5 bottom-1.5 left-0">
                                            {row.guideOffsets.map(
                                              (guideIndex) => (
                                                <span
                                                  key={`${row.node.id}-guide-${guideIndex}`}
                                                  className="absolute top-0 bottom-0 w-px bg-border/65"
                                                  style={{
                                                    left: `${0.52 + guideIndex * 0.9}rem`,
                                                  }}
                                                />
                                              ),
                                            )}
                                          </div>
                                        ) : null}

                                        <div className="flex items-start justify-between gap-2">
                                          <p className="truncate text-sm font-medium">
                                            {row.node.name}
                                          </p>
                                          <span
                                            className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${levelPillClass(row.node.level)}`}
                                          >
                                            {row.node.level <= 0
                                              ? "Leaf"
                                              : row.node.level === 1
                                                ? "Sub"
                                                : "Top"}
                                          </span>
                                        </div>
                                        <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
                                          <span>
                                            {levelLabel(row.node.level)}
                                          </span>
                                          <span>size {row.node.size}</span>
                                          <span>
                                            {formatShare(row.shareOfParent)} of
                                            parent
                                          </span>
                                        </div>
                                        <p className="mt-0.5 text-[10px] text-muted-foreground">
                                          {formatShare(row.shareOfRun)} of run
                                          volume
                                        </p>
                                      </button>
                                    );
                                  })}
                                </div>
                              ) : null}
                            </div>
                          );
                        })}
                        {hierarchyNavigationSections.length === 0 ? (
                          <p className="rounded-md border border-dashed border-border/70 px-3 py-4 text-xs text-muted-foreground">
                            No hierarchy nodes match this search.
                          </p>
                        ) : null}
                      </div>
                    </ScrollArea>
                  </div>

                  <div className="flex min-h-0 flex-col">
                    {selectedHierarchyNode ? (
                      <>
                        <div className="border-b px-6 py-4">
                          <div className="flex flex-wrap items-center gap-2">
                            {hierarchyBreadcrumb.map((node) => (
                              <Badge
                                key={node.id}
                                variant="outline"
                                className="border-transparent"
                                style={{
                                  color:
                                    nodeAccentById.get(node.id) ??
                                    ROOT_ACCENT_PALETTE[0],
                                  backgroundColor:
                                    node.id === selectedHierarchyNode.id
                                      ? hexToRgba(
                                          nodeAccentById.get(node.id) ??
                                            ROOT_ACCENT_PALETTE[0],
                                          0.23,
                                        )
                                      : hexToRgba(
                                          nodeAccentById.get(node.id) ??
                                            ROOT_ACCENT_PALETTE[0],
                                          0.11,
                                        ),
                                }}
                              >
                                {node.name}
                              </Badge>
                            ))}
                          </div>
                          <h3 className="mt-3 text-xl font-semibold">
                            {selectedHierarchyNode.name}
                          </h3>
                          <p className="mt-1 text-sm text-muted-foreground">
                            {selectedHierarchyNode.description ||
                              "No description available for this hierarchy node."}
                          </p>
                        </div>
                        <ScrollArea className="h-[calc(57dvh-40px)] xl:h-[calc(95dvh-238px)] px-6 py-4">
                          <div className="space-y-4">
                            {hierarchyGraphData ? (
                              <div className="rounded-xl border border-border/70 bg-muted/10 p-3">
                                <div className="flex flex-wrap items-start justify-between gap-2">
                                  <div>
                                    <p className="text-sm font-medium">
                                      Hierarchy graph
                                    </p>
                                    <p className="mt-1 text-[11px] text-muted-foreground">
                                      Focused on {hierarchyFocusLabel}
                                      {selectedHierarchyNode.level === 0
                                        ? " (leaf selected, parent context shown)"
                                        : ""}
                                    </p>
                                    {generatedHierarchyDepth !== null &&
                                    requestedHierarchyLevels !== null ? (
                                      <p className="mt-1 text-[11px] text-muted-foreground">
                                        Generated depth{" "}
                                        {generatedHierarchyDepth} of requested{" "}
                                        {requestedHierarchyLevels} levels
                                        {hierarchyDepthPolicy
                                          ? ` (${hierarchyDepthPolicy} policy).`
                                          : "."}
                                      </p>
                                    ) : null}
                                    {generatedHierarchyDepth !== null &&
                                    requestedHierarchyLevels !== null &&
                                    generatedHierarchyDepth <
                                      requestedHierarchyLevels &&
                                    hierarchyWhyNotDeeper ? (
                                      <p className="mt-1 text-[11px] text-muted-foreground">
                                        {hierarchyWhyNotDeeper}
                                      </p>
                                    ) : null}
                                  </div>
                                  <div className="inline-flex rounded-md border border-border/70 bg-card/70 p-0.5">
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setHierarchyGraphMode("sunburst")
                                      }
                                      className={`rounded px-2.5 py-1 text-[11px] ${
                                        hierarchyGraphMode === "sunburst"
                                          ? "bg-primary text-primary-foreground"
                                          : "text-muted-foreground"
                                      }`}
                                    >
                                      Sunburst
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setHierarchyGraphMode("icicle")
                                      }
                                      className={`rounded px-2.5 py-1 text-[11px] ${
                                        hierarchyGraphMode === "icicle"
                                          ? "bg-primary text-primary-foreground"
                                          : "text-muted-foreground"
                                      }`}
                                    >
                                      Icicle
                                    </button>
                                  </div>
                                </div>
                                {hierarchyGraphLegend.length > 0 ? (
                                  <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
                                    {hierarchyGraphLegend.map((item) => (
                                      <button
                                        type="button"
                                        key={`legend-${item.id}`}
                                        onClick={() =>
                                          setSelectedHierarchyNodeId(item.id)
                                        }
                                        className="inline-flex items-center gap-1 rounded-full border border-border/65 bg-card/70 px-2 py-0.5 text-left transition-colors hover:bg-muted/40"
                                      >
                                        <span
                                          className="size-2 rounded-full"
                                          style={{
                                            backgroundColor: item.color,
                                          }}
                                        />
                                        <span className="max-w-40 truncate">
                                          {item.name}
                                        </span>
                                        <span className="text-muted-foreground">
                                          {formatShare(item.share)}
                                        </span>
                                      </button>
                                    ))}
                                  </div>
                                ) : null}
                                <p className="mt-1 text-[11px] text-muted-foreground">
                                  Drill down using left tree rows, legend pills,
                                  or sub-group cards below.
                                </p>
                                {hierarchyGraphMode === "sunburst" ? (
                                  <div className="mt-2 h-[330px] 2xl:h-[400px]">
                                    <button
                                      type="button"
                                      className="h-full w-full rounded-md border border-transparent text-left transition-colors hover:border-border/70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                                      aria-label="Hierarchy sunburst chart. Click a segment to focus that branch."
                                      onClick={(event) => {
                                        const segmentId =
                                          resolveSegmentIdFromEventTarget(
                                            event.target,
                                          );
                                        if (segmentId) {
                                          setSelectedHierarchyNodeId(segmentId);
                                        }
                                      }}
                                      onMouseMove={(event) => {
                                        const segmentId =
                                          resolveSegmentIdFromEventTarget(
                                            event.target,
                                          );
                                        setActiveSunburstNodeId(
                                          segmentId ?? "",
                                        );
                                      }}
                                      onMouseLeave={() =>
                                        setActiveSunburstNodeId("")
                                      }
                                    >
                                      <svg
                                        viewBox="0 0 420 320"
                                        className="h-full w-full"
                                        aria-hidden="true"
                                      >
                                        <g transform="translate(210 160)">
                                          {hierarchySunburstSegments.map(
                                            (segment) => {
                                              const path = describeSunburstArc(
                                                0,
                                                0,
                                                segment.innerRadius,
                                                segment.outerRadius,
                                                segment.startAngle,
                                                segment.endAngle,
                                              );
                                              const isActive =
                                                activeSunburstNodeId ===
                                                segment.id;
                                              const isSelected =
                                                selectedHierarchyNodeId ===
                                                segment.id;
                                              const segmentShare =
                                                segment.size /
                                                Math.max(
                                                  1,
                                                  hierarchyFocusNodeSize,
                                                );
                                              const angleSpan =
                                                segment.endAngle -
                                                segment.startAngle;
                                              const showLabel =
                                                angleSpan >= 0.44 &&
                                                segment.outerRadius -
                                                  segment.innerRadius >=
                                                  15;
                                              const midAngle =
                                                (segment.startAngle +
                                                  segment.endAngle) /
                                                2;
                                              const labelPoint =
                                                polarToCartesian(
                                                  0,
                                                  0,
                                                  (segment.innerRadius +
                                                    segment.outerRadius) /
                                                    2,
                                                  midAngle,
                                                );
                                              const segmentLabel = `${segment.name} (${levelLabel(
                                                segment.level,
                                              )}, ${formatShare(
                                                segmentShare,
                                              )} of current focus)`;
                                              return (
                                                <g
                                                  key={`sunburst-${segment.id}`}
                                                  data-segment-id={segment.id}
                                                >
                                                  <path
                                                    d={path}
                                                    fill={segment.fill}
                                                    stroke={
                                                      isSelected
                                                        ? "var(--foreground)"
                                                        : segment.stroke
                                                    }
                                                    strokeWidth={
                                                      isSelected
                                                        ? 2
                                                        : isActive
                                                          ? 1.5
                                                          : 1
                                                    }
                                                    opacity={
                                                      isActive || isSelected
                                                        ? 1
                                                        : activeSunburstNodeId
                                                          ? 0.62
                                                          : 0.9
                                                    }
                                                    style={{
                                                      cursor: "pointer",
                                                    }}
                                                  >
                                                    <title>
                                                      {segmentLabel}
                                                    </title>
                                                  </path>
                                                  {showLabel ? (
                                                    <text
                                                      x={labelPoint.x}
                                                      y={labelPoint.y}
                                                      textAnchor="middle"
                                                      dominantBaseline="middle"
                                                      className="fill-foreground/80 text-[7.5px] font-medium"
                                                      style={{
                                                        pointerEvents: "none",
                                                      }}
                                                    >
                                                      {truncateSunburstLabel(
                                                        segment.name,
                                                        18,
                                                      )}
                                                    </text>
                                                  ) : null}
                                                </g>
                                              );
                                            },
                                          )}
                                          <title>
                                            Click sunburst segments to focus
                                            that branch.
                                          </title>
                                          <circle
                                            cx={0}
                                            cy={0}
                                            r={30}
                                            fill="var(--background)"
                                            stroke="var(--border)"
                                            strokeWidth={1}
                                          />
                                          <text
                                            x={0}
                                            y={4}
                                            textAnchor="middle"
                                            className="fill-muted-foreground text-[8px]"
                                          >
                                            CLIO
                                          </text>
                                        </g>
                                      </svg>
                                    </button>
                                    <div className="mt-2 rounded-md border border-border/60 bg-card/65 px-2 py-1 text-[11px] text-muted-foreground">
                                      {activeSunburstNode ? (
                                        <span>
                                          Hovering{" "}
                                          <span className="font-medium text-foreground">
                                            {activeSunburstNode.name}
                                          </span>{" "}
                                          -{" "}
                                          {hoveredSunburstNodeShare !== null
                                            ? formatShare(
                                                hoveredSunburstNodeShare,
                                              )
                                            : "n/a"}{" "}
                                          of current focus.
                                        </span>
                                      ) : (
                                        <span>
                                          Hover a slice to preview its name and
                                          share.
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                ) : (
                                  <div className="mt-3 space-y-2">
                                    {hierarchyIcicleLanes.map((lane) => (
                                      <div key={`lane-${lane.depth}`}>
                                        <p className="mb-1 text-[11px] font-medium text-muted-foreground">
                                          Depth {lane.depth}
                                        </p>
                                        <div className="overflow-x-auto rounded-md border border-border/60 bg-card/40 p-1">
                                          <div className="flex min-w-[560px] gap-1">
                                            {lane.nodes.map((node) => {
                                              const share =
                                                node.size /
                                                Math.max(
                                                  1,
                                                  hierarchyFocusNodeSize,
                                                );
                                              return (
                                                <button
                                                  type="button"
                                                  key={`lane-${lane.depth}-${node.id}`}
                                                  onClick={() =>
                                                    setSelectedHierarchyNodeId(
                                                      node.id,
                                                    )
                                                  }
                                                  className="h-12 rounded px-2 text-left text-[10px] text-foreground/90"
                                                  style={{
                                                    width: `${Math.max(2, share * 100)}%`,
                                                    backgroundColor: node.fill,
                                                    border: `1px solid ${node.stroke}`,
                                                  }}
                                                >
                                                  <span className="block truncate font-medium">
                                                    {node.name}
                                                  </span>
                                                  <span className="block text-[9px] text-foreground/70">
                                                    {formatShare(share)}
                                                  </span>
                                                </button>
                                              );
                                            })}
                                          </div>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {activeSunburstNode ? (
                                  <p className="mt-2 text-[11px] text-muted-foreground">
                                    Focused node: {activeSunburstNode.name} -{" "}
                                    {formatShare(
                                      activeSunburstNode.size /
                                        Math.max(1, hierarchyFocusNodeSize),
                                    )}{" "}
                                    of current focus
                                  </p>
                                ) : null}
                              </div>
                            ) : null}

                            <div className="flex flex-wrap gap-2 text-xs">
                              <Badge
                                variant="outline"
                                className={levelPillClass(
                                  selectedHierarchyNode.level,
                                )}
                              >
                                {levelLabel(selectedHierarchyNode.level)}
                              </Badge>
                              <Badge
                                variant="outline"
                                className="border-transparent"
                                style={{
                                  color: selectedHierarchyAccentColor,
                                  backgroundColor: hexToRgba(
                                    selectedHierarchyAccentColor,
                                    0.14,
                                  ),
                                }}
                              >
                                direct children {selectedChildren.length}
                              </Badge>
                              <Badge
                                variant="outline"
                                className="border-transparent"
                                style={{
                                  color: selectedHierarchyAccentColor,
                                  backgroundColor: hexToRgba(
                                    selectedHierarchyAccentColor,
                                    0.14,
                                  ),
                                }}
                              >
                                size {selectedHierarchyNode.size}
                              </Badge>
                              <Badge
                                variant="outline"
                                className="border-transparent"
                                style={{
                                  color: selectedHierarchyAccentColor,
                                  backgroundColor: hexToRgba(
                                    selectedHierarchyAccentColor,
                                    0.14,
                                  ),
                                }}
                              >
                                {formatShare(selectedShareOfParent)} of parent
                              </Badge>
                              <Badge
                                variant="outline"
                                className="border-transparent"
                                style={{
                                  color: selectedHierarchyAccentColor,
                                  backgroundColor: hexToRgba(
                                    selectedHierarchyAccentColor,
                                    0.14,
                                  ),
                                }}
                              >
                                {formatShare(selectedShareOfRun)} of run
                              </Badge>
                              {selectedHierarchyNode.sourceClusterId !==
                              null ? (
                                <Badge
                                  variant="outline"
                                  className="border-transparent"
                                  style={{
                                    color: selectedHierarchyAccentColor,
                                    backgroundColor: hexToRgba(
                                      selectedHierarchyAccentColor,
                                      0.14,
                                    ),
                                  }}
                                >
                                  cluster{" "}
                                  {selectedHierarchyNode.sourceClusterId}
                                </Badge>
                              ) : null}
                              {selectedHierarchyNode.level === 0 &&
                              selectedHierarchyNode.sourceClusterId !== null ? (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-7"
                                  onClick={() =>
                                    openLeafRecords(
                                      selectedHierarchyNode.id,
                                      selectedHierarchyNode.sourceClusterId as number,
                                    )
                                  }
                                >
                                  View chats
                                </Button>
                              ) : null}
                            </div>

                            {selectedGroups.length > 0 ? (
                              <div className="space-y-2">
                                <p className="text-sm font-medium">
                                  Sub-groups
                                </p>
                                <div className="grid gap-2 lg:grid-cols-2">
                                  {selectedGroups.map((node) => {
                                    const accent =
                                      nodeAccentById.get(node.id) ??
                                      ROOT_ACCENT_PALETTE[0];
                                    return (
                                      <button
                                        type="button"
                                        key={node.id}
                                        onClick={() =>
                                          setSelectedHierarchyNodeId(node.id)
                                        }
                                        className="rounded-lg border border-border/70 p-3 text-left hover:bg-muted/35"
                                        style={{
                                          borderLeftWidth: "4px",
                                          borderLeftColor: accent,
                                          backgroundColor: hexToRgba(
                                            accent,
                                            0.08,
                                          ),
                                        }}
                                      >
                                        <div className="flex items-center justify-between gap-2">
                                          <p className="text-sm font-medium">
                                            {node.name}
                                          </p>
                                          <span
                                            className={`rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${levelPillClass(node.level)}`}
                                          >
                                            Sub-group
                                          </span>
                                        </div>
                                        <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                                          {node.description ||
                                            "No description available."}
                                        </p>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                          size {node.size} -{" "}
                                          {formatShare(
                                            node.size /
                                              Math.max(
                                                1,
                                                selectedHierarchyNode.size,
                                              ),
                                          )}{" "}
                                          of parent
                                        </p>
                                      </button>
                                    );
                                  })}
                                </div>
                              </div>
                            ) : null}

                            {selectedLeaves.length > 0 ? (
                              <div className="space-y-2">
                                <p className="text-sm font-medium">
                                  Leaf clusters
                                </p>
                                <div className="space-y-2">
                                  {selectedLeaves.map((leaf) => {
                                    const accent =
                                      nodeAccentById.get(leaf.id) ??
                                      ROOT_ACCENT_PALETTE[0];
                                    return (
                                      <div
                                        key={leaf.id}
                                        className="rounded-lg border border-border/70 p-3"
                                        style={{
                                          borderLeftWidth: "4px",
                                          borderLeftColor: accent,
                                          backgroundColor: hexToRgba(
                                            accent,
                                            0.07,
                                          ),
                                        }}
                                      >
                                        <div className="flex items-center justify-between gap-2">
                                          <p className="text-sm font-medium">
                                            {leaf.name}
                                          </p>
                                          <Badge
                                            variant="outline"
                                            className={`border ${levelPillClass(0)}`}
                                          >
                                            size {leaf.size}
                                          </Badge>
                                        </div>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                          {leaf.description ||
                                            "No description available."}
                                        </p>
                                        <p className="mt-1 text-[11px] text-muted-foreground">
                                          {formatShare(
                                            leaf.size /
                                              Math.max(
                                                1,
                                                selectedHierarchyNode.size,
                                              ),
                                          )}{" "}
                                          of parent
                                        </p>
                                        {leaf.sourceClusterId !== null ? (
                                          <div className="mt-1 flex flex-wrap items-center justify-between gap-2">
                                            <p className="text-[11px] text-muted-foreground">
                                              cluster id {leaf.sourceClusterId}
                                            </p>
                                            <Button
                                              size="sm"
                                              variant="outline"
                                              className="h-7"
                                              onClick={() =>
                                                openLeafRecords(
                                                  leaf.id,
                                                  leaf.sourceClusterId as number,
                                                )
                                              }
                                            >
                                              View chats
                                            </Button>
                                          </div>
                                        ) : null}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            ) : null}

                            {selectedHierarchyNode.level === 0 &&
                            leafSiblingLeaves.length > 0 ? (
                              <div className="space-y-2">
                                <p className="text-sm font-medium">
                                  Sibling leaf clusters
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {selectedParentNode
                                    ? `Selected leaf belongs to ${selectedParentNode.name}.`
                                    : "Selected leaf has no parent metadata."}
                                </p>
                                <div className="space-y-2">
                                  {leafSiblingLeaves.map((leaf) => {
                                    const accent =
                                      nodeAccentById.get(leaf.id) ??
                                      ROOT_ACCENT_PALETTE[0];
                                    const isSelected =
                                      leaf.id === selectedHierarchyNode.id;
                                    return (
                                      <button
                                        type="button"
                                        key={`sibling-${leaf.id}`}
                                        onClick={() =>
                                          setSelectedHierarchyNodeId(leaf.id)
                                        }
                                        className="w-full rounded-lg border border-border/70 p-3 text-left hover:bg-muted/35"
                                        style={{
                                          borderLeftWidth: "4px",
                                          borderLeftColor: accent,
                                          backgroundColor: isSelected
                                            ? hexToRgba(accent, 0.14)
                                            : hexToRgba(accent, 0.07),
                                        }}
                                      >
                                        <div className="flex items-center justify-between gap-2">
                                          <p className="text-sm font-medium">
                                            {leaf.name}
                                          </p>
                                          <Badge
                                            variant="outline"
                                            className={`border ${levelPillClass(0)}`}
                                          >
                                            size {leaf.size}
                                          </Badge>
                                        </div>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                          {leaf.description ||
                                            "No description available."}
                                        </p>
                                        <p className="mt-1 text-[11px] text-muted-foreground">
                                          {formatShare(
                                            leaf.size /
                                              Math.max(
                                                1,
                                                selectedParentNode?.size ?? 1,
                                              ),
                                          )}{" "}
                                          of parent
                                        </p>
                                      </button>
                                    );
                                  })}
                                </div>
                              </div>
                            ) : null}

                            {selectedGroups.length === 0 &&
                            selectedLeaves.length === 0 &&
                            !(
                              selectedHierarchyNode.level === 0 &&
                              leafSiblingLeaves.length > 0
                            ) ? (
                              <p className="text-xs text-muted-foreground">
                                No child groups are available for this node.
                              </p>
                            ) : null}
                          </div>
                        </ScrollArea>
                      </>
                    ) : (
                      <div className="p-6 text-sm text-muted-foreground">
                        No hierarchy node selected.
                      </div>
                    )}
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog
              open={leafRecordsDialogOpen}
              onOpenChange={(open) => {
                if (!open) {
                  setLeafRecordsClusterId(null);
                  setLeafRecordsNodeId("");
                }
              }}
            >
              <DialogContent className="h-[88dvh] w-[96vw] max-w-[96vw] overflow-hidden p-0 sm:max-w-[96vw]">
                <DialogHeader className="border-b px-6 py-4">
                  <DialogTitle className="text-base">
                    Cluster conversations
                  </DialogTitle>
                  <DialogDescription>
                    {selectedLeafRecordsNode
                      ? selectedLeafRecordsNode.name
                      : "Leaf cluster"}{" "}
                    {leafRecordsClusterId !== null
                      ? `(cluster ${leafRecordsClusterId})`
                      : ""}{" "}
                    - conversation IDs, user metadata, and facets only.
                  </DialogDescription>
                </DialogHeader>
                <div className="flex h-[calc(88dvh-88px)] min-h-0 flex-col px-6 py-4">
                  <div className="mb-3 flex flex-wrap items-center gap-2">
                    <Badge variant="outline">
                      total {leafRecordsPayload?.totalConversations ?? 0}
                    </Badge>
                    <Badge variant="outline">
                      shown {filteredLeafRecords.length}
                    </Badge>
                    <Badge variant="outline">
                      metadata{" "}
                      {leafRecordsPayload?.metadataAvailable
                        ? "available"
                        : "limited"}
                    </Badge>
                  </div>
                  <div className="mb-3">
                    <Input
                      value={leafRecordSearch}
                      onChange={(event) =>
                        setLeafRecordSearch(event.target.value)
                      }
                      placeholder="Search by conversation ID, user ID, task, language, or metadata"
                      className="h-8 text-xs"
                    />
                  </div>

                  {leafRecordsLoading ? (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="size-4 animate-spin" />
                      Loading cluster conversations...
                    </div>
                  ) : null}

                  {leafRecordsError ? (
                    <Alert variant="destructive" className="mb-3">
                      <AlertTriangle className="size-4" />
                      <AlertTitle>Could not load conversations</AlertTitle>
                      <AlertDescription>{leafRecordsError}</AlertDescription>
                    </Alert>
                  ) : null}

                  {!leafRecordsLoading && !leafRecordsError ? (
                    <ScrollArea className="min-h-0 flex-1 rounded-md border border-border/70 bg-muted/10 p-3">
                      <div className="space-y-2">
                        {filteredLeafRecords.map((record) => (
                          <div
                            key={`${record.clusterId}-${record.conversationId}`}
                            className="rounded-lg border border-border/70 bg-card/70 p-3"
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge
                                variant="outline"
                                className="font-mono text-[11px]"
                              >
                                chat {record.conversationId}
                              </Badge>
                              <Badge
                                variant="outline"
                                className="font-mono text-[11px]"
                              >
                                user {record.userId || "n/a"}
                              </Badge>
                              <Badge variant="outline" className="text-[11px]">
                                {formatTimestamp(record.timestampUtc)}
                              </Badge>
                            </div>

                            {record.facet ? (
                              <div className="mt-2 grid gap-2 lg:grid-cols-[1fr_auto]">
                                <div>
                                  <p className="text-xs text-muted-foreground">
                                    {record.facet.summary ||
                                      "No facet summary."}
                                  </p>
                                  <p className="mt-1 text-[11px] text-muted-foreground">
                                    task {record.facet.task || "n/a"} - language{" "}
                                    {record.facet.language || "n/a"} - concern{" "}
                                    {record.facet.concerningScore !== null
                                      ? record.facet.concerningScore.toFixed(2)
                                      : "n/a"}
                                  </p>
                                </div>
                                <div className="text-[11px] text-muted-foreground">
                                  turns {record.facet.turnCount ?? "n/a"} -
                                  messages {record.facet.messageCount ?? "n/a"}
                                </div>
                              </div>
                            ) : (
                              <p className="mt-2 text-xs text-muted-foreground">
                                Facet record not found for this conversation.
                              </p>
                            )}

                            <div className="mt-2 rounded-md border border-border/60 bg-muted/20 p-2">
                              <p className="text-[11px] font-medium">
                                User metadata
                              </p>
                              {record.userMetadata ? (
                                <div className="mt-1 grid gap-1 text-[11px] text-muted-foreground md:grid-cols-2">
                                  {Object.entries(record.userMetadata)
                                    .slice(0, 8)
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
                              ) : (
                                <p className="mt-1 text-[11px] text-muted-foreground">
                                  No user metadata attached to this
                                  conversation.
                                </p>
                              )}
                            </div>
                          </div>
                        ))}
                        {filteredLeafRecords.length === 0 ? (
                          <p className="rounded-md border border-dashed border-border/70 px-3 py-5 text-center text-xs text-muted-foreground">
                            No conversations match this filter.
                          </p>
                        ) : null}
                      </div>
                    </ScrollArea>
                  ) : null}
                </div>
              </DialogContent>
            </Dialog>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
