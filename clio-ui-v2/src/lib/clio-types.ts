export type RunState =
  | "running"
  | "completed"
  | "completed_with_warnings"
  | "failed"
  | "partial";

export type PhaseStatus =
  | "pending"
  | "running"
  | "completed"
  | "resumed"
  | "failed"
  | "skipped";

export type RunListItem = {
  runId: string;
  runRoot: string;
  phase: string;
  createdAtUtc: string;
  updatedAtUtc: string;
  completedPhases: string[];
  conversationCountInput: number;
  conversationCountProcessed: number;
  clusterCountTotal: number;
  runLockActive: boolean;
  state: RunState;
  overallProgressPercent: number;
};

export type PhaseTimelineItem = {
  phase: string;
  label: string;
  status: PhaseStatus;
  processed: number | null;
  total: number | null;
  percent: number | null;
  note: string;
  updatedAtUtc: string;
  currentConcurrency: number | null;
};

export type ArtifactStatus = {
  artifactKey: string;
  relativePath: string;
  exists: boolean;
  required: boolean;
};

export type RunEventItem = {
  id: string;
  timestampUtc: string;
  source: "run_events" | "checkpoint" | "system";
  type: string;
  message: string;
  status: string | null;
};

export type RunDetailResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  run: RunListItem;
  manifest: Record<string, unknown>;
  runMetrics: Record<string, unknown> | null;
  checkpoints: Record<string, Record<string, unknown> | null>;
  artifactStatus: ArtifactStatus[];
  phaseTimeline: PhaseTimelineItem[];
  latestEvents: RunEventItem[];
  summary: {
    totalPhases: number;
    completedPhases: number;
    failedPhases: number;
    requiredArtifactsMissing: number;
    optionalArtifactsPresent: number;
    warningCount: number;
    lockOwnerPid: number | null;
  };
};

export type RunVisualMapPoint = {
  x: number;
  y: number;
  clusterId: number;
  clusterName: string;
  kept: boolean;
  language: string;
  concerningScore: number | null;
};

export type RunVisualMapCluster = {
  clusterId: number;
  x: number;
  y: number;
  size: number;
  kept: boolean;
};

export type RunVisualTopCluster = {
  id: string;
  name: string;
  childCount: number;
  description: string;
};

export type RunVisualHierarchyNode = {
  id: string;
  parentId: string | null;
  level: number;
  name: string;
  description: string;
  size: number;
  sourceClusterId: number | null;
  childIds: string[];
};

export type RunVisualPrivacyStage = {
  stage: string;
  total: number;
  threshold: number;
  passCount: number;
  failCount: number;
  passRate: number;
};

export type RunVisualEvalAblation = {
  name: string;
  accuracy: number;
  macroF1: number;
  weightedF1: number;
};

export type RunVisualsResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  runId: string;
  map: {
    projectionMethod: string | null;
    totalPoints: number;
    sampled: boolean;
    points: RunVisualMapPoint[];
    clusters: RunVisualMapCluster[];
  } | null;
  hierarchy: {
    topLevelCount: number;
    leafCount: number;
    maxLevel: number | null;
    requestedLevels: number | null;
    topLevelClusters: RunVisualTopCluster[];
    rootNodeIds: string[];
    nodes: RunVisualHierarchyNode[];
  } | null;
  privacy: {
    stages: RunVisualPrivacyStage[];
    validation: {
      totalCases: number;
      inRangeRate: number;
      meanAbsoluteError: number;
    } | null;
  } | null;
  evaluation: {
    syntheticCount: number;
    topicCount: number;
    languageCount: number;
    ablations: RunVisualEvalAblation[];
  } | null;
};

export type RunListResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  runs: RunListItem[];
};

export type RunLaunchOptions = {
  withFacets: boolean;
  withClustering: boolean;
  withLabeling: boolean;
  withHierarchy: boolean;
  withPrivacy: boolean;
  withEval: boolean;
  streaming: boolean;
  streamChunkSize: number;
  hierarchyLevels: number;
  strict: boolean;
  limit: number | null;
  evalCount: number | null;
};

export type RunJobStatus =
  | "running"
  | "finished"
  | "finished_ok"
  | "finished_with_warnings";

export type RunJobRecord = {
  runId: string;
  runRoot: string;
  pid: number;
  status: RunJobStatus;
  running: boolean;
  hasMetrics: boolean;
  hasWarnings: boolean;
  startedAtUtc: string;
  inputPath: string;
  logPath: string;
  command: string[];
};

export type RunJobsResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  jobs: RunJobRecord[];
};

export type RunLaunchResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  job: RunJobRecord;
};

export type RunTerminateResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  runId: string;
  ok: boolean;
  status: string;
  job: RunJobRecord | null;
};

export type RunLogResponse = {
  generatedAtUtc: string;
  runsRoot: string;
  runId: string;
  status: string;
  running: boolean;
  logPath: string;
  lineCount: number;
  logTail: string;
};
