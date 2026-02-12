import { spawn } from "node:child_process";
import { randomInt } from "node:crypto";
import { closeSync, promises as fs, openSync } from "node:fs";
import path from "node:path";

import { getRunsRootPath } from "@/lib/clio-runs";
import type { RunJobRecord, RunLaunchOptions } from "@/lib/clio-types";

const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;
const RUN_ID_ALPHABET =
  "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as JsonRecord;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function utcStamp(): string {
  return new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d{3}Z$/, "Z");
}

function runJobsRoot(runsRoot: string): string {
  return path.join(runsRoot, "_jobs");
}

function uploadsRoot(runsRoot: string): string {
  return path.join(runsRoot, "_uploads");
}

function projectRoot(): string {
  return path.resolve(process.cwd(), "..");
}

async function ensureDirectory(targetPath: string): Promise<void> {
  await fs.mkdir(targetPath, { recursive: true });
}

async function readJson(filePath: string): Promise<JsonRecord | null> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return asRecord(JSON.parse(raw));
  } catch {
    return null;
  }
}

async function writeJson(filePath: string, payload: JsonRecord): Promise<void> {
  await ensureDirectory(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(payload)}\n`, "utf-8");
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function isPidRunning(pid: number): boolean {
  if (!Number.isFinite(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function resolvePathFromProject(rawPath: string, fallbackPath: string): string {
  const trimmed = rawPath.trim();
  if (!trimmed) {
    return path.resolve(fallbackPath);
  }
  if (path.isAbsolute(trimmed)) {
    return path.resolve(trimmed);
  }
  return path.resolve(projectRoot(), trimmed);
}

async function readRunLockState(runRoot: string): Promise<{
  active: boolean;
  lockFilePresent: boolean;
  lockOwnerPid: number | null;
}> {
  const lockPath = path.join(runRoot, ".run.lock");
  const lockPayload = await readJson(lockPath);
  const lockFilePresent = lockPayload !== null || (await fileExists(lockPath));
  if (!lockFilePresent) {
    return {
      active: false,
      lockFilePresent: false,
      lockOwnerPid: null,
    };
  }

  const pid = asNumber(lockPayload?.pid, 0);
  if (pid > 0) {
    return {
      active: isPidRunning(pid),
      lockFilePresent: true,
      lockOwnerPid: pid,
    };
  }
  return {
    active: true,
    lockFilePresent: true,
    lockOwnerPid: null,
  };
}

function generateRunId(size = 12): string {
  let output = "";
  for (let index = 0; index < size; index += 1) {
    output += RUN_ID_ALPHABET[randomInt(0, RUN_ID_ALPHABET.length)];
  }
  return output;
}

function sanitizeRunId(value: string): string {
  const sanitized = value.trim().replace(/[^a-zA-Z0-9._-]/g, "");
  if (!sanitized) {
    return "";
  }
  return sanitized.slice(0, 64);
}

function normalizeLaunchOptions(
  options: Partial<RunLaunchOptions>,
): RunLaunchOptions {
  const withHierarchy = Boolean(options.withHierarchy);
  const withPrivacy = Boolean(options.withPrivacy);
  const withLabeling = Boolean(
    options.withLabeling || withHierarchy || withPrivacy,
  );
  const withClustering = Boolean(options.withClustering || withLabeling);
  const withFacets = Boolean(options.withFacets || withClustering);

  const streamChunkSizeRaw = asNumber(options.streamChunkSize, 32);
  const streamChunkSize = Math.max(1, Math.floor(streamChunkSizeRaw));
  const hierarchyLevelsRaw = asNumber(options.hierarchyLevels, 5);
  const hierarchyLevels = Math.max(
    2,
    Math.min(20, Math.floor(hierarchyLevelsRaw)),
  );
  const hierarchyDepthPolicy =
    options.hierarchyDepthPolicy === "strict_min" ? "strict_min" : "adaptive";
  const clusterStrategy =
    options.clusterStrategy === "kmeans" ||
    options.clusterStrategy === "hdbscan"
      ? options.clusterStrategy
      : "hybrid";
  const clusterLeafMode =
    options.clusterLeafMode === "fixed" ? "fixed" : "auto";
  const clusterTargetLeafSize = Math.max(
    1,
    Math.floor(asNumber(options.clusterTargetLeafSize, 25)),
  );
  const clusterMinLeafClusters = Math.max(
    1,
    Math.floor(asNumber(options.clusterMinLeafClusters, 20)),
  );
  const clusterMaxLeafClusters = Math.max(
    clusterMinLeafClusters,
    Math.floor(asNumber(options.clusterMaxLeafClusters, 600)),
  );
  const clusterHdbscanMinClusterSize = Math.max(
    2,
    Math.floor(asNumber(options.clusterHdbscanMinClusterSize, 12)),
  );
  const clusterHdbscanMinSamples = Math.max(
    1,
    Math.floor(asNumber(options.clusterHdbscanMinSamples, 6)),
  );
  const clusterNoisePolicy =
    options.clusterNoisePolicy === "singleton" ||
    options.clusterNoisePolicy === "drop"
      ? options.clusterNoisePolicy
      : "nearest";

  const limitRaw = asNumber(options.limit, 0);
  const evalCountRaw = asNumber(options.evalCount, 0);

  return {
    withFacets,
    withClustering,
    withLabeling,
    withHierarchy,
    withPrivacy,
    withEval: Boolean(options.withEval),
    streaming: Boolean(options.streaming),
    streamChunkSize,
    hierarchyLevels,
    hierarchyDepthPolicy,
    clusterStrategy,
    clusterLeafMode,
    clusterTargetLeafSize,
    clusterMinLeafClusters,
    clusterMaxLeafClusters,
    clusterHdbscanMinClusterSize,
    clusterHdbscanMinSamples,
    clusterNoisePolicy,
    strict: Boolean(options.strict),
    limit: limitRaw > 0 ? Math.floor(limitRaw) : null,
    evalCount: evalCountRaw > 0 ? Math.floor(evalCountRaw) : null,
  };
}

function resolveConfigPath(): string {
  const envValue = process.env.CLIO_CONFIG_PATH?.trim();
  if (!envValue) {
    return path.join(projectRoot(), "configs", "default.yaml");
  }
  return path.isAbsolute(envValue)
    ? envValue
    : path.resolve(projectRoot(), envValue);
}

function buildRunCommand(params: {
  runId: string;
  inputPath: string;
  options: RunLaunchOptions;
}): { command: string; args: string[] } {
  const { runId, inputPath, options } = params;
  const configPath = resolveConfigPath();
  const runArgs = [
    "--config",
    configPath,
    "run",
    "--run-id",
    runId,
    "--input",
    inputPath,
  ];

  if (options.withFacets) {
    runArgs.push("--with-facets");
  }
  if (options.withClustering) {
    runArgs.push("--with-clustering");
    runArgs.push("--cluster-strategy", String(options.clusterStrategy));
    runArgs.push("--cluster-leaf-mode", String(options.clusterLeafMode));
    runArgs.push(
      "--cluster-target-leaf-size",
      String(options.clusterTargetLeafSize),
    );
    runArgs.push(
      "--cluster-min-leaf-clusters",
      String(options.clusterMinLeafClusters),
    );
    runArgs.push(
      "--cluster-max-leaf-clusters",
      String(options.clusterMaxLeafClusters),
    );
    runArgs.push(
      "--cluster-hdbscan-min-cluster-size",
      String(options.clusterHdbscanMinClusterSize),
    );
    runArgs.push(
      "--cluster-hdbscan-min-samples",
      String(options.clusterHdbscanMinSamples),
    );
    runArgs.push("--cluster-noise-policy", String(options.clusterNoisePolicy));
  }
  if (options.withLabeling) {
    runArgs.push("--with-labeling");
  }
  if (options.withHierarchy) {
    runArgs.push("--with-hierarchy");
    runArgs.push("--hierarchy-levels", String(options.hierarchyLevels));
    runArgs.push(
      "--hierarchy-depth-policy",
      String(options.hierarchyDepthPolicy),
    );
  }
  if (options.withPrivacy) {
    runArgs.push("--with-privacy");
  }
  if (options.withEval) {
    runArgs.push("--with-eval");
  }
  if (options.limit !== null) {
    runArgs.push("--limit", String(options.limit));
  }
  if (options.evalCount !== null) {
    runArgs.push("--eval-count", String(options.evalCount));
  }
  if (options.streaming) {
    runArgs.push(
      "--streaming",
      "--stream-chunk-size",
      String(options.streamChunkSize),
    );
  }
  if (options.strict) {
    runArgs.push("--strict");
  }

  const runnerMode = (process.env.CLIO_RUNNER ?? "uv").trim().toLowerCase();
  if (runnerMode === "python") {
    const command =
      (process.env.CLIO_PYTHON_BIN ?? "python").trim() || "python";
    return {
      command,
      args: ["-m", "clio_pipeline.cli", ...runArgs],
    };
  }

  const uvCommand = (process.env.CLIO_UV_BIN ?? "uv").trim() || "uv";
  return {
    command: uvCommand,
    args: ["run", "clio", ...runArgs],
  };
}

function normalizedInputPath(rawInputPath: string): string {
  const trimmed = rawInputPath.trim();
  if (!trimmed) {
    return "";
  }
  if (path.isAbsolute(trimmed)) {
    return path.resolve(trimmed);
  }
  return path.resolve(projectRoot(), trimmed);
}

async function deriveJobRecord(
  payload: JsonRecord,
  runsRoot: string,
): Promise<RunJobRecord | null> {
  const runIdCandidate = asString(payload.runId) || asString(payload.run_id);
  const runRootCandidate =
    asString(payload.runRoot) || asString(payload.run_root);
  const fallbackRunId =
    runIdCandidate || path.basename(runRootCandidate || "") || "";
  const runRootFallback = path.join(runsRoot, fallbackRunId);
  const runRootRaw = runRootCandidate || runRootFallback;
  const runRoot = resolvePathFromProject(runRootRaw, runRootFallback);
  const runId = runIdCandidate || path.basename(runRoot);
  if (!runId) {
    return null;
  }

  const pid = asNumber(payload.pid, 0);
  const startedAtUtc =
    asString(payload.startedAtUtc) ||
    asString(payload.started_at_utc) ||
    new Date().toISOString();
  const inputPath =
    asString(payload.inputPath) || asString(payload.input_path) || "n/a";
  const rawLogPath =
    asString(payload.logPath) ||
    asString(payload.log_path) ||
    path.join(runRoot, "ui_run.log");
  const logPath = resolvePathFromProject(
    rawLogPath,
    path.join(runRoot, "ui_run.log"),
  );
  const command = asStringArray(payload.command);

  const lockState = await readRunLockState(runRoot);
  const hasMetrics = await fileExists(path.join(runRoot, "run_metrics.json"));
  const hasWarnings = await fileExists(path.join(runRoot, "run_warnings.json"));
  const running = isPidRunning(pid) || lockState.active;

  let status: RunJobRecord["status"] = running ? "running" : "finished";
  if (!running && hasMetrics && hasWarnings) {
    status = "finished_with_warnings";
  } else if (!running && hasMetrics) {
    status = "finished_ok";
  }

  return {
    runId,
    runRoot,
    pid,
    status,
    running,
    hasMetrics,
    hasWarnings,
    startedAtUtc,
    inputPath,
    logPath,
    command,
  };
}

async function readJobPayloadByRunId(
  runId: string,
  runsRoot: string,
): Promise<JsonRecord | null> {
  const jobPath = path.join(runJobsRoot(runsRoot), `${runId}.json`);
  const payload = await readJson(jobPath);
  if (payload) {
    return payload;
  }
  return null;
}

export async function resolveInputPathFromText(
  inputPath: string,
): Promise<string> {
  const normalized = normalizedInputPath(inputPath);
  if (!normalized) {
    throw new Error("inputPath is required when no file upload is provided.");
  }
  const stat = await fs.stat(normalized);
  if (!stat.isFile()) {
    throw new Error(`Input path is not a file: ${normalized}`);
  }
  return normalized;
}

export async function persistUploadedInput(
  file: File,
  runsRoot = getRunsRootPath(),
): Promise<{ inputPath: string; sizeBytes: number }> {
  const bytes = Buffer.from(await file.arrayBuffer());
  if (bytes.byteLength <= 0) {
    throw new Error("Uploaded file is empty.");
  }
  if (bytes.byteLength > MAX_UPLOAD_BYTES) {
    const limitMB = Math.round(MAX_UPLOAD_BYTES / (1024 * 1024));
    throw new Error(`Uploaded file exceeds ${limitMB} MB limit.`);
  }

  await ensureDirectory(uploadsRoot(runsRoot));
  const safeName = path
    .basename(file.name || "uploaded_input.jsonl")
    .replace(/\s+/g, "_");
  const outputPath = path.join(
    uploadsRoot(runsRoot),
    `${utcStamp()}_${safeName}`,
  );
  await fs.writeFile(outputPath, bytes);
  return {
    inputPath: outputPath,
    sizeBytes: bytes.byteLength,
  };
}

export async function startBackgroundRun(params: {
  inputPath: string;
  requestedRunId?: string | null;
  options: Partial<RunLaunchOptions>;
  runsRoot?: string;
}): Promise<RunJobRecord> {
  const runsRoot = params.runsRoot ?? getRunsRootPath();
  await ensureDirectory(runsRoot);
  await ensureDirectory(runJobsRoot(runsRoot));

  const runIdCandidate = sanitizeRunId(params.requestedRunId ?? "");
  const runId = runIdCandidate || generateRunId();
  const runRoot = path.join(runsRoot, runId);
  await ensureDirectory(runRoot);

  const options = normalizeLaunchOptions(params.options);
  const commandSpec = buildRunCommand({
    runId,
    inputPath: params.inputPath,
    options,
  });
  const commandLine = [commandSpec.command, ...commandSpec.args];
  const logPath = path.join(runRoot, "ui_run.log");
  await fs.writeFile(
    logPath,
    `[${new Date().toISOString()}] Starting run ${runId}\n$ ${commandLine.join(" ")}\n\n`,
    "utf-8",
  );

  const outFd = openSync(logPath, "a");
  const child = spawn(commandSpec.command, commandSpec.args, {
    cwd: projectRoot(),
    env: process.env,
    detached: true,
    stdio: ["ignore", outFd, outFd],
  });
  closeSync(outFd);
  child.unref();

  if (!child.pid || child.pid <= 0) {
    throw new Error("Failed to launch background run process.");
  }

  const payload: JsonRecord = {
    runId,
    runRoot,
    pid: child.pid,
    startedAtUtc: new Date().toISOString(),
    inputPath: params.inputPath,
    options,
    command: commandLine,
    logPath,
  };
  await writeJson(path.join(runJobsRoot(runsRoot), `${runId}.json`), payload);

  const job = await deriveJobRecord(payload, runsRoot);
  if (!job) {
    throw new Error("Failed to build run job payload.");
  }
  return job;
}

export async function collectRunJobs(
  runsRoot = getRunsRootPath(),
  limit = 120,
): Promise<RunJobRecord[]> {
  const jobsPath = runJobsRoot(runsRoot);
  try {
    const entries = await fs.readdir(jobsPath, { withFileTypes: true });
    const files = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
      .map((entry) => path.join(jobsPath, entry.name));
    const payloads = await Promise.all(
      files.map((filePath) => readJson(filePath)),
    );
    const jobs = await Promise.all(
      payloads
        .filter((payload): payload is JsonRecord => Boolean(payload))
        .map((payload) => deriveJobRecord(payload, runsRoot)),
    );

    const filtered = jobs.filter((item): item is RunJobRecord => Boolean(item));
    filtered.sort(
      (a, b) => Date.parse(b.startedAtUtc) - Date.parse(a.startedAtUtc),
    );
    if (limit <= 0) {
      return filtered;
    }
    return filtered.slice(0, limit);
  } catch {
    return [];
  }
}

export async function terminateRunJob(
  runId: string,
  runsRoot = getRunsRootPath(),
): Promise<{ ok: boolean; status: string; job: RunJobRecord | null }> {
  const terminatePid = (
    pid: number,
  ): { ok: boolean; status: string; terminatedPid: number | null } => {
    if (pid <= 0) {
      return { ok: false, status: "invalid_pid", terminatedPid: null };
    }
    if (!isPidRunning(pid)) {
      return { ok: false, status: "process_not_running", terminatedPid: pid };
    }

    try {
      process.kill(-pid, "SIGTERM");
    } catch {
      try {
        process.kill(pid, "SIGTERM");
      } catch (error) {
        return {
          ok: false,
          status: error instanceof Error ? error.message : "terminate_failed",
          terminatedPid: pid,
        };
      }
    }
    return { ok: true, status: "terminated", terminatedPid: pid };
  };

  const payload = await readJobPayloadByRunId(runId, runsRoot);
  if (!payload) {
    const runRoot = path.join(runsRoot, runId);
    const lockState = await readRunLockState(runRoot);
    if (lockState.lockOwnerPid && lockState.lockOwnerPid > 0) {
      const lockTermination = terminatePid(lockState.lockOwnerPid);
      return {
        ok: lockTermination.ok,
        status: lockTermination.ok
          ? "terminated_lock_pid"
          : lockTermination.status,
        job: null,
      };
    }

    return {
      ok: false,
      status: lockState.lockFilePresent
        ? "lock_owner_unknown"
        : "job_not_found",
      job: null,
    };
  }

  const job = await deriveJobRecord(payload, runsRoot);
  if (!job) {
    return { ok: false, status: "invalid_job_payload", job: null };
  }

  const termination = terminatePid(job.pid);
  return {
    ok: termination.ok,
    status: termination.status,
    job,
  };
}

export async function readRunJobLogTail(params: {
  runId: string;
  lines?: number;
  runsRoot?: string;
}): Promise<{
  runId: string;
  status: string;
  running: boolean;
  logPath: string;
  lineCount: number;
  logTail: string;
}> {
  const runsRoot = params.runsRoot ?? getRunsRootPath();
  const lines = Math.max(20, Math.min(500, Math.floor(params.lines ?? 120)));
  const payload = await readJobPayloadByRunId(params.runId, runsRoot);
  const job = payload ? await deriveJobRecord(payload, runsRoot) : null;

  const fallbackLogPath = path.join(runsRoot, params.runId, "ui_run.log");
  const logPath = job?.logPath || fallbackLogPath;
  try {
    const raw = await fs.readFile(logPath, "utf-8");
    const allLines = raw.split(/\r?\n/);
    const trimmed = allLines.filter((line, index) => {
      if (index === allLines.length - 1 && line === "") {
        return false;
      }
      return true;
    });
    const tailLines = trimmed.slice(-lines);
    return {
      runId: params.runId,
      status: job?.status ?? "unknown",
      running: job?.running ?? false,
      logPath,
      lineCount: tailLines.length,
      logTail: tailLines.join("\n"),
    };
  } catch {
    return {
      runId: params.runId,
      status: job?.status ?? "unknown",
      running: job?.running ?? false,
      logPath,
      lineCount: 0,
      logTail: "",
    };
  }
}
