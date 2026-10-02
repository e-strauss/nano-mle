import "server-only";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { adapterFor } from "./adapters";
import { DASHBOARD_DIR, loadConfig } from "./config";
import { resolveRun } from "./runs";

// Draws an attempt's graph with the harness's own `draw` command (configured per
// adapter in launchers.json). Results are cached in .cache/graphs/, keyed by the
// attempt path and its plan.py, never written into the workspace.

const CACHE = path.join(DASHBOARD_DIR, ".cache", "graphs");
const TIMEOUT_MS = 120_000;
const inflight = new Map<string, Promise<string>>();

export async function drawAttempt(runId: string, attemptRel: string): Promise<string> {
  const run = resolveRun(runId);
  if (!run) throw new Error("Unknown run");
  const attempt = realpathSync(path.resolve(run.dir, attemptRel));
  if (!attempt.startsWith(run.dir + path.sep) || !statSync(attempt).isDirectory()) throw new Error("Outside the run");
  const plan = path.join(attempt, "plan.py");
  if (!existsSync(plan)) throw new Error("Attempt has no plan.py");

  const harness = loadConfig().harnesses[adapterFor(run.dir)!.name];
  if (!harness?.draw) throw new Error("No draw command configured for this harness");
  const key = createHash("sha256").update(attempt).update(readFileSync(plan)).digest("hex").slice(0, 32);
  const cached = path.join(CACHE, `${key}.svg`);
  if (existsSync(cached)) return readFileSync(cached, "utf8");
  if (inflight.has(key)) return inflight.get(key)!;

  const argv = harness.draw.map((a) => a.replaceAll("{attempt}", attempt));
  const job = new Promise<string>((resolve, reject) => {
    execFile(argv[0], argv.slice(1), { cwd: harness.cwd, timeout: TIMEOUT_MS, maxBuffer: 64 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          const lines = String(stderr).trim().split("\n");
          return reject(new Error(lines.slice(-3).join("\n") || error.message));
        }
        if (!stdout.includes("<svg")) return reject(new Error("Draw command produced no SVG"));
        mkdirSync(CACHE, { recursive: true });
        writeFileSync(cached, stdout);
        resolve(stdout);
      });
  }).finally(() => inflight.delete(key));
  inflight.set(key, job);
  return job;
}
