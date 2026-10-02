import "server-only";
import { existsSync, readdirSync, readFileSync, realpathSync, statSync } from "node:fs";
import path from "node:path";
import { adapterFor } from "./adapters";
import { loadConfig } from "./config";
import { listLaunches } from "./launch";
import type { RunDetail, RunSummary, Section } from "./types";

// Discovers runs under the configured roots. A run id is
// "<root basename>/<path below root>", so it maps 1:1 to a URL.

const MAX_DEPTH = 4;
const SKIP = new Set(["artifacts", "node_modules", ".git", ".venv", "mpl-cache"]);

function roots(): Map<string, string> {
  return new Map(loadConfig().runsRoots.filter(existsSync).map((r) => [path.basename(r), r]));
}

function walk(dir: string, depth: number, found: string[]) {
  if (adapterFor(dir)) {
    found.push(dir);
    return;
  }
  if (depth >= MAX_DEPTH) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory() && !SKIP.has(e.name) && !e.name.startsWith(".")) walk(path.join(dir, e.name), depth + 1, found);
  }
}

export function resolveRun(id: string): { dir: string; id: string } | undefined {
  const [rootName, ...rest] = id.split("/");
  const root = roots().get(rootName);
  if (!root || rest.some((p) => p === ".." || p === "." || !p)) return undefined;
  const dir = path.join(root, ...rest);
  if (!existsSync(dir)) return undefined;
  const real = realpathSync(dir);
  if (!real.startsWith(realpathSync(root) + path.sep)) return undefined;
  return adapterFor(real) ? { dir: real, id } : undefined;
}

function attachLaunch(summary: RunSummary, dir: string): RunSummary {
  const launch = listLaunches().find((l) => l.workspace && path.resolve(l.workspace) === dir);
  if (!launch) return summary;
  return { ...summary, model: summary.model ?? launch.values?.model, launchId: launch.id };
}

export function listRuns(): RunSummary[] {
  const out: RunSummary[] = [];
  for (const [name, root] of roots()) {
    const found: string[] = [];
    walk(root, 0, found);
    for (const dir of found) {
      const id = [name, ...path.relative(root, dir).split(path.sep)].join("/");
      try {
        out.push(attachLaunch(adapterFor(dir)!.summary(dir, id), dir));
      } catch (error) {
        out.push({ id, adapter: adapterFor(dir)!.name, title: id, state: "unknown", live: false,
          budgets: [], notes: [`Could not read: ${(error as Error).message}`],
          updatedAt: statSync(dir).mtimeMs });
      }
    }
  }
  return out.sort((a, b) => Number(b.live) - Number(a.live) || (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
}

export function runDetail(id: string): RunDetail | undefined {
  const run = resolveRun(id);
  if (!run) return undefined;
  const detail = adapterFor(run.dir)!.detail(run.dir, id);
  return { ...detail, summary: attachLaunch(detail.summary, run.dir) };
}

export function nodeDetail(id: string, nodeId: string): Section[] {
  const run = resolveRun(id);
  if (!run) throw new Error("Unknown run");
  return adapterFor(run.dir)!.node(run.dir, nodeId);
}

const MAX_FILE = 2_000_000;

export function readRunFile(id: string, rel: string): { text: string; truncated: boolean } {
  const run = resolveRun(id);
  if (!run) throw new Error("Unknown run");
  const full = path.resolve(run.dir, rel);
  if (!existsSync(full)) throw new Error("No such file");
  const real = realpathSync(full);
  if (!real.startsWith(run.dir + path.sep) || !statSync(real).isFile()) throw new Error("Outside the run");
  const buf = readFileSync(real);
  const truncated = buf.length > MAX_FILE;
  let text = buf.subarray(0, MAX_FILE).toString("utf8");
  if (real.endsWith(".json") && !truncated) {
    try {
      text = JSON.stringify(JSON.parse(text), null, 2);
    } catch { /* show raw */ }
  }
  return { text, truncated };
}
