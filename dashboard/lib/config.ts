import "server-only";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

// Configuration lives in launchers.json (committed defaults) and an optional
// launchers.local.json (machine-specific, gitignored) merged on top of it.
// Relative paths resolve against the dashboard directory.

export type Field =
  | { name: string; label?: string; type: "slug"; default?: string; help?: string }
  | { name: string; label?: string; type: "text"; default?: string; pattern?: string; help?: string }
  | { name: string; label?: string; type: "int"; default?: number; min?: number; max?: number; help?: string }
  | { name: string; label?: string; type: "select"; options: string[]; default?: string; help?: string }
  | { name: string; label?: string; type: "file"; glob: string; roots: string[]; help?: string };

export type Launcher = {
  id: string;
  label: string;
  description?: string;
  cwd: string;
  root: string;               // which runs root new runs go to
  workspace: string;          // template, e.g. "{root}/{name}"
  fields: Field[];
  steps: string[][];          // argv templates, run in order; stop at first failure
  env?: Record<string, string>;
};

// Optional per-adapter commands, e.g. drawing an attempt's graph on demand.
export type Harness = { cwd: string; draw?: string[] };

export type Config = {
  runsRoots: string[];
  harnesses: Record<string, Harness>;
  launchers: Launcher[];
};

export const DASHBOARD_DIR = process.cwd();

function read(file: string): Partial<Config> {
  const full = path.join(DASHBOARD_DIR, file);
  return existsSync(full) ? JSON.parse(readFileSync(full, "utf8")) : {};
}

export function loadConfig(): Config {
  const base = read("launchers.json"), local = read("launchers.local.json");
  // Local launchers replace same-id defaults and append new ones; local roots replace defaults.
  const ids = new Set((local.launchers ?? []).map((l) => l.id));
  const merged = {
    runsRoots: local.runsRoots ?? base.runsRoots ?? [],
    harnesses: { ...(base.harnesses ?? {}), ...(local.harnesses ?? {}) },
    launchers: [...(base.launchers ?? []).filter((l) => !ids.has(l.id)), ...(local.launchers ?? [])],
  };
  return {
    runsRoots: merged.runsRoots.map((r) => path.resolve(DASHBOARD_DIR, r)),
    harnesses: Object.fromEntries(Object.entries(merged.harnesses)
      .map(([k, h]) => [k, { ...h, cwd: path.resolve(DASHBOARD_DIR, h.cwd) }])),
    launchers: merged.launchers.map((l) => ({
      ...l,
      cwd: path.resolve(DASHBOARD_DIR, l.cwd),
      root: path.resolve(DASHBOARD_DIR, l.root),
      fields: l.fields.map((f) => (f.type === "file"
        ? { ...f, roots: f.roots.map((r) => path.resolve(DASHBOARD_DIR, r)) } : f)),
    })),
  };
}

export const LAUNCH_DIR = path.join(DASHBOARD_DIR, ".launches");
