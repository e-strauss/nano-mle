import "server-only";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { DASHBOARD_DIR, LAUNCH_DIR, loadConfig, type Field, type Launcher } from "./config";
import { pidAlive } from "./procs";

// A launch is a detached scripts/launch-runner.mjs process executing the
// launcher's argv templates in order. Its record lives in .launches/<id>/,
// never inside the harness's workspace, and survives dashboard restarts.

export type LaunchRecord = {
  id: string;
  launcher: string;
  values: Record<string, string>;
  workspace?: string;
  steps: string[][];
  createdAt: number;
  status?: { state: "running" | "succeeded" | "failed" | "stopped"; step?: number; exitCode?: number | null;
    pid?: number; finishedAt?: number };
  live: boolean;
};

const SKIP = new Set(["artifacts", "node_modules", ".git", ".venv", "mpl-cache"]);

export function fileOptions(field: Extract<Field, { type: "file" }>): string[] {
  const out: string[] = [];
  const pattern = new RegExp("^" + field.glob.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$");
  const walk = (dir: string, depth: number) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, e.name);
      if (e.isDirectory() && depth < 4 && !SKIP.has(e.name) && !e.name.startsWith(".")) walk(full, depth + 1);
      else if (e.isFile() && pattern.test(e.name)) out.push(full);
    }
  };
  for (const root of field.roots) if (existsSync(root)) walk(root, 0);
  return out.sort();
}

function validate(launcher: Launcher, raw: Record<string, string>): Record<string, string> {
  const values: Record<string, string> = {};
  for (const f of launcher.fields) {
    const v = (raw[f.name] ?? "").trim();
    const label = f.label ?? f.name;
    switch (f.type) {
      case "slug":
        if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/.test(v)) throw new Error(`${label}: letters, digits, _ . - only`);
        break;
      case "text":
        if (!v || (f.pattern && !new RegExp(f.pattern).test(v))) throw new Error(`${label}: invalid value`);
        break;
      case "int": {
        const n = Number(v);
        if (!Number.isInteger(n) || (f.min !== undefined && n < f.min) || (f.max !== undefined && n > f.max)) {
          throw new Error(`${label}: integer in [${f.min ?? "-∞"}, ${f.max ?? "∞"}]`);
        }
        break;
      }
      case "select":
        if (!f.options.includes(v)) throw new Error(`${label}: choose one of the options`);
        break;
      case "file":
        if (!fileOptions(f).includes(v)) throw new Error(`${label}: choose a listed file`);
        break;
    }
    values[f.name] = v;
  }
  return values;
}

function render(template: string, values: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_, k) => {
    if (!(k in values)) throw new Error(`Template references unknown field {${k}}`);
    return values[k];
  });
}

export function startLaunch(launcherId: string, raw: Record<string, string>): LaunchRecord {
  const launcher = loadConfig().launchers.find((l) => l.id === launcherId);
  if (!launcher) throw new Error("Unknown launcher");
  const values: Record<string, string> = { ...validate(launcher, raw), root: launcher.root };
  const workspace = path.resolve(launcher.root, path.relative(launcher.root, render(launcher.workspace, values)));
  if (!workspace.startsWith(launcher.root + path.sep)) throw new Error("Workspace must stay under the runs root");
  if (existsSync(workspace)) throw new Error(`Workspace already exists: ${workspace}`);
  values.workspace = workspace;
  const steps = launcher.steps.map((argv) => argv.map((a) => render(a, values)));

  const id = `${new Date().toISOString().replace(/[-:]/g, "").slice(0, 15)}-${randomBytes(3).toString("hex")}`;
  const dir = path.join(LAUNCH_DIR, id);
  mkdirSync(dir, { recursive: true });
  const record = { id, launcher: launcher.id, values, workspace, steps, cwd: launcher.cwd, env: launcher.env ?? {},
    createdAt: Date.now() };
  writeFileSync(path.join(dir, "launch.json"), JSON.stringify(record, null, 2));
  const child = spawn(process.execPath, [path.join(DASHBOARD_DIR, "scripts", "launch-runner.mjs"), dir], {
    detached: true, stdio: "ignore", cwd: launcher.cwd, env: { ...harnessEnv(), ...(launcher.env ?? {}) },
  });
  child.unref();
  return { ...record, live: true, status: { state: "running", pid: child.pid } };
}

// The dashboard's own login secrets never reach harness processes.
function harnessEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const key of ["AUTH_USER", "AUTH_PASSWORD_HASH", "SESSION_SECRET"]) delete env[key];
  for (const key of Object.keys(env)) if (key.startsWith("NEXT_") || key.startsWith("__NEXT")) delete env[key];
  return env;
}

function readJson(file: string): any {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return undefined;
  }
}

export function listLaunches(): LaunchRecord[] {
  if (!existsSync(LAUNCH_DIR)) return [];
  return readdirSync(LAUNCH_DIR).flatMap((id) => {
    const record = readJson(path.join(LAUNCH_DIR, id, "launch.json"));
    if (!record) return [];
    const status = readJson(path.join(LAUNCH_DIR, id, "status.json"));
    return [{ ...record, status, live: status?.state === "running" && pidAlive(status.pid) }];
  }).sort((a, b) => b.createdAt - a.createdAt);
}

export function launchLog(id: string, maxBytes = 200_000): string {
  if (!/^[\w-]+$/.test(id)) throw new Error("Bad launch id");
  const file = path.join(LAUNCH_DIR, id, "output.log");
  if (!existsSync(file)) return "";
  const text = readFileSync(file, "utf8");
  return text.length > maxBytes ? "… " + text.slice(-maxBytes) : text;
}
