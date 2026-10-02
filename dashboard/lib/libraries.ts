import "server-only";
import { existsSync, readFileSync } from "node:fs";
import { loadConfig } from "./config";

export type MissingLibrary = {
  harness: string; module: string; count: number; firstSeen: number; lastSeen: number; workspaces: string[];
};

// Libraries that plans imported but were not installed, across all harnesses.
export function missingLibraries(): MissingLibrary[] {
  const out: MissingLibrary[] = [];
  for (const [harness, h] of Object.entries(loadConfig().harnesses)) {
    if (!h.missingLibraries || !existsSync(h.missingLibraries)) continue;
    try {
      const data = JSON.parse(readFileSync(h.missingLibraries, "utf8")) as Record<string, Record<string, any>>;
      for (const [module, e] of Object.entries(data)) {
        out.push({ harness, module, count: e.count ?? 0, firstSeen: (e.first_seen ?? 0) * 1000,
          lastSeen: (e.last_seen ?? 0) * 1000, workspaces: e.workspaces ?? [] });
      }
    } catch { /* unreadable record: show nothing rather than fail the page */ }
  }
  return out.sort((a, b) => b.count - a.count || b.lastSeen - a.lastSeen);
}
