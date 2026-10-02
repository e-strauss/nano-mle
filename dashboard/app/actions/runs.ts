"use server";

import { redirect } from "next/navigation";
import { verifySession } from "@/lib/dal";
import { launchLog, listLaunches, startLaunch } from "@/lib/launch";
import { pidAlive } from "@/lib/procs";
import { nodeDetail, readRunFile, runDetail } from "@/lib/runs";
import type { Section } from "@/lib/types";

export async function getNode(runId: string, nodeId: string): Promise<Section[]> {
  await verifySession();
  return nodeDetail(runId, nodeId);
}

export async function getFile(runId: string, rel: string): Promise<{ text: string; truncated: boolean }> {
  await verifySession();
  return readRunFile(runId, rel);
}

export async function getLaunchLog(id: string): Promise<string> {
  await verifySession();
  return launchLog(id);
}

export type LaunchState = { error?: string } | undefined;

export async function launch(_: LaunchState, form: FormData): Promise<LaunchState> {
  await verifySession();
  const launcher = String(form.get("_launcher") ?? "");
  const values: Record<string, string> = {};
  for (const [k, v] of form.entries()) if (!k.startsWith("_") && !k.startsWith("$")) values[k] = String(v);
  try {
    startLaunch(launcher, values);
  } catch (error) {
    return { error: (error as Error).message };
  }
  redirect("/");
}

// Ask the process that owns a run to stop: SIGINT lets the harness record the
// interruption itself. Only pids the dashboard can attribute to a run are signalled.
export async function stopRun(runId: string): Promise<{ error?: string }> {
  await verifySession();
  const pid = runDetail(runId)?.summary.pid;
  if (!pid || !pidAlive(pid)) return { error: "No live process owns this run" };
  process.kill(pid, "SIGINT");
  return {};
}

export async function stopLaunch(id: string): Promise<{ error?: string }> {
  await verifySession();
  const record = listLaunches().find((l) => l.id === id);
  const pid = record?.status?.pid;
  if (!record?.live || !pid) return { error: "Launch is not running" };
  process.kill(pid, "SIGINT");
  return {};
}
