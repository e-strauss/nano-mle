import "server-only";
import { existsSync, readFileSync, statSync } from "node:fs";

// Liveness without touching locks: a harness that holds flock/fcntl on a lock
// file shows up in /proc/locks with that file's inode. Reading never takes the
// lock, so the dashboard cannot race a runner that is starting.

export function lockHolder(lockFile: string): number | undefined {
  if (!existsSync(lockFile)) return undefined;
  let locks: string;
  try {
    locks = readFileSync("/proc/locks", "utf8");
  } catch {
    return undefined;
  }
  const inode = statSync(lockFile).ino;
  for (const line of locks.split("\n")) {
    // "1: FLOCK  ADVISORY  WRITE 12345 00:2f:987654 0 EOF"
    const parts = line.trim().split(/\s+/);
    const pidIndex = parts.findIndex((p) => p === "READ" || p === "WRITE") + 1;
    if (pidIndex <= 0 || pidIndex + 1 >= parts.length) continue;
    const ino = Number(parts[pidIndex + 1].split(":").pop());
    if (ino === inode) {
      const pid = Number(parts[pidIndex]);
      if (pidAlive(pid)) return pid;
    }
  }
  return undefined;
}

export function pidAlive(pid: number | undefined): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function cmdline(pid: number): string[] {
  try {
    return readFileSync(`/proc/${pid}/cmdline`, "utf8").split("\0").filter(Boolean);
  } catch {
    return [];
  }
}

export function argValue(argv: string[], flag: string): string | undefined {
  const i = argv.indexOf(flag);
  return i >= 0 ? argv[i + 1] : undefined;
}
