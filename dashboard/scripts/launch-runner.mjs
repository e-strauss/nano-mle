#!/usr/bin/env node
// Detached executor for one dashboard launch: runs the recorded argv steps in
// order (no shell), appends their output to output.log and keeps status.json
// current. Stops at the first failing step. SIGINT/SIGTERM are forwarded to
// the running step so the harness can record an interruption itself.
import { spawn } from "node:child_process";
import { appendFileSync, openSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const dir = process.argv[2];
const record = JSON.parse(readFileSync(path.join(dir, "launch.json"), "utf8"));
const log = path.join(dir, "output.log");
const status = (s) => writeFileSync(path.join(dir, "status.json"), JSON.stringify({ pid: process.pid, ...s }, null, 2));

let current;
let stopped = false;
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    stopped = true;
    current?.kill("SIGINT");
  });
}

for (let i = 0; i < record.steps.length; i++) {
  const argv = record.steps[i];
  status({ state: "running", step: i });
  appendFileSync(log, `\n$ ${argv.join(" ")}\n`);
  const fd = openSync(log, "a");
  const code = await new Promise((resolve) => {
    current = spawn(argv[0], argv.slice(1), {
      cwd: record.cwd, stdio: ["ignore", fd, fd],
      env: { ...process.env, ...record.env, PYTHONUNBUFFERED: "1" },
    });
    current.on("error", (e) => { appendFileSync(log, `spawn failed: ${e.message}\n`); resolve(127); });
    current.on("exit", (c, sig) => resolve(c ?? (sig ? 128 : 1)));
  });
  appendFileSync(log, `[exit ${code}]\n`);
  if (code !== 0 || stopped) {
    status({ state: stopped ? "stopped" : "failed", step: i, exitCode: code, finishedAt: Date.now() });
    process.exit(0);
  }
}
status({ state: "succeeded", step: record.steps.length - 1, exitCode: 0, finishedAt: Date.now() });
