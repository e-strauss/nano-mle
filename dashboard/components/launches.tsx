"use client";

import { Fragment, useEffect, useState, useTransition } from "react";
import { getLaunchLog, stopLaunch } from "@/app/actions/runs";
import { ago } from "@/lib/format";

export type LaunchRow = {
  id: string; launcher: string; workspace?: string; createdAt: number; live: boolean;
  state?: string; step?: number; steps: number; exitCode?: number | null; runHref?: string;
};

function LaunchLog({ id, live }: { id: string; live: boolean }) {
  const [text, setText] = useState<string>("");
  useEffect(() => {
    let alive = true;
    const load = () => getLaunchLog(id).then((t) => alive && setText(t));
    load();
    const timer = live ? setInterval(load, 4000) : undefined;
    return () => { alive = false; if (timer) clearInterval(timer); };
  }, [id, live]);
  return <pre className="wrap">{text || "(empty)"}</pre>;
}

export default function Launches({ rows }: { rows: LaunchRow[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const [pending, start] = useTransition();
  if (!rows.length) return <p className="muted">No runs launched from the dashboard yet.</p>;
  return (
    <table className="list">
      <thead>
        <tr><th>Launch</th><th>Launcher</th><th>Workspace</th><th>State</th><th>Step</th><th>Started</th><th /></tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <Fragment key={r.id}>
            <tr>
              <td className="mono">{r.id}</td>
              <td>{r.launcher}</td>
              <td className="mono small">{r.runHref ? <a href={r.runHref}>{r.workspace}</a> : r.workspace}</td>
              <td>
                <span className={`badge ${r.live ? "live" : r.state === "failed" ? "failed" : ""}`}>
                  {r.live ? "running" : r.state ?? "unknown"}{r.exitCode ? ` (exit ${r.exitCode})` : ""}
                </span>
              </td>
              <td className="num">{(r.step ?? 0) + 1}/{r.steps}</td>
              <td>{ago(r.createdAt)}</td>
              <td className="row">
                <button onClick={() => setOpen(open === r.id ? null : r.id)}>{open === r.id ? "Hide log" : "Log"}</button>
                {r.live && (
                  <button className="danger" disabled={pending} onClick={() => {
                    if (confirm("Send SIGINT to this launch?")) start(async () => { await stopLaunch(r.id); });
                  }}>Stop</button>
                )}
              </td>
            </tr>
            {open === r.id && (
              <tr><td colSpan={7}><LaunchLog id={r.id} live={r.live} /></td></tr>
            )}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}
