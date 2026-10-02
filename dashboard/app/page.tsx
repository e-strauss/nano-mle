import Link from "next/link";
import AutoRefresh from "@/components/auto-refresh";
import { BudgetMeters } from "@/components/budgets";
import Header from "@/components/header";
import Launches, { type LaunchRow } from "@/components/launches";
import { verifySession } from "@/lib/dal";
import { ago, runHref, score } from "@/lib/format";
import { listLaunches } from "@/lib/launch";
import { missingLibraries } from "@/lib/libraries";
import { listRuns } from "@/lib/runs";

export const dynamic = "force-dynamic";

export default async function RunsPage() {
  await verifySession();
  const runs = listRuns();
  const launches = listLaunches();
  const byWorkspace = new Map(runs.map((r) => [r.launchId, r]));
  const launchRows: LaunchRow[] = launches.slice(0, 20).map((l) => {
    const run = byWorkspace.get(l.id);
    return { id: l.id, launcher: l.launcher, workspace: l.workspace?.split("/").slice(-2).join("/"),
      createdAt: l.createdAt, live: l.live, state: l.status?.state, step: l.status?.step, steps: l.steps.length,
      exitCode: l.status?.exitCode, runHref: run ? runHref(run.id) : undefined };
  });
  const missing = missingLibraries();
  const anyLive = runs.some((r) => r.live) || launches.some((l) => l.live);
  return (
    <>
      <Header />
      <main className="page">
        <div className="row">
          <h1>Runs</h1>
          <AutoRefresh active={anyLive} />
          <span className="spacer" />
          <Link href="/new" className="button">New run</Link>
        </div>
        <table className="list" style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Run</th><th>State</th><th>Policy</th><th>Model</th><th>Metric</th><th className="num">Best</th>
              <th>Budgets</th><th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>
                  <Link href={runHref(r.id)} className="mono">{r.id}</Link>
                  <div className="small muted">{r.title}</div>
                  {r.notes.map((n) => <div key={n} className="small error">{n}</div>)}
                </td>
                <td><span className={`badge ${r.live ? "live" : ""}`}>{r.live ? "running" : r.state}</span>
                  <div className="small muted">{r.adapter}</div></td>
                <td>{r.policy ?? "—"}</td>
                <td className="mono small">{r.model ?? "—"}</td>
                <td className="small">{r.metric ?? "—"}</td>
                <td className="num mono">{score(r.best?.score)}</td>
                <td style={{ minWidth: 320 }}><BudgetMeters budgets={r.budgets} compact /></td>
                <td className="small">{ago(r.updatedAt)}</td>
              </tr>
            ))}
            {!runs.length && <tr><td colSpan={8} className="muted">No runs found under the configured roots.</td></tr>}
          </tbody>
        </table>
        <section>
          <h2>Dashboard launches</h2>
          <Launches rows={launchRows} />
        </section>
        <section>
          <h2>Requested libraries that are not installed</h2>
          <p className="small muted">Imports in agent plans that failed because the library is missing.
            Harnesses do not install packages; add them to the environment if they are worth having.</p>
          {missing.length ? (
            <table className="list">
              <thead><tr><th>Library</th><th>Harness</th><th className="num">Requests</th>
                <th className="num">Workspaces</th><th>First seen</th><th>Last seen</th></tr></thead>
              <tbody>
                {missing.map((m) => (
                  <tr key={`${m.harness}:${m.module}`}>
                    <td className="mono">{m.module}</td><td>{m.harness}</td><td className="num">{m.count}</td>
                    <td className="num" title={m.workspaces.join("\n")}>{m.workspaces.length}</td>
                    <td className="small">{ago(m.firstSeen)}</td><td className="small">{ago(m.lastSeen)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="muted">None recorded.</p>}
        </section>
      </main>
    </>
  );
}
