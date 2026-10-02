"use client";

import { useEffect, useState, useTransition } from "react";
import { getNode, stopRun } from "@/app/actions/runs";
import { clock } from "@/lib/format";
import type { RunDetail, Section } from "@/lib/types";
import ScoreChart from "./score-chart";
import { SectionView } from "./sections";
import TreeView from "./tree-view";

export default function RunView({ detail }: { detail: RunDetail }) {
  const { summary } = detail;
  const [selected, setSelected] = useState<string | null>(summary.best?.node ?? "root");
  const [sections, setSections] = useState<Section[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();

  // Refetch the selected node whenever the run changes (auto-refresh).
  useEffect(() => {
    if (!selected) return;
    let alive = true;
    getNode(summary.id, selected)
      .then((s) => { if (alive) { setSections(s); setError(null); } })
      .catch((e) => alive && setError((e as Error).message));
    return () => { alive = false; };
  }, [selected, summary.id, summary.updatedAt]);

  const node = detail.nodes.find((n) => n.id === selected);
  return (
    <div className="run-layout">
      <div>
        <TreeView nodes={detail.nodes} edges={detail.edges} selected={selected} onSelect={setSelected}
          bestId={summary.best?.node} />
        <section>
          <h2>Score progress</h2>
          <ScoreChart points={detail.scores} onSelect={setSelected} />
        </section>
        <section>
          <h2>Run overview</h2>
          <div className="panel">
            {detail.overview.map((s, i) => <SectionView key={i} s={s} runId={summary.id} />)}
          </div>
        </section>
        <section>
          <details>
            <summary>Event timeline ({detail.events.length})</summary>
            <table className="list small">
              <tbody>
                {[...detail.events].reverse().slice(0, 400).map((e, i) => (
                  <tr key={i}><td className="mono">{clock(e.time)}</td><td>{e.kind}</td><td className="mono">{e.text}</td></tr>
                ))}
              </tbody>
            </table>
          </details>
        </section>
        {summary.live && (
          <section>
            <button className="danger" disabled={pending} onClick={() => {
              if (confirm(`Send SIGINT to pid ${summary.pid}? The harness records the run as interrupted.`)) {
                start(async () => { const r = await stopRun(summary.id); if (r.error) alert(r.error); });
              }
            }}>Stop run (pid {summary.pid})</button>
          </section>
        )}
      </div>
      <div className="panel sticky">
        {node ? (
          <>
            <h2>{node.label} <span className="muted small">{node.kind} · {node.status}</span></h2>
            <div className="mono small muted">{node.id}</div>
            {error && <p className="error">{error}</p>}
            {sections?.map((s, i) => <SectionView key={`${selected}-${i}`} s={s} runId={summary.id} />)}
          </>
        ) : <p className="muted">Select a node.</p>}
      </div>
    </div>
  );
}
