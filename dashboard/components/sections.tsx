"use client";

import { useState } from "react";
import { getFile } from "@/app/actions/runs";
import { bytes } from "@/lib/format";
import type { FileRef, Section } from "@/lib/types";
import GraphViewer from "./graph-viewer";

function Files({ runId, files }: { runId: string; files: FileRef[] }) {
  const [open, setOpen] = useState<{ path: string; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (!files.length) return <p className="muted small">none</p>;
  return (
    <>
      <ul className="files">
        {files.map((f) => (
          <li key={f.path}>
            <button onClick={async () => {
              if (open?.path === f.path) return setOpen(null);
              try {
                const r = await getFile(runId, f.path);
                setOpen({ path: f.path, text: r.text + (r.truncated ? "\n… truncated" : "") });
                setError(null);
              } catch (e) {
                setError((e as Error).message);
              }
            }}>{f.path}</button> <span className="muted">{bytes(f.size)}</span>
            {open?.path === f.path && <pre>{open.text}</pre>}
          </li>
        ))}
      </ul>
      {error && <p className="error small">{error}</p>}
    </>
  );
}

export function SectionView({ s, runId, depth = 0 }: { s: Section; runId: string; depth?: number }) {
  const body = (() => {
    switch (s.kind) {
      case "text": return <pre className="wrap">{s.content || "—"}</pre>;
      case "code": return <pre>{s.content}</pre>;
      case "json": return <pre className="wrap">{JSON.stringify(s.content, null, 2)}</pre>;
      case "kv": return (
        <table className="kv"><tbody>
          {/* labels can repeat (e.g. verify_contract runs twice), so key by position */}
          {s.content.map(([k, v], i) => <tr key={i}><td>{k}</td><td className="mono">{v}</td></tr>)}
        </tbody></table>
      );
      case "files": return <Files runId={runId} files={s.content} />;
      case "graph": return <GraphViewer runId={runId} attempt={s.content} />;
      case "group": return s.content.map((c, i) => <SectionView key={i} s={c} runId={runId} depth={depth + 1} />);
    }
  })();
  const startOpen = s.kind === "kv" || (depth === 0 && s.kind !== "group" && s.kind !== "files" && s.kind !== "code");
  return (
    <details open={startOpen}>
      <summary>{s.title}</summary>
      {body}
    </details>
  );
}
