"use client";

import { useEffect, useRef, useState } from "react";
import { drawGraph } from "@/app/actions/runs";

// On-demand drawing of an attempt's DataOps graph. The SVG is shown as an <img>
// (scripts inside it never run). Scroll pans; pinch or Ctrl/⌘ + scroll zooms.

export default function GraphViewer({ runId, attempt }: { runId: string; attempt: string }) {
  const [state, setState] = useState<"idle" | "drawing" | "ready" | "error">("idle");
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scale, setScale] = useState(1);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const pane = useRef<HTMLDivElement>(null);

  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  const fit = (n = natural) => {
    if (n && pane.current) setScale(Math.min(1, (pane.current.clientWidth - 4) / n.w));
  };

  useEffect(() => {
    const el = pane.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      setScale((s) => Math.min(4, Math.max(0.02, s * (e.deltaY > 0 ? 0.9 : 1.1))));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [state]);

  const draw = async () => {
    setState("drawing");
    setError(null);
    const r = await drawGraph(runId, attempt);
    if (r.error || !r.svg) {
      setError(r.error ?? "No drawing");
      setState("error");
      return;
    }
    setUrl(URL.createObjectURL(new Blob([r.svg], { type: "image/svg+xml" })));
    setState("ready");
  };

  if (state === "idle" || state === "error") {
    return (
      <div>
        <button onClick={draw}>Draw Skrub graph</button>
        <span className="small muted"> rebuilds the plan lazily via the harness; no data is read</span>
        {error && <pre className="wrap error">{error}</pre>}
      </div>
    );
  }
  if (state === "drawing") return <p className="muted small">Drawing graph… (a few seconds)</p>;
  return (
    <div>
      <div className="row small" style={{ marginBottom: 4 }}>
        <button onClick={() => setScale((s) => Math.min(4, s * 1.25))} aria-label="Zoom in">+</button>
        <button onClick={() => setScale((s) => Math.max(0.02, s / 1.25))} aria-label="Zoom out">−</button>
        <button onClick={() => fit()}>Fit width</button>
        <button onClick={() => setScale(1)}>100%</button>
        <a href={url!} target="_blank" rel="noopener noreferrer">Open full size</a>
        <span className="muted">{Math.round(scale * 100)}%{natural ? ` · ${natural.w}×${natural.h}px` : ""}</span>
      </div>
      <div ref={pane} className="graph-pane">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url!} alt="Skrub DataOps graph" draggable={false}
          style={natural ? { width: natural.w * scale, height: natural.h * scale } : undefined}
          onLoad={(e) => {
            const n = { w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight };
            setNatural(n);
            fit(n);
          }} />
      </div>
    </div>
  );
}
