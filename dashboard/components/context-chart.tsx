"use client";

import { useState } from "react";
import { clock } from "@/lib/format";
import type { ContextSizes } from "@/lib/types";

// Size of the controller's input per control call, stacked by part. Colors follow
// the part (fixed order), the remainder is neutral gray.

const W = 640, H = 200, M = { l: 56, r: 12, t: 10, b: 28 };
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
  "var(--series-5)", "var(--series-6)"];
const color = (i: number) => COLORS[i] ?? "var(--neutral-node)";
const kb = (chars: number) => `${(chars / 1000).toFixed(1)} KB`;

export default function ContextChart({ data }: { data: ContextSizes }) {
  const [hover, setHover] = useState<number | null>(null);
  const { parts, points } = data;
  if (!points.length) return <p className="muted">No controller calls yet.</p>;
  const total = (s: number[]) => s.reduce((a, b) => a + b, 0);
  const max = Math.max(...points.map((p) => total(p.sizes)));
  const step = 10 ** Math.floor(Math.log10(max || 1));
  const top = Math.ceil((max || 1) / step) * step;
  const n = points.length;
  const slot = (W - M.l - M.r) / n;
  const bar = Math.max(2, Math.min(24, slot - 4));
  const x = (i: number) => M.l + slot * i + (slot - bar) / 2;
  const y = (v: number) => M.t + (1 - v / top) * (H - M.t - M.b);
  const ticks = [0, top / 2, top];
  const h = hover !== null ? points[hover] : null;

  return (
    <div style={{ position: "relative" }}>
      <div className="legend">
        {parts.map((p, i) => (
          <span key={p}><svg width="12" height="12"><rect width="12" height="12" rx="3" fill={color(i)} /></svg> {p}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, background: "var(--chart-surface)", borderRadius: 6 }}
        role="img" aria-label="Controller context size per control call">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end" fontSize="10" fill="var(--muted)">{kb(t)}</text>
          </g>
        ))}
        <text x={(W + M.l) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--muted)">control call</text>
        {points.map((p, i) => {
          let acc = 0;
          const last = p.sizes.reduce((k, v, j) => (v > 0 ? j : k), -1);
          return (
            <g key={p.order} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={M.l + slot * i} y={M.t} width={slot} height={H - M.t - M.b} fill="transparent" />
              {p.sizes.map((v, j) => {
                if (v <= 0) return null;
                const y0 = y(acc), y1 = y(acc + v);
                acc += v;
                // 2px surface gap between segments; the top segment gets the rounded data end.
                const height = Math.max(0, y0 - y1 - (j === last ? 0 : 2));
                return j === last
                  ? <path key={j} d={`M${x(i)} ${y0}V${y1 + Math.min(4, height)}q0 -4 4 -4h${bar - 8}q4 0 4 4V${y0}Z`}
                      fill={color(j)} opacity={hover === null || hover === i ? 1 : 0.55} />
                  : <rect key={j} x={x(i)} y={y1 + 2} width={bar} height={height} fill={color(j)}
                      opacity={hover === null || hover === i ? 1 : 0.55} />;
              })}
            </g>
          );
        })}
      </svg>
      {h && (
        <div className="tooltip" style={{ left: `${((x(hover!) + bar / 2) / W) * 100}%`, top: 24 }}>
          <b>call {h.order}</b> {h.time ? clock(h.time) : ""} · {kb(total(h.sizes))}<br />
          {parts.map((p, j) => <span key={p}>{p}: {kb(h.sizes[j])}<br /></span>)}
        </div>
      )}
      <details>
        <summary className="muted small">table</summary>
        <table className="list small">
          <thead><tr><th>call</th><th>time</th><th>total</th>{parts.map((p) => <th key={p}>{p}</th>)}</tr></thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.order}><td>{p.order}</td><td className="mono">{p.time ? clock(p.time) : ""}</td>
                <td>{kb(total(p.sizes))}</td>{p.sizes.map((v, j) => <td key={j}>{kb(v)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
