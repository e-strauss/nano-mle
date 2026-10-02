"use client";

import { useState } from "react";
import { score as fmtScore } from "@/lib/format";
import type { ScorePoint } from "@/lib/types";

// Candidate scores in evaluation order, with the best-so-far step line.
// Failed candidates sit as ✕ markers on the bottom band.

const W = 640, H = 200, M = { l: 56, r: 12, t: 10, b: 28 };

export default function ScoreChart({ points, onSelect }: { points: ScorePoint[]; onSelect: (id: string) => void }) {
  const [hover, setHover] = useState<number | null>(null);
  const valid = points.filter((p) => p.score !== null) as (ScorePoint & { score: number })[];
  if (!points.length) return <p className="muted">No scored candidates yet.</p>;
  const lo = Math.min(...valid.map((p) => p.score)), hi = Math.max(...valid.map((p) => p.score));
  const span = hi - lo || Math.abs(hi) * 0.01 || 1;
  const y0 = lo - span * 0.1, y1 = hi + span * 0.1;
  const n = points.length;
  const x = (i: number) => M.l + (n === 1 ? (W - M.l - M.r) / 2 : ((i - 1) / (n - 1)) * (W - M.l - M.r));
  const y = (s: number) => M.t + (1 - (s - y0) / (y1 - y0)) * (H - M.t - M.b - 14);
  const failY = H - M.b - 2;

  let best = -Infinity;
  const steps: string[] = [];
  for (const p of points) {
    if (p.score !== null && p.score > best) {
      if (best !== -Infinity) steps.push(`H${x(p.order)}`);
      best = p.score;
      steps.push(steps.length ? `V${y(best)}` : `M${x(p.order)} ${y(best)}`);
    }
  }
  if (steps.length) steps.push(`H${x(n)}`);
  const ticks = [y0 + (y1 - y0) * 0.1, (y0 + y1) / 2, y1 - (y1 - y0) * 0.1];
  const h = hover !== null ? points[hover] : null;

  return (
    <div style={{ position: "relative" }}>
      <div className="legend">
        <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--series-1)" /></svg> candidate score</span>
        <span><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="var(--series-2)" strokeWidth="2" /></svg> best so far</span>
        <span>✕ failed</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, background: "var(--chart-surface)", borderRadius: 6 }}
        role="img" aria-label="Score by evaluation order">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end" fontSize="10" fill="var(--muted)">{t.toFixed(4)}</text>
          </g>
        ))}
        <text x={(W + M.l) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--muted)">evaluation order</text>
        <path d={steps.join("")} fill="none" stroke="var(--series-2)" strokeWidth={2} />
        {points.map((p, i) => (
          <g key={p.node} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}
            onClick={() => onSelect(p.node)} style={{ cursor: "pointer" }}>
            <rect x={x(p.order) - 8} y={M.t} width={16} height={H - M.t - M.b} fill="transparent" />
            {p.score === null
              ? <text x={x(p.order)} y={failY} textAnchor="middle" fontSize="12" fill="var(--critical)">✕</text>
              : <circle cx={x(p.order)} cy={y(p.score)} r={hover === i ? 6 : 4.5} fill="var(--series-1)"
                  stroke="var(--chart-surface)" strokeWidth={2} />}
          </g>
        ))}
      </svg>
      {h && (
        <div className="tooltip" style={{ left: `${(x(h.order) / W) * 100}%`, top: 24 }}>
          <b>{h.label}</b> #{h.order}<br />{h.score === null ? "failed" : `score ${fmtScore(h.score)}`}
        </div>
      )}
    </div>
  );
}
