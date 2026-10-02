import type { GraphNode } from "./types";

// Left-to-right tidy tree over the layout parents: depth -> x, leaf slot -> y,
// parents centred over their children, children in creation order.

export type Placed = GraphNode & { x: number; y: number; depth: number };

export const X_STEP = 250;
export const Y_STEP = 40;

export function layout(nodes: GraphNode[]): { placed: Map<string, Placed>; width: number; height: number } {
  const children = new Map<string | null, GraphNode[]>();
  const ids = new Set(nodes.map((n) => n.id));
  for (const n of nodes) {
    const parent = n.parent && ids.has(n.parent) ? n.parent : null;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent)!.push(n);
  }
  for (const list of children.values()) list.sort((a, b) => a.order - b.order);

  const placed = new Map<string, Placed>();
  let slot = 0;
  let maxDepth = 0;
  const visit = (n: GraphNode, depth: number): number => {
    maxDepth = Math.max(maxDepth, depth);
    const kids = children.get(n.id) ?? [];
    let y: number;
    if (!kids.length) {
      y = slot++ * Y_STEP;
    } else {
      const ys = kids.map((k) => visit(k, depth + 1));
      y = (ys[0] + ys[ys.length - 1]) / 2;
    }
    placed.set(n.id, { ...n, x: depth * X_STEP, y, depth });
    return y;
  };
  for (const r of children.get(null) ?? []) {
    visit(r, 0);
    slot += 0.5;
  }
  return { placed, width: (maxDepth + 1) * X_STEP, height: Math.max(1, slot) * Y_STEP };
}
