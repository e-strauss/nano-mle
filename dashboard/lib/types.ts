// Harness-neutral run model. Adapters translate a harness's own storage into
// these shapes; pages and components only ever see these.

export type RunState = "ready" | "running" | "complete" | "interrupted" | "failed" | "unknown";

export type Budget = { name: string; used: number; limit: number | null };

export type RunSummary = {
  id: string;               // "<root name>/<relative path>", used in URLs
  adapter: string;
  title: string;
  state: RunState;
  live: boolean;            // a process currently owns the run
  pid?: number;
  policy?: string;
  model?: string;
  metric?: string;
  best?: { score: number; node: string };
  budgets: Budget[];
  createdAt?: number;       // ms since epoch
  updatedAt?: number;
  launchId?: string;        // set when the dashboard launched this run
  notes: string[];          // adapter warnings, e.g. "state says running, no live process"
};

export type NodeStatus = "ok" | "failed" | "running" | "rejected" | "interrupted" | "info";

export type GraphNode = {
  id: string;
  kind: string;             // adapter vocabulary: root, candidate, exploration, setup, expansion ...
  shape: "circle" | "diamond" | "square";
  label: string;
  subtitle?: string;
  status: NodeStatus;
  score?: number | null;
  parent: string | null;    // layout parent (a tree); other relations are edges
  group?: string;           // siblings evaluated together (e.g. one grid)
  order: number;            // creation order
};

export type EdgeKind = "primary" | "reference" | "evidence" | "requested";
export type GraphEdge = { from: string; to: string; kind: EdgeKind };

export type TimelineEvent = { time: number; kind: string; text: string };

export type FileRef = { path: string; size: number }; // relative to the run directory

export type Section =
  | { title: string; kind: "text"; content: string }
  | { title: string; kind: "code"; content: string; lang?: string }
  | { title: string; kind: "json"; content: unknown }
  | { title: string; kind: "kv"; content: [string, string][] }
  | { title: string; kind: "files"; content: FileRef[] }
  | { title: string; kind: "graph"; content: string }   // attempt dir, relative to the run; drawn on demand
  | { title: string; kind: "group"; content: Section[] };

export type ScorePoint = { node: string; order: number; score: number | null; label: string };

export type RunDetail = {
  summary: RunSummary;
  nodes: GraphNode[];
  edges: GraphEdge[];
  events: TimelineEvent[];
  scores: ScorePoint[];     // scored nodes in evaluation order (null = failed)
  overview: Section[];
};

export interface Adapter {
  name: string;
  detect(dir: string): boolean;
  summary(dir: string, id: string): RunSummary;
  detail(dir: string, id: string): RunDetail;
  node(dir: string, nodeId: string): Section[];
}
