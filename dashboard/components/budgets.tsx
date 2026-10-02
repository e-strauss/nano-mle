import type { Budget } from "@/lib/types";

export function BudgetMeters({ budgets, compact = false }: { budgets: Budget[]; compact?: boolean }) {
  const shown = compact ? budgets.filter((b) => b.limit !== null) : budgets;
  return (
    <div className="budgets">
      {shown.map((b) => {
        const frac = b.limit ? Math.min(1, b.used / b.limit) : 0;
        return (
          <span key={b.name} title={`${b.name}: ${b.used}${b.limit !== null ? ` of ${b.limit}` : ""}`}>
            {b.limit !== null && (
              <span className="meter"><span className={frac >= 1 ? "full" : ""} style={{ width: `${frac * 100}%` }} /></span>
            )}
            <span className="num">{b.used}{b.limit !== null ? `/${b.limit}` : ""}</span>{" "}
            <span className="muted">{b.name}</span>
          </span>
        );
      })}
    </div>
  );
}
