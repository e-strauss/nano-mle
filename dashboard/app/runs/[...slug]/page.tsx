import { notFound } from "next/navigation";
import AutoRefresh from "@/components/auto-refresh";
import { BudgetMeters } from "@/components/budgets";
import Header from "@/components/header";
import RunView from "@/components/run-view";
import { verifySession } from "@/lib/dal";
import { ago, score } from "@/lib/format";
import { runDetail } from "@/lib/runs";

export const dynamic = "force-dynamic";

export default async function RunPage(props: PageProps<"/runs/[...slug]">) {
  await verifySession();
  const { slug } = await props.params;
  const detail = runDetail(slug.map(decodeURIComponent).join("/"));
  if (!detail) notFound();
  const s = detail.summary;
  return (
    <>
      <Header />
      <main className="page">
        <div className="row">
          <h1 className="mono">{s.id}</h1>
          <span className={`badge ${s.live ? "live" : ""}`}>{s.live ? "running" : s.state}</span>
          <AutoRefresh active={s.live} />
        </div>
        <div className="muted">{s.title}</div>
        <div className="row small" style={{ margin: "6px 0 10px" }}>
          <span>adapter <b>{s.adapter}</b></span>
          <span>policy <b>{s.policy ?? "—"}</b></span>
          <span>model <b className="mono">{s.model ?? "—"}</b></span>
          <span>metric <b>{s.metric ?? "—"}</b></span>
          <span>best <b className="mono">{score(s.best?.score)}</b></span>
          <span>updated {ago(s.updatedAt)}</span>
        </div>
        {s.notes.map((n) => <p key={n} className="error small">{n}</p>)}
        <div className="panel" style={{ marginBottom: 12 }}><BudgetMeters budgets={s.budgets} /></div>
        <RunView detail={detail} />
      </main>
    </>
  );
}
