import Header from "@/components/header";
import LaunchForm, { type FormField } from "@/components/launch-form";
import { loadConfig } from "@/lib/config";
import { verifySession } from "@/lib/dal";
import { fileOptions } from "@/lib/launch";

export const dynamic = "force-dynamic";

export default async function NewRunPage() {
  await verifySession();
  const { launchers } = loadConfig();
  return (
    <>
      <Header />
      <main className="page">
        <h1>New run</h1>
        {!launchers.length && <p className="muted">No launchers configured in launchers.json.</p>}
        {launchers.map((l) => {
          const fields: FormField[] = l.fields.map((f) => ({
            name: f.name, label: f.label ?? f.name, type: f.type, help: f.help,
            default: "default" in f && f.default !== undefined ? String(f.default) : undefined,
            options: f.type === "select" ? f.options : f.type === "file" ? fileOptions(f) : undefined,
            min: f.type === "int" ? f.min : undefined, max: f.type === "int" ? f.max : undefined,
          }));
          return (
            <section key={l.id} className="panel">
              <h2>{l.label}</h2>
              {l.description && <p className="muted">{l.description}</p>}
              <p className="small muted">New runs go to <span className="mono">{l.workspace.replace("{root}", l.root)}</span></p>
              <LaunchForm launcher={l.id} fields={fields} steps={l.steps} />
            </section>
          );
        })}
      </main>
    </>
  );
}
