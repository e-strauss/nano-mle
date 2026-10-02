"use client";

import { useActionState } from "react";
import { launch } from "@/app/actions/runs";

export type FormField = {
  name: string; label: string; type: "slug" | "text" | "int" | "select" | "file";
  default?: string; options?: string[]; help?: string; min?: number; max?: number;
};

export default function LaunchForm({ launcher, fields, steps }: {
  launcher: string; fields: FormField[]; steps: string[][];
}) {
  const [state, action, pending] = useActionState(launch, undefined);
  return (
    <form action={action}>
      <input type="hidden" name="_launcher" value={launcher} />
      <div className="form-grid">
        {fields.map((f) => (
          <FieldInput key={f.name} f={f} />
        ))}
      </div>
      <details style={{ marginTop: 12 }}>
        <summary>Command templates</summary>
        <pre className="wrap">{steps.map((s) => s.join(" ")).join("\n\n")}</pre>
      </details>
      {state?.error && <p className="error">{state.error}</p>}
      <p className="small muted">Launching runs the commands above as a detached process; model calls cost money.</p>
      <button type="submit" className="primary" disabled={pending}>{pending ? "Starting…" : "Start run"}</button>
    </form>
  );
}

function FieldInput({ f }: { f: FormField }) {
  const input = (() => {
    switch (f.type) {
      case "select":
      case "file":
        return (
          <select name={f.name} defaultValue={f.default} required>
            {f.type === "file" && <option value="">— choose —</option>}
            {(f.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        );
      case "int":
        return <input name={f.name} type="number" min={f.min} max={f.max} defaultValue={f.default} required />;
      default:
        return <input name={f.name} defaultValue={f.default} required
          placeholder={f.type === "slug" ? "e.g. housing-greedy-20261002" : undefined} />;
    }
  })();
  return (
    <>
      <label htmlFor={f.name}>{f.label}</label>
      {input}
      {f.help && <span className="help small muted">{f.help}</span>}
    </>
  );
}
