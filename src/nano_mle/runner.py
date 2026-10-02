"""One sequential controller; explorations are evidence, candidates are search nodes."""

import fcntl
import json
from pathlib import Path

from .contracts import create_contract, digest, source_manifest, verify_contract
from .execution import run_plan
from .models import Budget, Task
from .plans import evaluation_source, resolve_source, validate_source
from .search import policy, update_stats, valid
from .store import Store, new_id


class ModelCallBudgetExceeded(RuntimeError):
    pass


def initialize(workspace: Path, task: Task, budget: Budget, search_policy="greedy"):
    manifest = source_manifest(task)
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "artifacts").mkdir()
    store = Store(workspace)
    try:
        store.set_meta("task", task.model_dump())
        store.set_meta("sources", manifest)
        store.set_meta("budget", budget.model_dump())
        store.set_meta("policy", search_policy)
        store.set_meta("state", "ready")
        store.set_meta("actions", 0)
        store.set_meta("search_stats", {})
        store.event("initialized", policy=search_policy)
        export_workspace(store)
    finally:
        store.close()


def export_workspace(store):
    workspace = store.workspace
    candidates = store.records("candidate")
    metadata = {"task": store.meta("task"), "sources": store.meta("sources"),
                "budget": store.meta("budget"), "policy": store.meta("policy"), "model": store.meta("model"),
                "state": store.meta("state"), "evaluation": store.meta("contract")}
    (workspace / "workspace.json").write_text(json.dumps(metadata, indent=2))
    graph = {"root": "root", "candidates": candidates,
             "primary_edges": [[c["parent_id"], c["id"]] for c in candidates],
             "reference_edges": [[r, c["id"]] for c in candidates for r in c["reference_ids"]],
             "evidence_links": [[e, c["id"]] for c in candidates for e in c["finding_ids"]],
             "search_stats": store.meta("search_stats")}
    (workspace / "graph.json").write_text(json.dumps(graph, indent=2))
    lines = ["# Experiment report", "",
             f"Policy: {store.meta('policy')}; model: {store.meta('model')}; state: {store.meta('state')}", "",
             "| Candidate | Parent | Status | Score | Configuration |", "|---|---|---|---:|---|"]
    for candidate in sorted(candidates, key=lambda c: c["score"] if c["score"] is not None else float("-inf"), reverse=True):
        lines.append(f"| {candidate['id']} | {candidate['parent_id']} | {candidate['status']} | "
                     f"{candidate['score']} | {candidate.get('configuration_description', {})} |")
    contract = store.meta("contract")
    if contract and "audit" in contract:
        lines += ["", "## Evaluation audit", "", json.dumps(contract["audit"], indent=2),
                  "", "Inputs are assumed static/frozen; source contents are not checked."]
    rejected = [e for e in store.records("expansion") if e.get("status") == "rejected"]
    if rejected:
        lines += ["", "## Evaluation drift warnings", ""]
        lines.extend(f"- {e['id']}: {e['result']['error']}" for e in rejected)
    lines += ["", "## Findings", ""]
    for finding in store.records("finding"):
        lines.append(f"- {finding['id']}: {finding['statement']} ({finding['scope']})")
    lines += ["", "All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.", ""]
    (workspace / "report.md").write_text("\n".join(lines))


class Runner:
    def __init__(self, workspace: Path, backend):
        self.workspace = workspace.resolve()
        if not (self.workspace / "state.db").exists():
            raise ValueError("Not an initialized workspace")
        self.store = Store(self.workspace)
        self.backend = backend
        self.task = Task.model_validate(self.store.meta("task"))
        self.budget = Budget.model_validate(self.store.meta("budget"))
        self.policy = policy(self.store.meta("policy"), self.budget.max_expansions)
        self.notify = print

    def counts(self):
        return {"evaluation_setups": len(self.store.records("evaluation_setup")), "explorations": len(self.store.records("exploration")),
                "expansions": len(self.store.records("expansion")),
                "evaluations": sum(a.get("evaluation_count", 0) for a in self.store.records("attempt")),
                "actions": self.store.meta("actions", 0),
                "model_calls": len(self.store.records("model_call"))}

    def context(self, selection=None, requested=0):
        candidates = self.store.records("candidate")
        findings = self.store.records("finding")
        superseded = {f for record in findings for f in record["supersedes"]}
        active_findings = [f for f in findings if f["id"] not in superseded]
        contract = self.store.meta("contract")
        summary = ({k: contract[k] for k in ("id", "spec", "rows", "fold_fingerprint")}
                   if contract else None)
        context = {"task": self.task.model_dump(), "contract": summary,
                   "sources": self.store.meta("sources"),
                   "locked_evaluation_source": contract["setup_source"] if contract else None,
                   "findings": active_findings, "counts": self.counts(),
                   "budget": self.budget.model_dump(),
                   "remaining_evaluations": self.budget.max_evaluations - self.counts()["evaluations"],
                   "leaderboard": sorted(valid(candidates), key=lambda c: c["score"], reverse=True)[:8],
                   "recent_failures": [c for c in candidates if c["status"] != "ok"][-4:],
                   "recent_explorations": self.store.records("exploration")[-4:],
                   "requested_explorations": requested, "parent": None}
        if selection:
            context["selection"] = {"parent_id": selection.parent_id, "reference_ids": selection.reference_ids}
            context["requested_explorations_remaining"] = self.budget.max_requested_explorations - requested
            context["references"] = [self._candidate_context(self.store.get(i)) for i in selection.reference_ids]
            if selection.parent_id != "root":
                parent = self.store.get(selection.parent_id)
                context["parent"] = self._candidate_context(parent)
                trajectory = []
                cursor = parent
                while cursor["id"] != "root":
                    trajectory.append(cursor)
                    if cursor["parent_id"] == "root":
                        break
                    cursor = self.store.get(cursor["parent_id"])
                context["trajectory"] = list(reversed(trajectory[:6]))
        return context

    def _candidate_context(self, candidate):
        path = self.workspace / candidate["source_path"]
        if digest(path) != candidate["source_sha256"]:
            raise ValueError("Historical candidate source was modified")
        return {**candidate, "resolved_source": resolve_source(path.read_text(), candidate["configuration"])}

    def call(self, method, **kwargs):
        if self.counts()["model_calls"] >= self.budget.max_model_calls:
            raise ModelCallBudgetExceeded("Model-call budget exhausted")
        call_id = new_id("call")
        directory = self.workspace / "artifacts" / "calls"
        directory.mkdir(exist_ok=True)
        request = {"method": method, "inputs": kwargs}
        (directory / f"{call_id}.input.json").write_text(json.dumps(request, indent=2))
        self.store.put("model_call", {"id": call_id, "method": method, "status": "started"})
        self.store.event("model_call_started", id=call_id, method=method)
        result = getattr(self.backend, method)(**kwargs)
        encoded = (result.model_dump() if hasattr(result, "model_dump") else
                   [f.model_dump() for f in result] if isinstance(result, list) else result)
        (directory / f"{call_id}.output.json").write_text(json.dumps(encoded, indent=2))
        self.store.put("model_call", {"id": call_id, "method": method, "status": "finished"})
        self.store.event("model_call_finished", id=call_id, method=method)
        return result

    def execute(self, kind, record, context, intent):
        source = self.call("implement", kind=kind, context=context, intent=intent)
        result = {"status": "failed", "error": "No attempt executed"}
        for repair_number in range(self.budget.max_repairs + 1):
            attempt_id = new_id("attempt")
            directory = self.workspace / "artifacts" / record["id"] / attempt_id
            request = {"kind": kind, "task": self.task.model_dump(), "sources": self.store.meta("sources"),
                       "remaining_evaluations": self.budget.max_evaluations - self.counts()["evaluations"]}
            if kind == "evaluation":
                request["expected_scoring"] = intent["scoring"]
            if kind == "pipeline":
                request["contract"] = self.store.meta("contract")
            attempt = {"id": attempt_id, "owner_id": record["id"], "status": "running",
                       "path": str(directory.relative_to(self.workspace)), "repair_number": repair_number,
                       "evaluation_count": 0}
            self.store.put("attempt", attempt)
            try:
                validate_source(source)
                result = run_plan(directory, source, request, self.budget.execution_timeout)
            except (SyntaxError, ValueError) as error:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "plan.py").write_text(source)
                result = {"status": "failed", "error": str(error), "evaluation_count": 0}
                (directory / "response.json").write_text(json.dumps(result))
            attempt.update(status=result["status"], evaluation_count=result.get("evaluation_count", 0))
            if result.get("warning"):
                attempt.update(warning=result["warning"], changed_components=result.get("changed_components"))
            self.store.put("attempt", attempt)
            record.setdefault("attempt_ids", []).append(attempt_id)
            self.store.put({"exploration": "exploration", "evaluation": "evaluation_setup", "pipeline": "expansion"}[kind], record)
            self.store.event("execution_finished", owner_id=record["id"], attempt_id=attempt_id,
                             status=result["status"], evaluations=attempt["evaluation_count"])
            if result["status"] == "ok":
                return result, attempt
            if (repair_number == self.budget.max_repairs or
                    (kind == "pipeline" and self.counts()["evaluations"] >= self.budget.max_evaluations)):
                break
            self.notify(f"  Repair {repair_number + 1}: {result['error'].splitlines()[0][:180]}")
            source = self.call("repair", kind=kind, context=context, intent=intent, source=source,
                               error=result.get("traceback", result["error"]))
        return result, attempt

    def explore(self, question, stopping_condition, expansion_id=None, context=None):
        if self.counts()["explorations"] >= self.budget.max_explorations:
            raise ValueError("Exploration budget exhausted")
        record = {"id": new_id("exploration"), "question": question,
                  "stopping_condition": stopping_condition, "expansion_id": expansion_id,
                  "status": "running", "attempt_ids": []}
        self.store.put("exploration", record)
        self.notify(f"Explore: {question}")
        context = context or self.context()
        result, attempt = self.execute("exploration", record, context,
                                       {"question": question, "stopping_condition": stopping_condition})
        record.update(status=result["status"], result=result, artifact_path=attempt["path"])
        self.store.put("exploration", record)
        if result["status"] == "ok":
            findings = self.call("interpret", context=context, question=question, result=result)
            existing = {f["id"] for f in self.store.records("finding")}
            for finding in findings[:8]:
                if not set(finding.supersedes) <= existing:
                    raise ValueError("Finding supersedes an unknown finding")
                saved = {**finding.model_dump(), "id": new_id("finding"), "exploration_id": record["id"],
                         "artifact_path": attempt["path"]}
                self.store.put("finding", saved)
        export_workspace(self.store)
        return record["id"]

    def establish(self, spec):
        if self.store.meta("contract") is not None:
            raise ValueError("Evaluation is already locked; use a new workspace to change it")
        if not any(e["status"] == "ok" for e in self.store.records("exploration")):
            raise ValueError("Successful exploration is required before locking evaluation")
        if self.counts()["evaluation_setups"] >= self.budget.max_evaluation_setups:
            raise ValueError("Evaluation setup budget exhausted")
        record = {"id": new_id("setup"), "status": "running", "spec": spec.model_dump(), "attempt_ids": []}
        self.store.put("evaluation_setup", record)
        self.notify("Audit agent-authored evaluation setup")
        result, attempt = self.execute("evaluation", record, self.context(), spec.model_dump())
        record.update(status=result["status"], result=result, artifact_path=attempt["path"])
        self.store.put("evaluation_setup", record)
        if result["status"] != "ok":
            export_workspace(self.store)
            return
        if result["snapshot"]["scoring"] != spec.scoring:
            raise ValueError("Setup scoring must match the proposed scorer")
        source = (self.workspace / attempt["path"] / "plan.py").read_text()
        contract = create_contract(result["snapshot"], evaluation_source(source), spec)
        self.store.set_meta("contract", contract)
        self.store.event("evaluation_locked", contract_id=contract["id"])
        self.notify(f"Locked {spec.scoring}: {contract['rows']} rows")
        export_workspace(self.store)

    def expand(self):
        contract = self.store.meta("contract")
        if contract is None:
            raise ValueError("Establish evaluation before expanding")
        verify_contract(contract)
        counts = self.counts()
        if counts["expansions"] >= self.budget.max_expansions or counts["evaluations"] >= self.budget.max_evaluations:
            raise ValueError("Search budget exhausted")
        old_candidates = self.store.records("candidate")
        selection = self.policy.select(old_candidates, self.store.meta("search_stats"), counts["expansions"])
        record = {"id": new_id("expansion"), "parent_id": selection.parent_id,
                  "reference_ids": selection.reference_ids, "status": "running", "attempt_ids": [],
                  "exploration_ids": []}
        self.store.put("expansion", record)
        self.notify(f"Expand {selection.parent_id} ({self.policy.name})")
        requested = 0
        while True:
            context = self.context(selection, requested)
            proposal = self.call("plan", context=context)
            if proposal.action == "experiment":
                break
            if requested >= self.budget.max_requested_explorations:
                record.update(status="failed", error="Planner exceeded requested exploration limit")
                self.store.put("expansion", record)
                return
            record["exploration_ids"].append(self.explore(proposal.question, proposal.stopping_condition,
                                                         record["id"], context))
            self.store.put("expansion", record)
            requested += 1
        record.update(proposal=proposal.model_dump(), finding_ids=[f["id"] for f in context["findings"]])
        self.store.put("expansion", record)
        result, attempt = self.execute("pipeline", record, context, proposal.model_dump())
        record.update(status=result["status"], result=result)
        if result.get("warning") == "contract_drift":
            record.update(status="rejected", candidate_ids=[])
            self.store.put("expansion", record)
            self.store.event("contract_drift", error=result["error"], changed_components=result.get("changed_components"))
            self.notify(result["error"])
            export_workspace(self.store)
            return
        source_path = self.workspace / attempt["path"] / "plan.py"
        variants = result.get("variants", [{"index": 0, "status": "failed", "score": None,
                                            "configuration": {}, "error": result.get("error")}])
        new_candidates = []
        for variant in variants:
            new_candidates.append({**variant, "id": f"{record['id']}_variant_{variant['index'] + 1}",
                                   "parent_id": selection.parent_id, "reference_ids": selection.reference_ids,
                                   "finding_ids": record["finding_ids"], "batch_id": record["id"],
                                   "description": proposal.description, "contract_id": contract["id"],
                                   "source_path": str(source_path.relative_to(self.workspace)),
                                   "source_sha256": digest(source_path)})
        baseline = max((c["score"] for c in valid(old_candidates)), default=None)
        stats = update_stats(self.store.meta("search_stats"), old_candidates + new_candidates,
                             new_candidates, baseline)
        record["candidate_ids"] = [c["id"] for c in new_candidates]
        self.store.commit_expansion(record, new_candidates, stats)
        self.notify(f"  Recorded {len(new_candidates)} candidate(s): " +
                    ", ".join(str(c["score"]) if c["status"] == "ok" else "failed" for c in new_candidates))
        export_workspace(self.store)

    def recover(self):
        """Interrupted work is retained and charged; never silently replay executions."""
        for attempt in self.store.records("attempt"):
            if attempt["status"] == "running":
                usage = self.workspace / attempt["path"] / "usage.json"
                attempt.update(status="interrupted", evaluation_count=json.loads(usage.read_text())["evaluation_count"]
                               if usage.exists() else 0)
                self.store.put("attempt", attempt)
        for kind in ("expansion", "exploration", "evaluation_setup"):
            for record in self.store.records(kind):
                if record["status"] == "running":
                    record.update(status="interrupted", error="Previous runner interrupted; artifacts retained")
                    self.store.put(kind, record)
        for call in self.store.records("model_call"):
            if call["status"] == "started":
                call["status"] = "interrupted"
                self.store.put("model_call", call)

    def fail_active(self, error):
        for kind in ("expansion", "exploration", "evaluation_setup"):
            for record in self.store.records(kind):
                if record["status"] == "running":
                    record.update(status="failed", error=error)
                    self.store.put(kind, record)

    def run(self):
        owns_lock = False
        try:
            with (self.workspace / ".run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("Another runner owns this workspace") from None
                owns_lock = True
                if self.store.meta("state") == "complete":
                    return
                if self.store.meta("contract"):
                    verify_contract(self.store.meta("contract"))
                self.recover()
                # A resumed run may use a different model; each session is recorded.
                model = getattr(self.backend, "model", type(self.backend).__name__)
                self.store.set_meta("model", model)
                self.store.event("run_started", model=model)
                self.store.set_meta("state", "running")
                while self.counts()["actions"] < self.budget.max_actions:
                    if self.counts()["evaluations"] >= self.budget.max_evaluations:
                        self.store.event("stopped", reason="Evaluation budget exhausted")
                        break
                    self.store.set_meta("actions", self.counts()["actions"] + 1)
                    decision = self.call("control", context=self.context())
                    self.store.event("controller_decision", **decision.model_dump())
                    if decision.action == "stop":
                        self.store.event("stopped", reason=decision.reason)
                        break
                    try:
                        if decision.action == "explore":
                            self.explore(decision.question, decision.stopping_condition)
                        elif decision.action == "establish_evaluation":
                            self.establish(decision.evaluation)
                        else:
                            self.expand()
                    except ValueError as error:
                        self.fail_active(str(error))
                        self.store.event("action_rejected", error=str(error))
                        self.notify(f"Action rejected: {error}")
                self.store.set_meta("state", "complete")
                export_workspace(self.store)
        except ModelCallBudgetExceeded as error:
            self.fail_active(str(error))
            self.store.event("stopped", reason=str(error))
            self.store.set_meta("state", "complete")
            export_workspace(self.store)
        except BaseException:
            if owns_lock:
                self.store.set_meta("state", "interrupted")
                export_workspace(self.store)
            raise
        finally:
            self.store.close()
