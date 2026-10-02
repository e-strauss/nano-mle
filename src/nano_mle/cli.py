"""Explicit live runs; the demo path is entirely offline."""

import argparse
import json
from pathlib import Path


def load_task(path):
    from .models import Task

    path = path.resolve()
    data = json.loads(path.read_text())
    data["sources"] = {name: str((path.parent / value).resolve()) if "://" not in value else value
                       for name, value in data["sources"].items()}
    return Task.model_validate(data)


def make_demo(directory, search_policy="greedy"):
    import numpy as np
    import pandas as pd

    from .demo import DemoBackend
    from .models import Budget, Task
    from .runner import Runner, initialize

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 3))
    table = pd.DataFrame(X, columns=["a", "b", "c"])
    table["target"] = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.2, size=len(X))
    table.to_csv(directory / "train.csv", index=False)
    task = Task(description="Independent synthetic regression rows; minimize RMSE",
                sources={"train": str(directory / "train.csv")}, target="target")
    workspace = directory / "workspace"
    initialize(workspace, task, Budget(max_expansions=2, max_evaluations=4), search_policy)
    Runner(workspace, DemoBackend()).run()
    return workspace


def main():
    parser = argparse.ArgumentParser(description="Sequential DSPy/Skrub ML experiment harness")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a workspace; makes no model calls")
    init.add_argument("workspace", type=Path)
    init.add_argument("--task", required=True, type=Path)
    init.add_argument("--policy", choices=["greedy", "mcts", "mcgs"], default="greedy")
    init.add_argument("--max-expansions", type=int, default=6)
    init.add_argument("--max-explorations", type=int, default=8)
    init.add_argument("--max-evaluations", type=int, default=24)
    init.add_argument("--max-evaluation-setups", type=int, default=3)
    init.add_argument("--max-actions", type=int, default=30)
    init.add_argument("--max-repairs", type=int, default=2)
    init.add_argument("--max-probes", type=int, default=4)
    init.add_argument("--max-model-calls", type=int, default=80)
    init.add_argument("--execution-timeout", type=int, default=120)
    run = sub.add_parser("run", help="Run/resume a workspace using the explicitly supplied model (API calls)")
    run.add_argument("workspace", type=Path)
    run.add_argument("--model", required=True, help="DSPy/LiteLLM model ID, e.g. gemini/gemini-3.8-flash")
    run.add_argument("--max-tokens", type=int, default=16000, help="Completion-token cap per call, incl. reasoning")
    run.add_argument("--request-timeout", type=int, default=180, help="Seconds per model request")
    draw = sub.add_parser("draw", help="Print an attempt's Skrub DataOps graph as SVG (lazy; reads no data)")
    draw.add_argument("attempt", type=Path, help="Attempt directory containing plan.py and request.json")
    show = sub.add_parser("show", help="Read the exported experiment report")
    show.add_argument("workspace", type=Path)
    demo = sub.add_parser("demo", help="Offline deterministic demonstration; no API calls")
    demo.add_argument("directory", type=Path)
    demo.add_argument("--policy", choices=["greedy", "mcts", "mcgs"], default="greedy")
    args = parser.parse_args()
    if args.command == "init":
        from .models import Budget
        from .runner import initialize

        budget = Budget(max_expansions=args.max_expansions, max_explorations=args.max_explorations,
                        max_evaluations=args.max_evaluations, max_actions=args.max_actions,
                        max_evaluation_setups=args.max_evaluation_setups,
                        max_repairs=args.max_repairs, max_probes=args.max_probes,
                        execution_timeout=args.execution_timeout,
                        max_model_calls=args.max_model_calls)
        initialize(args.workspace.resolve(), load_task(args.task), budget, args.policy)
        print(f"Initialized {args.workspace.resolve()}")
    elif args.command == "run":
        from .agents import DSPyBackend
        from .runner import Runner

        Runner(args.workspace, DSPyBackend(args.model, args.max_tokens, args.request_timeout)).run()
    elif args.command == "draw":
        from .draw import main as draw_main

        draw_main(args.attempt)
    elif args.command == "show":
        print((args.workspace / "report.md").read_text())
    else:
        print(f"Offline demo workspace: {make_demo(args.directory, args.policy)}")


if __name__ == "__main__":
    main()
