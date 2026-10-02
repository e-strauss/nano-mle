"""DSPy backend. Importing the package never configures or calls a language model."""

import json
from typing import Protocol

from .models import Decision, Finding, Proposal


def lm_settings(model: str, max_tokens: int) -> dict:
    settings = {"max_tokens": max_tokens, "temperature": 0.2, "num_retries": 0,
                "timeout": 60, "cache": False}
    if model.removeprefix("openai/").startswith("gpt-6"):
        # DSPy's current reasoning-family detection only covers GPT-5/o-series.
        # Set the Chat Completions reasoning token cap explicitly for GPT-6.
        settings.update(temperature=None, max_tokens=None, max_completion_tokens=max_tokens,
                        reasoning_effort="low")
    return settings


class Backend(Protocol):
    def control(self, context: dict) -> Decision: ...
    def plan(self, context: dict) -> Proposal: ...
    def implement(self, kind: str, context: dict, intent: dict) -> str: ...
    def repair(self, kind: str, context: dict, intent: dict, source: str, error: str) -> str: ...
    def interpret(self, context: dict, question: str, result: dict) -> list[Finding]: ...


class DSPyBackend:
    def __init__(self, model: str, max_tokens=6000):
        import os
        import dspy
        from dotenv import load_dotenv

        from .prompts import CONTROL_INSTRUCTIONS, PLAN_INSTRUCTIONS, PLANNING_INSTRUCTIONS

        load_dotenv()
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        kwargs = lm_settings(model, max_tokens)
        if model.startswith("gemini/"):
            kwargs["api_key"] = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.lm = dspy.LM(model, **kwargs)
        self.dspy = dspy
        self.adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)

        class Control(dspy.Signature):
            context: str = dspy.InputField()
            decision: Decision = dspy.OutputField()

        class Plan(dspy.Signature):
            context: str = dspy.InputField()
            proposal: Proposal = dspy.OutputField()

        class Write(dspy.Signature):
            kind: str = dspy.InputField()
            context: str = dspy.InputField()
            intent: str = dspy.InputField()
            source: str = dspy.OutputField(desc="Python source only, defining zero-argument build()")

        class Repair(dspy.Signature):
            kind: str = dspy.InputField()
            context: str = dspy.InputField()
            intent: str = dspy.InputField()
            source: str = dspy.InputField()
            error: str = dspy.InputField()
            fixed_source: str = dspy.OutputField()

        class Interpret(dspy.Signature):
            """Extract evidence-backed, scoped findings; distinguish observations and hypotheses."""
            context: str = dspy.InputField()
            question: str = dspy.InputField()
            result: str = dspy.InputField()
            findings: list[Finding] = dspy.OutputField()

        self.controller = dspy.Predict(Control.with_instructions(CONTROL_INSTRUCTIONS))
        self.planner = dspy.Predict(Plan.with_instructions(PLANNING_INSTRUCTIONS))
        self.writer = dspy.Predict(Write.with_instructions(PLAN_INSTRUCTIONS))
        self.repairer = dspy.Predict(Repair.with_instructions(
            PLAN_INSTRUCTIONS + "\nRepair the error while preserving the planned experiment."))
        self.interpreter = dspy.Predict(Interpret)

    def _call(self, module, **kwargs):
        with self.dspy.context(lm=self.lm, adapter=self.adapter):
            return module(**{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in kwargs.items()})

    def control(self, context):
        return Decision.model_validate(self._call(self.controller, context=context).decision)

    def plan(self, context):
        return Proposal.model_validate(self._call(self.planner, context=context).proposal)

    def implement(self, kind, context, intent):
        return self._call(self.writer, kind=kind, context=context, intent=intent).source

    def repair(self, kind, context, intent, source, error):
        return self._call(self.repairer, kind=kind, context=context, intent=intent,
                          source=source, error=error).fixed_source

    def interpret(self, context, question, result):
        return [Finding.model_validate(f) for f in self._call(self.interpreter, context=context,
                                                            question=question, result=result).findings]
