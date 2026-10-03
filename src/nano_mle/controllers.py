"""Action strategies return the same decisions as the model controller."""

import math

from .models import Decision


class LLM:
    name = "llm"

    def decide(self, context, call):
        return call("control", context=context)


class Auto:
    """Use the model for evaluation preparation, then expand until a budget ends."""

    name = "auto"

    def __init__(self, time_margin=1.0):
        if not math.isfinite(time_margin) or time_margin < 0:
            raise ValueError("time_margin must be finite and non-negative")
        self.time_margin = time_margin

    def decide(self, context, call):
        counts, budget = context["counts"], context["budget"]
        for kind in ("expansions", "evaluations"):
            if counts[kind] >= budget[f"max_{kind}"]:
                return Decision(action="stop", reason=f"auto: {kind.capitalize()} budget exhausted")
        if context["contract"] is None:
            return call("control", context=context)
        timing = context["time"]
        remaining = timing["remaining_s"]
        typical = timing["typical_attempt_s"].get("expansion")
        if remaining is not None and (remaining <= 0 or
                (typical is not None and remaining < self.time_margin * typical)):
            return Decision(action="stop", reason="auto: insufficient time for another expansion")
        parent = context["next_expansion_parent"]
        if parent["id"] == "root":
            progress = (f" {parent['draft_number']}/{parent['num_drafts']}"
                        if "draft_number" in parent else "")
            reason = f"auto: draft{progress}: propose a new, simple solution, different from the existing drafts."
        else:
            reason = "auto: improve: propose one change to the selected parent whose effect on the score can be measured."
        if budget["max_requested_explorations"] == 0:
            reason += " Propose an experiment directly; do not request exploration."
        return Decision(action="expand", reason=reason)


CONTROLLERS = {"llm": LLM, "auto": Auto}


def controller(name, params=None):
    return CONTROLLERS[name](**(params or {}))
