"""Typed decisions exchanged with models; durable state is owned by the harness."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Task(Model):
    description: str
    sources: dict[str, str]
    train_source: str = "train"
    target: str
    drop_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_source(self):
        if self.train_source not in self.sources:
            raise ValueError("train_source must name a source")
        return self


class EvaluationSpec(Model):
    scoring: str
    cv: Literal["kfold", "stratified", "group", "time"]
    folds: int = Field(default=3, ge=2, le=20)
    seed: int = 42
    group_column: str | None = None
    time_column: str | None = None
    rationale: str

    @model_validator(mode="after")
    def check_columns(self):
        if self.cv == "group" and not self.group_column:
            raise ValueError("group CV needs group_column")
        if self.cv == "time" and not self.time_column:
            raise ValueError("time CV needs time_column")
        return self


class Decision(Model):
    action: Literal["explore", "establish_evaluation", "expand", "stop"]
    reason: str
    question: str | None = None
    stopping_condition: str | None = None
    evaluation: EvaluationSpec | None = None

    @model_validator(mode="after")
    def check_payload(self):
        if self.action == "explore" and not (self.question and self.stopping_condition):
            raise ValueError("explore needs a question and stopping_condition")
        if self.action == "establish_evaluation" and self.evaluation is None:
            raise ValueError("establish_evaluation needs evaluation")
        return self


class Proposal(Model):
    action: Literal["experiment", "explore"]
    description: str
    changes: list[str] = Field(default_factory=list)
    question: str | None = None
    stopping_condition: str | None = None

    @model_validator(mode="after")
    def check_question(self):
        if self.action == "explore" and not (self.question and self.stopping_condition):
            raise ValueError("exploration request needs question and stopping_condition")
        return self


class Finding(Model):
    statement: str
    kind: Literal["observation", "hypothesis"] = "observation"
    scope: str
    evidence: str
    supersedes: list[str] = Field(default_factory=list)


class Budget(Model):
    max_actions: int = Field(default=30, ge=1)
    max_expansions: int = Field(default=6, ge=1)
    max_explorations: int = Field(default=8, ge=1)
    max_requested_explorations: int = Field(default=2, ge=0)
    max_evaluations: int = Field(default=24, ge=1)
    max_repairs: int = Field(default=2, ge=0)
    execution_timeout: int = Field(default=120, ge=1)
