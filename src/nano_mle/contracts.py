"""Immutable evaluation contract, local input identities, and frozen folds."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from .models import EvaluationSpec, Task


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_manifest(task: Task):
    manifest = {}
    for name, location in task.sources.items():
        if "://" in location:
            raise ValueError("Prototype sources must be local CSV files; remote source adapters are not implemented")
        path = Path(location).resolve(strict=True)
        manifest[name] = {"path": str(path), "sha256": digest(path)}
    return manifest


def verify_sources(manifest):
    for name, source in manifest.items():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Source {name!r} changed; create a new workspace")


def create_contract(task: Task, manifest, spec: EvaluationSpec):
    get_scorer(spec.scoring)
    verify_sources(manifest)
    frame = pd.read_csv(manifest[task.train_source]["path"])
    if task.target not in frame:
        raise ValueError(f"Target {task.target!r} missing")
    if frame[task.target].isna().any():
        raise ValueError("Missing labels: define the modelling population before creating a workspace")
    y = frame[task.target]
    positions = np.arange(len(frame))
    if spec.cv == "kfold":
        splitter = KFold(spec.folds, shuffle=True, random_state=spec.seed)
        splits = splitter.split(frame)
    elif spec.cv == "stratified":
        if y.value_counts().min() < spec.folds:
            raise ValueError("Each class needs at least folds rows")
        splits = StratifiedKFold(spec.folds, shuffle=True, random_state=spec.seed).split(frame, y)
    elif spec.cv == "group":
        groups = frame[spec.group_column]
        if groups.isna().any():
            raise ValueError("Missing groups")
        splits = GroupKFold(spec.folds).split(frame, y, groups)
    else:
        times = pd.to_datetime(frame[spec.time_column], errors="raise")
        if times.isna().any() or times.duplicated().any():
            raise ValueError("Time CV prototype requires nonmissing unique timestamps")
        positions = np.argsort(times.to_numpy(), kind="stable")
        splits = ((positions[a], positions[b]) for a, b in TimeSeriesSplit(spec.folds).split(frame))
    frozen = [{"train": a.tolist(), "test": b.tolist()} for a, b in splits]
    contract = {"task": task.model_dump(), "sources": manifest, "spec": spec.model_dump(),
                "rows": len(frame), "row_identity": "CSV position in SHA256-pinned training file",
                "splits": frozen, "fold_fingerprint": canonical_hash(frozen)}
    contract["id"] = canonical_hash(contract)
    return contract


def verify_contract(contract):
    body = {key: value for key, value in contract.items() if key != "id"}
    if canonical_hash(body) != contract["id"]:
        raise ValueError("Evaluation contract was modified")
    verify_sources(contract["sources"])
