"""Lock agent-authored evaluation graphs. Input files are assumed frozen."""

import hashlib
import json
from pathlib import Path


class ContractDrift(ValueError):
    def __init__(self, changes):
        self.changes = changes
        super().__init__("Evaluation contract drift: " + ", ".join(changes) +
                         ". Restore the locked setup or start a new workspace with a new evaluation lock.")


def digest(path: Path) -> str:
    """Integrity of generated artifacts only; never used to inspect source data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_manifest(task):
    return {name: {"path": location if "://" in location else str(Path(location).resolve())}
            for name, location in task.sources.items()}


def create_contract(snapshot, setup_source, spec):
    contract = {**snapshot, "version": 2, "spec": spec.model_dump(),
                "setup_source": setup_source, "input_assumption": "static/frozen; content checks out of scope"}
    contract["id"] = canonical_hash(contract)
    return contract


def verify_contract(contract):
    if contract.get("version") != 2:
        raise ValueError(f"Unsupported evaluation contract version {contract.get('version')!r}; start a new workspace")
    if canonical_hash({k: v for k, v in contract.items() if k != "id"}) != contract["id"]:
        raise ValueError("Evaluation contract was modified")


def check_snapshot(contract, snapshot):
    verify_contract(contract)
    changes = [name for name in ("X", "y", "cv", "scoring", "row_keys")
               if contract["components"].get(name) != snapshot["components"].get(name)]
    if contract["rows"] != snapshot["rows"]:
        changes.append("row count")
    if contract["fold_fingerprint"] != snapshot["fold_fingerprint"]:
        changes.append("fold membership")
    if changes:
        raise ContractDrift(changes)
