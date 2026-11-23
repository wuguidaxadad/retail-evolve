from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

from .config import ROOT


class SnapshotRunner:
    """Adapt real, completed tau evaluations to the original auto-harness gate."""

    def __init__(self, results):
        self.results = results

    def run(self, task_ids=None):
        ids = task_ids if task_ids is not None else self.results.keys()
        return {str(t): self.results.get(str(t)) for t in ids}

    def val_score(self, results):
        return sum(v or 0 for v in results.values()) / len(results) if results else 0


def run_gate(directory, incumbent_train, candidate_train, incumbent_val, candidate_val):
    directory = Path(directory)
    (directory / "workspace").mkdir(parents=True, exist_ok=True)
    (directory / "experiment_config.yaml").write_text("file_guard: false\n")
    # Candidate code is already validated as a pure AST; no arbitrary repo edits.
    suite = {
        "tasks": [t for t, r in incumbent_train.items() if r >= 0.5],
        "threshold": 1.0,
        "last_results": {},
    }
    (directory / "workspace/eval_suite.json").write_text(json.dumps(suite))
    (directory / "workspace/train_results.json").write_text(
        json.dumps({"results": incumbent_train})
    )
    baseline = SnapshotRunner(incumbent_val).val_score(incumbent_val)
    vendor = ROOT / "vendor/auto-harness"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("retail_original_gate", vendor / "gating.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Upstream gate uses relative files. This runs only in the single-job worker.
    previous = Path.cwd()
    output = io.StringIO()
    try:
        os.chdir(directory)
        # Use the upstream constant instead of depending on a filename guess.
        Path(module.SUITE_FILE).write_text(json.dumps(suite))
        Path(module.RESULTS_FILE).write_text("val_score\n" + str(baseline) + "\n")
        with contextlib.redirect_stdout(output):
            status = module.run_gate(SnapshotRunner(candidate_train), SnapshotRunner(candidate_val))
    finally:
        os.chdir(previous)
    log = output.getvalue()
    (directory / "gate.log").write_text(log)
    return {
        "passed": status == 0,
        "incumbent_score": baseline,
        "candidate_score": SnapshotRunner(candidate_val).val_score(candidate_val),
        "regression_tasks": suite["tasks"],
        "log": log,
        "scope": "selected training regression + validation; final holdout unused",
    }
