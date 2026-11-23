from __future__ import annotations

import json

from .config import ROOT, settings

DATA = ROOT / "vendor/tau2/data/tau2/domains/retail"


def raw_database():
    return json.loads((DATA / "db.json").read_text())


def raw_tasks():
    return json.loads((DATA / "tasks.json").read_text())


def splits():
    """Keep official train/test; split official test into validation and final."""
    if settings.experiment_profile == "aligned-v1":
        import hashlib
        frozen = json.loads((ROOT / "configs/aligned_retail_split.json").read_text())
        if hashlib.sha256((DATA / "tasks.json").read_bytes()).hexdigest() != frozen["task_sha256"]:
            raise ValueError("Frozen task pool changed")
        parts = frozen["parts"]
        sets = [set(parts[k]) for k in ("learn", "validation", "final")]
        if [len(x) for x in sets] != [20, 20, 74] or len(set.union(*sets)) != 114:
            raise ValueError("Invalid or overlapping aligned splits")
        return parts
    source = json.loads((DATA / "split_tasks.json").read_text())
    learn = [str(t) for t in source["train"]]
    test = [str(t) for t in source["test"]]
    middle = max(1, len(test) // 2)
    return {"learn": learn, "validation": test[:middle], "final": test[middle:]}


def choose_tasks(split, ids=None, limit=1):
    parts = splits()
    allowed = parts["validation"] + parts["final"] if split == "upstream_test" else parts.get(split)
    if allowed is None:
        raise ValueError("任务分组不存在")
    chosen = ids if ids is not None else allowed[:limit]
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(allowed):
        raise ValueError("任务必须来自所选分组，且不能重复")
    return chosen


def task_summary(task):
    scenario = task.get("user_scenario", {}).get("instructions", {})
    return (
        scenario.get("reason_for_call", str(scenario))
        if isinstance(scenario, dict)
        else str(scenario)
    )
