"""Fail closed before paid formal runs unless this exact build passed live probes."""
import hashlib
import json

from .config import ROOT, model_signature

FILES = ("app/transport_guard.py", "app/config.py", "app/dataset.py", "app/agent.py", "app/native_bridge.py",
         "app/full_rsi.py", "app/failure_analysis.py", "app/simulation.py",
         "app/learning.py", "app/native_harness.py", "app/experiment_metrics.py",
         "app/api.py", "app/jobs.py", "app/codex_gateway.py",
         "scripts/aligned_preflight.py", "app/preflight.py", "scripts/aligned_customer_experiment.py",
         "configs/aligned_retail_split.json")


def build_signature():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}


def require_preflight():
    path = ROOT / "docs/aligned-preflight.json"
    if not path.exists():
        raise ValueError("Live preflight has not completed")
    record = json.loads(path.read_text())
    if (record.get("status") != "passed" or record.get("model_config") != model_signature()
            or record.get("build") != build_signature()):
        raise ValueError("Live preflight missing, failed or stale; refusing formal run")
    required = {"customer_tool_call", "optimizer_code_edit", "baseline_episodes",
                "ace_learning", "training_failure_analysis", "offline_tests"}
    if not all(record.get("checks", {}).get(k) is True for k in required):
        raise ValueError("Incomplete preflight evidence")
    return record
