"""Three repair-only live episodes; no scorecard access and no automatic fallback."""
import json
from concurrent.futures import ThreadPoolExecutor

from ace import Skillbook
from tau2.data_model.tasks import Task

from app.config import ROOT, model_signature, settings
from app.dataset import raw_tasks, splits
from app.evaluation_validity import require_scoreable
from app.native_bridge import atomic_json, sanitize
from app.preflight import build_signature
from app.simulation import simulate_native


def main():
    path = ROOT / "docs/aligned-preflight.json"
    record = {"status": "running", "model_config": model_signature(),
              "build": build_signature(), "checks": {}, "episodes": []}
    if path.exists():
        import time
        path.rename(path.with_name(f"aligned-preflight-{time.time_ns()}.json"))
    atomic_json(path, record)
    code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    version = {"id": "aligned-preflight", "harness_code": code,
               "skillbook": Skillbook().to_dict()}
    tasks = {str(t["id"]): t for t in raw_tasks()}
    ids = splits()["learn"][:3]
    output = ROOT / "artifacts/aligned-preflight"
    output.mkdir(exist_ok=True)

    def episode(task_id):
        try:
            reward, data = simulate_native(Task.model_validate(tasks[task_id]), version, "baseline", 0)
            atomic_json(output / (task_id + ".json"), sanitize(data))
            require_scoreable(data)
            calls = sum(len(m.get("tool_calls") or []) for m in data.get("messages", []))
            if not calls:
                raise ValueError("Customer produced no real tool calls")
            return {"task_id": task_id, "status": "passed", "reward": reward, "tool_calls": calls}
        except Exception as exc:
            return {"task_id": task_id, "status": "failed", "error": sanitize(str(exc))}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(episode, ids):
            record["episodes"].append(result)
            atomic_json(path, record)
    valid = all(e["status"] == "passed" for e in record["episodes"])
    record["checks"]["baseline_episodes"] = valid
    record["checks"]["customer_tool_call"] = valid
    probe_path = ROOT / ".runtime/deepseek-optimizer-smoke.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    record["checks"]["optimizer_code_edit"] = (probe.get("status") == "succeeded"
        and probe.get("model") == settings.optimizer_model)
    try:
        if valid:
            import hashlib

            from app.failure_analysis import analyze_training
            trajectories = [json.loads((output / (t + ".json")).read_text()) for t in ids]
            analysis = analyze_training(trajectories, hashlib.sha256(code.encode()).hexdigest(), ids)
            atomic_json(output / "failure-analysis.json", analysis)
            record["checks"]["training_failure_analysis"] = True
            from app.learning import learn_skills
            runs = [{"id": "preflight-" + d["task_id"], "task_id": d["task_id"], "job_id": "aligned-preflight",
                     "reward": (d.get("reward_info") or {}).get("reward"), "data": d}
                    for d in trajectories]
            learned = learn_skills(version, runs, lambda *args: None)
            atomic_json(output / "ace-result.json", sanitize({"skillbook": learned[0], "audit": learned[1]}))
            record["checks"]["ace_learning"] = True
    except Exception as exc:
        valid = False
        record["error"] = sanitize(str(exc))
    # Only a subsequent reviewed offline-test record can seal this as passed.
    record["status"] = "ready_for_review" if valid else "blocked_model_behavior"
    atomic_json(path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
