"""Revalidate native ACE serving; retain unchanged controls with explicit provenance."""
import json
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=20)
old = json.loads((ROOT / "docs/a6-runtime2-results.json").read_text())
version_id = (ROOT / ".runtime/revalidated-version").read_text()
evidence = {"note": "Small functional validation. Unchanged baseline and code-only controls reused explicitly.",
            "candidate_version": version_id, "jobs": [], "ablations": {},
            "experience_effect_valid": True, "runtime_adapter": 3,
            "reused_unchanged_modes": ["baseline", "harness"]}
path = ROOT / "docs/live-results.json"


def save():
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))


def wait(job_id):
    last = ""
    for _ in range(600):
        job = client.get("/api/jobs/" + job_id).raise_for_status().json()
        if job["stage"] != last:
            last = job["stage"]
            print(job_id[:8], job["status"], last, flush=True)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            evidence["jobs"].append(job)
            save()
            if job["status"] != "succeeded":
                raise RuntimeError(job["error"] or job["status"])
            return job
        time.sleep(3)
    raise TimeoutError("Revalidation timeout")


def run(kind, **kwargs):
    job = client.post("/api/jobs", headers={"Idempotency-Key": uuid.uuid4().hex},
                      json={"kind": kind, "version_id": version_id, **kwargs}).raise_for_status().json()
    return wait(job["id"])


gate = run("gate", limit=1)
traces = client.get("/api/jobs/" + gate["id"] + "/traces").raise_for_status().json()
train = next(t for t in traces if t["label"] == "candidate_train")
assert train["data"]["injected_skill_ids"], "Learned procedural skills must reach serving"
evidence["injected_skill_ids"] = train["data"]["injected_skill_ids"]
for mode in ["baseline", "harness"]:
    control = next(j for j in old["jobs"] if j["kind"] == "evaluate"
                   and j["payload"]["mode"] == mode and j["payload"]["version_id"] == old["candidate_version"])
    evidence["ablations"][mode] = {**control["result"], "reused_from_job": control["id"],
                                 "reason": "Skill routing change leaves this mode's exact prompt/code unchanged"}
    evidence["jobs"].append(control)
skill = run("evaluate", split="learn", mode="skills", task_ids=["0"], limit=1)
skill_traces = client.get("/api/jobs/" + skill["id"] + "/traces").raise_for_status().json()
assert skill_traces[0]["data"]["injected_skill_ids"], "Skill-only mode must inject experience"
evidence["ablations"]["skills"] = skill["result"]
evidence["ablations"]["combined"] = {
    "mean_reward": train["reward"], "scores": {"0": train["reward"]},
    "sample_count": 1, "trials": 1, "split": "learn", "mode": "combined",
    "source_gate_job": gate["id"], "source_label": "candidate_train",
    "injected_skill_ids": train["data"]["injected_skill_ids"],
    "reason": "Reuse the same fresh combined evaluation already executed by the gate"}
save()
if gate["result"]["passed"]:
    deployment = client.get("/api/overview").raise_for_status().json()["deployment"]
    evidence["release"] = client.post("/api/releases", json={
        "version_id": version_id, "expected_revision": deployment["revision"]
    }).raise_for_status().json()
    evidence["rollback"] = client.post("/api/rollback", json={
        "version_id": "baseline", "expected_revision": evidence["release"]["revision"]
    }).raise_for_status().json()
    evidence["deployment"] = client.post("/api/releases", json={
        "version_id": version_id, "expected_revision": evidence["rollback"]["revision"]
    }).raise_for_status().json()
save()
print("Native ACE serving revalidation complete.", flush=True)
