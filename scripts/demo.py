"""Real API demonstration. Uses public synthetic records and the locally configured LLM."""

import json
import os
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=30)
evidence = {"note": "Small live smoke demonstration, not an improvement benchmark", "jobs": []}
path = ROOT / "docs/live-results.json"


def save():
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))


def wait(job_id):
    last = ""
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).raise_for_status().json()
        status = job["status"] + " " + job["stage"]
        if status != last:
            print(job_id[:8], status, flush=True)
            last = status
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            evidence["jobs"].append(job)
            save()
            if job["status"] != "succeeded":
                raise RuntimeError(job["error"] or job["status"])
            return job
        time.sleep(3)
    raise TimeoutError("demo job timeout")


def run(kind, version_id, **kwargs):
    response = client.post(
        "/api/jobs",
        headers={"Idempotency-Key": "demo-" + uuid.uuid4().hex},
        json={"kind": kind, "version_id": version_id, **kwargs},
    )
    response.raise_for_status()
    return wait(response.json()["id"])


base = (
    wait(os.environ["DEMO_SOURCE_JOB"])
    if os.getenv("DEMO_SOURCE_JOB")
    else run("evaluate", "baseline", mode="baseline", split="learn", limit=1)
)
if os.getenv("DEMO_SKILL_VERSION"):
    skills_id = os.environ["DEMO_SKILL_VERSION"]
    wait(skills_id)
    skill_eval = wait(os.environ["DEMO_SKILL_SOURCE_JOB"])
else:
    skills = run("learn", "baseline", source_job_id=base["id"])
    skills_id = skills["result"]["version_id"]
    skill_eval = run("evaluate", skills_id, split="learn", mode="combined", limit=1)
code = run("evolve", skills_id, source_job_id=skill_eval["id"])
combined = code["result"]["version_id"]
gate = run("gate", combined, limit=1)
evidence["candidate_version"] = combined
# Four modes use identical task IDs, model configuration and repeat counts.
evidence["ablations"] = {}
for mode in ["baseline", "skills", "harness", "combined"]:
    result = run("evaluate", combined, split="learn", mode=mode, limit=1)
    evidence["ablations"][mode] = result["result"]
    save()
if gate["result"]["passed"]:
    deployment = client.get("/api/overview").raise_for_status().json()["deployment"]
    released = (
        client.post(
            "/api/releases",
            json={"version_id": combined, "expected_revision": deployment["revision"]},
        )
        .raise_for_status()
        .json()
    )
    evidence["release"] = released
    # Exercise a real rollback, then restore the validated candidate.
    rollback = (
        client.post(
            "/api/rollback",
            json={"version_id": "baseline", "expected_revision": released["revision"]},
        )
        .raise_for_status()
        .json()
    )
    evidence["rollback"] = rollback
    evidence["deployment"] = (
        client.post(
            "/api/releases",
            json={"version_id": combined, "expected_revision": rollback["revision"]},
        )
        .raise_for_status()
        .json()
    )
conversation = client.post("/api/conversations").raise_for_status().json()
order = client.get("/api/records?kind=orders&limit=1").raise_for_status().json()[0]
message = (
    "My user ID is "
    + order["data"]["user_id"]
    + ". Please tell me the status and items of order "
    + order["id"]
    + "."
)
if order.get("customer"):
    customer = order["customer"]
    message = (
        "My name is "
        + customer["name"]["first_name"]
        + " "
        + customer["name"]["last_name"]
        + ". My zip code is "
        + customer["zip"]
        + ". Please tell me the status and items of order "
        + order["id"]
        + "."
    )
chat = (
    client.post("/api/conversations/" + conversation["id"] + "/messages", json={"message": message})
    .raise_for_status()
    .json()
)
evidence["chat"] = wait(chat["id"])["result"]
save()
print("Live demonstration complete:", path, flush=True)
