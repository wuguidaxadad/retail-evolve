"""Verify a real multi-turn Agent write is persisted, while the public fixture stays intact."""

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.db import Conversation, RetailRecord, transaction  # noqa: E402

client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=15)
order = next(
    o
    for o in client.get("/api/records?kind=orders&limit=20").raise_for_status().json()
    if o["data"]["status"] == "pending"
)
customer = order["customer"]
address = {**order["data"]["address"], "address1": "100 Demo Road", "address2": ""}
conversation = client.post("/api/conversations").raise_for_status().json()
messages = [
    "My name is "
    + customer["name"]["first_name"]
    + " "
    + customer["name"]["last_name"]
    + ", and my zip code is "
    + customer["zip"]
    + ". Please change the SHIPPING ADDRESS "
    + "of my pending order "
    + order["id"]
    + " to "
    + json.dumps(address)
    + ". Keep my user profile address unchanged. Please list the change for my confirmation.",
    "Yes, I explicitly confirm changing only the shipping address of order "
    + order["id"]
    + " to "
    + json.dumps(address)
    + ". Please execute this change now.",
]
job_ids = []
for message in messages:
    response = (
        client.post(
            "/api/conversations/" + conversation["id"] + "/messages", json={"message": message}
        )
        .raise_for_status()
        .json()
    )
    job_ids.append(response["id"])
    for _ in range(200):
        job = client.get("/api/jobs/" + response["id"]).raise_for_status().json()
        if job["status"] in {"failed", "cancelled"}:
            raise RuntimeError(job["error"] or job["status"])
        if job["status"] == "succeeded":
            print("客服回复：", job["result"]["reply"], flush=True)
            break
        time.sleep(2)
    else:
        raise TimeoutError("Chat smoke timeout")
with transaction() as s:
    conv = s.get(Conversation, conversation["id"])
    actual = conv.state["orders"][order["id"]]["address"]
    canonical = s.get(RetailRecord, ("orders", order["id"])).data["address"]
    profile = conv.state["users"][order["data"]["user_id"]]["address"]
    canonical_profile = s.get(RetailRecord, ("users", order["data"]["user_id"])).data["address"]
    evidence = {
        "conversation_id": conv.id,
        "version_id": conv.version_id,
        "order_id": order["id"],
        "job_ids": job_ids,
        "requested_address": address,
        "persisted_address": actual,
        "public_fixture_unchanged": canonical == order["data"]["address"],
        "user_profile_unchanged": profile == canonical_profile,
        "revision": conv.revision,
        "active_job": conv.active_job,
        "messages": conv.messages,
        "passed": actual == address
        and profile == canonical_profile
        and canonical == order["data"]["address"]
        and conv.revision == 2
        and conv.active_job is None,
    }
(ROOT / "docs/chat-state-verification.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2)
)
if not evidence["passed"]:
    raise RuntimeError("The Agent did not persist the requested order change; see evidence.")
print("真实订单地址修改已持久化，公开数据镜像保持原样。")
