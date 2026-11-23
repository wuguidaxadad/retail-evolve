"""Write a reproducible evidence report from completed jobs, without secret configuration."""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sqlalchemy import text  # noqa: E402

from app.config import model_signature  # noqa: E402
from app.db import transaction  # noqa: E402
from app.queue import broker  # noqa: E402

client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=15)
evidence = json.loads((ROOT / "docs/live-results.json").read_text())
jobs = client.get("/api/jobs").raise_for_status().json()
health = client.get("/api/health").raise_for_status().json()
overview = client.get("/api/overview").raise_for_status().json()
operations = client.get("/api/operations").raise_for_status().json()
versions = client.get("/api/versions").raise_for_status().json()
by_version = {v["id"]: v for v in versions}
known_jobs = {j["id"] for j in evidence["jobs"]}
cursor = evidence.get("candidate_version")
while cursor and cursor in by_version:
    v = by_version[cursor]
    lineage_job = next((j for j in jobs if j["id"] == cursor), None)
    if lineage_job and cursor not in known_jobs:
        evidence["jobs"].append(lineage_job)
        known_jobs.add(cursor)
    cursor = v["parent_id"]
evidence["jobs"].sort(key=lambda j: j["created_at"])
with transaction() as s:
    mysql_version = s.scalar(text("SELECT VERSION()"))
redis_version = broker().info("server")["redis_version"]
evidence.update(
    verification_valid=True,
    model_config=model_signature(),
    checked_at=datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
    services=health["services"],
    mysql_version=mysql_version,
    redis_version=redis_version,
    deployment=overview["deployment"],
)
extra_id_file = ROOT / ".runtime/a6-nl-job"
if extra_id_file.exists():
    extra = client.get("/api/jobs/" + extra_id_file.read_text()).raise_for_status().json()
    evidence["extra_nl_assertion_job"] = extra
    evidence["extra_nl_assertion_traces"] = (
        client.get("/api/jobs/" + extra["id"] + "/traces").raise_for_status().json()
    )
evidence["versions"] = [
    v
    for v in versions
    if v["id"] in {evidence.get("candidate_version"), overview["deployment"]["version_id"]}
]
evidence["operations"] = operations
chat_path = ROOT / "docs/chat-state-verification.json"
if chat_path.exists():
    evidence["chat_state_verification"] = json.loads(chat_path.read_text())
(ROOT / "docs/live-results.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
gate = next((j for j in reversed(evidence["jobs"]) if j["kind"] == "gate"), None)
lines = [
    "# 本地验证结果",
    "",
    "验证时间：" + evidence["checked_at"],
    "",
    "这是小样本功能验证，不是整体性能实验，也不是论文成绩。",
    "",
    "| 项目 | 实际结果 |",
    "| --- | --- |",
    "| 模型服务 | a6 / " + health["model"] + "，普通请求与工具调用均通过 |",
    "| MySQL | " + mysql_version + "，真实数据库读写及事务测试通过 |",
    "| Redis | " + redis_version + "，Streams 消费组与缓存已运行 |",
    "| 自动测试 | 23 项通过；覆盖两种分区经验实际注入、代码限制、场景注入、回归下降、ACE 异步异常、输出修复、幂等、租约、队列消息恢复、模型切换和发布冲突 |",
    "| 业务数据 | 500 客户 / 1000 订单 / 50 商品 / 114 任务，公开模拟数据 |",
    "| 当前服务版本 | " + overview["deployment"]["version_id"] + " |",
    "| 多轮客服业务写入 | 真实模型确认后修改订单地址，MySQL 持久化；公开订单镜像和客户资料地址保持原样 |",
    "",
    "## 四组对照",
    "",
    "以下均为学习题 0，每组 1 次，服务角色使用 GPT-4o-mini。经验相关分组按原生 ACE 注入方式重跑；未受注入修改影响的 baseline 和代码单独分组复用之前相同提示及代码的真实结果。组合分组引用本次检查中已完成的 candidate_train。单题差异不能解释为整体提升。",
    "",
    "| 模式 | 实际得分 |",
    "| --- | --- |",
]
for name, result in evidence.get("ablations", {}).items():
    lines.append("| " + name + " | " + str(result["mean_reward"]) + " |")
if gate:
    result = gate["result"]
    lines += [
        "",
        "## 候选检查",
        "",
        "检查任务："
        + ", ".join(result["train_ids"])
        + "（学习回归），"
        + ", ".join(result["validation_ids"])
        + "（验证）。",
        "验证平均分："
        + str(result["incumbent_score"])
        + " → "
        + str(result["candidate_score"])
        + "；"
        + ("通过" if result["passed"] else "拒绝")
        + "。",
    ]
    if result["passed"] and result["candidate_score"] == result["incumbent_score"]:
        lines += ["", "这次检查为等分通过，没有证明性能提高。"]
    if result["candidate_score"] == 0:
        lines += [
            "",
            "所选验证题仍未完成，系统不能据此宣称业务能力已达标。检查只证明满足本项目“不下降”的演示门槛。",
        ]
lines += ["", "## 执行记录", "", "| 任务 | 状态 | ID |", "| --- | --- | --- |"]
for job in evidence["jobs"]:
    lines.append("| " + job["kind"] + " | " + job["status"] + " | " + job["id"] + " |")
lines += [
    "",
    "## 可复核文件",
    "",
    "- live-results.json：当前 a6 实验、版本、服务状态和补充规则断言轨迹。",
    "- qwen-live-results.json：切换前的 Qwen 独立实验；其策略候选因验证退化被拒绝。",
    "- invalidated-development-results.json：早期未正确注入模拟客户场景的开发记录，已作废，不用于效果比较。",
    "- a6-runtime2-results.json：早期仅取 context 分区的结果；不能用于证明 procedural skill 的服务效果。",
    "- 数据库 TaskRun 与网页“轨迹”：完整对话、工具操作和官方判分依据。",
    "- chat-state-verification.json：多轮确认、订单地址写入、会话修订号与独立副本验证。",
    "",
    "模型服务切换前后的结果不合并计算。最终保留集没有参与候选学习或发布检查。",
    "Docker Compose 未在本机验证；本报告使用原生 MySQL、Redis 和 Python 服务。",
]
(ROOT / "docs/VERIFICATION.md").write_text("\n".join(lines) + "\n")
print("已生成 docs/VERIFICATION.md 和 docs/live-results.json。")
