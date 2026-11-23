"""Quarantine observable protocol failures without changing official rewards.

This is a conservative diagnostic, not a judge of business correctness. An
unflagged conversation can still deviate from its scenario and require review.
"""

from __future__ import annotations

import re


class EvaluationValidityError(ValueError):
    pass


_IDENTITY = re.compile(
    r"(?:can(?:not|'t|’t)|won['’]t|will not|must decline|prohibit)[\s\S]{0,260}"
    r"(?:core (?:identity|guidelines|instructions)|core system instructions|"
    r"role[- ]play|simulate (?:this|the|any|customer)|"
    r"(?:override|adopt|assume|switch)[\s\S]{0,80}(?:identity|persona|role))"
    r"|(?:core (?:identity|instructions)|core system instructions)[\s\S]{0,180}"
    r"(?:cannot|can't|will not|prohibit|decline)"
    r"|I['’]m an AI assistant in this chat only, not a customer service representative",
    re.IGNORECASE,
)


def audit_protocol(data):
    issues = []
    if (data.get("info") or {}).get("error_type") == "SimulatorProtocolError":
        issues.append({"kind": "simulator_draft_validation_failed",
                       "reason": "Frozen simulator could not produce a valid utterance"})
    messages = data.get("messages", [])
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    for i, message in enumerate(messages):
        content = message.get("content") or ""
        if not isinstance(content, str) or message.get("role") not in {"assistant", "user"}:
            continue
        if _IDENTITY.search(content):
            issues.append(
                {
                    "kind": "role_identity_refusal",
                    "message_index": i,
                    "role": message["role"],
                    "excerpt": content[:500],
                }
            )
        if user_indices and i == user_indices[0] and content.strip() == "###STOP###":
            issues.append(
                {
                    "kind": "user_stopped_before_request",
                    "message_index": i,
                    "role": "user",
                    "excerpt": content,
                }
            )
    return {
        "status": "invalid" if issues else "not_flagged",
        "audit_version": 1,
        "issues": issues,
        "business_score_eligible": not bool(issues),
    }


def require_scoreable(data):
    recorded = data.get("evaluation_validity") or {}
    audit = audit_protocol(data)
    if recorded.get("status") == "invalid" or audit["status"] == "invalid":
        raise EvaluationValidityError(
            "Invalid observation: role/simulator protocol failure; "
            "preserve raw reward but do not score or learn from this observation"
        )
    return audit
