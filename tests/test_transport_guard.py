from types import SimpleNamespace as NS

import pytest

from app import transport_guard as guard


def reply(content=None, calls=None):
    r = NS(choices=[NS(message=NS(content=content, tool_calls=calls))])
    r.model_dump = lambda **kw: {"choices": [{"content": content, "tool_calls": calls}]}
    return r


def test_empty_retry_preserves_exact_request_and_tool_call(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    (tmp_path / ".runtime").mkdir()
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    seen = []
    responses = iter([reply(), reply(calls=[{"name": "write_order"}])])
    def complete(**kw):
        seen.append(kw)
        return next(responses)
    messages = [{"role": "user", "content": "confirm"}]
    result = guard.guarded_completion(complete, model="openai/" + guard.settings.agent_model, messages=messages)
    assert result.choices[0].message.tool_calls
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0]["messages"] is messages


def test_empty_exhaustion_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    (tmp_path / ".runtime").mkdir()
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    calls = []
    def complete(**kw):
        calls.append(kw)
        return reply()
    with pytest.raises(ConnectionError, match="3 identical"):
        guard.guarded_completion(complete, model="openai/" + guard.settings.agent_model)
    assert len(calls) == 3


def test_nonempty_business_answer_is_never_retried():
    calls = []
    def complete(**kw):
        calls.append(kw)
        return reply("I cannot do that")
    guard.guarded_completion(complete, model="openai/" + guard.settings.agent_model)
    assert len(calls) == 1
