"""Retry malformed empty customer responses without replaying conversation tools."""
import json
import threading
import time

from .config import ROOT, settings

_lock = threading.Lock()


def guarded_completion(completion, *args, **kwargs):
    model = kwargs.get("model", "")
    if model != "openai/" + settings.agent_model:
        return completion(*args, **kwargs)
    for attempt in range(3):
        response = completion(*args, **kwargs)
        choices = getattr(response, "choices", None)
        message = choices[0].message if choices else None
        if message and (getattr(message, "content", None) or getattr(message, "tool_calls", None)):
            return response
        record = {"time": time.time(), "model": model, "attempt": attempt + 1,
                  "response": response.model_dump(mode="json"),
                  "message_count": len(kwargs.get("messages", []))}
        with _lock:
            path = ROOT / ".runtime/transport-audits/empty-customer-responses.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
        if attempt < 2:
            time.sleep(1)
    raise ConnectionError("Customer provider returned empty content and tool calls on 3 identical requests")


def install():
    from tau2.utils import llm_utils

    with _lock:
        if getattr(llm_utils.completion, "_empty_response_guard", False):
            return
        original = llm_utils.completion

        def completion(*args, **kwargs):
            return guarded_completion(original, *args, **kwargs)

        completion._empty_response_guard = True
        llm_utils.completion = completion
