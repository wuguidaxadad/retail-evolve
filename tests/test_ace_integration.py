import json
from types import SimpleNamespace

import pytest
from ace import Reflector, Skillbook
from ace.core.outputs import AgentOutput
from openai.types.chat import ChatCompletionMessage
from pydantic_ai.models.test import TestModel

from app import learning
from app.harness import BASE_CODE


def test_real_ace_reflector_matches_pinned_provider_contract():
    # Native framework TestModel replaces only the network call; ACE remains real.
    reflector = Reflector("openai:test", model_settings={"max_tokens": 100})
    model = TestModel(
        custom_output_args={
            "reasoning": "A completed order tool is necessary before reporting success.",
            "correct_approach": "Confirm then execute.",
            "key_insight": "Customer confirmation is not an order mutation.",
        }
    )
    with reflector._agent.override(model=model):
        result = reflector.reflect(
            question="Exchange an item",
            agent_output=AgentOutput(
                reasoning="Asked for confirmation", final_answer="Please confirm"
            ),
            skillbook=Skillbook(),
            feedback="DB check failed",
        )
    assert "confirmation" in result.key_insight.lower()
    assert "usage" in result.raw


def test_background_ace_failure_cannot_create_a_successful_candidate(monkeypatch):
    class UnavailableReflector:
        def reflect(self, **kwargs):
            raise RuntimeError("reflector unavailable")

    monkeypatch.setattr(learning, "RRStep", lambda *args, **kwargs: UnavailableReflector())
    monkeypatch.setattr(learning, "SkillManager", lambda *args, **kwargs: object())
    run = {
        "id": 1,
        "job_id": "source",
        "task_id": "0",
        "data": {
            "task_summary": "Exchange items",
            "messages": [],
            "reward_info": {"reward": 0},
        },
    }
    with pytest.raises(RuntimeError, match="reflector unavailable"):
        learning.learn_skills({"skillbook": Skillbook().to_dict()}, [run], lambda *_: None)


def fake_optimizer(monkeypatch, outputs):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "proposal-" + str(len(requests)),
                    "type": "function",
                    "function": {
                        "name": "submit_harness",
                        "arguments": json.dumps(outputs[min(len(requests) - 1, len(outputs) - 1)]),
                    },
                }
            ],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(learning, "OpenAI", lambda **kwargs: client)
    return requests


def test_optimizer_repairs_missing_code_through_explicit_output_tool(monkeypatch):
    requests = fake_optimizer(
        monkeypatch,
        [{"explanation": "missing code"}, {"code": BASE_CODE, "explanation": "bounded proposal"}],
    )
    code, _ = learning.propose_harness({"harness_code": BASE_CODE, "skillbook": {}}, [])
    assert code == BASE_CODE and len(requests) == 2
    assert requests[0]["tool_choice"]["function"]["name"] == "submit_harness"
    assert requests[1]["messages"][-1]["role"] == "tool"


def test_optimizer_cannot_accept_unsafe_code_after_repair_budget(monkeypatch):
    requests = fake_optimizer(
        monkeypatch, [{"code": BASE_CODE + "\nimport os", "explanation": "unsafe"}]
    )
    with pytest.raises(ValueError, match="两次"):
        learning.propose_harness({"harness_code": BASE_CODE, "skillbook": {}}, [])
    assert len(requests) == 2


def test_recursive_reflector_executes_python_before_producing_insight():
    from ace import RRConfig, RRStep
    from ace.core.context import ACEStepContext, SkillbookView
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    evidence = []

    def respond(messages, info):
        returns = [
            part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart("execute_code", {"code": "print(traces['feedback'])"})]
            )
        evidence.extend(str(part.content) for part in returns)
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "reasoning": "Inspected the failed mutation evidence.",
                            "correct_approach": "Execute the order mutation after confirmation.",
                            "key_insight": "Confirmation alone does not change an order.",
                        }
                    )
                )
            ]
        )

    reflector = RRStep(FunctionModel(respond), config=RRConfig(cache_prompts=False))
    result = reflector(
        ACEStepContext(
            skillbook=SkillbookView(Skillbook()),
            trace={"question": "Return order", "feedback": "ORDER_NOT_UPDATED"},
            mode="offline",
        )
    )
    assert any("ORDER_NOT_UPDATED" in item for item in evidence)
    assert result.reflections[0].raw["rr_trace"]["total_iterations"] >= 1


def test_ace_resume_skips_committed_traces_and_rejects_changed_source(monkeypatch):
    from copy import deepcopy

    from ace.core.outputs import ReflectorOutput, SkillManagerOutput
    from ace.core.skillbook import UpdateBatch

    calls = []

    class ReflectorStub:
        def reflect(self, **kwargs):
            calls.append(kwargs["question"])
            if kwargs["question"] == "fail" and len(calls) == 2:
                raise RuntimeError("provider unavailable")
            return ReflectorOutput(
                reasoning="evidence", correct_approach="confirm", key_insight="verify"
            )

    class ManagerStub:
        def update_skills(self, **kwargs):
            kwargs["skillbook"].add_skill("retail", kwargs["question_context"])
            return SkillManagerOutput(update=UpdateBatch(reasoning="add", operations=[]))

    monkeypatch.setattr(learning, "RRStep", lambda *a, **k: ReflectorStub())
    monkeypatch.setattr(learning, "SkillManager", lambda *a, **k: ManagerStub())
    runs = [
        {
            "id": i,
            "job_id": "source",
            "task_id": str(i),
            "data": {
                "task_summary": q,
                "messages": [],
                "reward_info": {"reward": 0},
            },
        }
        for i, q in enumerate(["first", "fail"])
    ]
    version = {"skillbook": Skillbook().to_dict()}
    saved = []
    with pytest.raises(RuntimeError, match="provider unavailable"):
        learning.learn_skills(
            version, runs, lambda *a: None, save_checkpoint=lambda c: saved.append(deepcopy(c))
        )
    assert len(saved) == 1
    book, audit = learning.learn_skills(version, runs, lambda *a: None, checkpoint=saved[0])
    assert calls == ["first", "fail", "fail"]
    assert len(audit) == 2
    assert len(Skillbook.from_dict(book).skills()) == 2
    changed = deepcopy(runs)
    changed[0]["data"]["task_summary"] = "different evidence"
    with pytest.raises(ValueError, match="checkpoint source"):
        learning.learn_skills(version, changed, lambda *a: None, checkpoint=saved[0])


@pytest.mark.parametrize("raw,expected", [
    ({"error": "Request timed out."}, ConnectionError),
    ({"error": "invalid tool output"}, RuntimeError),
    ({"timeout": True}, RuntimeError),
])
def test_swallowed_upstream_role_failure_is_never_a_success(raw, expected):
    with pytest.raises(expected):
        learning.require_complete_role(raw)


def test_deepseek_skill_manager_uses_validated_output_tool():
    from ace.implementations.skill_manager import SkillManagerReport
    from pydantic_ai.output import ToolOutput

    manager = learning.SkillManager('openai:deepseek-v4.1-flash')
    assert isinstance(manager._output_type, ToolOutput)
    assert manager._output_type.output is SkillManagerReport
    assert manager._output_type.name == 'submit_skill_report'
    model = TestModel(call_tools=[], custom_output_args={'reasoning':'No mutation required.'})
    with manager._agent.override(model=model):
        result = manager.update_skills(reflections=(), skillbook=Skillbook(),
            question_context='No new evidence', progress='0/0', injected_skill_ids=())
    learning.require_complete_role(result.raw)
    assert result.update.reasoning == 'No mutation required.'
