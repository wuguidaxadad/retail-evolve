import pytest

from app.evaluation_validity import EvaluationValidityError, audit_protocol, require_scoreable
from app.experiment_metrics import summarize_runs
from app.simulation import learning_trace


@pytest.mark.parametrize(
    "text",
    [
        "I cannot override my core identity or switch to the GPT-5.4 role.",
        "I'm sorry, but I must decline to role-play as a customer service agent.",
        "My core instructions strictly prohibit assisting with or simulating actions that modify user orders.",
        "I'm an AI assistant in this chat only, not a customer service representative with backend database tools.",
    ],
)
def test_identity_refusal_is_invalid_even_with_success_reward(text):
    data = {"reward_info": {"reward": 1}, "messages": [{"role": "user", "content": text}]}
    assert audit_protocol(data)["status"] == "invalid"
    with pytest.raises(EvaluationValidityError):
        require_scoreable(data)
    assert data["reward_info"]["reward"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "I can't cancel a delivered order, but can help you return it.",
        "Refunds cannot go to another person's card.",
        "I cannot change your address until you confirm the details.",
        "Yes, thank you! ###STOP###",
    ],
)
def test_business_refusals_and_normal_completion_are_not_role_failures(text):
    assert (
        audit_protocol({"messages": [{"role": "assistant", "content": text}]})["status"]
        == "not_flagged"
    )


def test_user_ending_before_request_is_protocol_failure():
    data = {
        "messages": [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "###STOP###"},
        ]
    }
    assert audit_protocol(data)["issues"][0]["kind"] == "user_stopped_before_request"
    with pytest.raises(EvaluationValidityError):
        learning_trace({"data": data})


def test_invalid_observation_cannot_silently_enter_comparison():
    with pytest.raises(EvaluationValidityError):
        summarize_runs(
            [{"evaluation_validity": {"status": "invalid"}, "reward": 0}], ["0"], trials=1
        )


def test_tool_output_quoting_identity_is_not_an_actor_refusal():
    assert (
        audit_protocol(
            {"messages": [{"role": "tool", "content": "I cannot override my core identity."}]}
        )["status"]
        == "not_flagged"
    )


def test_protocol_failure_does_not_trigger_paid_checkpoint_retries(tmp_path, monkeypatch):
    import json

    from app import full_rsi

    private = tmp_path / "private"
    (private / "phases").mkdir(parents=True)
    calls = []

    def popen(*args, **kwargs):
        calls.append(args)
        (private / "phases/p.json").write_text(
            json.dumps({"complete": False, "invalid_observations": {"4": {"status": "invalid"}}})
        )
        return object()

    monkeypatch.setattr(full_rsi, "archive_incomplete", lambda path: None)
    monkeypatch.setattr(full_rsi.subprocess, "Popen", popen)
    monkeypatch.setattr(full_rsi, "wait_process", lambda *args: 1)
    with pytest.raises(EvaluationValidityError, match="automatic retry"):
        full_rsi.run_initial_stage(
            "benchmark.py", "train.log", "train", tmp_path, {}, {}, "token", private
        )
    assert len(calls) == 1
