from types import SimpleNamespace

import pytest
from ace import Skillbook
from tau2.domains.retail.environment import get_environment

from app.config import ROOT
from app.harness import validate_code
from app.native_bridge import sanitize
from app.native_harness import load_native_class, make_native_agent


def test_native_versions_cannot_reuse_a_mixed_role_baseline(monkeypatch):
    from app import native_harness

    signature = {"agent": "gpt-6-luna", "user": "gpt-5.4", "learning": "gpt-5.4"}
    monkeypatch.setattr(native_harness, "model_signature", lambda: signature)
    mixed = native_harness.native_code_version("same agent source")
    signature.update(user="gpt-6-luna", learning="gpt-6-luna")
    assert native_harness.native_code_version("same agent source") != mixed


def test_full_python_state_and_architecture_are_editable_without_legacy_dsl():
    original = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    code = (
        original
        + """
@dataclass
class ExtendedState(HarnessState):
    confirmations: dict[str, bool] = field(default_factory=dict)

class HarnessAgent(HarnessAgent):
    def get_init_state(self, message_history=None):
        return ExtendedState(messages=list(message_history or []))

    @property
    def system_prompt(self):
        return super().system_prompt + "\\nArchitecture edit supported."
"""
    )
    with pytest.raises(ValueError):
        validate_code(code)
    cls = load_native_class(code)
    env = get_environment()
    agent = cls(
        tools=env.get_tools(), domain_policy=env.get_policy(), llm="openai:test", llm_args={}
    )
    assert agent.get_init_state().confirmations == {}
    assert "Architecture edit supported" in agent.system_prompt


def test_native_three_arms_use_complete_agent_and_canonical_ace_context():
    book = Skillbook()
    skill = book.add_skill(
        "harness",
        issue="Confirmation alone does not execute a change",
        insight="Execute the confirmed order tool before reporting completion.",
    )
    code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    version = {"harness_code": code, "skillbook": book.to_dict()}
    env = get_environment()
    for mode in ["baseline", "harness"]:
        agent = make_native_agent(env, version, mode)
        assert not agent.injected_skill_ids
        assert skill.insight not in agent.system_prompt
    learned = make_native_agent(env, version, "skills")
    assert learned.injected_skill_ids == [skill.id]
    assert skill.insight in learned.system_prompt
    from tau2.config import DEFAULT_MAX_STEPS

    assert learned.program["MAX_STEPS"] == DEFAULT_MAX_STEPS


def test_native_bridge_never_persists_api_secrets(monkeypatch):
    from app import native_bridge

    monkeypatch.setattr(native_bridge, "settings", SimpleNamespace(api_key="unit-secret"))
    clean = sanitize(
        {
            "api_key": "unit-secret",
            "nested": [{"authorization": "Bearer unit-secret", "text": "error with unit-secret"}],
            "tokens": 50,
        }
    )
    assert clean == {"nested": [{"text": "error with [redacted]"}], "tokens": 50}


def test_gate_audit_allows_only_proven_original_file_guard_early_exit():
    import pytest

    from app.full_rsi import gate_validation_value

    suite = {"tasks": [], "threshold": 0.8}
    gate = {
        "exit_code": 1,
        "suite_before": suite,
        "suite_after": suite,
        "file_guard_violations": ["forbidden.py"],
    }
    assert gate_validation_value(gate, []) is None
    with pytest.raises(ValueError, match="full validation"):
        gate_validation_value({**gate, "file_guard_violations": []}, [])
    with pytest.raises(ValueError, match="early file-guard"):
        gate_validation_value({**gate, "exit_code": 0}, [])
    # A regression failure still requires the original full test.
    assert (
        gate_validation_value(
            {**gate, "file_guard_violations": []},
            [{"split": "train", "mean_reward": 0}, {"split": "test", "mean_reward": 0.4}],
        )
        == 0.4
    )


def test_native_factory_uses_runner_environment():
    code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    agent = make_native_agent(
        None,
        {"harness_code": code, "skillbook": Skillbook().to_dict()},
        "harness",
        tools=[],
        domain_policy="Runner-specific policy",
    )
    assert agent.tools == []
    assert "Runner-specific policy" in agent.system_prompt


def test_native_source_cache_mismatch_is_rejected(tmp_path, monkeypatch):
    import hashlib

    from app import native_harness

    monkeypatch.setattr(native_harness, "ROOT", tmp_path)
    code = "# cache corruption regression\nclass HarnessAgent:\n def generate_next_message(self): pass\n"
    directory = tmp_path / ".runtime/native-modules"
    directory.mkdir(parents=True)
    (directory / (hashlib.sha256(code.encode()).hexdigest() + ".py")).write_text("wrong source")
    with pytest.raises(ValueError, match="cache"):
        load_native_class(code)


def test_fresh_install_initializes_full_upstream_baseline(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import bootstrap, db

    engine = create_engine("sqlite://")
    monkeypatch.setattr(bootstrap, "engine", engine)
    monkeypatch.setattr(db, "Session", sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(bootstrap, "raw_database", lambda: {})
    bootstrap.initialize()
    bootstrap.initialize()
    with db.transaction() as session:
        deployment = session.get(db.Deployment, 1)
        version = session.get(db.Version, deployment.version_id)
        assert version.metadata_json["runtime_kind"] == "auto_harness_full"
        assert (
            version.harness_code
            == (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
        )
    engine.dispose()
