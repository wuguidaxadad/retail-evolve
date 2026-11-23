import hashlib
import json

import pytest

from app import full_rsi


def test_resume_rejects_changed_runtime_before_touching_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(full_rsi, "ROOT", tmp_path)
    monkeypatch.setattr(full_rsi, "model_signature", lambda: {"runtime_adapter": 5})
    run = tmp_path / ".runtime/full-rsi/old-run"
    run.mkdir(parents=True)
    (run / "setup.json").write_text(
        json.dumps(
            {
                "upstream_commit": full_rsi.UPSTREAM_COMMIT,
                "model_config": {"runtime_adapter": 4},
            }
        )
    )
    with pytest.raises(ValueError, match="configuration changed"):
        full_rsi.setup({"id": "new-job", "payload": {"resume_run_id": "old-run"}})
    assert not (run / "repo").exists()
    assert not (run / "adapter").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"code_hash": "different-source"},
        {"process_id": "different-command"},
        {"complete": False},
    ],
)
def test_gate_rejects_unrelated_or_incomplete_evaluation(change):
    gate = {"id": "gate-command", "code_hash": "candidate-source"}
    phase = {"complete": True, "process_id": gate["id"], "code_hash": gate["code_hash"]}
    full_rsi.validate_gate_sources(gate, [phase])
    with pytest.raises(ValueError, match="evaluated source/command"):
        full_rsi.validate_gate_sources(gate, [{**phase, **change}])


def test_final_source_cannot_change_after_successful_evaluation(tmp_path):
    (tmp_path / "agent").mkdir()
    source = tmp_path / "agent/agent.py"
    source.write_text("accepted candidate")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    full_rsi.validate_final_source(tmp_path, digest)
    source.write_text("different candidate that has not passed evaluation")
    with pytest.raises(ValueError, match="last accepted"):
        full_rsi.validate_final_source(tmp_path, digest)
