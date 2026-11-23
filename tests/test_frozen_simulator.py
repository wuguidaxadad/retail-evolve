import hashlib
from dataclasses import replace

import pytest

from app import frozen_simulator as frozen
from app.config import ROOT
from app.evaluation_validity import audit_protocol

DIGEST = "784c9e706ca8fe785a2ba63f8746715ee4e131c4108bae87753ef46927e005e9"


def test_frozen_policy_and_repeat_registration(monkeypatch):
    monkeypatch.setattr(frozen, "settings", replace(frozen.settings,
        simulator_sha256=DIGEST, simulator_protocol="scenario-faithfulness-v10"))
    source = ROOT / "artifacts/frozen-simulators" / DIGEST / "simulator.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == DIGEST
    assert frozen.register_simulator() == frozen.register_simulator()
    module, cls = frozen._load(DIGEST)
    assert cls.__mro__[1] is module.ScenarioFaithfulUser
    assert frozen.is_simulator_error(module.SimulatorProtocolError("invalid draft"))
    assert not frozen.is_simulator_error(ValueError("other failure"))


def test_tampering_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "settings", replace(frozen.settings,
        simulator_sha256=DIGEST, simulator_protocol="scenario-faithfulness-v10"))
    path = tmp_path / "artifacts/frozen-simulators" / DIGEST / "simulator.py"
    path.parent.mkdir(parents=True)
    path.write_text("tampered")
    monkeypatch.setattr(frozen, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        frozen.register_simulator()


def test_exhausted_draft_is_invalid_observation():
    result = audit_protocol({"info": {"error_type": "SimulatorProtocolError"}, "messages": []})
    assert result["status"] == "invalid"


def test_formal_api_accepts_hy3_optimizer():
    from app.api import JobRequest
    request = JobRequest(kind="rsi", version_id="test-baseline", optimizer_model="hy3")
    assert request.optimizer_model == "hy3"
