from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from app import api


def test_readiness_distinguishes_process_health_from_dependency_failure(monkeypatch):
    monkeypatch.setattr(api, "health", lambda: {"services": {"mysql": True, "redis": False}})
    assert api.live() == {"alive": True}
    with pytest.raises(HTTPException) as exc:
        api.ready()
    assert exc.value.status_code == 503
    monkeypatch.setattr(api, "health", lambda: {"services": {"mysql": True, "redis": True}})
    assert api.ready() == {"ready": True}


def test_k8s_storage_routing_and_model_isolation():
    root = Path(__file__).resolve().parents[1]
    objects = [obj for path in (root / "deploy/k8s").glob("*.yaml") for obj in yaml.safe_load_all(path.read_text())]
    indexed = {(obj["kind"], obj["metadata"]["name"]): obj for obj in objects}
    worker = indexed["Deployment", "worker"]["spec"]
    assert worker["replicas"] == 1
    assert worker["strategy"]["type"] == "Recreate"
    pod = worker["template"]["spec"]
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "worker-state"
    gateway = next(c for c in pod["containers"] if c["name"] == "gateway")
    assert "127.0.0.1" in gateway["command"]
    cfg = indexed["ConfigMap", "retail-config"]["data"]
    assert {cfg[k] for k in ["AGENT_MODEL", "USER_MODEL", "LEARNING_MODEL"]} == {"gpt-6-luna"}
    assert cfg["INITIALIZE_DATABASE"] == "0"
    assert not any(obj["kind"] == "Secret" for obj in objects)
    for name in ["mysql", "redis"]:
        assert indexed["StatefulSet", name]["spec"]["volumeClaimTemplates"]
    service = indexed["Service", "api"]["spec"]
    assert service["selector"] == indexed["Deployment", "api"]["spec"]["selector"]["matchLabels"]
