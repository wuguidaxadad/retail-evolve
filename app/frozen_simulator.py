"""Content-addressed simulator policy shared by every formal experiment arm."""

import hashlib
import importlib.util
import json
import sys
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from .config import ROOT, settings

_audit_scope = ContextVar("simulator_audit_scope", default="unscoped")
_lock = threading.RLock()


def simulator_signature():
    digest = settings.simulator_sha256
    if not digest:
        return {"protocol": "upstream", "sha256": None}
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid frozen simulator digest")
    path = ROOT / "artifacts/frozen-simulators" / digest / "simulator.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen simulator hash mismatch; refusing mixed evaluation")
    return {"protocol": settings.simulator_protocol, "sha256": digest}


@lru_cache(maxsize=4)
def _load(digest):
    path = ROOT / "artifacts/frozen-simulators" / digest / "simulator.py"
    name = "retail_frozen_simulator_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.SIMULATOR_PROTOCOL_VERSION != settings.simulator_protocol:
        raise ValueError("Frozen simulator protocol version mismatch")

    class AuditedUser(module.ScenarioFaithfulUser):
        @staticmethod
        def save_audit(audit, accepted):
            folder = ROOT / "artifacts/simulator-audits" / digest / _audit_scope.get()
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (uuid.uuid4().hex + ".json")).write_text(json.dumps(
                {"simulator": simulator_signature(), "accepted": accepted, "attempts": audit},
                ensure_ascii=False, indent=2,
            ))

    return module, AuditedUser


def register_simulator():
    signature = simulator_signature()
    if signature["sha256"] is None:
        return "user_simulator"
    from tau2.registry import registry

    name = "frozen_user_" + signature["sha256"]
    with _lock:
        _, cls = _load(signature["sha256"])
        if name not in registry._users:
            registry.register_user(cls, name)
        elif registry._users[name] is not cls:
            raise ValueError("Frozen simulator registry collision")
    return name


def is_simulator_error(exc):
    if not settings.simulator_sha256:
        return False
    module, _ = _load(settings.simulator_sha256)
    return isinstance(exc, module.SimulatorProtocolError)


@contextmanager
def simulator_scope(label):
    token = _audit_scope.set(label)
    try:
        yield
    finally:
        _audit_scope.reset(token)
