"""Execute the complete upstream HarnessAgent source, without the legacy AST DSL."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading

from ace import Skillbook
from ace.implementations.prompts import wrap_skillbook_for_external_agent

from .config import ROOT, agent_model_route, model_signature


def native_code_version(code):
    import json

    return hashlib.sha256(
        (json.dumps(model_signature(), sort_keys=True) + "\n" + code).encode()
    ).hexdigest()[:32]


_load_lock = threading.RLock()


def load_native_class(code):
    from .transport_guard import install

    install()
    with _load_lock:
        return _load_native_class(code)


def _load_native_class(code):
    compile(code, "agent/agent.py", "exec")
    digest = hashlib.sha256(code.encode()).hexdigest()
    name = "native_harness_" + digest
    if name not in sys.modules:
        directory = ROOT / ".runtime/native-modules"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (digest + ".py")
        if path.exists() and path.read_text() != code:
            raise ValueError("Native source cache does not match requested code")
        if not path.exists():
            path.write_text(code)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    cls = getattr(sys.modules[name], "HarnessAgent", None)
    if cls is None or not callable(getattr(cls, "generate_next_message", None)):
        raise ValueError("完整 Agent 必须保留 HarnessAgent 接口")
    return cls


def selected_source(version, mode):
    return (
        version["harness_code"]
        if mode in {"harness", "combined"}
        else (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    )


def make_native_agent(environment, version, mode, *, tools=None, domain_policy=None):
    code = selected_source(version, mode)
    cls = load_native_class(code)
    book = Skillbook.from_dict(version["skillbook"])
    insights = (
        [s for s in book.skills() if s.active and s.insight]
        if mode in {"skills", "combined"}
        else []
    )
    allowed = {s.id for s in insights}
    for skill in list(book.skills(include_invalid=True)):
        if skill.id not in allowed:
            book.remove_skill(skill.id)
    context = wrap_skillbook_for_external_agent(book) if insights else ""
    if context:
        base = cls

        class WithACESkills(base):
            @property
            def system_prompt(self):
                return super().system_prompt + "\n\n" + context

        cls = WithACESkills
    from .agent import llm_args

    agent = cls(
        tools=environment.get_tools() if tools is None else tools,
        domain_policy=environment.get_policy() if domain_policy is None else domain_policy,
        llm=agent_model_route(),
        llm_args=llm_args(),
    )
    agent.executed_source_hash = hashlib.sha256(code.encode()).hexdigest()
    agent.injected_skill_ids = [s.id for s in insights]
    from tau2.config import DEFAULT_MAX_STEPS

    agent.program = {"MAX_STEPS": DEFAULT_MAX_STEPS}
    return agent


def make_agent(environment, version, mode="combined"):
    if version.get("metadata", {}).get("runtime_kind") == "auto_harness_full":
        return make_native_agent(environment, version, mode)
    from .agent import RetailAgent

    return RetailAgent(environment, version, mode)


def bind_source_snapshot(code):
    """Register the exact source snapshot for an upstream benchmark invocation."""
    from tau2.registry import registry

    cls = load_native_class(code)
    name = "rsi_snapshot_" + hashlib.sha256(code.encode()).hexdigest()
    if registry.get_agent_factory(name) is None:

        def factory(tools, domain_policy, **kwargs):
            return cls(
                tools=tools,
                domain_policy=domain_policy,
                llm=kwargs.get("llm"),
                llm_args=kwargs.get("llm_args"),
            )

        registry.register_agent_factory(factory, name)
    return name
