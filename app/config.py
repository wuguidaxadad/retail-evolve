from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ADAPTER_VERSION = 9
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT / "vendor/tau2/data"))
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("DO_NOT_TRACK", "1")


@dataclass(frozen=True)
class Settings:
    agent_transport: str = os.getenv("AGENT_TRANSPORT", "chat")
    experiment_profile: str = os.getenv("EXPERIMENT_PROFILE", "official")
    optimizer_model: str = os.getenv("OPTIMIZER_MODEL", os.getenv("LEARNING_MODEL", "gpt-4o-mini"))
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("LEARNING_MODEL", "gpt-4o-mini"))
    simulator_sha256: str = os.getenv("SIMULATOR_SHA256", "")
    simulator_protocol: str = os.getenv("SIMULATOR_PROTOCOL", "upstream")
    mysql_url: str = os.getenv(
        "MYSQL_URL",
        "mysql+pymysql://retail:retail_local@127.0.0.1:3307/retail_evolve?charset=utf8mb4",
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6381/0")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    agent_model: str = os.getenv("AGENT_MODEL", "gpt-4o-mini")
    user_model: str = os.getenv("USER_MODEL", os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    learning_model: str = os.getenv("LEARNING_MODEL", os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    lease_seconds: int = int(os.getenv("JOB_LEASE_SECONDS", "180"))
    max_attempts: int = int(os.getenv("MAX_JOB_ATTEMPTS", "3"))
    max_tasks: int = int(os.getenv("MAX_TASKS_PER_JOB", "12"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    stream: str = "retail:jobs"
    group: str = "retail-workers"
    dead_stream: str = "retail:dead"


settings = Settings()


def model_signature():
    from .frozen_simulator import simulator_signature

    return {
        "agent": settings.agent_model,
        "user": settings.user_model,
        "learning": settings.learning_model,
        "optimizer": settings.optimizer_model,
        "agent_transport": settings.agent_transport,
        "judge": settings.judge_model,
        "experiment_profile": settings.experiment_profile,
        "split_sha256": __import__("hashlib").sha256((ROOT / "configs/aligned_retail_split.json").read_bytes()).hexdigest() if settings.experiment_profile == "aligned-v1" else None,
        "provider": urlparse(settings.base_url).hostname,
        "runtime_adapter": RUNTIME_ADAPTER_VERSION,
        "simulator": simulator_signature(),
    }


def agent_model_route():
    if settings.agent_transport not in {"chat", "responses"}:
        raise ValueError("Unsupported agent transport")
    route = "openai/" + settings.agent_model
    if settings.agent_transport == "responses":
        # A recognized Gemini model overrides LiteLLM's responses/ prefix heuristic.
        # Register the exact customer route, preserving the real wire model name.
        import litellm
        litellm.register_model({route: {"mode": "responses", "litellm_provider": "openai"}})
    return route
