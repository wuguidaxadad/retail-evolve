from __future__ import annotations

from ace import Skillbook
from sqlalchemy import func, select

from .config import ROOT, model_signature
from .dataset import raw_database
from .db import Base, Deployment, RetailRecord, Version, engine, transaction
from .native_harness import native_code_version


def initialize():
    Base.metadata.create_all(engine)
    with transaction() as session:
        if not session.get(Deployment, 1):
            code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
            baseline = Version(
                id=native_code_version(code),
                name="原始客服 Agent",
                origin="baseline",
                status="validated",
                harness_code=code,
                skillbook=Skillbook().to_dict(),
                metadata_json={
                    "description": "τ 零售客服 + auto-harness 原生完整模板",
                    "runtime_kind": "auto_harness_full",
                    "model_config": model_signature(),
                },
            )
            session.add(baseline)
            session.flush()
            session.add(Deployment(id=1, version_id=baseline.id, revision=0))
        if not session.scalar(select(func.count()).select_from(RetailRecord)):
            for kind, records in raw_database().items():
                session.add_all([RetailRecord(kind=kind, id=k, data=v) for k, v in records.items()])


if __name__ == "__main__":
    initialize()
    print("MySQL schema, public retail records, and baseline version initialized.")
