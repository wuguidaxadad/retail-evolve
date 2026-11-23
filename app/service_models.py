"""Product data is independent from benchmark datasets and experiment checkpoints."""

from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, identifier, now


class ServiceRecord(Base):
    __tablename__ = "service_records"
    kind: Mapped[str] = mapped_column(String(24), primary_key=True)
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    revision: Mapped[int] = mapped_column(Integer, default=0)


class Ticket(Base):
    __tablename__ = "service_tickets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    title: Mapped[str] = mapped_column(String(160))
    order_id: Mapped[str] = mapped_column(String(80), index=True)
    customer_id: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    priority: Mapped[str] = mapped_column(String(16), default="normal")
    owner: Mapped[str] = mapped_column(String(80), default="Agent")
    version_id: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(16), default="combined")
    messages: Mapped[list] = mapped_column(JSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    active_turn: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime, default=now)
    updated_at: Mapped[object] = mapped_column(DateTime, default=now, onupdate=now)


class ServiceTurn(Base):
    __tablename__ = "service_turns"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    ticket_id: Mapped[str] = mapped_column(String(32), index=True)
    key: Mapped[str] = mapped_column(String(128), unique=True)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    stage: Mapped[str] = mapped_column(String(160), default="等待客服执行")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace: Mapped[list] = mapped_column(JSON, default=list)
    lease: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lease_until: Mapped[object | None] = mapped_column(DateTime, nullable=True)
    published: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[object] = mapped_column(DateTime, default=now)


class SupportConversation(Base):
    __tablename__ = "support_conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    token_hash: Mapped[str] = mapped_column(String(64))
    customer_id: Mapped[str] = mapped_column(String(80))
    version_id: Mapped[str] = mapped_column(String(32))
    next_seq: Mapped[int] = mapped_column(Integer, default=1)


class SupportMessage(Base):
    __tablename__ = "support_messages"
    __table_args__ = (UniqueConstraint("conversation_id", "seq"), UniqueConstraint("conversation_id", "key"))
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    conversation_id: Mapped[str] = mapped_column(String(32), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    reply: Mapped[str] = mapped_column(Text, default="")
    lease: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lease_until: Mapped[object | None] = mapped_column(DateTime, nullable=True)
    recoveries: Mapped[int] = mapped_column(Integer, default=0)


class SupportClarification(Base):
    __tablename__ = "support_clarifications"
    message_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    clarifies_id: Mapped[str] = mapped_column(String(32))


class SupportTask(Base):
    __tablename__ = "support_tasks"
    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(24))


class SupportDispatch(Base):
    __tablename__ = "support_dispatches"
    __table_args__ = (UniqueConstraint("message_id", "ticket_id"),)
    turn_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(32), index=True)
    ticket_id: Mapped[str] = mapped_column(String(32), index=True)
    quote: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(16))
    recoveries: Mapped[int] = mapped_column(Integer, default=0)


class Proposal(Base):
    __tablename__ = "service_proposals"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    ticket_id: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(24))
    parameters: Mapped[dict] = mapped_column(JSON)
    order_revision: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[object] = mapped_column(DateTime, default=now)


class Knowledge(Base):
    __tablename__ = "service_knowledge"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    title: Mapped[str] = mapped_column(String(160))
    content: Mapped[str] = mapped_column(Text)
    tags: Mapped[str] = mapped_column(String(240), default="")
    status: Mapped[str] = mapped_column(String(16), default="draft")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[object] = mapped_column(DateTime, default=now, onupdate=now)


class ServiceAudit(Base):
    __tablename__ = "service_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(48))
    actor: Mapped[str] = mapped_column(String(80))
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[object] = mapped_column(DateTime, default=now)


class SpecialistRun(Base):
    __tablename__ = "service_specialist_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=identifier)
    ticket_id: Mapped[str] = mapped_column(String(32), index=True)
    turn_id: Mapped[str] = mapped_column(String(32), index=True)
    role: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20))
    evidence: Mapped[dict] = mapped_column(JSON)
    output: Mapped[dict] = mapped_column(JSON)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[object] = mapped_column(DateTime, default=now)


class RetrievalVector(Base):
    """Content-addressed vectors; published revisions are selected at query time."""

    __tablename__ = "service_retrieval_vectors"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(160))
    vector: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[object] = mapped_column(DateTime, default=now)


class KnowledgeScope(Base):
    __tablename__ = "service_knowledge_scopes"
    knowledge_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_states: Mapped[list] = mapped_column(JSON, default=list)
    product_ids: Mapped[list] = mapped_column(JSON, default=list)


class TicketMemory(Base):
    __tablename__ = "service_ticket_memory"
    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[object] = mapped_column(DateTime, default=now, onupdate=now)
