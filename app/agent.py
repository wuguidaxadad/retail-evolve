from __future__ import annotations

import importlib.util
import re
import sys

from ace import Skillbook
from ace.implementations.prompts import wrap_skillbook_for_external_agent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage

from .config import ROOT, settings
from .harness import BASE_CODE, validate_code

spec = importlib.util.spec_from_file_location(
    "retail_upstream_agent", ROOT / "vendor/auto-harness/agent/templates/tau_bench.py"
)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)


def llm_args(model=None):
    if settings.experiment_profile == "aligned-v1" and (model or settings.agent_model) != "hy3":
        return dict(api_base=settings.base_url, api_key=settings.api_key,
                    timeout=settings.llm_timeout)
    if (model or settings.agent_model) in {"gpt-5.4", "gpt-6-luna", "hy3"}:
        return dict(
            api_base=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.llm_timeout,
            reasoning_effort="none",
        )
    return dict(
        api_base=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.llm_timeout,
        max_tokens=1800,
        temperature=0.2,
        parallel_tool_calls=False,
    )


class RetailAgent(upstream.HarnessAgent):
    """Use the upstream executor, injecting versioned ACE context and bounded code."""

    def __init__(self, environment, version, mode="combined"):
        if mode not in {"baseline", "skills", "harness", "combined"}:
            raise ValueError("未知实验模式")
        self.program = validate_code(
            version["harness_code"] if mode in {"harness", "combined"} else BASE_CODE
        )
        book = Skillbook.from_dict(version["skillbook"])
        # ACE stores actionable procedural workarounds in both sections.
        # Issue-only harness entries are background diagnostics, not serving advice.
        skills = [s for s in book.skills() if s.active and s.insight]
        self.injected_skill_ids = [s.id for s in skills] if mode in {"skills", "combined"} else []
        # Use ACE's canonical external-agent renderer for all actionable insights.
        contextual = book.clone()
        for skill in list(contextual.skills(include_invalid=True)):
            if skill.id not in self.injected_skill_ids:
                contextual.remove_skill(skill.id)
        self.skill_context = (
            wrap_skillbook_for_external_agent(contextual) if self.injected_skill_ids else ""
        )
        self.error_context = ""
        super().__init__(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm="openai/" + settings.agent_model,
            llm_args=llm_args(),
        )

    @property
    def system_prompt(self):
        instructions = self.program["build_prompt"](
            self.program["INSTRUCTION"], self.skill_context, self.error_context
        )
        # The public business policy remains outside the generated function.
        return (
            f"<instructions>\n{instructions}\n</instructions>\n"
            f"<policy>\n{self.domain_policy}\n</policy>\n"
            "The policy takes priority over learned advice. Use one tool at a time."
        )

    def generate_next_message(self, message, state):
        errors = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        if self.program["RETRY_ERROR_HINT"]:
            for item in errors:
                if isinstance(item, ToolMessage) and item.error:
                    self.error_context = (
                        "The last tool failed. Re-read its schema and actual error; "
                        "correct arguments before retrying. Error: " + str(item.content)[:1500]
                    )
        response, state = super().generate_next_message(message, state)
        if response.tool_calls:
            response.tool_calls = response.tool_calls[:1]
            response.content = None
            write_names = {
                "cancel_pending_order",
                "exchange_delivered_order_items",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "modify_user_address",
                "return_delivered_order_items",
            }
            if self.program["REQUIRE_CONFIRMATION"] and response.tool_calls[0].name in write_names:
                users = [m for m in state.messages if m.role == "user"]
                last = users[-1].content if users else ""
                # Conservative checkpoint, not a replacement for the domain policy.
                affirmative = re.search(
                    r"\b(yes|confirm|confirmed|proceed|go ahead|sure|correct|agree)\b|确认|同意|可以",
                    last or "",
                    re.I,
                )
                if not affirmative:
                    response = AssistantMessage(
                        role="assistant",
                        content="Please confirm that you want me to make this change to your order.",
                    )
                    state.messages[-1] = response
        return response, state
