from __future__ import annotations

import hashlib
import json
import uuid

from ace import RRConfig, Skillbook
from ace import RRStep as UpstreamRRStep
from ace import SkillManager as UpstreamSkillManager
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.recursive_agent import AgenticConfig
from ace.implementations.skill_manager import SkillManagerReport
from ace.steps import learning_tail
from openai import OpenAI
from pipeline import Pipeline
from pydantic_ai.output import ToolOutput

from .config import ROOT, model_signature, settings
from .harness import validate_code
from .simulation import learning_trace


def require_complete_role(raw):
    error = raw.get("error")
    if error:
        from .native_bridge import sanitize
        folder = ROOT / ".runtime/ace-role-failures"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (uuid.uuid4().hex + ".json")).write_text(
            json.dumps(sanitize(raw), ensure_ascii=False, default=str))
        if any(
            word in str(error).lower()
            for word in ["timeout", "timed out", "connection", "429", "503"]
        ):
            raise ConnectionError("ACE provider failure: " + str(error))
        raise RuntimeError("ACE role failed: " + str(error))
    if raw.get("timeout") or raw.get("rr_trace", {}).get("timed_out"):
        raise RuntimeError("ACE role exhausted its budget; partial output cannot be committed")


class RRStep(UpstreamRRStep):
    def __call__(self, ctx):
        result = super().__call__(ctx)
        for reflection in result.reflections:
            require_complete_role(reflection.raw)
        return result


class SkillManager(UpstreamSkillManager):
    def _create_agent(self, depth=0):
        # Preserve the upstream report schema and atomic skill mutation tools.
        # Require a final tool submission instead of free-text/fenced JSON.
        if depth == 0 and self._model == "openai:deepseek-v4.1-flash":
            self._output_type = ToolOutput(SkillManagerReport, name="submit_skill_report")
        return super()._create_agent(depth=depth)

    def update_skills(self, **kwargs):
        result = super().update_skills(**kwargs)
        require_complete_role(result.raw)
        return result


def learn_skills(version, runs, on_progress, *, checkpoint=None, save_checkpoint=None):
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "version": {key: version.get(key) for key in ["id", "harness_code", "skillbook"]},
                "traces": [learning_trace(run) for run in runs],
                "model": model_signature(),
                "pipeline": "ace-recursive-v1",
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if checkpoint and checkpoint.get("fingerprint") != fingerprint:
        raise ValueError("ACE checkpoint source, model, or pipeline changed")
    audit = list(checkpoint["audit"]) if checkpoint else []
    if [a["trace_id"] for a in audit] != [str(r["id"]) for r in runs[: len(audit)]]:
        raise ValueError("ACE checkpoint trace order mismatch")
    book = Skillbook.from_dict(checkpoint["skillbook"] if checkpoint else version["skillbook"])
    model_settings = {"timeout": settings.llm_timeout}
    if settings.learning_model in {"gpt-5.4", "gpt-6-luna", "hy3"}:
        model_settings["openai_reasoning_effort"] = "none"
    reflector = RRStep(
        "openai:" + settings.learning_model, config=RRConfig(), model_settings=model_settings
    )
    manager = SkillManager(
        "openai:" + settings.learning_model,
        model_settings=model_settings,
        config=AgenticConfig(),
    )
    pipeline = Pipeline(learning_tail(reflector, manager, book))
    for i, run in enumerate(runs[len(audit) :], start=len(audit)):
        on_progress(i / len(runs), f"ACE 分析学习轨迹 {i + 1}/{len(runs)}")
        trace = learning_trace(run)
        ctx = ACEStepContext(
            sample=None,
            skillbook=SkillbookView(book),
            trace=trace,
            mode="offline",
            epoch=1,
            total_epochs=1,
            step_index=i + 1,
            total_steps=len(runs),
            injected_skill_ids=tuple(run["data"].get("injected_skill_ids", [])),
            metadata={
                "trace_identity": {
                    "source_system": "retail-evolve",
                    "trace_id": trace["trace_id"],
                    "display_name": "零售任务 " + run["task_id"],
                },
                "source": trace["source"],
            },
        )
        outcome = pipeline.run([ctx], workers=1)[0]
        pipeline.wait_for_background()
        if outcome.error:
            raise outcome.error
        result = outcome.output
        if result is None:
            raise RuntimeError("ACE 未返回完整的学习结果")
        audit.append(
            {
                "trace_id": trace["trace_id"],
                "task_id": run["task_id"],
                "reflections": [
                    r.model_dump(mode="json") if hasattr(r, "model_dump") else str(r)
                    for r in result.reflections
                ],
                "updates": result.skill_manager_output.to_json()
                if result.skill_manager_output
                else None,
            }
        )
        if save_checkpoint:
            save_checkpoint(
                {
                    "fingerprint": fingerprint,
                    "skillbook": book.to_dict(),
                    "audit": audit,
                    "pipeline": "ace-recursive-v1",
                    "output_transport": "skill-manager-tool-output-v1",
                }
            )
    return book.to_dict(), audit


def propose_harness(version, runs):
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.llm_timeout,
        max_retries=1,
    )
    prompt = """Improve a retail customer-service Agent execution harness from training traces.
Call submit_harness with explanation and code. code MUST be a standalone Python source
containing only literal assignments INSTRUCTION (string), REQUIRE_CONFIRMATION (bool),
RETRY_ERROR_HINT (bool), MAX_STEPS (integer 10..80), and exactly one pure function
def build_prompt(base_instruction, skill_context, error_context):
    return base_instruction + "\\n" + skill_context + "\\n" + error_context
You may rewrite instruction literals, booleans and the pure return expression using only
string constants, these three arguments, concatenation, or f-strings. No calls, imports,
attributes, loops, conditionals or other variables. Include base_instruction and skill_context.
Business policy is enforced separately. Do not encode task identifiers, customer names,
order numbers, gold answers, or specific benchmark cases in the prompt.
This is bounded code/prompt improvement, not weight training. Explain observed evidence,
and make only a modest generalizable change."""
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "current_code": version["harness_code"],
                    "skillbook": version["skillbook"],
                    "training_traces": [learning_trace(r) for r in runs],
                },
                ensure_ascii=False,
            ),
        },
    ]
    tool = {
        "type": "function",
        "function": {
            "name": "submit_harness",
            "description": "Submit a bounded Python harness candidate.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "explanation": {"type": "string"}},
                "required": ["code", "explanation"],
                "additionalProperties": False,
            },
        },
    }
    error = ""
    for _ in range(2):
        response = client.chat.completions.create(
            model=settings.learning_model,
            temperature=0.2,
            max_tokens=2800,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "submit_harness"}},
            parallel_tool_calls=False,
            messages=messages,
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        try:
            if len(calls) != 1 or calls[0].function.name != "submit_harness":
                raise ValueError("优化器未按约定提交结构化候选")
            result = json.loads(calls[0].function.arguments)
            if not isinstance(result.get("code"), str) or not isinstance(
                result.get("explanation"), str
            ):
                raise ValueError("候选缺少代码或解释字段")
            validate_code(result["code"])
            return result["code"], result["explanation"]
        except (ValueError, KeyError, SyntaxError, TypeError) as exc:
            error = str(exc)
            messages.append(message.model_dump(exclude_none=True))
            if calls:
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": c.id,
                        "content": "Candidate rejected: " + error,
                    }
                    for c in calls
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Candidate rejected: "
                        + error
                        + ". Use the submit_harness tool.",
                    }
                )
    raise ValueError("优化器两次输出均未通过格式或代码检查：" + error)
