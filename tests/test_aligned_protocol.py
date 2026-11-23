from dataclasses import replace

import pytest

from app import dataset, failure_analysis, preflight
from app.experiment_metrics import MODES, summarize_runs


def test_aligned_split_is_frozen_disjoint_and_complete(monkeypatch):
    monkeypatch.setattr(dataset, "settings", replace(dataset.settings, experiment_profile="aligned-v1"))
    parts = dataset.splits()
    assert [len(parts[k]) for k in ("learn", "validation", "final")] == [20, 20, 74]
    assert len(set.union(*(set(v) for v in parts.values()))) == 114
    with pytest.raises(ValueError):
        dataset.choose_tasks("learn", [parts["final"][0]])
    with pytest.raises(ValueError):
        dataset.choose_tasks("validation", [parts["final"][0]])


def test_failure_analysis_cannot_cite_hidden_tasks():
    data = {"clusters": [{"id": "c1", "failed_task_ids": ["hidden"],
                          "control_task_ids": [], "edit_surface": "state",
                          "root_cause": "lost confirmation", "hypothesis": "track pending work"}]}
    with pytest.raises(ValueError, match="non-training"):
        failure_analysis.validate_analysis(data, ["train"], "source")
    data["clusters"][0]["failed_task_ids"] = ["train"]
    result = failure_analysis.validate_analysis(data, ["train"], "source")
    assert result["source_hash"] == "source"


def test_formal_run_cannot_start_without_live_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="not completed"):
        preflight.require_preflight()


def test_four_rollouts_all_success_metric_is_not_average():
    rows = [dict(mode=m, task_id="t", trial=t, reward=int(m != "baseline" or t < 3),
                 seconds=1, input_tokens=1, output_tokens=1, cached_input_tokens=0)
            for m in MODES for t in range(4)]
    result = summarize_runs(rows, ["t"], trials=4, bootstrap_samples=20)
    assert result["baseline"]["success_rate"] == .75
    assert result["baseline"]["pass_all_repeats"] == 0
    assert result["harness"]["pass_all_repeats"] == 1


def test_gemini_responses_route_preserves_wire_model(monkeypatch):
    from litellm.main import responses_api_bridge_check

    from app import config

    monkeypatch.setattr(config, "settings", replace(config.settings,
        agent_model="gemini-3.5-flash-lite", agent_transport="responses"))
    route = config.agent_model_route()
    info, wire_model = responses_api_bridge_check(route.removeprefix("openai/"), "openai")
    assert info["mode"] == "responses"
    assert wire_model == "gemini-3.5-flash-lite"
    user_info, user_model = responses_api_bridge_check("hy3", "openai")
    assert user_info.get("mode") != "responses"
    assert user_model == "hy3"
