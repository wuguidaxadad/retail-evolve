import pytest

from app.experiment_metrics import summarize_runs


def sample(mode, task, trial, reward):
    return dict(
        mode=mode,
        task_id=task,
        trial=trial,
        reward=reward,
        seconds=1,
        input_tokens=10,
        output_tokens=2,
        cached_input_tokens=0,
        estimated_agent_user_usd=0.001,
    )


def test_repeated_success_and_task_paired_percentage_points():
    runs = []
    scores = {
        "baseline": [[1, 0, 0], [1, 1, 1]],
        "harness": [[1, 1, 1], [1, 1, 1]],
        "skills": [[0, 0, 0], [0, 0, 0]],
    }
    for mode, tasks in scores.items():
        for task, rewards in zip(["a", "b"], tasks, strict=True):
            runs.extend(sample(mode, task, trial, reward) for trial, reward in enumerate(rewards))
    result = summarize_runs(runs, ["a", "b"], bootstrap_samples=1000)
    assert result["baseline"]["success_rate"] == pytest.approx(2 / 3)
    assert result["baseline"]["pass_all_repeats"] == 0.5
    assert result["harness"]["success_delta_percentage_points"] == pytest.approx(100 / 3)
    assert result["harness"]["relative_change_percent"] == pytest.approx(50)
    assert result["skills"]["success_delta_percentage_points"] == pytest.approx(-200 / 3)


def test_missing_and_duplicate_observations_are_never_silently_dropped():
    rows = [
        sample(mode, "a", trial, 0)
        for mode in ["baseline", "harness", "skills"]
        for trial in range(3)
    ]
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_runs(rows[:-1], ["a"])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_runs(rows + [rows[0]], ["a"])
    result = summarize_runs(rows, ["a"], bootstrap_samples=20)
    assert result["harness"]["relative_change_percent"] is None


def test_one_trial_reports_success_without_claiming_repeat_stability():
    rows = [sample(mode, "a", 0, 1) for mode in ["baseline", "harness", "skills"]]
    result = summarize_runs(rows, ["a"], trials=1, bootstrap_samples=20)
    assert result["baseline"]["runs"] == 1
    assert result["baseline"]["success_rate"] == 1
    assert result["baseline"]["pass_all_repeats"] is None


@pytest.mark.parametrize("reason", ["timeout", "infrastructure_error"])
def test_censored_observations_cannot_be_counted_as_business_failures(reason):
    rows = [sample(mode, "a", 0, 0) for mode in ["baseline", "harness", "skills"]]
    rows[0]["termination_reason"] = reason
    with pytest.raises(ValueError, match="Incomplete observation"):
        summarize_runs(rows, ["a"], trials=1)
