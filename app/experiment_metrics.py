"""Task-paired metrics for a frozen retail customer-service ablation."""

from __future__ import annotations

import random

from .evaluation_validity import require_scoreable

MODES = ("baseline", "harness", "skills")


def summarize_runs(runs, task_ids, trials=3, bootstrap_samples=10000):
    grouped = {mode: {} for mode in MODES}
    expected = {(task, trial) for task in task_ids for trial in range(trials)}
    for row in runs:
        require_scoreable(row)
        if isinstance(row.get("data"), dict):
            require_scoreable(row["data"])
        if row.get("termination_reason") in {"timeout", "infrastructure_error"}:
            raise ValueError("Incomplete observation: timeout/infrastructure error cannot be scored")
        mode = row["mode"]
        if mode not in grouped:
            raise ValueError("Unexpected experiment mode")
        key = (row["task_id"], row["trial"])
        if key not in expected or key in grouped[mode]:
            raise ValueError("Unexpected or duplicate experiment observation")
        grouped[mode][key] = row
    if any(set(values) != expected for values in grouped.values()):
        raise ValueError("Incomplete experiment: do not omit failed or missing observations")

    def quantile(values, q):
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    results = {}
    rates = {}
    for mode, values in grouped.items():
        task_rates = []
        task_all_pass = []
        for task in task_ids:
            passed = [values[(task, trial)]["reward"] >= 1 - 1e-9 for trial in range(trials)]
            task_rates.append(sum(passed) / trials)
            task_all_pass.append(float(all(passed)))
        rates[mode] = task_rates
        count = len(values)
        results[mode] = {
            "tasks": len(task_ids),
            "trials_per_task": trials,
            "runs": count,
            "successes": sum(row["reward"] >= 1 - 1e-9 for row in values.values()),
            "success_rate": sum(task_rates) / len(task_rates),
            "mean_official_reward": sum(row["reward"] for row in values.values()) / count,
            "pass_all_repeats": sum(task_all_pass) / len(task_ids) if trials > 1 else None,
            "mean_seconds": sum(row["seconds"] for row in values.values()) / count,
            "input_tokens": sum(row["input_tokens"] for row in values.values()),
            "output_tokens": sum(row["output_tokens"] for row in values.values()),
            "cached_input_tokens": sum(row["cached_input_tokens"] for row in values.values()),
            "estimated_agent_user_usd": (
                sum(row["estimated_agent_user_usd"] for row in values.values())
                if all(row.get("estimated_agent_user_usd") is not None for row in values.values())
                else None
            ),
            "per_task_success_rates": dict(zip(task_ids, task_rates, strict=True)),
        }
    for mode in ("harness", "skills"):
        deltas = [a - b for a, b in zip(rates[mode], rates["baseline"], strict=True)]
        rng = random.Random(20261013)
        samples = [
            sum(rng.choice(deltas) for _ in task_ids) / len(task_ids)
            for _ in range(bootstrap_samples)
        ]
        base = results["baseline"]["success_rate"]
        delta = results[mode]["success_rate"] - base
        results[mode].update(
            success_delta_percentage_points=100 * delta,
            pass_all_delta_percentage_points=(100 * (results[mode]["pass_all_repeats"] - results["baseline"]["pass_all_repeats"])) if trials > 1 else None,
            relative_change_percent=(100 * delta / base) if base else None,
            delta_95ci_percentage_points=[
                100 * quantile(samples, 0.025),
                100 * quantile(samples, 0.975),
            ],
        )
    return results
