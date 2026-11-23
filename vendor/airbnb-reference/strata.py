def build_tau_strata(domain: str, pool: list[int], idx_to_task: dict) -> dict[int, str]:
    """Return a strata dict mapping pool_index → category label.

    telecom: category extracted from bracket-prefixed task ID e.g. "[mms_issue]task_name"
    retail:  inferred from user_scenario text — 3 buckets: exchange / return / other
    airline: no meaningful categories — returns empty dict (no stratification)
    others:  returns empty dict
    """
    if domain == "telecom":
        strata = {}
        for idx in pool:
            task = idx_to_task.get(idx)
            tid = str(task.id) if task else ""
            cat = tid.split("]")[0].lstrip("[") if tid.startswith("[") else "unknown"
            strata[idx] = cat
        return strata

    if domain == "retail":
        strata = {}
        for idx in pool:
            task = idx_to_task.get(idx)
            sc = str(getattr(task, "user_scenario", "")).lower() if task else ""
            if "exchange" in sc:
                cat = "exchange"
            elif "return" in sc:
                cat = "return"
            else:
                cat = "other"
            strata[idx] = cat
        return strata

    # airline and others: no stratification
    return {}
