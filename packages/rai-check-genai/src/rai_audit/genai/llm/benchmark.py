from __future__ import annotations

from collections import defaultdict


def summarize_reports(reports) -> dict:
    """Summarize repeated LLM suite reports by provider and model."""
    groups = defaultdict(
        lambda: {"runs": 0, "requests": 0, "tokens": 0, "cost_usd": 0.0, "latency_ms": []}
    )
    for report in reports:
        metrics = report.metadata.get("response_metrics", ())
        if not metrics:
            groups["captured/<unknown>"]["runs"] += 1
            continue
        seen = set()
        for item in metrics:
            key = f"{item.get('provider', '<unknown>')}/{item.get('model', '<unknown>')}"
            group = groups[key]
            group["requests"] += 1
            group["tokens"] += int(item.get("total_tokens", 0))
            group["cost_usd"] += float(item.get("cost_usd") or 0)
            group["latency_ms"].append(float(item.get("latency_ms", 0)))
            seen.add(key)
        for key in seen:
            groups[key]["runs"] += 1
    return {
        key: {
            "runs": value["runs"],
            "requests": value["requests"],
            "total_tokens": value["tokens"],
            "total_cost_usd": round(value["cost_usd"], 8),
            "mean_latency_ms": round(sum(value["latency_ms"]) / len(value["latency_ms"]), 3)
            if value["latency_ms"]
            else None,
        }
        for key, value in sorted(groups.items())
    }
