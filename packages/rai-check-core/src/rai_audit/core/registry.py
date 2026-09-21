from __future__ import annotations

from typing import Callable

_CHECK_REGISTRY: dict[str, dict] = {}
_METRIC_REGISTRY: dict[str, Callable] = {}


def register_check(
    check_id: str,
    title: str,
    category: str,
    description: str = "",
) -> None:
    _CHECK_REGISTRY[check_id] = {
        "check_id": check_id,
        "title": title,
        "category": category,
        "description": description,
    }


def register_metric(name: str, fn: Callable) -> None:
    _METRIC_REGISTRY[name] = fn


def get_check(check_id: str) -> dict | None:
    return _CHECK_REGISTRY.get(check_id)


def list_checks(category: str | None = None) -> list[dict]:
    checks = list(_CHECK_REGISTRY.values())
    if category:
        checks = [c for c in checks if c["category"] == category]
    return checks
