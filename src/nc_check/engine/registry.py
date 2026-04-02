from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import xarray as xr

from ..contracts import SummaryStatus

RunReportFn = Callable[[xr.Dataset, dict[str, Any]], dict[str, Any]]
ResolveStatusFn = Callable[[dict[str, Any]], SummaryStatus]
ResolveDetailFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class RegisteredCheck:
    key: str
    run_report: RunReportFn
    resolve_status: ResolveStatusFn
    resolve_detail: ResolveDetailFn


_REGISTRY: dict[str, RegisteredCheck] = {}


def register_check(check: RegisteredCheck) -> None:
    _REGISTRY[check.key] = check


def get_check(key: str) -> RegisteredCheck:
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown check '{key}'. Registered checks: {known}.") from exc


def has_check(key: str) -> bool:
    return key in _REGISTRY


def registered_check_keys() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())
