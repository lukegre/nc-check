from __future__ import annotations

from typing import Any

from . import compliance as _compliance
from .check import Check, CheckInfo, CheckResult, CheckStatus, FixResult

# Re-export compatibility API from the historical nc_check.core module.
for _name in dir(_compliance):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_compliance, _name)

# Keep this alias monkeypatch-friendly for existing test and user patterns.
_run_cfchecker_on_dataset = _compliance._run_cfchecker_on_dataset


def check_dataset_compliant(*args: Any, **kwargs: Any) -> dict[str, Any] | str | None:
    original_runner = _compliance._run_cfchecker_on_dataset
    _compliance._run_cfchecker_on_dataset = _run_cfchecker_on_dataset
    try:
        return _compliance.check_dataset_compliant(*args, **kwargs)
    finally:
        _compliance._run_cfchecker_on_dataset = original_runner


__all__ = [
    "Check",
    "CheckInfo",
    "CheckResult",
    "CheckStatus",
    "FixResult",
    "check_dataset_compliant",
]
