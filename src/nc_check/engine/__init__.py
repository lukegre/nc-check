from typing import Any

import xarray as xr

from ..contracts import SuiteReport
from .suite import Suite, SuiteCheck


def default_check_order() -> tuple[str, ...]:
    from .defaults import default_check_order as _default_check_order

    return _default_check_order()


def register_default_checks() -> None:
    from .defaults import register_default_checks as _register_default_checks

    _register_default_checks()


def run_suite_checks(
    ds: xr.Dataset,
    *,
    checks_enabled: dict[str, bool],
    options_by_check: dict[str, dict[str, Any]],
) -> SuiteReport:
    from .runner import run_suite_checks as _run_suite_checks

    return _run_suite_checks(
        ds,
        checks_enabled=checks_enabled,
        options_by_check=options_by_check,
    )


__all__ = [
    "Suite",
    "SuiteCheck",
    "default_check_order",
    "register_default_checks",
    "run_suite_checks",
]
