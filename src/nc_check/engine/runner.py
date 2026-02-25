from __future__ import annotations

from typing import Any

import xarray as xr

from ..contracts import CheckSummaryItem, SummaryStatus, SuiteReport
from .defaults import default_check_order, register_default_checks
from .registry import get_check


def _combine_suite_statuses(statuses: list[SummaryStatus]) -> SummaryStatus:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def run_suite_checks(
    ds: xr.Dataset,
    *,
    checks_enabled: dict[str, bool],
    options_by_check: dict[str, dict[str, Any]],
) -> SuiteReport:
    register_default_checks()
    if not any(bool(enabled) for enabled in checks_enabled.values()):
        raise ValueError("At least one check must be enabled.")

    reports: dict[str, dict[str, Any]] = {}
    check_summary: list[CheckSummaryItem] = []

    for key in default_check_order():
        if not bool(checks_enabled.get(key)):
            continue
        registration = get_check(key)
        report = registration.run_report(ds, options_by_check.get(key, {}))
        reports[key] = report
        status = registration.resolve_status(report)
        detail = registration.resolve_detail(report)
        check_summary.append({"check": key, "status": status, "detail": detail})

    statuses = [entry["status"] for entry in check_summary]
    overall_status = _combine_suite_statuses(statuses)
    return {
        "checks_enabled": checks_enabled,
        "check_summary": check_summary,
        "reports": reports,
        "summary": {
            "checks_run": len(check_summary),
            "failing_checks": sum(1 for status in statuses if status == "fail"),
            "warnings_or_skips": sum(1 for status in statuses if status == "warn"),
            "overall_status": overall_status,
            "overall_ok": overall_status == "pass",
        },
    }
