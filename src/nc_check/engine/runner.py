from __future__ import annotations

from typing import Any, cast

import xarray as xr

from ..contracts import (
    AtomicCheckItem,
    CheckSummaryItem,
    GroupSummaryItem,
    SuiteReport,
    SummaryStatus,
)
from .defaults import default_check_order, register_default_checks
from .registry import RegisteredCheck, get_check


def _status_kind(status: Any) -> str:
    normalized = str(status).strip().lower()
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return "fail"
    if normalized in {"warn", "warning"}:
        return "warn"
    if normalized in {"skip", "skipped"} or normalized.startswith("skip"):
        return "skip"
    return "pass"


def _combine_status_kinds(status_kinds: list[str]) -> str:
    if any(status == "fail" for status in status_kinds):
        return "fail"
    if any(status == "warn" for status in status_kinds):
        return "warn"
    return "pass"


def _summary_from_atomic_checks(checks: list[AtomicCheckItem]) -> dict[str, Any]:
    kinds = [_status_kind(item.get("status")) for item in checks]
    failing_checks = sum(1 for kind in kinds if kind == "fail")
    warnings_or_skips = sum(1 for kind in kinds if kind in {"warn", "skip"})
    overall_status = _combine_status_kinds(kinds)
    return {
        "checks_run": len(checks),
        "failing_checks": failing_checks,
        "warnings_or_skips": warnings_or_skips,
        "overall_status": overall_status,
        "overall_ok": overall_status == "pass",
    }


def _group_summary_from_atomic_checks(
    checks: list[AtomicCheckItem],
) -> GroupSummaryItem:
    summary = _summary_from_atomic_checks(checks)
    return {
        "checks_run": int(summary["checks_run"]),
        "failing_checks": int(summary["failing_checks"]),
        "warnings_or_skips": int(summary["warnings_or_skips"]),
        "status": cast(SummaryStatus, summary["overall_status"]),
        "overall_ok": bool(summary["overall_ok"]),
    }


def _flatten_atomic_checks(
    *,
    group_key: str,
    report: dict[str, Any],
    registration: RegisteredCheck,
) -> list[AtomicCheckItem]:
    checks = report.get("checks")
    if isinstance(checks, list):
        flattened: list[AtomicCheckItem] = []
        for index, raw_check in enumerate(checks):
            if not isinstance(raw_check, dict):
                continue
            check_id = str(raw_check.get("id") or f"{group_key}.check_{index + 1}")
            check_name = str(raw_check.get("name") or check_id)
            status = str(raw_check.get("status", "unknown"))
            detail = str(raw_check.get("detail", ""))
            item: AtomicCheckItem = {
                "id": check_id,
                "name": check_name,
                "group": group_key,
                "status": status,
                "detail": detail,
            }
            if isinstance(raw_check.get("result"), dict):
                item["result"] = raw_check["result"]
            if raw_check.get("variable") is not None:
                item["variable"] = str(raw_check["variable"])
            flattened.append(item)
        if flattened:
            return flattened

    status = registration.resolve_status(report)
    detail = registration.resolve_detail(report)
    return [
        {
            "id": group_key,
            "name": group_key,
            "group": group_key,
            "status": status,
            "detail": detail,
        }
    ]


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
    checks: list[AtomicCheckItem] = []
    groups: dict[str, GroupSummaryItem] = {}
    check_summary: list[CheckSummaryItem] = []

    for key in default_check_order():
        if not bool(checks_enabled.get(key)):
            continue
        registration = get_check(key)
        report = registration.run_report(ds, options_by_check.get(key, {}))
        reports[key] = report
        group_checks = _flatten_atomic_checks(
            group_key=key,
            report=report,
            registration=registration,
        )
        checks.extend(group_checks)

        group_summary = _group_summary_from_atomic_checks(group_checks)
        groups[key] = group_summary
        check_summary.append(
            {
                "check": key,
                "status": group_summary["status"],
                "detail": (
                    f"checks={group_summary['checks_run']} "
                    f"fail={group_summary['failing_checks']} "
                    f"warn_or_skip={group_summary['warnings_or_skips']}"
                ),
            }
        )

    summary = _summary_from_atomic_checks(checks)
    return {
        "checks_enabled": checks_enabled,
        "checks": checks,
        "groups": groups,
        "check_summary": check_summary,
        "reports": reports,
        "summary": summary,
    }
