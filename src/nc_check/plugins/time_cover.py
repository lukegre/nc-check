from __future__ import annotations

from typing import Any

from ..dataset import CanonicalDataset
from ..models import AtomicCheckResult

_MISSING_SLICES_CHECK = "time.missing_slices"
_MONOTONIC_CHECK = "time.monotonic_increasing"
_REGULAR_SPACING_CHECK = "time.regular_spacing"


def _normalize_status_kind(raw: Any) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return "fail"
    if normalized in {"pass", "passed", "ok", "success", "true"}:
        return "pass"
    if normalized in {"skip", "skipped"} or normalized.startswith("skip"):
        return "skip"
    return "skip"


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return 0


def _collect_check_items(
    report: dict[str, Any], *, check_id: str
) -> list[dict[str, Any]]:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return []

    fallback_variable = (
        str(report.get("variable")) if report.get("variable") is not None else None
    )

    items: list[dict[str, Any]] = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "")) != check_id:
            continue

        variable = item.get("variable")
        normalized: dict[str, Any] = {
            "status": _normalize_status_kind(item.get("status")),
            "variable": str(variable) if variable is not None else fallback_variable,
            "result": item.get("result")
            if isinstance(item.get("result"), dict)
            else {},
        }
        items.append(normalized)
    return items


def _status_from_items(items: list[dict[str, Any]]) -> str:
    statuses = [str(item.get("status", "skip")) for item in items]
    if any(status == "fail" for status in statuses):
        return "failed"
    if any(status == "pass" for status in statuses):
        return "passed"
    return "skipped"


def _summarize_check(
    *,
    check_name: str,
    check_label: str,
    check_id: str,
    issue_count_key: str,
    report: dict[str, Any],
) -> AtomicCheckResult:
    items = _collect_check_items(report, check_id=check_id)
    if not items:
        return AtomicCheckResult.skipped_result(
            name=check_name,
            info=f"{check_label} check did not run.",
            details={"checked": 0, "failed": 0, "skipped": 0, "issue_count": 0},
        )

    checked = len(items)
    failed = sum(1 for item in items if str(item.get("status")) == "fail")
    skipped = sum(1 for item in items if str(item.get("status")) == "skip")
    issue_count = sum(
        _as_int(item.get("result", {}).get(issue_count_key, 0)) for item in items
    )

    failed_variables = sorted(
        {
            str(item.get("variable"))
            for item in items
            if str(item.get("status")) == "fail" and item.get("variable") is not None
        }
    )
    skipped_variables = sorted(
        {
            str(item.get("variable"))
            for item in items
            if str(item.get("status")) == "skip" and item.get("variable") is not None
        }
    )

    details: dict[str, Any] = {
        "checked": checked,
        "failed": failed,
        "skipped": skipped,
        "issue_count": issue_count,
    }
    if failed_variables:
        details["failed_variables"] = failed_variables
    if skipped_variables:
        details["skipped_variables"] = skipped_variables

    outcome = _status_from_items(items)
    if outcome == "failed":
        return AtomicCheckResult.failed_result(
            name=check_name,
            info=(f"{check_label} check found issues in {failed} variable outcome(s)."),
            details=details,
        )
    if outcome == "passed":
        return AtomicCheckResult.passed_result(
            name=check_name,
            info=f"{check_label} check passed.",
            details=details,
        )
    return AtomicCheckResult.skipped_result(
        name=check_name,
        info=f"{check_label} check skipped.",
        details=details,
    )


class TimeCoverPlugin:
    name = "time_cover"

    def __init__(self, *, var_name: str | None = None, time_name: str | None = "time"):
        self.var_name = var_name
        self.time_name = time_name

    def _run_report(
        self,
        ds: CanonicalDataset,
        *,
        check_time_monotonic: bool,
        check_time_regular_spacing: bool,
    ) -> dict[str, Any]:
        from ..checks.time_cover import TimeCoverConfig, run_time_cover_report

        return run_time_cover_report(
            ds,
            config=TimeCoverConfig(
                var_name=self.var_name,
                time_name=self.time_name,
                check_time_monotonic=check_time_monotonic,
                check_time_regular_spacing=check_time_regular_spacing,
            ),
        )

    def _missing_slices_check(self, ds: CanonicalDataset) -> AtomicCheckResult:
        report = self._run_report(
            ds,
            check_time_monotonic=False,
            check_time_regular_spacing=False,
        )
        return _summarize_check(
            check_name=_MISSING_SLICES_CHECK,
            check_label="Missing time slices",
            check_id=_MISSING_SLICES_CHECK,
            issue_count_key="missing_slice_count",
            report=report,
        )

    def _monotonic_increasing_check(self, ds: CanonicalDataset) -> AtomicCheckResult:
        report = self._run_report(
            ds,
            check_time_monotonic=True,
            check_time_regular_spacing=False,
        )
        return _summarize_check(
            check_name=_MONOTONIC_CHECK,
            check_label="Monotonic time order",
            check_id=_MONOTONIC_CHECK,
            issue_count_key="order_violation_count",
            report=report,
        )

    def _regular_spacing_check(self, ds: CanonicalDataset) -> AtomicCheckResult:
        report = self._run_report(
            ds,
            check_time_monotonic=False,
            check_time_regular_spacing=True,
        )
        return _summarize_check(
            check_name=_REGULAR_SPACING_CHECK,
            check_label="Regular time spacing",
            check_id=_REGULAR_SPACING_CHECK,
            issue_count_key="irregular_interval_count",
            report=report,
        )

    def register(self, registry: Any) -> None:
        registry.register_check(
            name=_MISSING_SLICES_CHECK,
            check=self._missing_slices_check,
            plugin=self.name,
        )
        registry.register_check(
            name=_MONOTONIC_CHECK,
            check=self._monotonic_increasing_check,
            plugin=self.name,
        )
        registry.register_check(
            name=_REGULAR_SPACING_CHECK,
            check=self._regular_spacing_check,
            plugin=self.name,
        )


def time_cover_check_names() -> tuple[str, ...]:
    return (
        _MISSING_SLICES_CHECK,
        _MONOTONIC_CHECK,
        _REGULAR_SPACING_CHECK,
    )
