from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from ..core.check import Check, CheckInfo, CheckResult
from ..core.coverage import (
    choose_time_vars,
    leaf_statuses,
    missing_mask,
    range_records,
    resolve_time_dim,
    status_from_leaf_statuses,
)
from ..engine.suite import Suite, SuiteCheck
from ..formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    print_pretty_time_cover_reports,
    render_pretty_time_cover_report_html,
    render_pretty_time_cover_reports_html,
    save_html_report,
)


class TimeCoverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    var_name: str | None = None
    time_name: str | None = "time"
    check_time_monotonic: bool = False
    check_time_regular_spacing: bool = False


@dataclass(frozen=True)
class TimeCheckContext:
    da: xr.DataArray
    time_dim: str | None
    time_coord: xr.DataArray | None


def _resolve_time_check_context(
    ds: xr.Dataset,
    *,
    var_name: str,
    time_name: str | None,
) -> TimeCheckContext:
    if var_name not in ds.data_vars:
        raise ValueError(f"Data variable '{var_name}' not found.")
    da = ds[var_name]
    time_dim = resolve_time_dim(da, time_name)
    time_coord = da.coords.get(time_dim) if time_dim is not None else None
    return TimeCheckContext(da=da, time_dim=time_dim, time_coord=time_coord)


def _time_values(context: TimeCheckContext) -> np.ndarray[Any, Any]:
    if context.time_dim is None:
        return np.asarray([], dtype=float)
    if context.time_coord is None:
        return np.arange(int(context.da.sizes[context.time_dim]))
    return np.asarray(context.time_coord.values)


def _intervals_match(left: Any, right: Any) -> bool:
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        try:
            return bool(np.isclose(float(left), float(right), equal_nan=True))
        except Exception:
            return False
    try:
        return bool(left == right)
    except Exception:
        return False


class MissingTimeSlicesCheck(Check):
    id = "time.missing_slices"
    description = "Check for missing slices along the time dimension."
    tags = ("time", "coverage")

    def __init__(
        self,
        *,
        var_name: str,
        time_name: str | None = "time",
        context: TimeCheckContext | None = None,
    ) -> None:
        self.var_name = var_name
        self.time_name = time_name
        self._context = context

    def _resolve_context(self, ds: xr.Dataset) -> TimeCheckContext:
        if self._context is not None:
            return self._context
        return _resolve_time_check_context(
            ds, var_name=self.var_name, time_name=self.time_name
        )

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        context = self._resolve_context(ds)
        if context.time_dim is None:
            return {
                "enabled": True,
                "status": "skipped_no_time",
                "missing_slice_count": 0,
                "missing_slice_ranges": [],
            }

        missing = missing_mask(context.da)
        reduce_dims = [dim for dim in missing.dims if dim != context.time_dim]
        if reduce_dims:
            missing = missing.all(dim=reduce_dims)
        missing_time_indices = np.flatnonzero(
            np.asarray(missing.values, dtype=bool)
        ).tolist()
        return {
            "enabled": True,
            "status": "fail" if missing_time_indices else "pass",
            "missing_slice_count": len(missing_time_indices),
            "missing_slice_ranges": range_records(
                missing_time_indices, context.time_coord
            ),
        }

    def check(self, ds: xr.Dataset) -> CheckResult:
        report = self.run_report(ds)
        status = status_from_leaf_statuses([str(report.get("status", ""))])
        missing_count = int(report.get("missing_slice_count", 0))
        if str(report.get("status", "")).startswith("skip"):
            message = "Missing time slices check skipped."
        elif missing_count > 0:
            message = f"Detected {missing_count} missing time slices."
        else:
            message = "No missing time slices detected."
        return CheckResult(
            check_id=self.id,
            status=status,
            info=CheckInfo(message=message, details={"report": report}),
            fixable=False,
            tags=list(self.tags),
        )


class TimeMonotonicIncreasingCheck(Check):
    id = "time.monotonic_increasing"
    description = "Check that time values are monotonic increasing."
    tags = ("time", "coverage")

    def __init__(
        self,
        *,
        var_name: str,
        time_name: str | None = "time",
        context: TimeCheckContext | None = None,
    ) -> None:
        self.var_name = var_name
        self.time_name = time_name
        self._context = context

    def _resolve_context(self, ds: xr.Dataset) -> TimeCheckContext:
        if self._context is not None:
            return self._context
        return _resolve_time_check_context(
            ds, var_name=self.var_name, time_name=self.time_name
        )

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        context = self._resolve_context(ds)
        if context.time_dim is None:
            return {
                "enabled": True,
                "status": "skipped_no_time",
                "order_violation_count": 0,
                "order_violation_ranges": [],
            }

        values = _time_values(context)
        violation_indices: list[int] = []
        for idx in range(1, int(values.size)):
            try:
                if bool(values[idx] < values[idx - 1]):
                    violation_indices.append(idx)
            except Exception:
                return {
                    "enabled": True,
                    "status": "skipped_uncomparable_time",
                    "order_violation_count": 0,
                    "order_violation_ranges": [],
                    "note": "Time values are not directly comparable for monotonic-order check.",
                }

        return {
            "enabled": True,
            "status": "fail" if violation_indices else "pass",
            "order_violation_count": len(violation_indices),
            "order_violation_ranges": range_records(
                violation_indices, context.time_coord
            ),
        }

    def check(self, ds: xr.Dataset) -> CheckResult:
        report = self.run_report(ds)
        status = status_from_leaf_statuses([str(report.get("status", ""))])
        violation_count = int(report.get("order_violation_count", 0))
        if str(report.get("status", "")).startswith("skip"):
            message = "Time monotonic-order check skipped."
        elif violation_count > 0:
            message = f"Detected {violation_count} time-order violations."
        else:
            message = "Time values are monotonic increasing."
        return CheckResult(
            check_id=self.id,
            status=status,
            info=CheckInfo(message=message, details={"report": report}),
            fixable=False,
            tags=list(self.tags),
        )


class TimeRegularSpacingCheck(Check):
    id = "time.regular_spacing"
    description = "Check that time spacing is regular."
    tags = ("time", "coverage")

    def __init__(
        self,
        *,
        var_name: str,
        time_name: str | None = "time",
        context: TimeCheckContext | None = None,
    ) -> None:
        self.var_name = var_name
        self.time_name = time_name
        self._context = context

    def _resolve_context(self, ds: xr.Dataset) -> TimeCheckContext:
        if self._context is not None:
            return self._context
        return _resolve_time_check_context(
            ds, var_name=self.var_name, time_name=self.time_name
        )

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        context = self._resolve_context(ds)
        if context.time_dim is None:
            return {
                "enabled": True,
                "status": "skipped_no_time",
                "irregular_interval_count": 0,
                "irregular_interval_ranges": [],
                "expected_interval": None,
                "interval_preview": [],
            }

        values = _time_values(context)
        if values.size <= 2:
            return {
                "enabled": True,
                "status": "pass",
                "irregular_interval_count": 0,
                "irregular_interval_ranges": [],
                "expected_interval": None,
                "interval_preview": [],
            }

        try:
            expected_interval = values[1] - values[0]
        except Exception:
            return {
                "enabled": True,
                "status": "skipped_uncomparable_time",
                "irregular_interval_count": 0,
                "irregular_interval_ranges": [],
                "expected_interval": None,
                "interval_preview": [],
                "note": "Time values do not support subtraction for regular-spacing check.",
            }

        irregular_indices: list[int] = []
        interval_preview: list[str] = []
        for idx in range(1, int(values.size)):
            try:
                interval = values[idx] - values[idx - 1]
            except Exception:
                return {
                    "enabled": True,
                    "status": "skipped_uncomparable_time",
                    "irregular_interval_count": 0,
                    "irregular_interval_ranges": [],
                    "expected_interval": str(expected_interval),
                    "interval_preview": interval_preview,
                    "note": "Time values do not support subtraction for regular-spacing check.",
                }
            if len(interval_preview) < 10:
                interval_preview.append(str(interval))
            if not _intervals_match(interval, expected_interval):
                irregular_indices.append(idx)

        return {
            "enabled": True,
            "status": "fail" if irregular_indices else "pass",
            "irregular_interval_count": len(irregular_indices),
            "irregular_interval_ranges": range_records(
                irregular_indices, context.time_coord
            ),
            "expected_interval": str(expected_interval),
            "interval_preview": interval_preview,
        }

    def check(self, ds: xr.Dataset) -> CheckResult:
        report = self.run_report(ds)
        status = status_from_leaf_statuses([str(report.get("status", ""))])
        irregular_count = int(report.get("irregular_interval_count", 0))
        if str(report.get("status", "")).startswith("skip"):
            message = "Time regular-spacing check skipped."
        elif irregular_count > 0:
            message = f"Detected {irregular_count} irregular time intervals."
        else:
            message = "Time spacing is regular."
        return CheckResult(
            check_id=self.id,
            status=status,
            info=CheckInfo(message=message, details={"report": report}),
            fixable=False,
            tags=list(self.tags),
        )


def _single_time_cover_report(
    ds: xr.Dataset,
    *,
    var_name: str,
    time_name: str | None,
    check_time_monotonic: bool,
    check_time_regular_spacing: bool,
) -> dict[str, Any]:
    context = _resolve_time_check_context(ds, var_name=var_name, time_name=time_name)
    missing_check = MissingTimeSlicesCheck(
        var_name=var_name,
        time_name=time_name,
        context=context,
    )
    monotonic_check = TimeMonotonicIncreasingCheck(
        var_name=var_name,
        time_name=time_name,
        context=context,
    )
    regular_spacing_check = TimeRegularSpacingCheck(
        var_name=var_name,
        time_name=time_name,
        context=context,
    )

    time_missing: dict[str, Any] = {}
    time_monotonic: dict[str, Any] = {}
    time_regular_spacing: dict[str, Any] = {}

    suite_checks: list[SuiteCheck] = [
        SuiteCheck(
            check_id=missing_check.id,
            name="Missing Time Slices",
            run=lambda: missing_check.run_report(ds),
            detail=lambda result: (
                f"missing_slices={int(result.get('missing_slice_count', 0))}"
            ),
        )
    ]
    if check_time_monotonic:
        suite_checks.append(
            SuiteCheck(
                check_id=monotonic_check.id,
                name="Monotonic Time Order",
                run=lambda: monotonic_check.run_report(ds),
                detail=lambda result: (
                    f"order_violations={int(result.get('order_violation_count', 0))}"
                ),
            )
        )
    if check_time_regular_spacing:
        suite_checks.append(
            SuiteCheck(
                check_id=regular_spacing_check.id,
                name="Regular Time Spacing",
                run=lambda: regular_spacing_check.run_report(ds),
                detail=lambda result: (
                    f"irregular_intervals={int(result.get('irregular_interval_count', 0))}"
                ),
            )
        )

    suite_report = Suite(name="time_cover", checks=suite_checks).run()
    for item in suite_report["checks"]:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("id", ""))
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        if check_id == "time.missing_slices":
            time_missing = result
        elif check_id == "time.monotonic_increasing":
            time_monotonic = result
        elif check_id == "time.regular_spacing":
            time_regular_spacing = result

    return {
        "group": suite_report["group"],
        "suite": suite_report["suite"],
        "variable": var_name,
        "time_dim": context.time_dim,
        "checks_enabled": {
            "time_missing": True,
            "time_monotonic": bool(check_time_monotonic),
            "time_regular_spacing": bool(check_time_regular_spacing),
        },
        "time_missing": time_missing,
        "time_monotonic": time_monotonic,
        "time_regular_spacing": time_regular_spacing,
        "checks": suite_report["checks"],
        "summary": suite_report["summary"],
        "ok": suite_report["ok"],
    }


def _build_time_cover_report(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    time_name: str | None,
    check_time_monotonic: bool,
    check_time_regular_spacing: bool,
) -> dict[str, Any]:
    selected = choose_time_vars(ds, var_name=var_name, time_name=time_name)
    reports: dict[str, dict[str, Any]] = {}
    for da, _ in selected:
        variable_name = str(da.name)
        reports[variable_name] = _single_time_cover_report(
            ds,
            var_name=variable_name,
            time_name=time_name,
            check_time_monotonic=check_time_monotonic,
            check_time_regular_spacing=check_time_regular_spacing,
        )

    if len(reports) == 1:
        return next(iter(reports.values()))

    suite_checks: list[dict[str, Any]] = []
    for variable_name, per_var in reports.items():
        raw_checks = per_var.get("checks")
        if not isinstance(raw_checks, list):
            continue
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            suite_item = dict(item)
            suite_item["variable"] = variable_name
            suite_checks.append(suite_item)

    suite_report = Suite.report_from_items("time_cover", suite_checks)
    return {
        "group": suite_report["group"],
        "suite": suite_report["suite"],
        "mode": "all_variables",
        "checked_variable_count": len(reports),
        "checked_variables": list(reports.keys()),
        "reports": reports,
        "checks": suite_report["checks"],
        "summary": suite_report["summary"],
        "ok": suite_report["ok"],
    }


def run_time_cover_report(
    ds: xr.Dataset,
    *,
    config: TimeCoverConfig | None = None,
    var_name: str | None = None,
    time_name: str | None = "time",
    check_time_monotonic: bool = False,
    check_time_regular_spacing: bool = False,
) -> dict[str, Any]:
    resolved_config = config or TimeCoverConfig(
        var_name=var_name,
        time_name=time_name,
        check_time_monotonic=check_time_monotonic,
        check_time_regular_spacing=check_time_regular_spacing,
    )
    return _build_time_cover_report(
        ds,
        var_name=resolved_config.var_name,
        time_name=resolved_config.time_name,
        check_time_monotonic=bool(resolved_config.check_time_monotonic),
        check_time_regular_spacing=bool(resolved_config.check_time_regular_spacing),
    )


def _time_statuses(report: dict[str, Any]) -> list[str]:
    if report.get("mode") == "all_variables":
        grouped = report.get("reports")
        if not isinstance(grouped, dict):
            return []
        statuses: list[str] = []
        for per_var in grouped.values():
            if not isinstance(per_var, dict):
                continue
            statuses.extend(_time_statuses(per_var))
        return statuses

    checks = report.get("checks")
    if isinstance(checks, list):
        statuses = [
            str(item.get("status", "")).strip().lower()
            for item in checks
            if isinstance(item, dict)
        ]
        return [status for status in statuses if status]

    return leaf_statuses(
        report,
        ("time_missing", "time_monotonic", "time_regular_spacing"),
    )


class TimeCoverCheck(Check):
    id = "nc_check.time_cover"
    description = "Time coverage checks."
    tags = ("time", "coverage")

    def __init__(
        self,
        *,
        var_name: str | None = None,
        time_name: str | None = "time",
        check_time_monotonic: bool = False,
        check_time_regular_spacing: bool = False,
    ) -> None:
        self.var_name = var_name
        self.time_name = time_name
        self.check_time_monotonic = check_time_monotonic
        self.check_time_regular_spacing = check_time_regular_spacing

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        return run_time_cover_report(
            ds,
            var_name=self.var_name,
            time_name=self.time_name,
            check_time_monotonic=self.check_time_monotonic,
            check_time_regular_spacing=self.check_time_regular_spacing,
        )

    def check(self, ds: xr.Dataset) -> CheckResult:
        report = self.run_report(ds)
        statuses = _time_statuses(report)
        status = status_from_leaf_statuses(statuses)
        return CheckResult(
            check_id=self.id,
            status=status,
            info=CheckInfo(
                message=(
                    f"Time cover check completed for {len(statuses)} check outcomes."
                    if statuses
                    else "Time cover check completed."
                ),
                details={"report": report, "statuses": statuses},
            ),
            fixable=False,
            tags=list(self.tags),
        )


def check_time_cover(
    ds: xr.Dataset,
    *,
    var_name: str | None = None,
    time_name: str | None = "time",
    check_time_monotonic: bool = False,
    check_time_regular_spacing: bool = False,
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    """Run time-coverage checks and report missing time-slice ranges."""
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    report = run_time_cover_report(
        ds,
        config=TimeCoverConfig(
            var_name=var_name,
            time_name=time_name,
            check_time_monotonic=check_time_monotonic,
            check_time_regular_spacing=check_time_regular_spacing,
        ),
    )
    if resolved_format == "tables":
        items = (
            list(report["reports"].values())
            if report.get("mode") == "all_variables"
            else [report]
        )
        print_pretty_time_cover_reports(items)
        return None
    if resolved_format == "html":
        if report.get("mode") == "all_variables":
            html_report = render_pretty_time_cover_reports_html(
                list(report["reports"].values())
            )
        else:
            html_report = render_pretty_time_cover_report_html(report)
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report
    return report
