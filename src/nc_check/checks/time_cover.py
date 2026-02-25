from __future__ import annotations

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


def _time_missing_check(da: xr.DataArray, *, time_dim: str | None) -> dict[str, Any]:
    if time_dim is None:
        return {
            "enabled": True,
            "status": "skipped_no_time",
            "missing_slice_count": 0,
            "missing_slice_ranges": [],
        }

    missing = missing_mask(da)
    reduce_dims = [dim for dim in missing.dims if dim != time_dim]
    if reduce_dims:
        missing = missing.all(dim=reduce_dims)
    missing_time_indices = np.flatnonzero(
        np.asarray(missing.values, dtype=bool)
    ).tolist()
    time_coord = da.coords.get(time_dim)
    return {
        "enabled": True,
        "status": "fail" if missing_time_indices else "pass",
        "missing_slice_count": len(missing_time_indices),
        "missing_slice_ranges": range_records(missing_time_indices, time_coord),
    }


def _single_time_cover_report(
    da: xr.DataArray,
    *,
    time_dim: str | None,
) -> dict[str, Any]:
    time_missing = _time_missing_check(da, time_dim=time_dim)
    suite_report = Suite(
        name="time_cover",
        checks=[
            SuiteCheck(
                check_id="time.missing_slices",
                name="Missing Time Slices",
                run=lambda: time_missing,
                detail=lambda result: (
                    f"missing_slices={int(result.get('missing_slice_count', 0))}"
                ),
            )
        ],
    ).run()
    return {
        "suite": suite_report["suite"],
        "variable": str(da.name),
        "time_dim": time_dim,
        "time_missing": time_missing,
        "checks": suite_report["checks"],
        "summary": suite_report["summary"],
        "ok": suite_report["ok"],
    }


def _build_time_cover_report(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    time_name: str | None,
) -> dict[str, Any]:
    selected = choose_time_vars(ds, var_name=var_name, time_name=time_name)
    reports: dict[str, dict[str, Any]] = {}
    for da, time_dim in selected:
        reports[str(da.name)] = _single_time_cover_report(da, time_dim=time_dim)

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
) -> dict[str, Any]:
    resolved_config = config or TimeCoverConfig(var_name=var_name, time_name=time_name)
    return _build_time_cover_report(
        ds,
        var_name=resolved_config.var_name,
        time_name=resolved_config.time_name,
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
    ) -> None:
        self.var_name = var_name
        self.time_name = time_name

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        return run_time_cover_report(
            ds,
            var_name=self.var_name,
            time_name=self.time_name,
        )

    def check(self, ds: xr.Dataset) -> CheckResult:
        report = self.run_report(ds)
        statuses = leaf_statuses(report, ("time_missing",))
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
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    """Run time-coverage checks and report missing time-slice ranges."""
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    report = run_time_cover_report(
        ds,
        config=TimeCoverConfig(var_name=var_name, time_name=time_name),
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
