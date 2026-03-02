from __future__ import annotations

from collections.abc import Iterable
from os import PathLike

import xarray as xr

from .api import (
    run_cfchecker_report,
    run_cf_compliance,
    run_ocean_cover,
    run_time_cover,
)
from .models import AtomicCheckResult, CheckStatus, SuiteReport, SuiteSummary
from .reporting import save_html_report


def _merge_results(
    reports: Iterable[SuiteReport],
) -> dict[str, dict[str, dict[str, AtomicCheckResult]]]:
    merged: dict[str, dict[str, dict[str, AtomicCheckResult]]] = {}
    for report in reports:
        for data_scope, scope_items in report.results.items():
            scope_bucket = merged.setdefault(data_scope, {})
            for scope_item, checks_by_name in scope_items.items():
                item_bucket = scope_bucket.setdefault(scope_item, {})
                for check_name, check in checks_by_name.items():
                    unique_name = check_name
                    counter = 2
                    while unique_name in item_bucket:
                        unique_name = f"{check_name}__{counter}"
                        counter += 1
                    item_bucket[unique_name] = check
    return merged


def _summary_from_checks(checks: list[AtomicCheckResult]) -> SuiteSummary:
    passed = sum(1 for item in checks if item.status == CheckStatus.passed)
    skipped = sum(1 for item in checks if item.status == CheckStatus.skipped)
    failed = sum(1 for item in checks if item.status == CheckStatus.failed)

    if failed > 0:
        overall_status = CheckStatus.failed
    elif passed > 0:
        overall_status = CheckStatus.passed
    else:
        overall_status = CheckStatus.skipped

    return SuiteSummary(
        checks_run=len(checks),
        passed=passed,
        skipped=skipped,
        failed=failed,
        overall_status=overall_status,
    )


def _maybe_save_report(
    report: SuiteReport,
    html_report_fname: str | PathLike[str] | None,
) -> None:
    if html_report_fname is None:
        return
    save_html_report(report, html_report_fname)


@xr.register_dataset_accessor("check")
class CheckAccessor:
    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def compliance(
        self,
        *,
        html_report_fname: str | PathLike[str] | None = None,
        **kwargs: object,
    ) -> SuiteReport:
        report = run_cf_compliance(self._obj, **kwargs)
        _maybe_save_report(report, html_report_fname)
        return report

    def cfchecker_report(
        self,
        *,
        html_report_fname: str | PathLike[str] | None = None,
        **kwargs: object,
    ) -> SuiteReport:
        report = run_cfchecker_report(self._obj, **kwargs)
        _maybe_save_report(report, html_report_fname)
        return report

    def ocean_cover(
        self,
        *,
        html_report_fname: str | PathLike[str] | None = None,
        **kwargs: object,
    ) -> SuiteReport:
        report = run_ocean_cover(self._obj, **kwargs)
        _maybe_save_report(report, html_report_fname)
        return report

    def time_cover(
        self,
        *,
        html_report_fname: str | PathLike[str] | None = None,
        **kwargs: object,
    ) -> SuiteReport:
        report = run_time_cover(self._obj, **kwargs)
        _maybe_save_report(report, html_report_fname)
        return report

    def all(
        self,
        *,
        compliance: bool = True,
        cfchecker_report: bool = False,
        ocean_cover: bool = True,
        time_cover: bool = True,
        time_cover_var_name: str | None = None,
        time_cover_time_name: str | None = "time",
        time_cover_check_monotonic: bool = False,
        time_cover_check_regular_spacing: bool = False,
        html_report_fname: str | PathLike[str] | None = None,
    ) -> SuiteReport:
        reports: list[SuiteReport] = []

        if compliance:
            reports.append(run_cf_compliance(self._obj))
        if cfchecker_report:
            reports.append(run_cfchecker_report(self._obj))
        if ocean_cover:
            reports.append(run_ocean_cover(self._obj))
        if time_cover:
            reports.append(
                run_time_cover(
                    self._obj,
                    var_name=time_cover_var_name,
                    time_name=time_cover_time_name,
                    check_time_monotonic=time_cover_check_monotonic,
                    check_time_regular_spacing=time_cover_check_regular_spacing,
                )
            )

        checks: list[AtomicCheckResult] = []
        for report in reports:
            checks.extend(report.checks)

        dataset_html: str | None = None
        for report in reports:
            if report.dataset_html is not None:
                dataset_html = report.dataset_html
                break

        report = SuiteReport(
            suite_name="all_checks",
            plugin="check_accessor",
            checks=checks,
            summary=_summary_from_checks(checks),
            results=_merge_results(reports),
            dataset_html=dataset_html,
        )
        _maybe_save_report(report, html_report_fname)
        return report
