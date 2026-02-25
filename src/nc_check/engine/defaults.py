from __future__ import annotations

from typing import Any, cast

import xarray as xr

from ..checks.ocean import check_ocean_cover
from ..checks.time_cover import TimeCoverConfig, run_time_cover_report
from ..core import check_dataset_compliant
from ..contracts import SummaryStatus
from .registry import RegisteredCheck, has_check, register_check

_DEFAULT_ORDER = ("compliance", "ocean_cover", "time_cover")


def _status_kind(status: Any) -> str:
    if isinstance(status, bool):
        return "pass" if status else "fail"
    normalized = str(status).strip().lower()
    if normalized in {"pass", "passed", "ok", "success", "true"}:
        return "pass"
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return "fail"
    if normalized in {"skip", "skipped"} or normalized.startswith("skip"):
        return "skip"
    return "warn"


def _combine_statuses(statuses: list[str]) -> SummaryStatus:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _count_to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return 0


def _summary_status(report: dict[str, Any]) -> SummaryStatus | None:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return None
    status = str(summary.get("overall_status", "")).strip().lower()
    if status == "fail":
        return "fail"
    if status == "warn":
        return "warn"
    if status == "pass":
        return "pass"
    return None


def _status_from_compliance_report(report: dict[str, Any]) -> SummaryStatus:
    checker_error = report.get("checker_error")
    if checker_error is not None:
        return "fail"

    counts = report.get("counts")
    if isinstance(counts, dict):
        fatal = _count_to_int(counts.get("fatal"))
        error = _count_to_int(counts.get("error"))
        warn = _count_to_int(counts.get("warn"))
        if fatal + error > 0:
            return "fail"
        if warn > 0:
            return "warn"
        return "pass"

    finding_statuses: list[str] = []
    for item in report.get("global", []) or []:
        if isinstance(item, dict):
            finding_statuses.append(_status_kind(item.get("severity")))
        else:
            finding_statuses.append("warn")
    for scope in ("coordinates", "variables"):
        entries = report.get(scope)
        if not isinstance(entries, dict):
            continue
        for findings in entries.values():
            if not isinstance(findings, list):
                continue
            for finding in findings:
                if isinstance(finding, dict):
                    finding_statuses.append(_status_kind(finding.get("severity")))
                else:
                    finding_statuses.append("warn")
    if not finding_statuses:
        return "pass"
    return _combine_statuses(finding_statuses)


def _status_from_ocean_report(report: dict[str, Any]) -> SummaryStatus:
    summary_status = _summary_status(report)
    if summary_status is not None:
        return summary_status

    if report.get("mode") == "all_variables":
        grouped = report.get("reports")
        if not isinstance(grouped, dict) or not grouped:
            return cast(SummaryStatus, _status_kind(report.get("ok")))
        return _combine_statuses(
            [
                _status_from_ocean_report(per_var)
                for per_var in grouped.values()
                if isinstance(per_var, dict)
            ]
        )

    statuses: list[str] = []
    for check_name in ("edge_of_map", "land_ocean_offset"):
        check_report = report.get(check_name)
        if isinstance(check_report, dict):
            statuses.append(_status_kind(check_report.get("status")))
    if not statuses:
        statuses.append(_status_kind(report.get("ok")))
    return _combine_statuses(statuses)


def _status_from_time_cover_report(report: dict[str, Any]) -> SummaryStatus:
    summary_status = _summary_status(report)
    if summary_status is not None:
        return summary_status

    if report.get("mode") == "all_variables":
        grouped = report.get("reports")
        if not isinstance(grouped, dict) or not grouped:
            return cast(SummaryStatus, _status_kind(report.get("ok")))
        return _combine_statuses(
            [
                _status_from_time_cover_report(per_var)
                for per_var in grouped.values()
                if isinstance(per_var, dict)
            ]
        )
    statuses: list[str] = []
    time_missing = report.get("time_missing")
    if isinstance(time_missing, dict):
        statuses.append(_status_kind(time_missing.get("status")))
    if not statuses:
        statuses.append(_status_kind(report.get("ok")))
    return _combine_statuses(statuses)


def _compliance_detail(report: dict[str, Any]) -> str:
    counts = report.get("counts")
    if isinstance(counts, dict):
        return (
            f"fatal={_count_to_int(counts.get('fatal'))} "
            f"error={_count_to_int(counts.get('error'))} "
            f"warn={_count_to_int(counts.get('warn'))}"
        )
    checker_error = report.get("checker_error")
    if checker_error is not None:
        return "checker_error=true"
    return "completed"


def _ocean_cover_detail(report: dict[str, Any]) -> str:
    summary = report.get("summary")
    if isinstance(summary, dict):
        return (
            f"checks={_count_to_int(summary.get('checks_run'))} "
            f"fail={_count_to_int(summary.get('failing_checks'))} "
            f"warn_or_skip={_count_to_int(summary.get('warnings_or_skips'))}"
        )
    if report.get("mode") == "all_variables":
        checked = _count_to_int(report.get("checked_variable_count"))
        return f"variables={checked}"
    edge = (
        report.get("edge_of_map") if isinstance(report.get("edge_of_map"), dict) else {}
    )
    offset = (
        report.get("land_ocean_offset")
        if isinstance(report.get("land_ocean_offset"), dict)
        else {}
    )
    return (
        f"missing_longitudes={_count_to_int(edge.get('missing_longitude_count'))} "
        f"mismatches={_count_to_int(offset.get('mismatch_count'))}"
    )


def _time_cover_detail(report: dict[str, Any]) -> str:
    summary = report.get("summary")
    if isinstance(summary, dict):
        return (
            f"checks={_count_to_int(summary.get('checks_run'))} "
            f"fail={_count_to_int(summary.get('failing_checks'))} "
            f"warn_or_skip={_count_to_int(summary.get('warnings_or_skips'))}"
        )
    if report.get("mode") == "all_variables":
        checked = _count_to_int(report.get("checked_variable_count"))
        return f"variables={checked}"
    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )
    return f"missing_slices={_count_to_int(time_missing.get('missing_slice_count'))}"


def _run_compliance_report(ds: xr.Dataset, options: dict[str, Any]) -> dict[str, Any]:
    report = check_dataset_compliant(
        ds,
        cf_version=str(options.get("cf_version", "1.12")),
        standard_name_table_xml=options.get("standard_name_table_xml"),
        cf_area_types_xml=options.get("cf_area_types_xml"),
        cf_region_names_xml=options.get("cf_region_names_xml"),
        cache_tables=bool(options.get("cache_tables", False)),
        domain=options.get("domain"),
        fallback_to_heuristic=bool(options.get("fallback_to_heuristic", True)),
        engine=str(options.get("engine", "auto")),
        conventions=options.get("conventions"),
        report_format="python",
    )
    return report if isinstance(report, dict) else {}


def _run_ocean_cover_report(ds: xr.Dataset, options: dict[str, Any]) -> dict[str, Any]:
    report = check_ocean_cover(
        ds,
        var_name=options.get("var_name"),
        lon_name=options.get("lon_name"),
        lat_name=options.get("lat_name"),
        time_name=options.get("time_name", "time"),
        check_edge_of_map=bool(options.get("check_edge_of_map", True)),
        check_land_ocean_offset=bool(options.get("check_land_ocean_offset", True)),
        report_format="python",
    )
    return report if isinstance(report, dict) else {}


def _run_time_cover_v2_report(
    ds: xr.Dataset, options: dict[str, Any]
) -> dict[str, Any]:
    config = TimeCoverConfig(
        var_name=options.get("var_name"),
        time_name=options.get("time_name", "time"),
    )
    return run_time_cover_report(ds, config=config)


def register_default_checks() -> None:
    if has_check("compliance"):
        return

    register_check(
        RegisteredCheck(
            key="compliance",
            run_report=_run_compliance_report,
            resolve_status=_status_from_compliance_report,
            resolve_detail=_compliance_detail,
        )
    )
    register_check(
        RegisteredCheck(
            key="ocean_cover",
            run_report=_run_ocean_cover_report,
            resolve_status=_status_from_ocean_report,
            resolve_detail=_ocean_cover_detail,
        )
    )
    register_check(
        RegisteredCheck(
            key="time_cover",
            run_report=_run_time_cover_v2_report,
            resolve_status=_status_from_time_cover_report,
            resolve_detail=_time_cover_detail,
        )
    )


def default_check_order() -> tuple[str, ...]:
    return _DEFAULT_ORDER
