from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import xarray as xr

from ..models import AtomicCheckResult, CheckStatus, SuiteReport, SuiteSummary
from ..suite import CheckSuite

_SUITE_NAME = "cfchecker_report"
_SEVERITY_PATTERN = re.compile(
    r"^\s*(ERROR|WARNING|WARN|INFO|FATAL)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(r"^\(([^)]+)\):\s*(.+?)\s*$")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_CANONICAL_COORD_NAMES = {"time", "lat", "lon"}


class CheckSuiteCFchecker(CheckSuite):
    def run(self, dataset: xr.Dataset) -> SuiteReport:
        return build_cfchecker_suite_report(dataset)


def _normalize_level(raw_level: str) -> str:
    level = raw_level.strip().upper()
    if level == "WARN":
        return "WARNING"
    if level == "FATAL":
        return "ERROR"
    return level


def _status_for_level(level: str) -> CheckStatus:
    normalized = _normalize_level(level)
    if normalized == "ERROR":
        return CheckStatus.failed
    if normalized == "WARNING":
        return CheckStatus.skipped
    return CheckStatus.passed


def _slug(value: str) -> str:
    text = _SLUG_PATTERN.sub("_", value.lower()).strip("_")
    return text or "message"


def _parse_code_and_message(message: str) -> tuple[str, str]:
    cleaned = message.strip()
    match = _CODE_PATTERN.match(cleaned)
    if match is None:
        return "message", cleaned
    return match.group(1).strip(), match.group(2).strip()


def _split_message_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    for raw_line in text.splitlines():
        for chunk in re.split(r"\s+\|\s+", raw_line):
            cleaned = chunk.strip()
            if cleaned:
                chunks.append(cleaned)
    return chunks


def _explode_levelled_message(
    message: str, *, default_level: str | None
) -> list[tuple[str, str]]:
    expanded: list[tuple[str, str]] = []
    for chunk in _split_message_chunks(message):
        match = _SEVERITY_PATTERN.match(chunk)
        if match is not None:
            level = _normalize_level(match.group(1))
            value = match.group(2).strip()
            if value:
                expanded.append((level, value))
            continue
        if default_level is None:
            continue
        expanded.append((default_level, chunk))
    return expanded


def _parse_cfchecker_messages(output: str) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for level, message in _explode_levelled_message(output, default_level=None):
        messages.append((level, message))
    return messages


def _messages_from_result(result: Any) -> list[tuple[str, str, str, str]]:
    if not isinstance(result, Mapping):
        return []

    messages: list[tuple[str, str, str, str]] = []

    global_results = result.get("global")
    if isinstance(global_results, Mapping):
        for level in ("FATAL", "ERROR", "WARN", "INFO"):
            entries = global_results.get(level, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                default_level = _normalize_level(level)
                for parsed_level, parsed_message in _explode_levelled_message(
                    str(entry),
                    default_level=default_level,
                ):
                    messages.append(
                        ("dataset", "dataset", parsed_level, parsed_message)
                    )

    variables = result.get("variables")
    if isinstance(variables, Mapping):
        for variable_name, variable_results in variables.items():
            if not isinstance(variable_results, Mapping):
                continue
            variable = str(variable_name)
            for level in ("FATAL", "ERROR", "WARN", "INFO"):
                entries = variable_results.get(level, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    default_level = _normalize_level(level)
                    for parsed_level, parsed_message in _explode_levelled_message(
                        str(entry),
                        default_level=default_level,
                    ):
                        messages.append(
                            ("data_vars", variable, parsed_level, parsed_message)
                        )

    return messages


def _summary_from_checks(checks: list[AtomicCheckResult]) -> SuiteSummary:
    passed = sum(1 for check in checks if check.status == CheckStatus.passed)
    skipped = sum(1 for check in checks if check.status == CheckStatus.skipped)
    failed = sum(1 for check in checks if check.status == CheckStatus.failed)

    if failed > 0:
        overall = CheckStatus.failed
    elif passed > 0:
        overall = CheckStatus.passed
    else:
        overall = CheckStatus.skipped

    return SuiteSummary(
        checks_run=len(checks),
        passed=passed,
        skipped=skipped,
        failed=failed,
        overall_status=overall,
    )


def _is_string_diff_type_error(exc: Exception) -> bool:
    message = str(exc)
    return "unsupported operand type(s) for -: 'str' and 'str'" in message


def _sanitize_dataset_for_cfchecker(
    dataset: xr.Dataset,
) -> tuple[xr.Dataset, list[str]]:
    sanitized = dataset.copy(deep=False)
    removed_or_replaced: list[str] = []

    for coord_name in list(sanitized.coords):
        if coord_name in _CANONICAL_COORD_NAMES:
            continue
        coord = sanitized.coords[coord_name]
        if getattr(coord.dtype, "kind", "") not in {"U", "S", "O"}:
            continue

        removed_or_replaced.append(str(coord_name))
        if coord_name in sanitized.dims:
            sanitized = sanitized.assign_coords(
                {
                    coord_name: np.arange(
                        int(sanitized.sizes[coord_name]), dtype=np.int64
                    )
                }
            )
        else:
            sanitized = sanitized.drop_vars(coord_name)

    return sanitized, removed_or_replaced


def _run_cfchecker_with_fallback(
    dataset: xr.Dataset,
) -> tuple[dict[str, Any], str, list[str]]:
    source_path = _dataset_source_path(dataset)
    if source_path is not None:
        try:
            run_result, combined_output = _run_cfchecker_in_process(source_path)
            return run_result, combined_output, []
        except TypeError as exc:
            if not _is_string_diff_type_error(exc):
                raise

            sanitized, removed_or_replaced = _sanitize_dataset_for_cfchecker(dataset)
            if not removed_or_replaced:
                raise

            with tempfile.TemporaryDirectory(prefix="nc-check-cfchecker-") as tmpdir:
                path = Path(tmpdir) / "input.nc"
                sanitized.to_netcdf(path)
                run_result, combined_output = _run_cfchecker_in_process(path)
                return run_result, combined_output, removed_or_replaced

    with tempfile.TemporaryDirectory(prefix="nc-check-cfchecker-") as tmpdir:
        path = Path(tmpdir) / "input.nc"
        dataset.to_netcdf(path)

        try:
            run_result, combined_output = _run_cfchecker_in_process(path)
            return run_result, combined_output, []
        except TypeError as exc:
            if not _is_string_diff_type_error(exc):
                raise

            sanitized, removed_or_replaced = _sanitize_dataset_for_cfchecker(dataset)
            if not removed_or_replaced:
                raise

            sanitized.to_netcdf(path)
            run_result, combined_output = _run_cfchecker_in_process(path)
            return run_result, combined_output, removed_or_replaced


def _dataset_source_path(dataset: xr.Dataset) -> Path | None:
    source = dataset.attrs.get("source")
    if source is None:
        return None
    source_path = Path(str(source).strip())
    if not source_path.name.endswith(".nc"):
        return None
    if not source_path.exists() or not source_path.is_file():
        return None
    return source_path


def _run_cfchecker_in_process(path: Path) -> tuple[dict[str, Any], str]:
    from cfchecker.cfchecks import CFChecker

    checker = CFChecker(silent=False)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        raw_result = checker.checker(str(path))
    totals_raw = checker.get_total_counts()

    result: dict[str, Any] = {
        "raw_result": raw_result,
        "fatal_count": int(totals_raw.get("FATAL", 0)),
        "error_count": int(totals_raw.get("ERROR", 0)),
        "warning_count": int(totals_raw.get("WARN", 0)),
        "info_count": int(totals_raw.get("INFO", 0)),
    }
    output = "\n".join(
        part for part in (stdout_buffer.getvalue(), stderr_buffer.getvalue()) if part
    ).strip()
    return result, output


def _convert_to_checks(
    *,
    raw_result: Any,
    combined_output: str,
) -> tuple[
    list[AtomicCheckResult],
    dict[str, dict[str, dict[str, AtomicCheckResult]]],
]:
    scoped_messages = _messages_from_result(raw_result)

    if not scoped_messages and combined_output:
        parsed_output_messages = _parse_cfchecker_messages(combined_output)
        scoped_messages = [
            ("dataset", "dataset", level, message)
            for level, message in parsed_output_messages
        ]

    checks: list[AtomicCheckResult] = []
    hierarchy: dict[str, dict[str, dict[str, AtomicCheckResult]]] = {}
    counters: dict[tuple[str, str, str, str], int] = defaultdict(int)

    for data_scope, scope_item, level, raw_message in scoped_messages:
        code, message = _parse_code_and_message(raw_message)
        level_slug = _slug(level)
        code_slug = _slug(code)
        counter_key = (data_scope, scope_item, level_slug, code_slug)
        counters[counter_key] += 1
        idx = counters[counter_key]

        base_name = f"cfchecker.{level_slug}.{code_slug}.{idx}"
        scoped_name = f"{base_name}[{data_scope}:{scope_item}]"
        status = _status_for_level(level)

        result = AtomicCheckResult(
            name=scoped_name,
            status=status,
            info=message,
            details={
                "data_scope": data_scope,
                "scope_item": scope_item,
                "cfchecker_level": level,
                "cfchecker_code": code,
            },
        )
        checks.append(result)

        scope_bucket = hierarchy.setdefault(data_scope, {})
        item_bucket = scope_bucket.setdefault(scope_item, {})
        item_bucket[base_name] = result

    if checks:
        return checks, hierarchy

    no_issues = AtomicCheckResult.passed_result(
        name="cfchecker.no_issues[dataset:dataset]",
        info="cfchecker returned no ERROR/WARNING/INFO messages.",
        details={"data_scope": "dataset", "scope_item": "dataset"},
    )
    return (
        [no_issues],
        {"dataset": {"dataset": {"cfchecker.no_issues": no_issues}}},
    )


def build_cfchecker_suite_report(dataset: xr.Dataset) -> SuiteReport:
    try:
        run_result, combined_output, removed_or_replaced = _run_cfchecker_with_fallback(
            dataset
        )
    except Exception as exc:
        failed_check = AtomicCheckResult.failed_result(
            name="cfchecker.execution_error[dataset:dataset]",
            info=f"cfchecker execution failed: {type(exc).__name__}: {exc}",
            details={
                "data_scope": "dataset",
                "scope_item": "dataset",
                "exception_type": type(exc).__name__,
            },
        )
        checks = [failed_check]
        return SuiteReport(
            suite_name=_SUITE_NAME,
            checks=checks,
            summary=_summary_from_checks(checks),
            results={
                "dataset": {"dataset": {"cfchecker.execution_error": failed_check}}
            },
        )

    checks, hierarchy = _convert_to_checks(
        raw_result=run_result.get("raw_result"),
        combined_output=combined_output,
    )

    if removed_or_replaced:
        removed_text = ", ".join(sorted(removed_or_replaced))
        preprocess_check = AtomicCheckResult.skipped_result(
            name="cfchecker.preprocess.removed_string_coords[dataset:dataset]",
            info=(
                "Removed/replaced non-numeric coordinates before cfchecker run: "
                f"{removed_text}"
            ),
            details={
                "data_scope": "dataset",
                "scope_item": "dataset",
                "removed_or_replaced_coords": sorted(removed_or_replaced),
            },
        )
        checks.insert(0, preprocess_check)
        hierarchy.setdefault("dataset", {}).setdefault("dataset", {})[
            "cfchecker.preprocess.removed_string_coords"
        ] = preprocess_check

    return SuiteReport(
        suite_name=_SUITE_NAME,
        checks=checks,
        summary=_summary_from_checks(checks),
        results=hierarchy,
    )


cfchecker_report_suite = CheckSuiteCFchecker(name=_SUITE_NAME, checks=[])
