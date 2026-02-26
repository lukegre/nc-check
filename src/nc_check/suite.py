from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Iterable, Literal

import numpy as np
import xarray as xr


from .dataset import CanonicalDataset
from .models import AtomicCheckResult, CheckStatus, SuiteReport, SuiteSummary

DataScope = Literal["data_vars", "dims", "coords"]
AtomicCheckFn = Callable[[xr.DataArray], AtomicCheckResult]


@dataclass(frozen=True)
class CheckDefinition:
    name: str
    data_scope: DataScope
    plugin: str = "local"

    def run(self, data: xr.DataArray) -> AtomicCheckResult:
        raise NotImplementedError


@dataclass(frozen=True)
class CallableCheck(CheckDefinition):
    fn: AtomicCheckFn | None = None

    def run(self, data: xr.DataArray) -> AtomicCheckResult:
        if self.fn is None:
            raise ValueError("CallableCheck requires a callable 'fn'.")
        return self.fn(data)


def run_atomic_check(
    data: xr.DataArray,
    definition: CheckDefinition,
    *,
    scope_item: str | None = None,
) -> AtomicCheckResult:
    try:
        result = definition.run(data)
    except Exception as exc:  # defensive wrapper for third-party checks
        details = {"exception_type": type(exc).__name__}
        if scope_item is not None:
            details["scope_item"] = scope_item
            details["data_scope"] = definition.data_scope
        return AtomicCheckResult.failed_result(
            name=definition.name,
            info=f"Check raised {type(exc).__name__}: {exc}",
            details=details,
        )

    if not isinstance(result, AtomicCheckResult):
        details = {"expected": "AtomicCheckResult", "actual": type(result).__name__}
        if scope_item is not None:
            details["scope_item"] = scope_item
            details["data_scope"] = definition.data_scope
        return AtomicCheckResult.failed_result(
            name=definition.name,
            info="Check returned an invalid result type.",
            details=details,
        )

    if scope_item is not None and "scope_item" not in result.details:
        result = AtomicCheckResult(
            name=result.name,
            status=result.status,
            info=result.info,
            details={
                **result.details,
                "data_scope": definition.data_scope,
                "scope_item": scope_item,
            },
        )

    if result.name != definition.name:
        return AtomicCheckResult(
            name=definition.name,
            status=result.status,
            info=result.info,
            details={**result.details, "reported_name": result.name},
        )

    return result


class CheckSuite:
    def __init__(
        self,
        *,
        name: str,
        checks: Iterable[CheckDefinition],
        plugin: str | None = None,
    ) -> None:
        self.name = name
        self.checks = list(checks)
        self.plugin = plugin

    def run(self, dataset: CanonicalDataset) -> SuiteReport:
        results: list[AtomicCheckResult] = []
        hierarchy: dict[str, dict[str, dict[str, AtomicCheckResult]]] = {}

        checks_by_scope: dict[DataScope, list[CheckDefinition]] = defaultdict(list)
        for definition in self.checks:
            checks_by_scope[definition.data_scope].append(definition)

        for data_scope in ("data_vars", "coords", "dims"):
            scope_definitions = checks_by_scope.get(data_scope, [])
            if not scope_definitions:
                continue

            scope_targets = _scope_targets(dataset, data_scope)
            if not scope_targets:
                for definition in scope_definitions:
                    results.append(_empty_scope_targets(definition))
                continue

            scope_hierarchy: dict[str, dict[str, AtomicCheckResult]] = {}
            for scope_item, data in scope_targets:
                variable_results: dict[str, AtomicCheckResult] = {}
                for definition in scope_definitions:
                    scoped_definition = CallableCheck(
                        name=f"{definition.name}[{data_scope}:{scope_item}]",
                        data_scope=data_scope,
                        plugin=definition.plugin,
                        fn=definition.run,
                    )
                    result = run_atomic_check(
                        _with_dataset_attrs(data, dataset.attrs),
                        scoped_definition,
                        scope_item=scope_item,
                    )
                    results.append(result)
                    variable_results[definition.name] = result
                scope_hierarchy[str(scope_item)] = variable_results
            hierarchy[data_scope] = scope_hierarchy

        summary = _summary_from_results(results)
        return SuiteReport(
            suite_name=self.name,
            plugin=self.plugin,
            checks=results,
            summary=summary,
            results=hierarchy,
            dataset_html=_dataset_repr_html(dataset),
        )

    def make_web_report(
        self, dataset: CanonicalDataset, html_report_fname: str
    ) -> None:
        from .reporting import save_html_report

        report = self.run(dataset)

        save_html_report(report, html_report_fname)

    def __repr__(self) -> str:
        return f"CheckSuite(name={self.name}, checks={len(self.checks)}, plugin={self.plugin})"


def _empty_scope_targets(definition: CheckDefinition) -> AtomicCheckResult:
    return AtomicCheckResult.skipped_result(
        name=definition.name,
        info=(
            f"Check skipped (dataset has no items in '{definition.data_scope}' scope)."
        ),
        details={
            "reason": "no_scope_targets",
            "data_scope": definition.data_scope,
        },
    )


def _summary_from_results(results: list[AtomicCheckResult]) -> SuiteSummary:
    passed = sum(1 for result in results if result.status == CheckStatus.passed)
    skipped = sum(1 for result in results if result.status == CheckStatus.skipped)
    failed = sum(1 for result in results if result.status == CheckStatus.failed)

    if failed > 0:
        overall = CheckStatus.failed
    elif passed > 0:
        overall = CheckStatus.passed
    else:
        overall = CheckStatus.skipped

    return SuiteSummary(
        checks_run=len(results),
        passed=passed,
        skipped=skipped,
        failed=failed,
        overall_status=overall,
    )


def _with_dataset_attrs(
    data: xr.DataArray, dataset_attrs: dict[str, object]
) -> xr.DataArray:
    enriched = data.copy(deep=False)
    attrs = dict(dataset_attrs)
    attrs.update(data.attrs)
    enriched.attrs = attrs
    return enriched


def _scope_targets(
    dataset: CanonicalDataset, data_scope: DataScope
) -> list[tuple[str, xr.DataArray]]:
    if data_scope == "data_vars":
        return [(str(name), dataset[name]) for name in dataset.data_vars]

    if data_scope == "coords":
        return [(str(name), dataset.coords[name]) for name in dataset.coords]

    if data_scope == "dims":
        targets: list[tuple[str, xr.DataArray]] = []
        for dim_name, size in dataset.sizes.items():
            if dim_name in dataset.coords:
                targets.append((str(dim_name), dataset.coords[dim_name]))
                continue
            targets.append(
                (
                    str(dim_name),
                    xr.DataArray(
                        np.arange(int(size)),
                        dims=(str(dim_name),),
                        name=str(dim_name),
                    ),
                )
            )
        return targets

    raise ValueError(f"Invalid data_scope '{data_scope}'.")


def _dataset_repr_html(dataset: CanonicalDataset) -> str | None:
    repr_html = getattr(dataset, "_repr_html_", None)
    if not callable(repr_html):
        return None
    try:
        return str(repr_html())
    except Exception:
        return None
