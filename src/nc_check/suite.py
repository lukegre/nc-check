from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import xarray as xr


from .dataset import CanonicalDataset
from .models import AtomicCheckResult, CheckStatus, SuiteReport, SuiteSummary

DataScope = Literal["dims", "coords", "dataset", "data_vars"]
DataArrayCheckFn = Callable[[xr.DataArray], AtomicCheckResult]
DatasetCheckFn = Callable[[CanonicalDataset], AtomicCheckResult]
AtomicCheckFn = DataArrayCheckFn | DatasetCheckFn
FixCheckFn = Callable[[CanonicalDataset, str | None], "FixOutcome"]


@dataclass
class FixOutcome:
    data: xr.Dataset | CanonicalDataset
    applied: bool
    info: str
    details: dict[str, object] = field(default_factory=dict)

    @classmethod
    def applied_result(
        cls,
        *,
        data: xr.Dataset | CanonicalDataset,
        info: str,
        details: dict[str, object] | None = None,
    ) -> "FixOutcome":
        return cls(
            data=data,
            applied=True,
            info=info,
            details={} if details is None else dict(details),
        )

    @classmethod
    def skipped_result(
        cls,
        *,
        data: xr.Dataset | CanonicalDataset,
        info: str,
        details: dict[str, object] | None = None,
    ) -> "FixOutcome":
        return cls(
            data=data,
            applied=False,
            info=info,
            details={} if details is None else dict(details),
        )


@dataclass
class CheckDefinition:
    name: str
    data_scope: DataScope
    variables: list[str] | None = None
    plugin: str | None = None

    def check(self, data: xr.DataArray | CanonicalDataset) -> AtomicCheckResult:
        raise NotImplementedError

    def run(self, data: xr.DataArray | CanonicalDataset) -> AtomicCheckResult:
        return self.check(data)

    def has_fix(self) -> bool:
        return False

    def fix(
        self, dataset: CanonicalDataset, *, scope_item: str | None = None
    ) -> FixOutcome:
        raise NotImplementedError("This check does not implement fix().")


@dataclass
class CallableCheck(CheckDefinition):
    fn: AtomicCheckFn | None = None

    def check(self, data: xr.DataArray | CanonicalDataset) -> AtomicCheckResult:
        if self.fn is None:
            raise ValueError("CallableCheck requires a callable 'fn'.")
        return self.fn(data)  # type: ignore[arg-type]  # runtime dispatch via data_scope guarantees correct type


@dataclass
class FixableCheck(CheckDefinition):
    def has_fix(self) -> bool:
        return type(self).fix is not FixableCheck.fix


@dataclass
class CallableFixCheck(FixableCheck):
    fn: AtomicCheckFn | None = None
    fix_fn: FixCheckFn | None = None

    def check(self, data: xr.DataArray | CanonicalDataset) -> AtomicCheckResult:
        if self.fn is None:
            raise ValueError("CallableFixCheck requires a callable 'fn'.")
        return self.fn(data)  # type: ignore[arg-type]  # runtime dispatch via data_scope guarantees correct type

    def has_fix(self) -> bool:
        return self.fix_fn is not None

    def fix(
        self, dataset: CanonicalDataset, *, scope_item: str | None = None
    ) -> FixOutcome:
        if self.fix_fn is None:
            raise NotImplementedError("CallableFixCheck requires a callable 'fix_fn'.")
        return self.fix_fn(dataset, scope_item)


def run_atomic_check(
    data: xr.DataArray | CanonicalDataset,
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
        result = replace(
            result,
            details={
                **result.details,
                "data_scope": definition.data_scope,
                "scope_item": scope_item,
            },
        )

    if result.name != definition.name:
        return replace(
            result,
            name=definition.name,
            details={**result.details, "reported_name": result.name},
        )

    return result


class CheckSuite:
    def __init__(
        self,
        *,
        name: str,
        checks: Iterable[CheckDefinition],
    ) -> None:
        self.name = name
        self.checks = list(checks)

    def run(
        self, dataset: CanonicalDataset, *, apply_fixes: bool = False
    ) -> SuiteReport:
        assert isinstance(dataset, CanonicalDataset), (
            "dataset must be a CanonicalDataset"
        )
        working_dataset = _copy_canonical_dataset(dataset) if apply_fixes else dataset
        results: list[AtomicCheckResult] = []
        hierarchy: dict[str, dict[str, dict[str, AtomicCheckResult]]] = {}

        checks_by_scope: dict[DataScope, list[CheckDefinition]] = defaultdict(list)
        for definition in self.checks:
            checks_by_scope[definition.data_scope].append(definition)

        for data_scope in ("dataset", "dims", "coords", "data_vars"):
            scope_definitions = checks_by_scope.get(data_scope, [])
            if not scope_definitions:
                continue

            scope_items = _scope_item_names(working_dataset, data_scope)
            if not scope_items:
                for definition in scope_definitions:
                    results.append(_empty_scope_targets(definition))
                continue

            scope_hierarchy: dict[str, dict[str, AtomicCheckResult]] = {}
            for scope_item in scope_items:
                data = _scope_target(working_dataset, data_scope, scope_item)
                variable_results: dict[str, AtomicCheckResult] = {}
                for definition in scope_definitions:
                    if (
                        definition.variables is not None
                        and definition.variables != [scope_item]
                        and scope_item not in definition.variables
                    ):
                        continue
                    scoped_definition = CallableCheck(
                        name=f"{definition.name}[{data_scope}:{scope_item}]",
                        data_scope=data_scope,
                        fn=definition.run,
                    )
                    result = run_atomic_check(
                        _with_dataset_attrs(data, working_dataset.attrs),
                        scoped_definition,
                        scope_item=scope_item,
                    )
                    if apply_fixes and definition.has_fix():
                        working_dataset, result = _run_fix_cycle(
                            working_dataset,
                            definition,
                            data_scope=data_scope,
                            scope_item=scope_item,
                            initial_result=result,
                        )
                    results.append(result)
                    variable_results[definition.name] = result
                scope_hierarchy[str(scope_item)] = variable_results
            hierarchy[data_scope] = scope_hierarchy

        summary = _summary_from_results(results)
        return SuiteReport(
            suite_name=self.name,
            checks=results,
            summary=summary,
            results=hierarchy,
            source_file=_dataset_source_file(working_dataset),
            _dataset_html=_dataset_repr_html(working_dataset),
        )

    def __repr__(self) -> str:
        return f"CheckSuite(name={self.name}, checks={len(self.checks)})"


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
    warnings = sum(1 for result in results if result.status == CheckStatus.warning)
    failed = sum(1 for result in results if result.status == CheckStatus.failed)
    fatal = sum(1 for result in results if result.status == CheckStatus.fatal)
    fixed = sum(1 for result in results if result.fixed)

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
        warnings=warnings,
        failed=failed,
        fatal=fatal,
        fixed=fixed,
        overall_status=overall,
    )


def _copy_canonical_dataset(dataset: CanonicalDataset) -> CanonicalDataset:
    return _canonicalize_dataset(dataset.copy(deep=True))


def _canonicalize_dataset(dataset: xr.Dataset | CanonicalDataset) -> CanonicalDataset:
    if isinstance(dataset, CanonicalDataset):
        return dataset
    return CanonicalDataset.from_xarray(dataset, rename_aliases=False, strict=False)


def _run_fix_cycle(
    dataset: CanonicalDataset,
    definition: CheckDefinition,
    *,
    data_scope: DataScope,
    scope_item: str,
    initial_result: AtomicCheckResult,
) -> tuple[CanonicalDataset, AtomicCheckResult]:
    if initial_result.status not in {CheckStatus.warning, CheckStatus.failed}:
        return dataset, initial_result

    try:
        outcome = definition.fix(dataset, scope_item=scope_item)
    except NotImplementedError:
        return dataset, initial_result
    except Exception as exc:
        return dataset, _annotate_fix_result(
            initial_result,
            fixed=False,
            original_status=initial_result.status,
            fix_info=f"Fix raised {type(exc).__name__}: {exc}",
        )

    if not isinstance(outcome, FixOutcome):
        return dataset, _annotate_fix_result(
            initial_result,
            fixed=False,
            original_status=initial_result.status,
            fix_info="Fix returned an invalid result type.",
        )

    next_dataset = _canonicalize_dataset(outcome.data)
    if not outcome.applied:
        return next_dataset, _annotate_fix_result(
            initial_result,
            fixed=False,
            original_status=initial_result.status,
            fix_info=outcome.info,
        )

    refreshed = _scope_target(next_dataset, data_scope, scope_item)
    scoped_definition = CallableCheck(
        name=f"{definition.name}[{data_scope}:{scope_item}]",
        data_scope=data_scope,
        variables=definition.variables,
        plugin=definition.plugin,
        fn=definition.run,
    )
    post_result = run_atomic_check(
        _with_dataset_attrs(refreshed, next_dataset.attrs),
        scoped_definition,
        scope_item=scope_item,
    )
    was_fixed = (
        initial_result.status in {CheckStatus.warning, CheckStatus.failed}
        and post_result.status == CheckStatus.passed
    )
    return next_dataset, _annotate_fix_result(
        post_result,
        fixed=was_fixed,
        original_status=initial_result.status,
        fix_info=outcome.info,
    )


def _annotate_fix_result(
    result: AtomicCheckResult,
    *,
    fixed: bool,
    original_status: CheckStatus,
    fix_info: str,
) -> AtomicCheckResult:
    return replace(
        result,
        fixed=fixed,
        original_status=original_status,
        fix_info=fix_info,
    )


def _with_dataset_attrs(
    data: xr.DataArray | CanonicalDataset,
    dataset_attrs: dict[str, object],
) -> xr.DataArray | CanonicalDataset:
    if isinstance(data, CanonicalDataset):
        return data
    enriched = data.copy(deep=False)
    attrs = dict(dataset_attrs)
    attrs.update(data.attrs)
    enriched.attrs = attrs
    return enriched


def _scope_item_names(dataset: CanonicalDataset, data_scope: DataScope) -> list[str]:
    if data_scope == "dataset":
        return ["dataset"]
    if data_scope == "data_vars":
        return [str(name) for name in dataset.data_vars]
    if data_scope == "coords":
        return [str(name) for name in dataset.coords]
    if data_scope == "dims":
        return [str(name) for name in dataset.sizes]
    raise ValueError(f"Invalid data_scope '{data_scope}'.")


def _scope_target(
    dataset: CanonicalDataset, data_scope: DataScope, scope_item: str
) -> xr.DataArray | CanonicalDataset:
    if data_scope == "dataset":
        return dataset

    if data_scope == "data_vars":
        return dataset[scope_item]

    if data_scope == "coords":
        return dataset.coords[scope_item]

    if data_scope == "dims":
        if scope_item in dataset.coords:
            return dataset.coords[scope_item]
        size = int(dataset.sizes[scope_item])
        return xr.DataArray(
            np.arange(size),
            dims=(scope_item,),
            name=scope_item,
        )

    raise ValueError(f"Invalid data_scope '{data_scope}'.")


def _scope_targets(
    dataset: CanonicalDataset, data_scope: DataScope
) -> list[tuple[str, xr.DataArray | CanonicalDataset]]:
    if data_scope == "dataset":
        return [("dataset", dataset)]

    if data_scope == "data_vars":
        return [(str(name), dataset[name]) for name in dataset.data_vars]

    if data_scope == "coords":
        return [(str(name), dataset.coords[name]) for name in dataset.coords]

    if data_scope == "dims":
        targets: list[tuple[str, xr.DataArray | CanonicalDataset]] = []
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


def _dataset_source_file(dataset: CanonicalDataset) -> str | None:
    source = dataset.attrs.get("source")
    if source is None:
        return None
    source_text = str(source).strip()
    if not source_text:
        return None
    return Path(source_text).name
