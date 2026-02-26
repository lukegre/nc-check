from __future__ import annotations

from typing import Iterable

import xarray as xr

from .dataset import CanonicalDataset
from .models import SuiteReport
from .plugins import (
    CFCompliancePlugin,
    CheckRegistry,
    TimeCoverPlugin,
    cf_check_names,
    time_cover_check_names,
)


def canonicalize_dataset(
    ds: xr.Dataset,
    *,
    rename_aliases: bool = True,
    strict: bool = True,
) -> CanonicalDataset:
    return CanonicalDataset.from_xarray(
        ds,
        rename_aliases=rename_aliases,
        strict=strict,
    )


def create_registry(*, load_entrypoints: bool = True) -> CheckRegistry:
    registry = CheckRegistry()
    registry.register_plugin(CFCompliancePlugin())
    registry.register_plugin(TimeCoverPlugin())
    if load_entrypoints:
        registry.register_entrypoint_plugins()
    return registry


def run_suite(
    ds: xr.Dataset | CanonicalDataset,
    *,
    check_names: Iterable[str],
    suite_name: str = "custom",
    registry: CheckRegistry | None = None,
    plugin: str | None = None,
    rename_aliases: bool = True,
    strict_dataset: bool = True,
) -> SuiteReport:
    active_registry = registry or create_registry()
    canonical = (
        ds
        if isinstance(ds, CanonicalDataset)
        else canonicalize_dataset(
            ds,
            rename_aliases=rename_aliases,
            strict=strict_dataset,
        )
    )

    suite = active_registry.build_suite(
        name=suite_name,
        check_names=check_names,
        plugin=plugin,
    )
    return suite.run(canonical)


def run_cf_compliance(
    ds: xr.Dataset | CanonicalDataset,
    *,
    registry: CheckRegistry | None = None,
    rename_aliases: bool = True,
    strict_dataset: bool = True,
) -> SuiteReport:
    active_registry = registry or create_registry()
    return run_suite(
        ds,
        check_names=cf_check_names(),
        suite_name="cf_compliance",
        registry=active_registry,
        plugin="cf_compliance",
        rename_aliases=rename_aliases,
        strict_dataset=strict_dataset,
    )


def _clone_registry_with_time_cover_options(
    registry: CheckRegistry,
    *,
    var_name: str | None,
    time_name: str | None,
) -> CheckRegistry:
    time_cover_checks = set(time_cover_check_names())
    cloned = CheckRegistry()
    for check_name in registry.list_checks():
        if check_name in time_cover_checks:
            continue
        registered = registry.get_check(check_name)
        cloned.register_check(check=registered.check)
    cloned.register_plugin(TimeCoverPlugin(var_name=var_name, time_name=time_name))
    return cloned


def run_time_cover(
    ds: xr.Dataset | CanonicalDataset,
    *,
    var_name: str | None = None,
    time_name: str | None = "time",
    check_time_monotonic: bool = False,
    check_time_regular_spacing: bool = False,
    registry: CheckRegistry | None = None,
    rename_aliases: bool = True,
    strict_dataset: bool = False,
) -> SuiteReport:
    missing_check, monotonic_check, regular_spacing_check = time_cover_check_names()
    selected_checks = [missing_check]
    if check_time_monotonic:
        selected_checks.append(monotonic_check)
    if check_time_regular_spacing:
        selected_checks.append(regular_spacing_check)

    active_registry = registry or create_registry()
    registered = set(active_registry.list_checks())
    missing_time_checks = any(
        check_name not in registered for check_name in time_cover_check_names()
    )
    if var_name is not None or time_name != "time" or missing_time_checks:
        active_registry = _clone_registry_with_time_cover_options(
            active_registry,
            var_name=var_name,
            time_name=time_name,
        )

    return run_suite(
        ds,
        check_names=selected_checks,
        suite_name="time_cover",
        registry=active_registry,
        plugin="time_cover",
        rename_aliases=rename_aliases,
        strict_dataset=strict_dataset,
    )
