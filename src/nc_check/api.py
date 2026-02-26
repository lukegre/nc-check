from __future__ import annotations

from typing import Iterable

import xarray as xr

from .dataset import CanonicalDataset
from .models import SuiteReport
from .plugins import CFCompliancePlugin, CheckRegistry, cf_check_names


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
