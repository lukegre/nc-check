from __future__ import annotations

import re
from os import PathLike
from typing import Callable, Literal

from IPython.display import HTML
import xarray as xr

from .dataset import CanonicalDataset
from .suite import CheckSuite
from . import plugins


def get_check_suites() -> list[CheckSuite]:
    """Get all check suites."""
    list_of_suites = []
    plugin_vars = dir(plugins)
    for var_name in plugin_vars:
        var = getattr(plugins, var_name)
        if isinstance(var, CheckSuite):
            list_of_suites.append(var)

    if not list_of_suites:
        raise ValueError("No check suites found in plugins")
    return list_of_suites


@xr.register_dataset_accessor("check")
class CheckAccessor:
    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj
        self._method_names: set[str] = set()
        for suite in get_check_suites():
            method_name = self._suite_method_name(suite.name)
            if hasattr(self, method_name):
                raise ValueError(
                    f"Method name conflict: {method_name} already exists on CheckAccessor"
                )
            self._method_names.add(method_name)
            setattr(self, method_name, self._make_suite_method(suite))

    def _suite_method_name(self, suite_name: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", suite_name.strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_")

    def _make_suite_method(
        self, suite: CheckSuite
    ) -> Callable[
        [Literal["html", "json", "dict"], str | PathLike[str] | None, bool],
        dict | HTML | str | None,
    ]:
        method_name = self._suite_method_name(suite.name)

        def method(
            format: Literal["html", "json", "dict"] = "html",
            report_fname: str | PathLike[str] | None = None,
            apply_fixes: bool = False,
        ) -> dict | HTML | str | None:
            canonical = CanonicalDataset.from_xarray(self._obj)
            report = suite.run(canonical, apply_fixes=apply_fixes)
            if format == "json":
                return report.to_json(report_fname=report_fname)
            elif format == "dict":
                return report.to_dict()
            elif format == "html":
                return report.to_html(report_fname=report_fname)
            else:
                raise ValueError(f"Unsupported format: {format}")

        method.__name__ = method_name
        method.__doc__ = f"Run the '{suite.name}' check suite and optionally apply fixes before returning a report."
        return method

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | self._method_names)

    def __repr__(self) -> str:
        check_list = "\n\t" + "\n\t".join(sorted(self._method_names)) + "\n"
        return f"<CheckAccessor> for xarray.Dataset with plugins: {check_list}"
