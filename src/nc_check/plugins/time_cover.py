from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from ..models import AtomicCheckResult
from ..suite import CallableCheck

_MISSING_SLICES_CHECK = "time.missing_slices"
_MONOTONIC_CHECK = "time.monotonic_increasing"
_REGULAR_SPACING_CHECK = "time.regular_spacing"


def _resolve_time_dim(
    variable_dims: tuple[str, ...], preferred: str | None
) -> str | None:
    if preferred and preferred in variable_dims:
        return preferred
    if "time" in variable_dims:
        return "time"
    return None


def _time_points(data: xr.DataArray, dim_name: str) -> np.ndarray:
    if dim_name in data.coords:
        values = np.asarray(data.coords[dim_name].values)
        if values.ndim == 1:
            return values
    return np.arange(int(data.sizes.get(dim_name, 0)))


def _time_as_numeric(values: np.ndarray) -> np.ndarray | None:
    if values.ndim != 1:
        values = values.reshape(-1)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float)
    try:
        return values.astype(float)
    except Exception:
        return None


def _contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = indices[0]
    end = indices[0]
    for value in indices[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append((start, end))
        start = value
        end = value
    ranges.append((start, end))
    return ranges


def _as_time_strings(
    values: np.ndarray, start_idx: int, end_idx: int
) -> tuple[str | None, str | None]:
    if values.ndim != 1:
        return None, None
    if start_idx < 0 or end_idx >= values.size:
        return None, None
    return _stringify_value(values[start_idx]), _stringify_value(values[end_idx])


def _stringify_value(value: Any) -> str:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return "NaT"
        return np.datetime_as_string(value, unit="s")
    if isinstance(value, np.timedelta64):
        return str(value.astype("timedelta64[ns]"))
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


class TimeCoverPlugin:
    name = "time_cover"

    def __init__(self, *, var_name: str | None = None, time_name: str | None = "time"):
        self.var_name = var_name
        self.time_name = time_name

    def _resolve_target(
        self, data: xr.DataArray, *, check_name: str
    ) -> tuple[str, str] | AtomicCheckResult:
        variable_name = str(data.name) if data.name is not None else "<unnamed>"
        if self.var_name is not None and variable_name != self.var_name:
            return AtomicCheckResult.skipped_result(
                name=check_name,
                info=f"Skipped variable '{variable_name}' (filtered to '{self.var_name}').",
                details={
                    "data_var": variable_name,
                    "requested_variable": self.var_name,
                },
            )

        time_dim = _resolve_time_dim(data.dims, self.time_name)
        if time_dim is None:
            return AtomicCheckResult.skipped_result(
                name=check_name,
                info=f"Variable '{variable_name}' has no resolved time dimension.",
                details={
                    "data_var": variable_name,
                    "reason": "no_resolved_time_dimension",
                },
            )
        return variable_name, time_dim

    def _missing_slices_check(self, data: xr.DataArray) -> AtomicCheckResult:
        resolved = self._resolve_target(data, check_name=_MISSING_SLICES_CHECK)
        if isinstance(resolved, AtomicCheckResult):
            return resolved
        variable_name, time_dim = resolved

        isnull = data.isnull()
        other_dims = [name for name in data.dims if name != time_dim]
        if other_dims:
            missing_mask = isnull.all(dim=other_dims)
        else:
            missing_mask = isnull

        mask_values = np.asarray(missing_mask.values, dtype=bool).reshape(-1)
        missing_indices = np.flatnonzero(mask_values).astype(int).tolist()
        missing_ranges = _contiguous_ranges(missing_indices)
        time_values = _time_points(data, time_dim)

        ranges_payload: list[dict[str, Any]] = []
        for start_idx, end_idx in missing_ranges:
            start_time, end_time = _as_time_strings(time_values, start_idx, end_idx)
            item: dict[str, Any] = {
                "start_index": start_idx,
                "end_index": end_idx,
                "length": end_idx - start_idx + 1,
            }
            if start_time is not None:
                item["start_time"] = start_time
            if end_time is not None:
                item["end_time"] = end_time
            ranges_payload.append(item)

        missing_count = len(missing_indices)
        details = {
            "data_var": variable_name,
            "time_dim": time_dim,
            "time_point_count": int(data.sizes.get(time_dim, 0)),
            "missing_slice_count": missing_count,
            "missing_ranges": ranges_payload,
        }
        if missing_count > 0:
            return AtomicCheckResult.failed_result(
                name=_MISSING_SLICES_CHECK,
                info=f"Missing time slices found for '{variable_name}'.",
                details=details,
            )
        return AtomicCheckResult.passed_result(
            name=_MISSING_SLICES_CHECK,
            info=f"Missing time slices check passed for '{variable_name}'.",
            details=details,
        )

    def _monotonic_increasing_check(self, data: xr.DataArray) -> AtomicCheckResult:
        resolved = self._resolve_target(data, check_name=_MONOTONIC_CHECK)
        if isinstance(resolved, AtomicCheckResult):
            return resolved
        variable_name, time_dim = resolved

        values = _time_points(data, time_dim).reshape(-1)
        numeric = _time_as_numeric(values)
        if numeric is None:
            return AtomicCheckResult.skipped_result(
                name=_MONOTONIC_CHECK,
                info=f"Skipped monotonic check for '{variable_name}' (unorderable time values).",
                details={
                    "data_var": variable_name,
                    "time_dim": time_dim,
                    "reason": "unorderable_time_values",
                },
            )

        violations = np.flatnonzero(numeric[1:] <= numeric[:-1]).astype(int)
        violation_indices = (violations + 1).tolist()
        violation_count = len(violation_indices)
        details = {
            "data_var": variable_name,
            "time_dim": time_dim,
            "time_point_count": int(values.size),
            "order_violation_count": violation_count,
            "order_violation_indices": violation_indices,
        }
        if violation_count > 0:
            return AtomicCheckResult.failed_result(
                name=_MONOTONIC_CHECK,
                info=f"Monotonic time order check failed for '{variable_name}'.",
                details=details,
            )
        return AtomicCheckResult.passed_result(
            name=_MONOTONIC_CHECK,
            info=f"Monotonic time order check passed for '{variable_name}'.",
            details=details,
        )

    def _regular_spacing_check(self, data: xr.DataArray) -> AtomicCheckResult:
        resolved = self._resolve_target(data, check_name=_REGULAR_SPACING_CHECK)
        if isinstance(resolved, AtomicCheckResult):
            return resolved
        variable_name, time_dim = resolved

        values = _time_points(data, time_dim).reshape(-1)
        numeric = _time_as_numeric(values)
        if numeric is None:
            return AtomicCheckResult.skipped_result(
                name=_REGULAR_SPACING_CHECK,
                info=f"Skipped regular spacing check for '{variable_name}' (non-numeric time values).",
                details={
                    "data_var": variable_name,
                    "time_dim": time_dim,
                    "reason": "non_numeric_time_values",
                },
            )

        if numeric.size < 2:
            return AtomicCheckResult.passed_result(
                name=_REGULAR_SPACING_CHECK,
                info=f"Regular time spacing check passed for '{variable_name}'.",
                details={
                    "data_var": variable_name,
                    "time_dim": time_dim,
                    "time_point_count": int(numeric.size),
                    "irregular_interval_count": 0,
                    "irregular_interval_indices": [],
                },
            )

        diffs = np.diff(numeric)
        irregular = diffs <= 0
        if diffs.size > 1:
            irregular[1:] |= diffs[1:] != diffs[0]
        irregular_indices = np.flatnonzero(irregular).astype(int).tolist()
        irregular_count = len(irregular_indices)
        details = {
            "data_var": variable_name,
            "time_dim": time_dim,
            "time_point_count": int(numeric.size),
            "irregular_interval_count": irregular_count,
            "irregular_interval_indices": irregular_indices,
            "expected_interval": _stringify_value(diffs[0]),
        }
        if irregular_count > 0:
            return AtomicCheckResult.failed_result(
                name=_REGULAR_SPACING_CHECK,
                info=f"Regular time spacing check failed for '{variable_name}'.",
                details=details,
            )
        return AtomicCheckResult.passed_result(
            name=_REGULAR_SPACING_CHECK,
            info=f"Regular time spacing check passed for '{variable_name}'.",
            details=details,
        )

    def register(self, registry: Any) -> None:
        registry.register_check(
            check=CallableCheck(
                name=_MISSING_SLICES_CHECK,
                data_scope="data_vars",
                plugin=self.name,
                fn=self._missing_slices_check,
            )
        )
        registry.register_check(
            check=CallableCheck(
                name=_MONOTONIC_CHECK,
                data_scope="data_vars",
                plugin=self.name,
                fn=self._monotonic_increasing_check,
            )
        )
        registry.register_check(
            check=CallableCheck(
                name=_REGULAR_SPACING_CHECK,
                data_scope="data_vars",
                plugin=self.name,
                fn=self._regular_spacing_check,
            )
        )


def time_cover_check_names() -> tuple[str, ...]:
    return (
        _MISSING_SLICES_CHECK,
        _MONOTONIC_CHECK,
        _REGULAR_SPACING_CHECK,
    )
