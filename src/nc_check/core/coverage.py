from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .check import CheckStatus


def resolve_time_dim(da: xr.DataArray, preferred_name: str | None) -> str | None:
    if preferred_name and preferred_name in da.dims:
        return preferred_name
    if "time" in da.dims:
        return "time"
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is None:
            continue
        standard_name = str(coord.attrs.get("standard_name", "")).strip().lower()
        if standard_name == "time":
            return str(dim)
    return None


def choose_time_vars(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    time_name: str | None,
) -> list[tuple[xr.DataArray, str | None]]:
    if var_name is not None:
        if var_name not in ds.data_vars:
            raise ValueError(f"Data variable '{var_name}' not found.")
        da = ds[var_name]
        return [(da, resolve_time_dim(da, time_name))]

    selected = [(da, resolve_time_dim(da, time_name)) for _, da in ds.data_vars.items()]
    if not selected:
        raise ValueError("Dataset has no data variables to check.")
    return selected


def missing_mask(da: xr.DataArray) -> xr.DataArray:
    mask = da.isnull()
    for source in (da.attrs, da.encoding):
        fill_value = source.get("_FillValue")
        if fill_value is None:
            continue
        try:
            mask = mask | (da == fill_value)
        except Exception:
            continue
    return mask


def indices_to_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for idx in ordered[1:]:
        if idx == end + 1:
            end = idx
            continue
        ranges.append((start, end))
        start = idx
        end = idx
    ranges.append((start, end))
    return ranges


def value_label(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="s")
    return str(value)


def range_records(
    indices: list[int],
    coord: xr.DataArray | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start, end in indices_to_ranges(indices):
        if coord is not None:
            values = np.asarray(coord.values)
            start_label = value_label(values[start])
            end_label = value_label(values[end])
        else:
            start_label = str(start)
            end_label = str(end)
        out.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "start": start_label,
                "end": end_label,
            }
        )
    return out


def leaf_statuses(report: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    if report.get("mode") == "all_variables":
        grouped = report.get("reports")
        if not isinstance(grouped, dict):
            return []
        statuses: list[str] = []
        for per_var in grouped.values():
            if not isinstance(per_var, dict):
                continue
            statuses.extend(leaf_statuses(per_var, keys))
        return statuses

    statuses: list[str] = []
    for key in keys:
        entry = report.get(key)
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).strip().lower()
        if status:
            statuses.append(status)
    return statuses


def status_from_leaf_statuses(statuses: list[str]) -> CheckStatus:
    if not statuses:
        return CheckStatus.passed
    if any(status in {"fatal"} for status in statuses):
        return CheckStatus.fatal
    if any(status in {"fail", "failed", "error"} for status in statuses):
        return CheckStatus.error
    if any(status == "warn" or "warn" in status for status in statuses):
        return CheckStatus.warn
    if all(status.startswith("skip") for status in statuses):
        return CheckStatus.skipped
    return CheckStatus.passed
