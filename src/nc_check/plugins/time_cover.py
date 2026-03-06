from __future__ import annotations


import numpy as np
import xarray as xr

from ..models import AtomicCheckResult
from ..suite import CallableCheck, CheckSuite

_TIME_CHECK_SUITE_NAME = "Time coverage"
_MISSING_SLICES_CHECK = "Missing time slices"
_TIME_STEP_REGULAR_CHECK = "Time step regular"
_TIME_FORMAT_CHECK = "Time format readable"


################################
# Atomic Checks for time cover #
################################
def check_missing_slices(data: xr.DataArray) -> AtomicCheckResult:
    dims = set(data.dims)
    if "time" not in dims:
        return _no_time_dim(data)

    other_dims = dims - {"time"}
    missing_mask = data.isnull().all(dim=other_dims)

    missing_count = missing_mask.sum().values.item()
    if missing_count == 0:
        return AtomicCheckResult.passed_result(
            name=_MISSING_SLICES_CHECK,
            info="No missing time slices found.",
            details={},
        )

    missing_indices = np.flatnonzero(missing_mask.values).astype(int).tolist()
    missing_ranges = _contiguous_ranges(missing_indices)
    time_values = _get_pretty_time(data, "time")
    missing_time_ranges = _ranges_to_string(missing_ranges, time_values)

    return AtomicCheckResult.failed_result(
        name=_MISSING_SLICES_CHECK,
        info=f"Missing time slices found for '{data.name}'.",
        details={
            "time_point_count": int(data.sizes.get("time", 0)),
            "missing_slice_count": missing_count,
            "missing_ranges": missing_time_ranges,
        },
    )


def check_time_step_regular(data: xr.DataArray) -> AtomicCheckResult:
    dims = set(data.dims)
    if "time" not in dims:
        return _no_time_dim(data)

    time_values = data.coords["time"].values
    if not np.issubdtype(time_values.dtype, np.datetime64):
        return AtomicCheckResult.skipped_result(
            name=_TIME_STEP_REGULAR_CHECK,
            info="Skipped time step regularity check (time is not datetime64).",
            details={},
        )

    if time_values.size < 2:
        return AtomicCheckResult.skipped_result(
            name=_TIME_STEP_REGULAR_CHECK,
            info="Time step regularity check skipped (fewer than 2 time points).",
            details={"time_step": "n/a"},
        )

    diffs = np.diff(time_values)
    diffs_days = diffs / np.timedelta64(1, "D")
    time_step = _detect_time_step(diffs_days)

    if time_step == "irregular":
        return AtomicCheckResult.failed_result(
            name=_TIME_STEP_REGULAR_CHECK,
            info=f"Time is not regular for '{data.name}'.",
            details={"time_step": time_step},
        )
    else:
        return AtomicCheckResult.passed_result(
            name=_TIME_STEP_REGULAR_CHECK,
            info="Time is regular.",
            details={"time_step": time_step},
        )


def check_time_format_readable(data: xr.DataArray) -> AtomicCheckResult:
    """
    Checks that read in time is in np.datetime64 format or cftime,
    meaning that xarray successfully decoded the time dimension
    """
    dims = set(data.dims)
    if "time" not in dims:
        return _no_time_dim(data)
    else:
        time = data.coords["time"]

    time_type = time.dtype

    if time_type == "datetime64[ns]" or time_type == "cftime":
        passed = True
    else:
        passed = False

    if passed:
        return AtomicCheckResult.passed_result(
            name=_TIME_FORMAT_CHECK,
            info="Time dimension is in readable format.",
            details={},
        )
    else:
        example_times = ", ".join(time.values[:2].astype(str))
        time_units = time.attrs.get("units", "unknown")
        return AtomicCheckResult.failed_result(
            name=_TIME_FORMAT_CHECK,
            info="Time dimension is not readable.",
            details={
                "example_times": f"[{example_times}, ...]",
                "time_units": time_units,
            },
        )


##############################
# CheckSuite for ocean cover #
##############################
time_cover_suite = CheckSuite(
    name=_TIME_CHECK_SUITE_NAME,
    checks=[
        CallableCheck(
            name=_TIME_FORMAT_CHECK,
            data_scope="coords",
            fn=check_time_format_readable,
        ),
        CallableCheck(
            name=_TIME_STEP_REGULAR_CHECK,
            data_scope="coords",
            fn=check_time_step_regular,
        ),
        CallableCheck(
            name=_MISSING_SLICES_CHECK,
            data_scope="data_vars",
            fn=check_missing_slices,
        ),
    ],
)


####################
# Helper Functions #
####################
def _detect_time_step(diffs_days: np.ndarray) -> str:
    """Classify time step from an array of inter-step intervals in days.

    Handles monthly data (28-31 day gaps) and yearly data (365-366 day gaps)
    with appropriate tolerance.
    """
    median_diff = float(np.median(diffs_days))

    # Monthly: median between 28 and 31, all values within 27-32
    if 27 <= median_diff <= 32 and np.all((diffs_days >= 27) & (diffs_days <= 32)):
        return "monthly"

    # Yearly: median between 365 and 366, tolerance for leap years
    if 364 <= median_diff <= 367 and np.all((diffs_days >= 364) & (diffs_days <= 367)):
        return "yearly"

    # Exact uniform spacing (with 0.01-day tolerance)
    if np.all(np.abs(diffs_days - median_diff) < 0.01):
        if abs(median_diff - 1.0) < 0.01:
            return "daily"
        if abs(median_diff - 7.0) < 0.01:
            return "weekly"
        return f"{median_diff:.4g} days"

    return "irregular"


def _no_time_dim(data: xr.DataArray) -> AtomicCheckResult:
    dims = set(data.dims)
    if "time" not in dims:
        return AtomicCheckResult.skipped_result(
            name="No time step",
            info="No time dimension found.",
            details={},
        )
    else:
        raise RuntimeError(
            "_no_time_dim should not be called if the time dimension is present"
        )


def _ranges_to_string(ranges: list[slice], values: np.ndarray) -> str:
    if values.ndim != 1:
        return ""

    range_strings = []
    for r in ranges:
        start_val = str(values[r.start])
        end_val = str(values[r.stop])
        range_strings.append(f"{start_val} : {end_val}")
    return ", ".join(range_strings)


def _get_pretty_time(data: xr.DataArray, dim_name: str) -> np.ndarray:
    time = data.coords[dim_name]

    if time.dtype == "datetime64[ns]":
        return time.dt.strftime("%Y-%m-%d").values
    else:
        return time.values


def _contiguous_ranges(indices: list[int]) -> list[slice]:
    if not indices:
        return []
    ranges: list[slice] = []
    start = indices[0]
    end = indices[0]
    for value in indices[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append((start, end))
        start = value
        end = value
    ranges.append(slice(start, end))
    return ranges
