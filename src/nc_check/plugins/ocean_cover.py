import numpy as np
import xarray as xr

from ..models import AtomicCheckResult
from ..suite import CheckSuite, CallableCheck

_LON_SHIFT_CHECK_NAME = "Longitude shifted"
_MISSING_LONS_CHECK_NAME = "Missing longitude ranges"

_LAND_POINTS: dict[str, dict[str, float]] = {
    "land_australia": {"lat": -25.0, "lon": 135.0},
    "land_south_america": {"lat": -15.0, "lon": -60.0},
    "land_north_america": {"lat": 40.0, "lon": -100.0},
    "land_africa": {"lat": 0.0, "lon": 20.0},
}

_OCEAN_POINTS: dict[str, dict[str, float]] = {
    "ocean_pacific": {"lat": -20.0, "lon": -160.0},
    "ocean_atlantic": {"lat": -26.0, "lon": -37.0},
    "ocean_indian": {"lat": -20.0, "lon": 80.0},
}


def check_180_lon_shift(data: xr.DataArray) -> AtomicCheckResult:
    """
    Check whether data appears shifted by 180 degrees in longitude without
    corresponding coordinate changes.

    Uses known land/ocean sample points and expects ocean-like fields to be NaN
    over land and non-NaN over ocean.
    """
    if "lat" not in data.coords or "lon" not in data.coords:
        return AtomicCheckResult.skipped_result(
            name=_LON_SHIFT_CHECK_NAME,
            info="Skipped longitude-shift check; dataset must include lat/lon coordinates.",
        )

    sampled = _sample_time_slice(data, n_checking_timesteps=4)
    land_values = _collect_point_values(sampled, _LAND_POINTS)
    ocean_values = _collect_point_values(sampled, _OCEAN_POINTS)
    display_values = _format_display_values(land_values | ocean_values)

    all_land_nan = all(np.isnan(val) for val in land_values.values())
    all_ocean_nan = all(np.isnan(val) for val in ocean_values.values())

    if all_land_nan and not all_ocean_nan:
        return AtomicCheckResult.passed_result(
            name=_LON_SHIFT_CHECK_NAME,
            info="Data appears to be correctly aligned with coordinates.",
            details={**display_values},
        )

    if not all_land_nan and all_ocean_nan:
        return AtomicCheckResult.failed_result(
            name=_LON_SHIFT_CHECK_NAME,
            info="Data appears to be shifted by 180 degrees in longitude.",
            details={**display_values},
        )

    return AtomicCheckResult.failed_result(
        name=_LON_SHIFT_CHECK_NAME,
        info="Data has unexpected pattern of NaNs in land and ocean points.",
        details={**display_values},
    )


def check_missing_lons(
    data: xr.DataArray, n_checking_timesteps: int = 4
) -> AtomicCheckResult:
    info = {
        "skipped": "Dataset is not global, or does not have 'lon'",
        "passed": "No missing longitude values found in checked time steps",
        "failed": "Found missing longitude values",
    }

    if not _has_global_lon_coverage(data):
        return AtomicCheckResult.skipped_result(
            name=_MISSING_LONS_CHECK_NAME,
            info=info["skipped"],
        )

    sampled = _sample_time_slice(data, n_checking_timesteps)
    missing_mask = _compute_missing_lon_mask(sampled)
    missing_ranges = _format_missing_lon_ranges(data.coords["lon"].values, missing_mask)

    if not missing_ranges:
        return AtomicCheckResult.passed_result(
            name=_MISSING_LONS_CHECK_NAME,
            info=info["passed"],
        )

    return AtomicCheckResult.failed_result(
        name=_MISSING_LONS_CHECK_NAME,
        info=info["failed"],
        details={"missing_longitudes": missing_ranges},
    )


ocean_check_suite = CheckSuite(
    name="Ocean Cover Checks",
    checks=[
        CallableCheck(
            name=_LON_SHIFT_CHECK_NAME,
            data_scope="data_vars",
            plugin="ocean",
            fn=check_180_lon_shift,
        ),
        CallableCheck(
            name=_MISSING_LONS_CHECK_NAME,
            data_scope="data_vars",
            plugin="ocean",
            fn=check_missing_lons,
        ),
    ],
)


def _mean_over_time_at_point(data: xr.DataArray, point: dict[str, float]) -> float:
    selection = data.sel(lat=point["lat"], lon=point["lon"], method="nearest")
    if "time" in selection.dims:
        return float(selection.mean("time").compute().item())
    return float(selection.compute().item())


def _collect_point_values(
    data: xr.DataArray, points: dict[str, dict[str, float]]
) -> dict[str, float]:
    return {
        name: _mean_over_time_at_point(data, point) for name, point in points.items()
    }


def _format_display_values(values: dict[str, float]) -> dict[str, str]:
    return {name: f"{value:.3g}" for name, value in values.items()}


def _has_global_lon_coverage(data: xr.DataArray, min_span_deg: float = 350.0) -> bool:
    if "lon" not in data.coords:
        return False
    lons = data.coords["lon"]
    lon_span = float(lons.max() - lons.min())
    return lon_span >= min_span_deg


def _sample_time_slice(
    data: xr.DataArray, n_checking_timesteps: int = 4
) -> xr.DataArray:
    if "time" not in data.coords:
        return data
    n_timesteps = data.coords["time"].size
    step = max(1, n_timesteps // max(1, n_checking_timesteps))
    sampled_times = data.coords["time"].isel(time=slice(None, None, step))
    return data.sel(time=sampled_times)


def _compute_missing_lon_mask(data: xr.DataArray) -> np.ndarray:
    if "time" in data.dims:
        return data.isnull().all("lat").any("time").values
    return data.isnull().all("lat").values


def _format_missing_lon_ranges(lons: np.ndarray, missing_mask: np.ndarray) -> str:
    ranges: list[str] = []
    start = None
    lon_prev = None

    for lon, is_missing in zip(lons.tolist(), missing_mask):
        if is_missing and start is None:
            start = lon
        elif not is_missing and start is not None:
            ranges.append(f"{start}:{lon_prev}")
            start = None
        lon_prev = lon

    if start is not None:
        ranges.append(f"{start}:{lon_prev}")

    return ", ".join(ranges)
