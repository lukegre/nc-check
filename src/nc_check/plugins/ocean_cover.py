import numpy as np
import xarray as xr

from ..models import AtomicCheckResult


def check_180_lon_shift(data: xr.DataArray) -> AtomicCheckResult:
    """
    Checks if the datail hasn't been shifted by 180 degrees in longitude
    without the shifting of the coordinates.

    This test is performed by using selected points that are known land points
    and other points that are known ocean points. We assume ocean data that
    has nans over land. If the land points have nans and the ocean points
    have values, then we assume the data is not shifted. If the land points have
    values and the ocean points have nans, then we assume the data is shifted.
    """
    LAND_POINTS = {
        "land_australia": {"lat": -25.0, "lon": 135.0},
        "land_south_america": {"lat": -15.0, "lon": -60.0},
        "land_north_america": {"lat": 40.0, "lon": -100.0},
        "land_africa": {"lat": 0.0, "lon": 20.0},
    }
    OCEAN_POINTS = {
        "ocean_pacific": {"lat": -20, "lon": -160.0},
        "ocean_atlantic": {"lat": -26, "lon": -37.0},
        "ocean_indian": {"lat": -20, "lon": 80.0},
    }

    n_timesteps = data.time.size
    time_points = data.time.isel(time=slice(None, None, n_timesteps // 4))

    def select_points(data, point: dict[str, float]):
        point = point | {"time": time_points, "method": "nearest"}
        return data.sel(**point)

    def point_has_any_data(data: xr.DataArray):
        return data.mean("time").compute().item()

    land_values = {}
    for continent, point in LAND_POINTS.items():
        land_values[continent] = select_points(data, point).pipe(point_has_any_data)

    ocean_values = {}
    for ocean, point in OCEAN_POINTS.items():
        ocean_values[ocean] = select_points(data, point).pipe(point_has_any_data)

    all_land_nan = all(np.isnan(val) for val in land_values.values())
    all_ocean_nan = all(np.isnan(val) for val in ocean_values.values())

    # creating results
    display_values = {k: f"{v:.3g}" for k, v in (land_values | ocean_values).items()}
    if all_land_nan and not all_ocean_nan:
        return AtomicCheckResult.passed_result(
            name="demo.lon_shift_check",
            info="Data appears to be correctly aligned with coordinates.",
            details={**display_values},
        )
    elif not all_land_nan and all_ocean_nan:
        return AtomicCheckResult.failed_result(
            name="demo.lon_shift_check",
            info="Data appears to be shifted by 180 degrees in longitude.",
            details={**display_values},
        )
    else:
        return AtomicCheckResult.failed_result(
            name="demo.lon_shift_check",
            info="Data has unexpected pattern of NaNs in land and ocean points.",
            details={**display_values},
        )
