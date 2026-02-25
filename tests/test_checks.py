import numpy as np
import xarray as xr

from nc_check.checks import HeuristicCheck, OceanCoverCheck, TimeCoverCheck
from nc_check.core import CheckStatus


def test_heuristic_check_status_warn_for_missing_conventions() -> None:
    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("time",),
                [290.0],
                {"units": "K", "long_name": "temperature"},
            )
        },
        coords={"time": np.array(["2024-01-01"], dtype="datetime64[ns]")},
    )

    result = HeuristicCheck(cf_version="CF-1.12").check(ds)

    assert result.status == CheckStatus.warn
    assert result.info.details["report"]["engine"] == "heuristic"


def test_heuristic_check_status_error_for_invalid_variable_name() -> None:
    ds = xr.Dataset(
        data_vars={"bad-name": (("time",), [1.0])},
        coords={"time": [0]},
        attrs={"Conventions": "CF-1.12"},
    )

    result = HeuristicCheck(cf_version="CF-1.12").check(ds)

    assert result.status == CheckStatus.error


def test_ocean_cover_check_status_skipped_when_checks_disabled() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    result = OceanCoverCheck(
        var_name="sst",
        check_edge_of_map=False,
        check_land_ocean_offset=False,
    ).check(ds)

    assert result.status == CheckStatus.skipped


def test_ocean_cover_check_status_error_for_missing_edge_band() -> None:
    lon = np.arange(0.0, 360.0, 30.0)
    lat = np.array([-30.0, 0.0, 30.0])
    data = np.ones((lat.size, lon.size), dtype=float)
    data[:, [0, -1]] = np.nan
    ds = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon},
    )

    result = OceanCoverCheck(
        var_name="sst",
        check_land_ocean_offset=False,
    ).check(ds)

    assert result.status == CheckStatus.error


def test_time_cover_check_status_skipped_when_variable_has_no_time_dim() -> None:
    ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    result = TimeCoverCheck(var_name="mask").check(ds)

    assert result.status == CheckStatus.skipped


def test_time_cover_check_status_error_for_missing_time_slices() -> None:
    lon = np.arange(0.0, 360.0, 60.0)
    lat = np.array([-45.0, 45.0])
    time = np.arange(5)
    data = np.ones((time.size, lat.size, lon.size), dtype=float)
    data[1:3, :, :] = np.nan
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    result = TimeCoverCheck(var_name="sst").check(ds)

    assert result.status == CheckStatus.error
