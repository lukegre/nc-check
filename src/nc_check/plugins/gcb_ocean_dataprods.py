from nc_check.models import AtomicCheckResult
import xarray as xr
import numpy as np

from .. import config
from ..suite import CallableCheck, CheckSuite
from .ocean_cover import (
    check_180_lon_shift,
    check_missing_lons,
)
from .time_cover import (
    check_time_format_readable,
    check_time_step_regular,
    check_missing_slices,
)
from .cf_compliance import (
    _latitude_range_check,
)

_OCEAN_SURFACE_AREA_KM2 = 361900000
_GLOBAL_SURFACE_AREA_KM2 = 510100000


def latest_date_check(data: xr.DataArray) -> AtomicCheckResult:
    if "time" not in data.coords:
        return AtomicCheckResult.skipped_result(
            name="Latest Date Check",
            info="Dataset does not have 'time' coordinate.",
        )
    if not np.issubdtype(data.coords["time"].dtype, np.datetime64):
        return AtomicCheckResult.skipped_result(
            name="Latest Date Check",
            info="'time' coordinate is not of datetime type.",
        )

    gcb_year = config.GCB_YEAR
    last_year = gcb_year - 1
    last_month_start = np.datetime64(f"{last_year}-12-01")
    last_month_end = np.datetime64(f"{last_year}-12-31 23:59:59")

    time_values = data.coords["time"].values
    last_step = time_values[-1]

    passed = last_month_start <= last_step <= last_month_end
    display_values = {
        "last_time_step": str(last_step),
        "expected_range_start": str(last_month_start),
        "expected_range_end": str(last_month_end),
    }

    if passed:
        return AtomicCheckResult.passed_result(
            name="Latest Date Check",
            info="Latest time step is within expected range",
            details=display_values,
        )
    else:
        return AtomicCheckResult.failed_result(
            name="Latest Date Check",
            info="Latest time step is outside expected range",
            details=display_values,
        )


def longitude_0_360(data: xr.DataArray) -> AtomicCheckResult:
    lon_values = data.values
    outside_0_360 = np.any((lon_values < 0) | (lon_values > 360))

    if outside_0_360:
        return AtomicCheckResult.failed_result(
            name="Longitude 0-360 Check",
            info="Longitude values are outside the range [0, 360].",
            details={
                "lon_min": float(np.nanmin(lon_values)),
                "lon_max": float(np.nanmax(lon_values)),
            },
        )
    else:
        return AtomicCheckResult.passed_result(
            name="Longitude 0-360 Check",
            info="Longitude values are within the range [0, 360].",
            details={
                "lon_min": float(np.nanmin(lon_values)),
                "lon_max": float(np.nanmax(lon_values)),
            },
        )


def global_ocean_area_check(data: xr.DataArray) -> AtomicCheckResult:
    total_area_km2 = data.sum().compute().item() / 1e6
    coverage_percent_ocean = (total_area_km2 / _OCEAN_SURFACE_AREA_KM2) * 100
    coverage_percent_global = (total_area_km2 / _GLOBAL_SURFACE_AREA_KM2) * 100

    if coverage_percent_global < 90:
        # now use ocean coverage
        coverage_percent = coverage_percent_ocean
        area_domain = "ocean"
    else:
        coverage_percent = coverage_percent_global
        area_domain = "global surface"

    coverage_percent = round(coverage_percent, 2)

    result = dict(
        name="Global Ocean Area Check",
        details={
            "coverage_percent": coverage_percent,
            "area_domain": area_domain,
            "total_area_Mkm2": round(total_area_km2 / 1e6, 2),
        },
        info=f"Dataset covers {coverage_percent:.1f}% of {area_domain} area.",
    )

    if (coverage_percent >= 90.0) and (coverage_percent < 105):
        return AtomicCheckResult.passed_result(**result)  # type: ignore
    else:
        return AtomicCheckResult.failed_result(**result)  # type: ignore


gcb_ocean_dataprod_suite = CheckSuite(
    name=f"GCB {config.GCB_YEAR} - Ocean Checks",
    checks=[
        CallableCheck(
            name="Time Format Readable",
            data_scope="coords",
            variables=["time"],
            fn=check_time_format_readable,
        ),
        CallableCheck(
            name="Latest Date Check",
            data_scope="coords",
            variables=["time"],
            fn=latest_date_check,
        ),
        CallableCheck(
            name="Time Step Regular",
            data_scope="coords",
            variables=["time"],
            fn=check_time_step_regular,
        ),
        CallableCheck(
            name="Missing Time Slices",
            data_scope="data_vars",
            fn=check_missing_slices,
        ),
        CallableCheck(
            name="Latitude Range Check",
            data_scope="coords",
            variables=["lat"],
            fn=_latitude_range_check,
        ),
        CallableCheck(
            name="Longitude 0-360 Check",
            data_scope="coords",
            variables=["lon"],
            fn=longitude_0_360,
        ),
        CallableCheck(
            name="Global Ocean Area Check",
            data_scope="data_vars",
            variables=["area"],
            fn=global_ocean_area_check,
        ),
        CallableCheck(
            name="Longitude Shift",
            data_scope="data_vars",
            fn=check_180_lon_shift,
        ),
        CallableCheck(
            name="Missing Longitudes",
            data_scope="data_vars",
            fn=check_missing_lons,
        ),
    ],
)
