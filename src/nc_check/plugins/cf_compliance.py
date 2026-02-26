from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..dataset import CanonicalDataset
from ..models import AtomicCheckResult

_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?|months?|years?)\s+since\s+.+$",
    re.IGNORECASE,
)


def _conventions_check(ds: CanonicalDataset) -> AtomicCheckResult:
    conventions = str(ds.attrs.get("Conventions", "")).strip()
    if not conventions:
        return AtomicCheckResult.failed_result(
            name="cf.conventions",
            info="Dataset is missing global 'Conventions' attribute.",
        )

    tokens = [token.strip() for token in conventions.split(",") if token.strip()]
    if any(token.upper().startswith("CF-") for token in tokens):
        return AtomicCheckResult.passed_result(
            name="cf.conventions",
            info="Conventions includes a CF token.",
            details={"conventions": conventions},
        )

    return AtomicCheckResult.failed_result(
        name="cf.conventions",
        info="Conventions does not contain a CF token (for example CF-1.12).",
        details={"conventions": conventions},
    )


def _coordinate_presence_check(ds: CanonicalDataset) -> AtomicCheckResult:
    missing = [name for name in ("time", "lat", "lon") if name not in ds.coords]
    if missing:
        return AtomicCheckResult.failed_result(
            name="cf.coordinates_present",
            info="Dataset is missing one or more canonical coordinates.",
            details={"missing": missing},
        )

    return AtomicCheckResult.passed_result(
        name="cf.coordinates_present",
        info="Dataset exposes canonical coordinates time/lat/lon.",
    )


def _latitude_units_check(ds: CanonicalDataset) -> AtomicCheckResult:
    units = str(ds.coords["lat"].attrs.get("units", "")).strip().lower()
    if units == "degrees_north":
        return AtomicCheckResult.passed_result(
            name="cf.latitude_units",
            info="Latitude units are degrees_north.",
        )

    if not units:
        return AtomicCheckResult.failed_result(
            name="cf.latitude_units",
            info="Latitude coordinate is missing units.",
            details={"expected": "degrees_north"},
        )

    return AtomicCheckResult.failed_result(
        name="cf.latitude_units",
        info="Latitude units should be degrees_north.",
        details={"expected": "degrees_north", "actual": units},
    )


def _longitude_units_check(ds: CanonicalDataset) -> AtomicCheckResult:
    units = str(ds.coords["lon"].attrs.get("units", "")).strip().lower()
    if units == "degrees_east":
        return AtomicCheckResult.passed_result(
            name="cf.longitude_units",
            info="Longitude units are degrees_east.",
        )

    if not units:
        return AtomicCheckResult.failed_result(
            name="cf.longitude_units",
            info="Longitude coordinate is missing units.",
            details={"expected": "degrees_east"},
        )

    return AtomicCheckResult.failed_result(
        name="cf.longitude_units",
        info="Longitude units should be degrees_east.",
        details={"expected": "degrees_east", "actual": units},
    )


def _time_units_check(ds: CanonicalDataset) -> AtomicCheckResult:
    coord = ds.coords["time"]
    values = np.asarray(coord.values)

    if np.issubdtype(values.dtype, np.datetime64):
        return AtomicCheckResult.passed_result(
            name="cf.time_units",
            info="Time coordinate uses decoded datetime values.",
        )

    units = str(coord.attrs.get("units", "")).strip()
    if not units:
        return AtomicCheckResult.failed_result(
            name="cf.time_units",
            info="Time coordinate is missing CF-like units.",
        )

    if _TIME_UNITS_RE.match(units):
        return AtomicCheckResult.passed_result(
            name="cf.time_units",
            info="Time units match CF 'units since epoch' syntax.",
            details={"units": units},
        )

    return AtomicCheckResult.failed_result(
        name="cf.time_units",
        info="Time units do not match CF expected syntax.",
        details={"units": units},
    )


def _lat_lon_range_check(ds: CanonicalDataset) -> AtomicCheckResult:
    lat_values = np.asarray(ds.coords["lat"].values, dtype=float)
    lon_values = np.asarray(ds.coords["lon"].values, dtype=float)

    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))

    lat_ok = lat_min >= -90.0 and lat_max <= 90.0
    lon_ok = (lon_min >= -180.0 and lon_max <= 180.0) or (
        lon_min >= 0.0 and lon_max <= 360.0
    )

    if lat_ok and lon_ok:
        return AtomicCheckResult.passed_result(
            name="cf.coordinate_ranges",
            info="Latitude and longitude ranges are CF-compatible.",
            details={
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        )

    return AtomicCheckResult.failed_result(
        name="cf.coordinate_ranges",
        info="Latitude or longitude values are outside CF-compatible ranges.",
        details={
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
    )


class CFCompliancePlugin:
    """Example plugin that contributes atomic CF checks."""

    name = "cf_compliance"

    def register(self, registry: Any) -> None:
        registry.register_check(
            name="cf.conventions", check=_conventions_check, plugin=self.name
        )
        registry.register_check(
            name="cf.coordinates_present",
            check=_coordinate_presence_check,
            plugin=self.name,
        )
        registry.register_check(
            name="cf.latitude_units", check=_latitude_units_check, plugin=self.name
        )
        registry.register_check(
            name="cf.longitude_units", check=_longitude_units_check, plugin=self.name
        )
        registry.register_check(
            name="cf.time_units",
            check=_time_units_check,
            plugin=self.name,
        )
        registry.register_check(
            name="cf.coordinate_ranges", check=_lat_lon_range_check, plugin=self.name
        )


def cf_check_names() -> tuple[str, ...]:
    return (
        "cf.conventions",
        "cf.coordinates_present",
        "cf.latitude_units",
        "cf.longitude_units",
        "cf.time_units",
        "cf.coordinate_ranges",
    )
