from __future__ import annotations

import re

import numpy as np
import xarray as xr

from ..models import AtomicCheckResult
from ..suite import CallableCheck, CheckSuite

_CF_SUITE_NAME = "CF Compliance"

_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?|months?|years?)\s+since\s+.+$",
    re.IGNORECASE,
)
_STANDARD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CELL_METHODS_TOKEN_RE = re.compile(r"^[A-Za-z0-9_]+\s*:\s*[A-Za-z0-9_]+")
_VALID_CALENDARS = {
    "standard",
    "gregorian",
    "proleptic_gregorian",
    "noleap",
    "365_day",
    "all_leap",
    "366_day",
    "360_day",
    "julian",
    "none",
}
_VALID_FEATURE_TYPES = {
    "point",
    "timeseries",
    "trajectory",
    "profile",
    "timeseriesprofile",
    "trajectoryprofile",
}


####################################
# Atomic Checks for CF compliance  #
####################################
def _conventions_check(data: xr.DataArray) -> AtomicCheckResult:
    conventions = str(data.attrs.get("Conventions", "")).strip()
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


def _coordinate_presence_check(data: xr.DataArray) -> AtomicCheckResult:
    missing = [name for name in ("time", "lat", "lon") if name not in data.coords]
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


def _coordinate_variables_linked_check(data: xr.DataArray) -> AtomicCheckResult:
    missing_dim_coords: dict[str, list[str]] = {}
    missing_coordinate_refs: dict[str, list[str]] = {}

    for var_name, variable in data.data_vars.items():
        missing_dims = [dim for dim in variable.dims if dim not in data.coords]
        if missing_dims:
            missing_dim_coords[str(var_name)] = [str(dim) for dim in missing_dims]

        coordinates_attr = variable.attrs.get("coordinates")
        if coordinates_attr is None:
            continue

        missing_refs = [
            token
            for token in _string_tokens(coordinates_attr)
            if token not in data.coords and token not in data.data_vars
        ]
        if missing_refs:
            missing_coordinate_refs[str(var_name)] = missing_refs

    if missing_dim_coords or missing_coordinate_refs:
        return AtomicCheckResult.failed_result(
            name="cf.coordinate_variables_linked",
            info="Some variables reference missing coordinate metadata.",
            details={
                "missing_dim_coords": missing_dim_coords,
                "missing_coordinates_attr_refs": missing_coordinate_refs,
            },
        )

    return AtomicCheckResult.passed_result(
        name="cf.coordinate_variables_linked",
        info="Dimension and coordinates-attribute coordinate links are valid.",
    )


def _latitude_units_check(data: xr.DataArray) -> AtomicCheckResult:
    coord = data.coords.get("lat")
    if coord is None:
        return AtomicCheckResult.failed_result(
            name="cf.latitude_units",
            info="Latitude coordinate is missing.",
            details={"expected": "degrees_north"},
        )

    units = str(coord.attrs.get("units", "")).strip().lower()
    if units == "degrees_north":
        return AtomicCheckResult.passed_result(
            name="cf.latitude_units",
            info="Latitude units are degrees_north.",
        )

    if not units:
        return AtomicCheckResult.warn_result(
            name="cf.latitude_units",
            info="Latitude coordinate is missing units.",
            details={"expected": "degrees_north"},
        )

    return AtomicCheckResult.warn_result(
        name="cf.latitude_units",
        info="Latitude units should be degrees_north.",
        details={"expected": "degrees_north", "actual": units},
    )


def _longitude_units_check(data: xr.DataArray) -> AtomicCheckResult:
    coord = data.coords.get("lon")
    if coord is None:
        return AtomicCheckResult.failed_result(
            name="cf.longitude_units",
            info="Longitude coordinate is missing.",
            details={"expected": "degrees_east"},
        )

    units = str(coord.attrs.get("units", "")).strip().lower()
    if units == "degrees_east":
        return AtomicCheckResult.passed_result(
            name="cf.longitude_units",
            info="Longitude units are degrees_east.",
        )

    if not units:
        return AtomicCheckResult.warn_result(
            name="cf.longitude_units",
            info="Longitude coordinate is missing units.",
            details={"expected": "degrees_east"},
        )

    return AtomicCheckResult.warn_result(
        name="cf.longitude_units",
        info="Longitude units should be degrees_east.",
        details={"expected": "degrees_east", "actual": units},
    )


def _time_units_check(data: xr.DataArray) -> AtomicCheckResult:
    coord = data.coords.get("time")
    if coord is None:
        return AtomicCheckResult.failed_result(
            name="cf.time_units",
            info="Time coordinate is missing.",
        )

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


def _time_calendar_check(data: xr.DataArray) -> AtomicCheckResult:
    coord = data.coords.get("time")
    if coord is None:
        return AtomicCheckResult.failed_result(
            name="cf.time_calendar",
            info="Time coordinate is missing.",
        )

    calendar = str(coord.attrs.get("calendar", "")).strip().lower()
    if not calendar:
        return AtomicCheckResult.passed_result(
            name="cf.time_calendar",
            info="Time calendar is not set (CF default calendar assumed).",
        )

    if calendar in _VALID_CALENDARS:
        return AtomicCheckResult.passed_result(
            name="cf.time_calendar",
            info="Time calendar value is CF-compatible.",
            details={"calendar": calendar},
        )

    return AtomicCheckResult.failed_result(
        name="cf.time_calendar",
        info="Time calendar is not a recognized CF calendar value.",
        details={"calendar": calendar},
    )


def _time_monotonic_check(data: xr.DataArray) -> AtomicCheckResult:
    coord = data.coords.get("time")
    if coord is None:
        return AtomicCheckResult.failed_result(
            name="cf.time_monotonic",
            info="Time coordinate is missing.",
        )

    values = np.asarray(coord.values)
    if values.size <= 1:
        return AtomicCheckResult.passed_result(
            name="cf.time_monotonic",
            info="Time axis has <= 1 value; monotonic by definition.",
        )

    if np.issubdtype(values.dtype, np.datetime64):
        encoded = values.astype("datetime64[ns]").astype(np.int64)
        monotonic = bool(np.all(np.diff(encoded) >= 0))
    elif np.issubdtype(values.dtype, np.number):
        monotonic = bool(np.all(np.diff(values.astype(float)) >= 0))
    else:
        return AtomicCheckResult.failed_result(
            name="cf.time_monotonic",
            info="Time coordinate is not numeric or datetime-like.",
            details={"dtype": str(values.dtype)},
        )

    if monotonic:
        return AtomicCheckResult.passed_result(
            name="cf.time_monotonic",
            info="Time axis is monotonic non-decreasing.",
        )

    return AtomicCheckResult.failed_result(
        name="cf.time_monotonic",
        info="Time axis is not monotonic non-decreasing.",
    )


def _latitude_range_check(data: xr.DataArray) -> AtomicCheckResult:
    values = np.asarray(data.values, dtype=float)
    finite = _finite_min_max(values)
    if finite is None:
        return AtomicCheckResult.failed_result(
            name="cf.latitude_range",
            info="Latitude coordinate has no finite numeric values.",
        )

    lat_min, lat_max = finite
    lat_ok = lat_min >= -90.0 and lat_max <= 90.0

    if lat_ok:
        return AtomicCheckResult.passed_result(
            name="cf.latitude_range",
            info="Latitude range is CF-compatible.",
            details={"lat_min": lat_min, "lat_max": lat_max},
        )

    return AtomicCheckResult.failed_result(
        name="cf.latitude_range",
        info="Latitude values are outside CF-compatible range [-90, 90].",
        details={"lat_min": lat_min, "lat_max": lat_max},
    )


def _longitude_range_check(data: xr.DataArray) -> AtomicCheckResult:
    values = np.asarray(data.values, dtype=float)
    finite = _finite_min_max(values)
    if finite is None:
        return AtomicCheckResult.failed_result(
            name="cf.longitude_range",
            info="Longitude coordinate has no finite numeric values.",
        )

    lon_min, lon_max = finite
    lon_ok = (lon_min >= -180.0 and lon_max <= 180.0) or (
        lon_min >= 0.0 and lon_max <= 360.0
    )

    if lon_ok:
        return AtomicCheckResult.passed_result(
            name="cf.longitude_range",
            info="Longitude range is CF-compatible.",
            details={"lon_min": lon_min, "lon_max": lon_max},
        )

    return AtomicCheckResult.failed_result(
        name="cf.longitude_range",
        info="Longitude values are outside CF-compatible ranges [-180,180] or [0,360].",
        details={"lon_min": lon_min, "lon_max": lon_max},
    )


def _grid_mapping_reference_check(data: xr.DataArray) -> AtomicCheckResult:
    missing: dict[str, str] = {}
    incomplete: dict[str, str] = {}

    for var_name, variable in data.data_vars.items():
        mapping_name = str(variable.attrs.get("grid_mapping", "")).strip()
        if not mapping_name:
            continue

        mapping_var = data.coords.get(mapping_name) or data.data_vars.get(mapping_name)
        if mapping_var is None:
            missing[str(var_name)] = mapping_name
            continue

        mapping_name_attr = str(mapping_var.attrs.get("grid_mapping_name", "")).strip()
        if not mapping_name_attr:
            incomplete[str(var_name)] = mapping_name

    if missing or incomplete:
        return AtomicCheckResult.failed_result(
            name="cf.grid_mapping_references",
            info="grid_mapping references are missing or incomplete.",
            details={
                "missing_grid_mapping_vars": missing,
                "missing_grid_mapping_name_attr": incomplete,
            },
        )

    return AtomicCheckResult.passed_result(
        name="cf.grid_mapping_references",
        info="grid_mapping references are valid for all variables that declare them.",
    )


def _units_parseable_check(data: xr.DataArray) -> AtomicCheckResult:
    units = str(data.attrs.get("units", "")).strip()
    if not units:
        return AtomicCheckResult.passed_result(
            name="cf.units_parseable",
            info="units not provided; skipping parseability check.",
        )

    if not _looks_udunits_like(units):
        return AtomicCheckResult.failed_result(
            name="cf.units_parseable",
            info="units are not UDUNITS-like.",
            details={"units": units},
        )

    return AtomicCheckResult.passed_result(
        name="cf.units_parseable",
        info="units are parseable-like.",
        details={"units": units},
    )


def _standard_name_syntax_check(data: xr.DataArray) -> AtomicCheckResult:
    standard_name = str(data.attrs.get("standard_name", "")).strip()
    if not standard_name:
        return AtomicCheckResult.passed_result(
            name="cf.standard_name_syntax",
            info="standard_name not provided.",
        )

    if not _STANDARD_NAME_RE.match(standard_name):
        return AtomicCheckResult.warn_result(
            name="cf.standard_name_syntax",
            info="standard_name is malformed.",
            details={"standard_name": standard_name},
        )

    return AtomicCheckResult.passed_result(
        name="cf.standard_name_syntax",
        info="standard_name syntax is valid.",
        details={"standard_name": standard_name},
    )


def _missing_data_encoding_check(data: xr.DataArray) -> AtomicCheckResult:
    issues: list[str] = []

    fill_value = _attr_or_encoding(data, "_FillValue")
    missing_value = _attr_or_encoding(data, "missing_value")
    valid_range = _attr_or_encoding(data, "valid_range")
    valid_min = _attr_or_encoding(data, "valid_min")
    valid_max = _attr_or_encoding(data, "valid_max")

    if fill_value is not None and not np.isscalar(_as_scalar(fill_value)):
        issues.append("_FillValue must be scalar")
    if missing_value is not None and not np.isscalar(_as_scalar(missing_value)):
        issues.append("missing_value must be scalar")

    if fill_value is not None and missing_value is not None:
        fill_scalar = _as_scalar(fill_value)
        miss_scalar = _as_scalar(missing_value)
        if (
            np.isscalar(fill_scalar)
            and np.isscalar(miss_scalar)
            and fill_scalar != miss_scalar
        ):
            issues.append("_FillValue and missing_value differ")

    if fill_value is not None and valid_range is not None:
        arr = np.asarray(valid_range, dtype=float).reshape(-1)
        if arr.size == 2 and _is_scalar_numeric(fill_value):
            fill_float = _to_float(fill_value)
            if fill_float is not None:
                lo, hi = float(np.min(arr)), float(np.max(arr))
                if lo <= fill_float <= hi:
                    issues.append("_FillValue falls inside valid_range")

    if fill_value is not None and _is_scalar_numeric(fill_value):
        fill_float = _to_float(fill_value)
        if fill_float is not None:
            valid_min_float = _to_float(valid_min)
            valid_max_float = _to_float(valid_max)
            if valid_min_float is not None and fill_float >= valid_min_float:
                issues.append("_FillValue is >= valid_min")
            if valid_max_float is not None and fill_float <= valid_max_float:
                issues.append("_FillValue is <= valid_max")

    if issues:
        return AtomicCheckResult.failed_result(
            name="cf.missing_data_encoding",
            info="Missing-data metadata has type or range conflicts.",
            details={"issues": issues},
        )

    return AtomicCheckResult.passed_result(
        name="cf.missing_data_encoding",
        info="Missing-data metadata is internally consistent.",
    )


def _packed_data_encoding_check(data: xr.DataArray) -> AtomicCheckResult:
    issues: dict[str, list[str]] = {}

    for var_name, variable in data.data_vars.items():
        var_issues: list[str] = []
        scale_factor = _attr_or_encoding(variable, "scale_factor")
        add_offset = _attr_or_encoding(variable, "add_offset")

        if scale_factor is None and add_offset is None:
            continue

        if scale_factor is not None and not _is_scalar_numeric(scale_factor):
            var_issues.append("scale_factor must be finite numeric scalar")
        if add_offset is not None and not _is_scalar_numeric(add_offset):
            var_issues.append("add_offset must be finite numeric scalar")

        scale_factor_float = _to_float(scale_factor)
        if scale_factor_float is not None and scale_factor_float == 0.0:
            var_issues.append("scale_factor must not be zero")

        if var_issues:
            issues[str(var_name)] = var_issues

    if issues:
        return AtomicCheckResult.failed_result(
            name="cf.packed_data_encoding",
            info="Packed-data metadata is invalid.",
            details={"issues": issues},
        )

    return AtomicCheckResult.passed_result(
        name="cf.packed_data_encoding",
        info="Packed-data metadata is valid where present.",
    )


def _bounds_and_cell_methods_check(data: xr.DataArray) -> AtomicCheckResult:
    missing_bounds_vars: dict[str, str] = {}
    invalid_cell_methods: dict[str, str] = {}

    for coord_name, coord in data.coords.items():
        bounds_name = str(coord.attrs.get("bounds", "")).strip()
        if (
            bounds_name
            and bounds_name not in data.coords
            and bounds_name not in data.data_vars
        ):
            missing_bounds_vars[str(coord_name)] = bounds_name

    for var_name, variable in data.data_vars.items():
        cell_methods = str(variable.attrs.get("cell_methods", "")).strip()
        if not cell_methods:
            continue

        tokens = [tok.strip() for tok in cell_methods.split() if ":" in tok]
        if not tokens:
            invalid_cell_methods[str(var_name)] = cell_methods
            continue

        if not _CELL_METHODS_TOKEN_RE.search(cell_methods):
            invalid_cell_methods[str(var_name)] = cell_methods

    if missing_bounds_vars or invalid_cell_methods:
        return AtomicCheckResult.failed_result(
            name="cf.bounds_and_cell_methods",
            info="Bounds or cell_methods metadata is invalid.",
            details={
                "missing_bounds_vars": missing_bounds_vars,
                "invalid_cell_methods": invalid_cell_methods,
            },
        )

    return AtomicCheckResult.passed_result(
        name="cf.bounds_and_cell_methods",
        info="Bounds and cell_methods metadata is valid where present.",
    )


def _ancillary_variables_check(data: xr.DataArray) -> AtomicCheckResult:
    missing: dict[str, list[str]] = {}

    for var_name, variable in data.data_vars.items():
        ancillary = variable.attrs.get("ancillary_variables")
        if ancillary is None:
            continue

        missing_refs = [
            token
            for token in _string_tokens(ancillary)
            if token not in data.data_vars and token not in data.coords
        ]
        if missing_refs:
            missing[str(var_name)] = missing_refs

    if missing:
        return AtomicCheckResult.failed_result(
            name="cf.ancillary_variables",
            info="ancillary_variables points to missing variables.",
            details={"missing_refs": missing},
        )

    return AtomicCheckResult.passed_result(
        name="cf.ancillary_variables",
        info="ancillary_variables references are valid where present.",
    )


def _feature_type_check(data: xr.DataArray) -> AtomicCheckResult:
    feature_type_raw = data.attrs.get("featureType")
    if feature_type_raw is None:
        return AtomicCheckResult.skipped_result(
            name="cf.feature_type",
            info="featureType not set; skipping discrete sampling geometry checks.",
        )

    feature_type = str(feature_type_raw).strip().lower()
    if feature_type not in _VALID_FEATURE_TYPES:
        return AtomicCheckResult.failed_result(
            name="cf.feature_type",
            info="featureType is not a recognized CF discrete sampling geometry.",
            details={"featureType": feature_type_raw},
        )

    coord_names = set(str(name) for name in data.coords)
    has_vertical = any(
        name in coord_names for name in {"z", "depth", "alt", "altitude"}
    )

    required: set[str] = {"lat", "lon"}
    if feature_type in {
        "timeseries",
        "trajectory",
        "timeseriesprofile",
        "trajectoryprofile",
    }:
        required.add("time")
    if feature_type in {"profile", "timeseriesprofile", "trajectoryprofile"}:
        if not has_vertical:
            return AtomicCheckResult.failed_result(
                name="cf.feature_type",
                info="Profile-like featureType requires a vertical coordinate.",
                details={"featureType": feature_type_raw},
            )

    missing_required = sorted(required - coord_names)
    if missing_required:
        return AtomicCheckResult.failed_result(
            name="cf.feature_type",
            info="featureType metadata is inconsistent with required coordinates.",
            details={
                "featureType": feature_type_raw,
                "missing_required_coords": missing_required,
            },
        )

    return AtomicCheckResult.passed_result(
        name="cf.feature_type",
        info="featureType metadata is consistent with required coordinates.",
        details={"featureType": feature_type_raw},
    )


def _attributes_valid_types_only(data: xr.DataArray) -> AtomicCheckResult:
    """
    Check that all attributes are of types that can be represented in netCDF files.
    """
    valid_types = (
        str,
        int,
        float,
        bool,
        np.number,
        np.bool_,
        np.integer,
        np.floating,
        list,
        np.ndarray,
    )

    invalid_attrs = {}
    for attr_name, attr_value in data.attrs.items():
        if not isinstance(attr_value, valid_types):
            invalid_attrs[str(attr_name)] = str(type(attr_value))

    if invalid_attrs:
        return AtomicCheckResult.fatal_result(
            name="cf.attributes_valid_types_only",
            info="Some attributes have types that may not be representable in netCDF files.",
            details=invalid_attrs,
        )
    else:
        return AtomicCheckResult.passed_result(
            name="cf.attributes_valid_types_only",
            info="All attributes are of types that can be represented in netCDF files.",
        )


def _coord_is_monotonic(data: xr.DataArray) -> AtomicCheckResult:
    values = np.asarray(data.values)
    if values.size <= 1:
        return AtomicCheckResult.passed_result(
            name="cf.coord_is_monotonic",
            info="Coordinate has <= 1 value; monotonic by definition.",
        )

    if np.issubdtype(values.dtype, np.datetime64):
        encoded = values.astype("datetime64[ns]").astype(np.int64)
        monotonic = bool(np.all(np.diff(encoded) >= 0))
    elif np.issubdtype(values.dtype, np.number):
        monotonic = bool(np.all(np.diff(values.astype(float)) >= 0))
    else:
        return AtomicCheckResult.fatal_result(
            name="cf.coord_is_monotonic",
            info="Coordinate is not numeric or datetime-like.",
            details={"dtype": str(values.dtype)},
        )

    if monotonic:
        return AtomicCheckResult.passed_result(
            name="cf.coord_is_monotonic",
            info="Coordinate is monotonic non-decreasing.",
        )

    return AtomicCheckResult.failed_result(
        name="cf.coord_is_monotonic",
        info="Coordinate is not monotonic non-decreasing.",
    )


##################################
# CheckSuite for CF compliance   #
##################################
cf_compliance_suite = CheckSuite(
    name=_CF_SUITE_NAME,
    checks=[
        CallableCheck(
            name="CF conventions",
            data_scope="dataset",
            fn=_conventions_check,
        ),
        CallableCheck(
            name="Spatial coodinates present",
            data_scope="dataset",
            fn=_coordinate_presence_check,
        ),
        CallableCheck(
            name="Coordinate is monotonic",
            data_scope="coords",
            fn=_coord_is_monotonic,
        ),
        CallableCheck(
            name="Valid attributes types",
            data_scope="dataset",
            fn=_attributes_valid_types_only,
        ),
        CallableCheck(
            name="Valid attributes types",
            data_scope="coords",
            fn=_attributes_valid_types_only,
        ),
        CallableCheck(
            name="Valid attributes types",
            data_scope="data_vars",
            fn=_attributes_valid_types_only,
        ),
        CallableCheck(
            name="Coordinate variables linked",
            data_scope="dataset",
            fn=_coordinate_variables_linked_check,
        ),
        CallableCheck(
            name="Grid mapping references",
            data_scope="dataset",
            fn=_grid_mapping_reference_check,
        ),
        CallableCheck(
            name="Parsable units",
            data_scope="data_vars",
            fn=_units_parseable_check,
        ),
        CallableCheck(
            name="Standard name syntax",
            data_scope="data_vars",
            fn=_standard_name_syntax_check,
        ),
        CallableCheck(
            name="Fill value encoding",
            data_scope="data_vars",
            fn=_missing_data_encoding_check,
        ),
        CallableCheck(
            name="Fill value encoding",
            data_scope="coords",
            fn=_missing_data_encoding_check,
        ),
        CallableCheck(
            name="Packed data encoding",
            data_scope="dataset",
            fn=_packed_data_encoding_check,
        ),
        CallableCheck(
            name="Bounds and cell methods",
            data_scope="dataset",
            fn=_bounds_and_cell_methods_check,
        ),
        CallableCheck(
            name="Ancillary variables",
            data_scope="dataset",
            fn=_ancillary_variables_check,
        ),
        CallableCheck(
            name="Feature types",
            data_scope="dataset",
            fn=_feature_type_check,
        ),
        CallableCheck(
            name="Latitude units",
            data_scope="coords",
            variables=["lat"],
            fn=_latitude_units_check,
        ),
        CallableCheck(
            name="Longitude units",
            data_scope="coords",
            variables=["lon"],
            fn=_longitude_units_check,
        ),
        CallableCheck(
            name="Time units",
            data_scope="coords",
            variables=["time"],
            fn=_time_units_check,
        ),
        CallableCheck(
            name="Time calendar",
            data_scope="coords",
            variables=["time"],
            fn=_time_calendar_check,
        ),
        CallableCheck(
            name="Time is monotonic",
            data_scope="coords",
            variables=["time"],
            fn=_time_monotonic_check,
        ),
        CallableCheck(
            name="Latitude range",
            data_scope="coords",
            variables=["lat"],
            fn=_latitude_range_check,
        ),
        CallableCheck(
            name="Longitude range",
            data_scope="coords",
            variables=["lon"],
            fn=_longitude_range_check,
        ),
    ],
)


##############################
# Internal Helper Functions  #
##############################
def _attr_or_encoding(data: xr.DataArray, key: str) -> object | None:
    if key in data.attrs:
        return data.attrs.get(key)
    return data.encoding.get(key)


def _string_tokens(value: object) -> list[str]:
    return [tok for tok in str(value).replace(",", " ").split() if tok]


def _as_scalar(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value
    return value


def _is_scalar_numeric(value: object) -> bool:
    scalar = _as_scalar(value)
    if isinstance(scalar, (np.number, int, float)):
        return np.isfinite(float(scalar))
    return False


def _to_float(value: object) -> float | None:
    scalar = _as_scalar(value)
    if isinstance(scalar, (np.number, int, float)):
        return float(scalar)
    return None


def _finite_min_max(values: np.ndarray) -> tuple[float, float] | None:
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _looks_udunits_like(units: str) -> bool:
    text = units.strip()
    if not text:
        return False

    if text.lower() in {"na", "n/a", "none", "unknown", "null", "nan"}:
        return False

    if "?" in text:
        return False

    if _TIME_UNITS_RE.match(text):
        return True

    unit_tokens = [tok for tok in text.split() if tok]
    if not unit_tokens:
        return False

    for tok in unit_tokens:
        if not re.match(r"^[A-Za-z0-9_./^*+-]+$", tok):
            return False

    alpha_only = [tok for tok in unit_tokens if re.match(r"^[A-Za-z_]+$", tok)]
    if len(alpha_only) >= 2 and any(
        tok.lower() in {"after", "before", "from", "at", "to", "by", "near"}
        for tok in alpha_only
    ):
        return False

    return True
