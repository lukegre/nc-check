from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import xarray as xr

from ..core.check import Check, CheckInfo, CheckResult
from ..core.coverage import (
    indices_to_ranges,
    leaf_statuses,
    missing_mask,
    resolve_time_dim,
    status_from_leaf_statuses,
    value_label,
)
from ..engine.suite import Suite, SuiteCheck
from ..formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    print_pretty_ocean_reports,
    render_pretty_ocean_report_html,
    render_pretty_ocean_reports_html,
    save_html_report,
)

_LON_CANDIDATES = ("lon", "longitude", "x")
_LAT_CANDIDATES = ("lat", "latitude", "y")
LongitudeConvention: TypeAlias = Literal["-180_180", "0_360", "other"]

_LAND_REFERENCE_POINTS = (
    ("sahara", 23.0, 13.0),
    ("australia_interior", -25.0, 134.0),
    ("mongolia", 47.0, 103.0),
    ("greenland_interior", 72.0, -40.0),
    ("south_america_interior", -15.0, -60.0),
)

_OCEAN_REFERENCE_POINTS = (
    ("equatorial_pacific", 0.0, -140.0),
    ("north_atlantic", 30.0, -40.0),
    ("indian_ocean", -30.0, 80.0),
    ("south_pacific", -45.0, -150.0),
    ("west_pacific", 10.0, 160.0),
)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _guess_coord_name(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
    units_token: str,
) -> str | None:
    candidate_set = set(candidates)
    for coord_name in ds.coords:
        if _normalize_name(str(coord_name)) in candidate_set:
            return str(coord_name)

    for coord_name, coord in ds.coords.items():
        units = _normalize_name(str(coord.attrs.get("units", "")))
        if units_token in units:
            return str(coord_name)
    return None


def _resolve_1d_coord(
    ds: xr.Dataset, coord_name: str
) -> tuple[str, np.ndarray[Any, Any]]:
    if coord_name not in ds.coords:
        raise ValueError(f"Coordinate '{coord_name}' not found.")
    coord = ds.coords[coord_name]
    if coord.ndim != 1:
        raise ValueError(f"Coordinate '{coord_name}' must be 1D.")
    return str(coord.dims[0]), np.asarray(coord.values)


def _choose_data_vars(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    lon_dim: str,
    lat_dim: str,
) -> list[xr.DataArray]:
    if var_name is not None:
        if var_name not in ds.data_vars:
            raise ValueError(f"Data variable '{var_name}' not found.")
        da = ds[var_name]
        if lon_dim not in da.dims or lat_dim not in da.dims:
            raise ValueError(
                f"Data variable '{var_name}' must include lon dim '{lon_dim}' and lat dim '{lat_dim}'."
            )
        return [da]

    selected: list[xr.DataArray] = []
    for _, da in ds.data_vars.items():
        if lon_dim in da.dims and lat_dim in da.dims:
            selected.append(da)

    if selected:
        return selected

    raise ValueError(
        "Could not infer ocean variable. Provide `var_name` for a variable that has lat/lon dimensions."
    )


def _value_ranges_from_indices(
    indices: list[int],
    values: np.ndarray[Any, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start, end in indices_to_ranges(indices):
        out.append(
            {
                "start": value_label(values[start]),
                "end": value_label(values[end]),
            }
        )
    return out


def _longitude_convention(lon_values: np.ndarray[Any, Any]) -> LongitudeConvention:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    eps = 1e-6
    if lon_min >= -180.0 - eps and lon_max <= 180.0 + eps:
        return "-180_180"
    if lon_min >= 0.0 - eps and lon_max <= 360.0 + eps:
        return "0_360"
    return "other"


def _normalize_lon_for_grid(lon: float, convention: LongitudeConvention) -> float:
    if convention == "0_360":
        return lon % 360.0
    if convention == "-180_180":
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _is_global_grid(
    lon_values: np.ndarray[Any, Any], lat_values: np.ndarray[Any, Any]
) -> bool:
    lon_span = float(np.nanmax(lon_values) - np.nanmin(lon_values))
    lat_span = float(np.nanmax(lat_values) - np.nanmin(lat_values))
    return lon_span >= 300.0 and lat_span >= 120.0


def _missing_lon_indices_for_time(
    da: xr.DataArray,
    *,
    lon_dim: str,
    time_dim: str | None,
    time_index: int | None,
) -> np.ndarray[Any, Any]:
    section = (
        da
        if time_dim is None or time_index is None
        else da.isel({time_dim: time_index})
    )
    mask = missing_mask(section)
    reduce_dims = [dim for dim in mask.dims if dim != lon_dim]
    if reduce_dims:
        mask = mask.all(dim=reduce_dims)
    return np.flatnonzero(np.asarray(mask.values, dtype=bool))


def _edge_of_map_check(
    da: xr.DataArray,
    *,
    lon_name: str,
    lon_dim: str,
    time_dim: str | None,
) -> dict[str, Any]:
    sampled_indices: list[int] = []
    persistent_missing_lon_indices = np.array([], dtype=int)

    if time_dim is None:
        sampled_indices = [0]
        persistent_missing_lon_indices = _missing_lon_indices_for_time(
            da,
            lon_dim=lon_dim,
            time_dim=None,
            time_index=None,
        )
    else:
        time_size = int(da.sizes[time_dim])
        last_idx = time_size - 1
        first_idx = 0

        sampled_indices.append(last_idx)
        missing_last = _missing_lon_indices_for_time(
            da,
            lon_dim=lon_dim,
            time_dim=time_dim,
            time_index=last_idx,
        )

        if first_idx != last_idx:
            sampled_indices.append(first_idx)
        missing_first = _missing_lon_indices_for_time(
            da,
            lon_dim=lon_dim,
            time_dim=time_dim,
            time_index=first_idx,
        )

        if missing_last.size and np.array_equal(missing_last, missing_first):
            persistent_missing_lon_indices = missing_last
            if time_size > 2:
                middle_idx = time_size // 2
                if middle_idx not in sampled_indices:
                    sampled_indices.append(middle_idx)
                missing_middle = _missing_lon_indices_for_time(
                    da,
                    lon_dim=lon_dim,
                    time_dim=time_dim,
                    time_index=middle_idx,
                )
                if not np.array_equal(missing_middle, persistent_missing_lon_indices):
                    persistent_missing_lon_indices = np.array([], dtype=int)

    lon_values = np.asarray(da.coords[lon_name].values)
    missing_lon_list = persistent_missing_lon_indices.tolist()
    missing_lon_values = [float(lon_values[idx]) for idx in missing_lon_list]

    return {
        "enabled": True,
        "status": "fail" if missing_lon_list else "pass",
        "sampled_time_indices": sampled_indices,
        "missing_longitude_count": len(missing_lon_list),
        "missing_longitudes": missing_lon_values,
        "missing_longitude_ranges": _value_ranges_from_indices(
            missing_lon_list, lon_values
        ),
    }


def _point_is_missing(point: xr.DataArray) -> bool:
    mask = missing_mask(point)
    reduce_dims = list(mask.dims)
    if reduce_dims:
        mask = mask.all(dim=reduce_dims)
    return bool(np.asarray(mask.values).item())


def _point_alignment_check(
    da: xr.DataArray,
    *,
    lon_name: str,
    lat_name: str,
    lon_convention: str,
    time_dim: str | None,
    lon_values: np.ndarray[Any, Any],
    lat_values: np.ndarray[Any, Any],
) -> dict[str, Any]:
    global_grid = _is_global_grid(lon_values, lat_values)
    if not global_grid:
        return {
            "enabled": True,
            "status": "skipped_non_global",
            "mismatch_count": 0,
            "land_points_checked": 0,
            "ocean_points_checked": 0,
            "land_mismatches": [],
            "ocean_mismatches": [],
            "note": "Skipped land/ocean sanity check because grid does not appear global.",
        }

    section = da if time_dim is None else da.isel({time_dim: -1})

    def check_points(
        points: tuple[tuple[str, float, float], ...],
        *,
        expected_missing: bool,
    ) -> list[dict[str, Any]]:
        mismatches: list[dict[str, Any]] = []
        for label, lat, lon in points:
            target_lon = _normalize_lon_for_grid(lon, lon_convention)
            selected = section.sel(
                {lat_name: lat, lon_name: target_lon},
                method="nearest",
            )
            observed_missing = _point_is_missing(selected)
            if observed_missing == expected_missing:
                continue
            actual_lat = float(np.asarray(selected.coords[lat_name].values).item())
            actual_lon = float(np.asarray(selected.coords[lon_name].values).item())
            mismatches.append(
                {
                    "point": label,
                    "requested_lat": float(lat),
                    "requested_lon": float(target_lon),
                    "actual_lat": actual_lat,
                    "actual_lon": actual_lon,
                    "expected_missing": expected_missing,
                    "observed_missing": observed_missing,
                }
            )
        return mismatches

    land_mismatches = check_points(_LAND_REFERENCE_POINTS, expected_missing=True)
    ocean_mismatches = check_points(_OCEAN_REFERENCE_POINTS, expected_missing=False)
    mismatch_count = len(land_mismatches) + len(ocean_mismatches)

    return {
        "enabled": True,
        "status": "fail" if mismatch_count else "pass",
        "mismatch_count": mismatch_count,
        "land_points_checked": len(_LAND_REFERENCE_POINTS),
        "ocean_points_checked": len(_OCEAN_REFERENCE_POINTS),
        "land_mismatches": land_mismatches,
        "ocean_mismatches": ocean_mismatches,
    }


def _single_ocean_report(
    da: xr.DataArray,
    *,
    lon_name: str,
    lat_name: str,
    lon_dim: str,
    lat_dim: str,
    lon_values: np.ndarray[Any, Any],
    lat_values: np.ndarray[Any, Any],
    time_name: str | None,
    check_edge_of_map: bool,
    check_land_ocean_offset: bool,
) -> dict[str, Any]:
    time_dim = resolve_time_dim(da, time_name)
    lon_convention = _longitude_convention(lon_values)
    edge_result: dict[str, Any] = {}
    offset_result: dict[str, Any] = {}

    def _run_edge() -> dict[str, Any]:
        if not check_edge_of_map:
            return {"enabled": False, "status": "skipped"}
        return _edge_of_map_check(
            da,
            lon_name=lon_name,
            lon_dim=lon_dim,
            time_dim=time_dim,
        )

    def _run_offset() -> dict[str, Any]:
        if not check_land_ocean_offset:
            return {"enabled": False, "status": "skipped"}
        return _point_alignment_check(
            da,
            lon_name=lon_name,
            lat_name=lat_name,
            lon_convention=lon_convention,
            time_dim=time_dim,
            lon_values=lon_values,
            lat_values=lat_values,
        )

    suite_report = Suite(
        name="ocean_cover",
        checks=[
            SuiteCheck(
                check_id="ocean.missing_longitude_bands",
                name="Missing Longitude Bands",
                run=_run_edge,
                detail=lambda result: (
                    f"missing_longitudes={int(result.get('missing_longitude_count', 0))}"
                ),
            ),
            SuiteCheck(
                check_id="ocean.land_ocean_offset",
                name="Land/Ocean Offset",
                run=_run_offset,
                detail=lambda result: f"mismatches={int(result.get('mismatch_count', 0))}",
            ),
        ],
    ).run()

    for item in suite_report["checks"]:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("id", ""))
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        if check_id == "ocean.missing_longitude_bands":
            edge_result = result
        elif check_id == "ocean.land_ocean_offset":
            offset_result = result

    report: dict[str, Any] = {
        "variable": str(da.name),
        "grid": {
            "lon_name": lon_name,
            "lat_name": lat_name,
            "lon_dim": lon_dim,
            "lat_dim": lat_dim,
            "time_dim": time_dim,
            "longitude_convention": lon_convention,
            "longitude_min": float(np.nanmin(lon_values)),
            "longitude_max": float(np.nanmax(lon_values)),
            "latitude_min": float(np.nanmin(lat_values)),
            "latitude_max": float(np.nanmax(lat_values)),
        },
        "checks_enabled": {
            "edge_of_map": bool(check_edge_of_map),
            "land_ocean_offset": bool(check_land_ocean_offset),
        },
        "edge_of_map": edge_result,
        "edge_sliver": edge_result,
        "land_ocean_offset": offset_result,
        "suite": suite_report["suite"],
        "checks": suite_report["checks"],
        "summary": suite_report["summary"],
        "ok": suite_report["ok"],
    }
    return report


def _build_ocean_cover_report(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    lon_name: str | None,
    lat_name: str | None,
    time_name: str | None,
    check_edge_of_map: bool,
    check_land_ocean_offset: bool,
) -> dict[str, Any]:
    lon_name = lon_name or _guess_coord_name(ds, _LON_CANDIDATES, "degrees_east")
    lat_name = lat_name or _guess_coord_name(ds, _LAT_CANDIDATES, "degrees_north")
    if lon_name is None or lat_name is None:
        raise ValueError(
            "Could not infer longitude/latitude coordinates. Pass `lon_name` and `lat_name`."
        )

    lon_dim, lon_values = _resolve_1d_coord(ds, lon_name)
    lat_dim, lat_values = _resolve_1d_coord(ds, lat_name)
    data_vars = _choose_data_vars(
        ds, var_name=var_name, lon_dim=lon_dim, lat_dim=lat_dim
    )

    reports: dict[str, dict[str, Any]] = {}
    for da in data_vars:
        reports[str(da.name)] = _single_ocean_report(
            da,
            lon_name=lon_name,
            lat_name=lat_name,
            lon_dim=lon_dim,
            lat_dim=lat_dim,
            lon_values=lon_values,
            lat_values=lat_values,
            time_name=time_name,
            check_edge_of_map=check_edge_of_map,
            check_land_ocean_offset=check_land_ocean_offset,
        )

    if len(reports) == 1:
        return next(iter(reports.values()))

    suite_checks: list[dict[str, Any]] = []
    for variable_name, per_var in reports.items():
        raw_checks = per_var.get("checks")
        if not isinstance(raw_checks, list):
            continue
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            suite_item = dict(item)
            suite_item["variable"] = variable_name
            suite_checks.append(suite_item)

    suite_report = Suite.report_from_items("ocean_cover", suite_checks)
    return {
        "suite": suite_report["suite"],
        "mode": "all_variables",
        "checked_variable_count": len(reports),
        "checked_variables": list(reports.keys()),
        "reports": reports,
        "checks": suite_report["checks"],
        "summary": suite_report["summary"],
        "ok": suite_report["ok"],
    }


class OceanCoverCheck(Check):
    id = "nc_check.ocean_cover"
    description = "Ocean coverage sanity checks."
    tags = ("ocean", "coverage")

    def __init__(
        self,
        *,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
    ) -> None:
        self.var_name = var_name
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.time_name = time_name
        self.check_edge_of_map = check_edge_of_map
        self.check_land_ocean_offset = check_land_ocean_offset

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        return _build_ocean_cover_report(
            ds,
            var_name=self.var_name,
            lon_name=self.lon_name,
            lat_name=self.lat_name,
            time_name=self.time_name,
            check_edge_of_map=self.check_edge_of_map,
            check_land_ocean_offset=self.check_land_ocean_offset,
        )

    def check(self, ds: xr.Dataset) -> CheckResult:
        report = self.run_report(ds)
        statuses = leaf_statuses(report, ("edge_of_map", "land_ocean_offset"))
        status = status_from_leaf_statuses(statuses)
        return CheckResult(
            check_id=self.id,
            status=status,
            info=CheckInfo(
                message=(
                    f"Ocean cover check completed for {len(statuses)} check outcomes."
                    if statuses
                    else "Ocean cover check completed."
                ),
                details={"report": report, "statuses": statuses},
            ),
            fixable=False,
            tags=list(self.tags),
        )


def check_ocean_cover(
    ds: xr.Dataset,
    *,
    var_name: str | None = None,
    lon_name: str | None = None,
    lat_name: str | None = None,
    time_name: str | None = "time",
    check_edge_of_map: bool = True,
    check_land_ocean_offset: bool = True,
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    """Run ocean-coverage sanity checks on one or more gridded variables."""
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    report = OceanCoverCheck(
        var_name=var_name,
        lon_name=lon_name,
        lat_name=lat_name,
        time_name=time_name,
        check_edge_of_map=check_edge_of_map,
        check_land_ocean_offset=check_land_ocean_offset,
    ).run_report(ds)

    if resolved_format == "tables":
        items = (
            list(report["reports"].values())
            if report.get("mode") == "all_variables"
            else [report]
        )
        print_pretty_ocean_reports(items)
        return None

    if resolved_format == "html":
        if report.get("mode") == "all_variables":
            html_report = render_pretty_ocean_reports_html(
                list(report["reports"].values())
            )
        else:
            html_report = render_pretty_ocean_report_html(report)
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report

    return report
