from __future__ import annotations

from dataclasses import dataclass
from html import escape
from uuid import uuid4

import numpy as np
import xarray as xr

_ALIAS_GROUPS = {
    "time": ("time", "t"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}
_GLOBAL_LAT_SPAN_DEG = 175.0
_GLOBAL_LON_SPAN_DEG = 355.0
_NEAR_GLOBAL_LAT_SPAN_DEG = 140.0
_NEAR_GLOBAL_LON_SPAN_DEG = 300.0


@dataclass(frozen=True)
class CoordinateNames:
    time: str = "time"
    lat: str = "lat"
    lon: str = "lon"


class CanonicalDataset(xr.Dataset):
    """Dataset enforcing canonical coordinate names: time, lat, lon."""

    __slots__ = ()

    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset,
        *,
        rename_aliases: bool = True,
        strict: bool = True,
    ) -> CanonicalDataset:
        if isinstance(ds, cls):
            return ds

        normalized = ds
        if rename_aliases:
            rename_map = _build_rename_map(normalized)
            if rename_map:
                normalized = normalized.rename(rename_map)

        normalized = _validate_canonical_coordinates(normalized, strict=strict)

        return cls(
            data_vars=normalized.data_vars,
            coords=normalized.coords,
            attrs=normalized.attrs,
        )

    @property
    def coordinate_names(self) -> CoordinateNames:
        return CoordinateNames()

    @property
    def canonical_info(self) -> dict[str, object]:
        return _canonical_info(self)

    def _repr_html_(self) -> str:
        display_ds = xr.Dataset(
            data_vars=self.data_vars,
            coords=self.coords,
            attrs=self.attrs,
        )
        html = xr.Dataset._repr_html_(display_ds)
        section = _canonical_info_section_html(self.canonical_info)
        marker = "</ul></div></div>"
        idx = html.rfind(marker)
        if idx == -1:
            return html
        return f"{html[:idx]}{section}{html[idx:]}"


def _build_rename_map(ds: xr.Dataset) -> dict[str, str]:
    names = [str(name) for name in list(ds.coords) + list(ds.dims)]
    lowered = {name.lower(): name for name in names}
    rename_map: dict[str, str] = {}

    for canonical, aliases in _ALIAS_GROUPS.items():
        if canonical in ds.coords or canonical in ds.dims:
            continue

        source_name: str | None = None
        for alias in aliases:
            found = lowered.get(alias)
            if found is not None:
                source_name = found
                break
        if source_name is None:
            continue

        rename_map[source_name] = canonical

    return rename_map


def _validate_canonical_coordinates(ds: xr.Dataset, *, strict: bool) -> xr.Dataset:
    missing: list[str] = []
    for name in ("time", "lat", "lon"):
        if name not in ds.coords:
            if name in ds.dims:
                size = int(ds.sizes[name])
                ds = ds.assign_coords({name: np.arange(size)})
            else:
                missing.append(name)

    if missing and strict:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Dataset must include canonical coordinates: time, lat, lon. "
            f"Missing: {missing_text}."
        )

    time_coord = ds.coords.get("time")
    if time_coord is not None and time_coord.ndim != 1:
        raise ValueError("Coordinate 'time' must be 1D.")

    for name in ("lat", "lon"):
        if name not in ds.coords:
            continue
        coord = ds.coords[name]
        if coord.ndim not in (1, 2):
            raise ValueError(f"Coordinate '{name}' must be 1D or 2D.")

    for name in ("lat", "lon"):
        if name in ds.coords and not np.issubdtype(ds.coords[name].dtype, np.number):
            raise ValueError(f"Coordinate '{name}' must be numeric.")
    return ds


def _finite_numeric(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _longitude_convention(values: np.ndarray) -> str:
    lon_min = float(np.nanmin(values))
    lon_max = float(np.nanmax(values))

    within_180 = lon_min >= -180.0 and lon_max <= 180.0
    within_360 = lon_min >= 0.0 and lon_max <= 360.0
    has_negative = bool(np.any(values < 0.0))
    has_over_180 = bool(np.any(values > 180.0))

    if within_360 and has_over_180 and not has_negative:
        return "0:360"
    if within_180 and has_negative and not has_over_180:
        return "-180:180"
    return "unknown"


def _longitude_coverage_span(values: np.ndarray) -> float:
    normalized = np.mod(values, 360.0)
    normalized = np.unique(normalized)
    if normalized.size <= 1:
        return 0.0

    diffs = np.diff(normalized)
    wrap_gap = float(normalized[0] + 360.0 - normalized[-1])
    max_gap = float(np.max(np.concatenate([diffs, np.array([wrap_gap])])))
    return float(360.0 - max_gap)


def _spatial_coverage(lat_span: float, lon_span: float) -> str:
    if lat_span >= _GLOBAL_LAT_SPAN_DEG and lon_span >= _GLOBAL_LON_SPAN_DEG:
        return "global"
    if lat_span >= _NEAR_GLOBAL_LAT_SPAN_DEG and lon_span >= _NEAR_GLOBAL_LON_SPAN_DEG:
        return "near-global"
    return "regional"


def _canonical_info(ds: xr.Dataset) -> dict[str, object]:
    lat = ds.coords.get("lat")
    lon = ds.coords.get("lon")
    if lat is None or lon is None:
        return {
            "spatial_coverage": "unknown",
            "longitude_convention": "unknown",
        }

    lat_values = _finite_numeric(lat.values)
    lon_values = _finite_numeric(lon.values)
    if lat_values.size == 0 or lon_values.size == 0:
        return {
            "spatial_coverage": "unknown",
            "longitude_convention": "unknown",
        }

    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    lat_span = float(np.nanmax(lat_values) - np.nanmin(lat_values))
    lon_span = _longitude_coverage_span(lon_values)

    return {
        "spatial_coverage": _spatial_coverage(lat_span, lon_span),
        "longitude_convention": _longitude_convention(lon_values),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_span_degrees": lat_span,
        "lon_span_degrees": lon_span,
    }


def _canonical_info_section_html(canonical_info: dict[str, object]) -> str:
    section_id = f"section-canonical-{uuid4()}"
    rows = "".join(
        f"<dt><span>{escape(str(key))} :</span></dt><dd>{escape(str(value))}</dd>"
        for key, value in canonical_info.items()
    )
    return (
        "<li class='xr-section-item'>"
        f"<input id='{section_id}' class='xr-section-summary-in' type='checkbox' checked />"
        f"<label for='{section_id}' class='xr-section-summary' title='Expand/collapse section'>"
        f"Canonical Info: <span>({len(canonical_info)})</span>"
        "</label>"
        "<div class='xr-section-inline-details'></div>"
        f"<div class='xr-section-details'><dl class='xr-attrs'>{rows}</dl></div>"
        "</li>"
    )
