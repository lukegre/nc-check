from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

_ALIAS_GROUPS = {
    "time": ("time", "t"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}


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

    for name in ("time", "lat", "lon"):
        if name not in ds.coords:
            continue
        coord = ds.coords[name]
        if coord.ndim != 1:
            raise ValueError(f"Coordinate '{name}' must be 1D.")

    for name in ("lat", "lon"):
        if name in ds.coords and not np.issubdtype(ds.coords[name].dtype, np.number):
            raise ValueError(f"Coordinate '{name}' must be numeric.")
    return ds
