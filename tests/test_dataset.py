from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from nc_check.dataset import CanonicalDataset


def test_canonical_dataset_renames_alias_coordinates() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("t", "latitude", "longitude"), [[[1.0]]])},
        coords={"t": [0], "latitude": [10.0], "longitude": [20.0]},
    )

    ds = CanonicalDataset.from_xarray(raw)

    assert isinstance(ds, CanonicalDataset)
    assert set(ds.coords) >= {"time", "lat", "lon"}
    assert ds["temp"].dims == ("time", "lat", "lon")


def test_canonical_dataset_requires_time_lat_lon_when_strict() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("lat", "lon"), [[1.0]])},
        coords={"lat": [10.0], "lon": [20.0]},
    )

    with pytest.raises(ValueError, match="Missing: time"):
        CanonicalDataset.from_xarray(raw, strict=True)


def test_canonical_dataset_validates_lat_lon_numeric() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), [[[1.0]]])},
        coords={"time": [0], "lat": ["north"], "lon": [20.0]},
    )

    with pytest.raises(ValueError, match="Coordinate 'lat' must be numeric"):
        CanonicalDataset.from_xarray(raw)


def test_canonical_dataset_keeps_decoded_time_dtype() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": [10.0],
            "lon": [20.0],
        },
    )

    ds = CanonicalDataset.from_xarray(raw)
    assert np.issubdtype(ds.coords["time"].dtype, np.datetime64)


def test_canonical_dataset_allows_2d_lat_lon_coordinates() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "y", "x"), np.ones((2, 2, 3)))},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": (("y", "x"), np.array([[10.0, 10.1, 10.2], [11.0, 11.1, 11.2]])),
            "lon": (("y", "x"), np.array([[20.0, 20.1, 20.2], [21.0, 21.1, 21.2]])),
        },
    )

    ds = CanonicalDataset.from_xarray(raw)

    assert ds.coords["time"].ndim == 1
    assert ds.coords["lat"].ndim == 2
    assert ds.coords["lon"].ndim == 2


def test_canonical_dataset_requires_1d_time_coordinate() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("y", "x"), np.ones((1, 1)))},
        coords={
            "time": (("y", "x"), np.array([[0.0]])),
            "lat": [10.0],
            "lon": [20.0],
        },
    )

    with pytest.raises(ValueError, match="Coordinate 'time' must be 1D"):
        CanonicalDataset.from_xarray(raw)


def test_canonical_dataset_sets_spatial_attrs_for_near_global_0_360() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((1, 2, 36)))},
        coords={
            "time": [0],
            "lat": ("lat", [-80.0, 80.0]),
            "lon": ("lon", np.arange(0.0, 360.0, 10.0)),
        },
    )

    ds = CanonicalDataset.from_xarray(raw)

    assert ds.canonical_info["spatial_coverage"] == "near-global"
    assert ds.canonical_info["longitude_convention"] == "0:360"


def test_canonical_dataset_sets_spatial_attrs_for_global_minus180_180() -> None:
    lon = np.linspace(-179.5, 179.5, 360)
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((1, 2, lon.size)))},
        coords={
            "time": [0],
            "lat": ("lat", [-89.0, 89.0]),
            "lon": ("lon", lon),
        },
    )

    ds = CanonicalDataset.from_xarray(raw)

    assert ds.canonical_info["spatial_coverage"] == "global"
    assert ds.canonical_info["longitude_convention"] == "-180:180"


def test_canonical_dataset_html_repr_includes_canonical_info() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((1, 2, 3)))},
        coords={
            "time": [0],
            "lat": ("lat", [-45.0, 45.0]),
            "lon": ("lon", [0.0, 120.0, 240.0]),
        },
    )

    ds = CanonicalDataset.from_xarray(raw)
    html = ds._repr_html_()

    assert "Canonical Info" in html
    assert "Canonical Info: <span>(" in html
    assert "spatial_coverage" in html
    assert "longitude_convention" in html
