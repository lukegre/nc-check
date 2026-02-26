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
