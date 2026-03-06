from __future__ import annotations

import numpy as np
import xarray as xr

from nc_check.dataset import CanonicalDataset
from nc_check.plugins.cf_compliance import cf_compliance_suite
from nc_check.plugins.time_cover import time_cover_suite


def _status_by_base_name(payload: dict[str, object]) -> dict[str, str]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return {}

    priority = {"failed": 0, "unknown": 1, "skipped": 2, "passed": 3}
    out: dict[str, str] = {}
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        base = name.split("[", 1)[0]
        status = str(item.get("status", "")).strip().lower()
        previous = out.get(base)
        if previous is None or priority.get(status, 1) < priority.get(previous, 1):
            out[base] = status
    return out


def test_cf_compliance_flags_bad_time_units_in_gnarly_netcdf(tmp_path) -> None:
    raw = xr.Dataset(
        data_vars={
            "pCO2 (sea) ???": (("t", "latitude", "longitude"), np.ones((2, 2, 2)))
        },
        coords={
            "t": (
                "t",
                np.array([0, 1], dtype=np.int32),
                {"units": "furlongs after lunch"},
            ),
            "latitude": ("latitude", [-45.0, 45.0], {"units": "degrees_north"}),
            "longitude": ("longitude", [10.0, 20.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )

    path = tmp_path / "gnarly-bad-time-units.nc"
    raw.to_netcdf(path)

    opened = xr.open_dataset(path, decode_times=False)
    canonical = CanonicalDataset.from_xarray(opened)
    payload = cf_compliance_suite.run(canonical).to_dict()

    statuses = _status_by_base_name(payload)
    assert statuses["cf.conventions"] == "passed"
    assert statuses["cf.coordinates_present"] == "passed"
    assert statuses["cf.time_units"] == "failed"


def test_time_cover_handles_unreadable_time_with_horrible_variable_name(
    tmp_path,
) -> None:
    bad_var_name = "__really__bad__var__name__v2026"
    raw = xr.Dataset(
        data_vars={bad_var_name: (("t", "y", "x"), np.ones((2, 1, 1)))},
        coords={
            "t": (
                "t",
                np.array(["today-ish", "tomorrow-ish"], dtype=str),
                {"units": "n/a"},
            ),
            "y": ("y", [12.0], {"units": "degrees_north"}),
            "x": ("x", [154.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )

    path = tmp_path / "gnarly-unreadable-time.nc"
    raw.to_netcdf(path)

    opened = xr.open_dataset(path, decode_times=False)
    canonical = CanonicalDataset.from_xarray(opened)
    payload = time_cover_suite.run(canonical).to_dict()

    statuses = _status_by_base_name(payload)
    assert statuses["Time format readable"] == "failed"
    assert statuses["Time step regular"] == "skipped"
    assert statuses["Missing time slices"] == "passed"

    assert bad_var_name in payload["results"]["data_vars"]


def test_canonical_dataset_from_file_normalizes_uppercase_alias_coords(
    tmp_path,
) -> None:
    raw = xr.Dataset(
        data_vars={"TMP__very_bad_name": (("T", "Y", "X"), np.ones((2, 2, 2)))},
        coords={
            "T": ("T", [0, 1]),
            "Y": ("Y", [-2.0, 2.0]),
            "X": ("X", [120.0, 121.0]),
        },
        attrs={"Conventions": "CF-1.12"},
    )

    path = tmp_path / "gnarly-uppercase-aliases.nc"
    raw.to_netcdf(path)

    ds = CanonicalDataset.from_file(str(path), decode_times=False)

    assert set(ds.coords) >= {"time", "lat", "lon"}
    assert ds["TMP__very_bad_name"].dims == ("time", "lat", "lon")


def test_cf_compliance_accepts_messy_but_cf_like_time_units(tmp_path) -> None:
    raw = xr.Dataset(
        data_vars={"temp__ugly": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={
            "time": (
                "time",
                np.array([0, 1], dtype=np.int32),
                {"units": "   HOURS  SINCE   2001-01-01 00:00:00   "},
            ),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [20.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )

    path = tmp_path / "gnarly-time-units-messy-but-valid.nc"
    raw.to_netcdf(path)

    opened = xr.open_dataset(path, decode_times=False)
    canonical = CanonicalDataset.from_xarray(opened)
    payload = cf_compliance_suite.run(canonical).to_dict()

    statuses = _status_by_base_name(payload)
    assert statuses["cf.time_units"] == "passed"


def test_cf_compliance_fails_when_time_units_are_missing(tmp_path) -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={
            "time": ("time", np.array([0, 1], dtype=np.int32)),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [20.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )

    path = tmp_path / "gnarly-time-units-missing.nc"
    raw.to_netcdf(path)

    opened = xr.open_dataset(path, decode_times=False)
    canonical = CanonicalDataset.from_xarray(opened)
    payload = cf_compliance_suite.run(canonical).to_dict()

    statuses = _status_by_base_name(payload)
    assert statuses["cf.time_units"] == "failed"


def test_time_cover_marks_numeric_undecoded_time_as_unreadable(tmp_path) -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={
            "time": (
                "time",
                np.array([0, 1, 2], dtype=np.int32),
                {"units": "days after launch"},
            ),
            "lat": ("lat", [5.0], {"units": "degrees_north"}),
            "lon": ("lon", [150.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )

    path = tmp_path / "gnarly-time-undecoded-numeric.nc"
    raw.to_netcdf(path)

    opened = xr.open_dataset(path, decode_times=False)
    canonical = CanonicalDataset.from_xarray(opened)
    payload = time_cover_suite.run(canonical).to_dict()

    statuses = _status_by_base_name(payload)
    assert statuses["Time format readable"] == "failed"
    assert statuses["Time step regular"] == "skipped"
