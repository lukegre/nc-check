"""Tests for extended heuristic checks added in Chunk 1."""
from __future__ import annotations

import numpy as np
import xarray as xr

from nc_check.checks.heuristic import HeuristicCheck, _heuristic_report


def _run(ds: xr.Dataset) -> dict:
    return _heuristic_report(ds, cf_version="CF-1.12")


def _all_findings(report: dict) -> list[dict]:
    findings = list(report.get("global", []))
    for items in report.get("coordinates", {}).values():
        findings.extend(items)
    for items in report.get("variables", {}).values():
        findings.extend(items)
    return findings


def _has_item(report: dict, item: str) -> bool:
    return any(
        isinstance(f, dict) and f.get("item") == item
        for f in _all_findings(report)
    )


def _has_severity(report: dict, item: str, severity: str) -> bool:
    return any(
        isinstance(f, dict) and f.get("item") == item and f.get("severity") == severity
        for f in _all_findings(report)
    )


# --- Bounds structure (CF §7.1) ---

def test_bounds_wrong_shape_produces_error() -> None:
    # bounds shape (N, 3) is wrong — should be (N, 2)
    ds = xr.Dataset(
        coords={
            "lat": (("lat",), [0.0, 1.0], {"bounds": "lat_bnds"}),
            "lat_bnds": (("lat", "nv"), np.zeros((2, 3))),
        },
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "bounds_wrong_shape", "ERROR")


def test_bounds_values_not_bracketing_produces_warn() -> None:
    lats = np.array([0.5, 1.5])
    # bounds do NOT bracket: lower=2, upper=3 for lat=0.5
    bnds = np.array([[2.0, 3.0], [2.5, 3.5]])
    ds = xr.Dataset(
        coords={
            "lat": (("lat",), lats, {"bounds": "lat_bnds"}),
            "lat_bnds": (("lat", "nv"), bnds),
        },
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "bounds_not_bracketing", "WARN")


def test_bounds_valid_produces_no_finding() -> None:
    lats = np.array([0.5, 1.5])
    bnds = np.array([[0.0, 1.0], [1.0, 2.0]])
    ds = xr.Dataset(
        coords={
            "lat": (("lat",), lats, {"bounds": "lat_bnds"}),
            "lat_bnds": (("lat", "nv"), bnds),
        },
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert not _has_item(report, "bounds_wrong_shape")
    assert not _has_item(report, "bounds_not_bracketing")


# --- Cell methods (CF §7.4) ---

def test_cell_methods_valid_produces_no_finding() -> None:
    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("time",),
                [290.0],
                {"units": "K", "long_name": "T", "cell_methods": "time: mean"},
            )
        },
        coords={"time": [0]},
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert not _has_item(report, "cell_methods_unknown_method")
    assert not _has_item(report, "cell_methods_unknown_dim")


def test_cell_methods_bad_method_produces_warn() -> None:
    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("time",),
                [290.0],
                {"units": "K", "long_name": "T", "cell_methods": "time: foobar"},
            )
        },
        coords={"time": [0]},
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "cell_methods_unknown_method", "WARN")


def test_cell_methods_bad_dim_produces_error() -> None:
    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("time",),
                [290.0],
                {"units": "K", "long_name": "T", "cell_methods": "nonexistent_dim: mean"},
            )
        },
        coords={"time": [0]},
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "cell_methods_unknown_dim", "ERROR")


# --- Calendar (CF §4.4.1) ---

def test_calendar_missing_produces_warn() -> None:
    ds = xr.Dataset(
        data_vars={"v": (("time",), [1.0], {"units": "K", "long_name": "v"})},
        coords={
            "time": (
                ("time",),
                np.array(["2024-01-01"], dtype="datetime64[ns]"),
                {"units": "days since 1970-01-01", "axis": "T"},
            )
        },
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "missing_calendar_attr", "WARN")


def test_calendar_invalid_value_produces_error() -> None:
    ds = xr.Dataset(
        data_vars={"v": (("time",), [1.0], {"units": "K", "long_name": "v"})},
        coords={
            "time": (
                ("time",),
                np.array(["2024-01-01"], dtype="datetime64[ns]"),
                {"units": "days since 1970-01-01", "axis": "T", "calendar": "not_a_calendar"},
            )
        },
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "invalid_calendar_attr", "ERROR")


# --- Vertical positive (CF §4.3) ---

def test_vertical_no_positive_produces_warn() -> None:
    ds = xr.Dataset(
        data_vars={"v": (("depth",), [1.0, 2.0], {"units": "m", "long_name": "v"})},
        coords={"depth": (("depth",), [10.0, 20.0], {"units": "m", "long_name": "depth"})},
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "missing_positive_attr", "WARN")


# --- Dimension order (COARDS/Ferret) ---

def test_dimension_order_wrong_produces_warn() -> None:
    # Wrong order: lon, lat, time — should be time, lat, lon
    ds = xr.Dataset(
        data_vars={"v": (("lon", "lat", "time"), np.ones((3, 2, 1)), {"units": "K", "long_name": "v"})},
        coords={
            "lon": (("lon",), [0.0, 120.0, 240.0]),
            "lat": (("lat",), [-30.0, 30.0]),
            "time": (("time",), np.array(["2024-01-01"], dtype="datetime64[ns]")),
        },
        attrs={"Conventions": "CF-1.12", "institution": "X", "source": "Y", "title": "T", "history": "H"},
    )
    report = _run(ds)
    assert _has_severity(report, "wrong_dimension_order", "WARN")


# --- CMIP6 global attributes ---

def test_cmip6_attrs_missing_produces_warns() -> None:
    ds = xr.Dataset(
        data_vars={"v": (("x",), [1.0], {"units": "K", "long_name": "v"})},
        coords={"x": [0]},
        attrs={"Conventions": "CF-1.12", "mip_era": "CMIP6"},
    )
    report = _run(ds)
    # Should warn about missing CMIP6 attrs: institution, source, tracking_id, etc.
    global_items = [
        f.get("item") for f in report.get("global", []) if isinstance(f, dict)
    ]
    assert "missing_global_attr:institution" in global_items
    assert "missing_global_attr:tracking_id" in global_items


def test_general_attrs_missing_produces_warns() -> None:
    ds = xr.Dataset(
        data_vars={"v": (("x",), [1.0], {"units": "K", "long_name": "v"})},
        coords={"x": [0]},
        attrs={"Conventions": "CF-1.12"},
    )
    report = _run(ds)
    global_items = [
        f.get("item") for f in report.get("global", []) if isinstance(f, dict)
    ]
    assert "missing_global_attr:institution" in global_items
    assert "missing_global_attr:title" in global_items
