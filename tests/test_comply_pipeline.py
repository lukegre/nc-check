"""Tests for nc-comply fix pipeline (Chunk 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from nc_check.checks.heuristic import HeuristicCheck
from nc_check.cli import run_comply
from nc_check.core.compliance import make_dataset_compliant


def _make_time_ds() -> xr.Dataset:
    """Dataset with a numeric time coordinate lacking calendar (CF-compliant serializable form)."""
    return xr.Dataset(
        data_vars={"v": (("time",), [1.0], {"units": "K", "long_name": "v"})},
        coords={
            "time": (
                ("time",),
                np.array([19723.0]),  # days since 1970-01-01 = 2024-01-01
                {"units": "days since 1970-01-01", "axis": "T"},
            )
        },
        attrs={
            "Conventions": "CF-1.12",
            "institution": "Test",
            "source": "Test",
            "title": "Test",
            "history": "Test",
        },
    )


def test_comply_adds_calendar() -> None:
    """make_dataset_compliant() should add calendar='standard' to time coords."""
    ds = _make_time_ds()
    fixed = make_dataset_compliant(ds)
    assert fixed["time"].attrs.get("calendar") == "standard"


def test_comply_calendar_not_overwritten() -> None:
    """If calendar is already set, make_dataset_compliant() should not overwrite it."""
    ds = _make_time_ds()
    ds["time"].attrs["calendar"] = "360_day"
    fixed = make_dataset_compliant(ds)
    assert fixed["time"].attrs.get("calendar") == "360_day"


def test_comply_roundtrip_no_new_calendar_errors() -> None:
    """After fixing, re-running heuristic check should not produce calendar errors."""
    ds = _make_time_ds()
    fixed = make_dataset_compliant(ds)
    check = HeuristicCheck(cf_version="CF-1.12")
    report = check.run_report(fixed)
    all_findings = []
    for items in report.get("coordinates", {}).values():
        all_findings.extend(items)
    calendar_errors = [
        f for f in all_findings
        if isinstance(f, dict) and f.get("item") in {"missing_calendar_attr", "invalid_calendar_attr"}
    ]
    assert calendar_errors == []


def test_heuristic_fix_returns_fix_result_with_unfixable_items() -> None:
    """HeuristicCheck.fix() should return FixResult with unfixable_items list."""
    ds = xr.Dataset(
        data_vars={"v": (("depth",), [1.0, 2.0], {"units": "m", "long_name": "v"})},
        coords={"depth": (("depth",), [10.0, 20.0], {"units": "m", "long_name": "depth"})},
        attrs={
            "Conventions": "CF-1.12",
            "institution": "Test",
            "source": "Test",
            "title": "Test",
            "history": "Test",
        },
    )
    check = HeuristicCheck(cf_version="CF-1.12")
    result = check.check(ds)
    fix_result = check.fix(ds, result=result)

    assert fix_result.applied is True
    assert isinstance(fix_result.unfixable_items, list)
    # depth coord missing positive attr is unfixable → should appear
    assert any("positive" in item for item in fix_result.unfixable_items)


def test_comply_cli_adds_calendar_to_output(tmp_path: Path) -> None:
    """nc-comply CLI should write a file with calendar='standard' on time coord."""
    ds = _make_time_ds()
    input_file = tmp_path / "input.nc"
    output_file = tmp_path / "output.nc"
    ds.to_netcdf(input_file)

    exit_code = run_comply([str(input_file), str(output_file)])

    assert output_file.exists()
    with xr.open_dataset(output_file) as out:
        # After round-trip through NetCDF, xarray stores calendar in encoding
        calendar = out["time"].attrs.get("calendar") or out["time"].encoding.get("calendar")
        assert calendar == "standard"


def test_comply_cli_exit_code_1_when_unfixable(tmp_path: Path) -> None:
    """nc-comply should return exit code 1 when unfixable issues are present."""
    # Dataset with depth coord (missing positive attr = unfixable)
    ds = xr.Dataset(
        data_vars={"v": (("depth",), [1.0, 2.0], {"units": "m", "long_name": "v"})},
        coords={"depth": (("depth",), [10.0, 20.0], {"units": "m", "long_name": "depth"})},
        attrs={
            "Conventions": "CF-1.12",
            "institution": "Test",
            "source": "Test",
            "title": "Test",
            "history": "Test",
        },
    )
    input_file = tmp_path / "input.nc"
    output_file = tmp_path / "output.nc"
    ds.to_netcdf(input_file)

    exit_code = run_comply([str(input_file), str(output_file)])

    assert exit_code == 1
