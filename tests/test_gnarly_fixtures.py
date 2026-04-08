"""Tests using deliberately broken NetCDF fixtures.

Each fixture is crafted to violate CF conventions in specific, realistic ways.
Run `uv run python tests/data/make_gnarly_fixtures.py` to regenerate the files.

Fixtures and their primary violations:
  lat_upside_down.nc      — latitude descending (90→-90), no coord attrs, no Conventions
  lon_offset_wrong_labels — lon labelled -180→180 but units='degrees_west', invalid cell method
  bad_var_names.nc        — cryptic dim/var names, no units, no long/standard names anywhere
  wrong_dim_order.nc      — dimensions (lon, depth, lat, time) instead of (time, depth, lat, lon)
  missing_units.nc        — numeric vars and all coords missing units
  missing_lon_slices.nc   — three persistent NaN longitude bands (0-30°, 150-180°, 330-355°)
  bad_time_units.nc       — 'parsecs since the Big Bang', bare 'days', and unknown calendar
  kitchen_sink.nc         — all of the above combined in one pathological file
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from nc_check.checks.heuristic import HeuristicCheck, _heuristic_report
from nc_check.checks.ocean import check_ocean_cover
from nc_check.checks.time_cover import run_time_cover_report
from nc_check.core import CheckStatus

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_heuristic_extended.py)
# ---------------------------------------------------------------------------


def _run(ds: xr.Dataset) -> dict:
    return _heuristic_report(ds, cf_version="CF-1.12")


def _all_findings(report: dict) -> list[dict]:
    findings = list(report.get("global", []))
    for items in report.get("coordinates", {}).values():
        findings.extend(items)
    for items in report.get("variables", {}).values():
        findings.extend(items)
    return findings


def _items(report: dict) -> set[str]:
    return {f.get("item") for f in _all_findings(report) if isinstance(f, dict)}


def _has_item(report: dict, item: str) -> bool:
    return item in _items(report)


def _has_severity(report: dict, item: str, severity: str) -> bool:
    return any(
        isinstance(f, dict) and f.get("item") == item and f.get("severity") == severity
        for f in _all_findings(report)
    )


# ---------------------------------------------------------------------------
# lat_upside_down.nc
# ---------------------------------------------------------------------------


class TestLatUpsideDown:
    """Latitude runs from 90 down to -90 (descending).

    No axis, units, or standard_name on lat/lon coordinates.
    No Conventions global attribute.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "lat_upside_down.nc")

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_missing_conventions_flagged(self, report):
        assert _has_item(report, "Conventions")

    def test_lat_missing_axis_attr(self, report):
        # lat coord exists but has no axis= attribute
        assert _has_item(report, "coord_attr:axis")

    def test_lat_missing_units_attr(self, report):
        assert _has_item(report, "coord_attr:units")

    def test_lat_missing_standard_name(self, report):
        assert _has_item(report, "coord_attr:standard_name")

    def test_lat_is_not_flagged_as_non_monotonic(self, ds):
        # Descending lat is still strictly monotonic — should not be flagged
        assert not _has_item(_run(ds), "coord_not_monotonic")

    def test_heuristic_check_status_is_warn_or_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status in (CheckStatus.warn, CheckStatus.error)


# ---------------------------------------------------------------------------
# lon_offset_wrong_labels.nc
# ---------------------------------------------------------------------------


class TestLonOffsetWrongLabels:
    """Longitude labelled -180→180 but units='degrees_west' (invalid CF unit).

    Also has an invalid cell method ('accumulation') and an incorrect standard
    name for the precipitation variable.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "lon_offset_wrong_labels.nc")

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_invalid_lon_units_degrees_west(self, report):
        # degrees_west is not a valid CF longitude unit (should be degrees_east)
        lon_findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict)
            and f.get("item") == "coord_attr:units"
            and f.get("current") == "degrees_west"
        ]
        assert lon_findings, "Expected a coord_attr:units finding for 'degrees_west'"

    def test_invalid_cell_method_accumulation(self, report):
        assert _has_item(report, "cell_methods_unknown_method")

    def test_cell_method_finding_severity_is_warn(self, report):
        assert _has_severity(report, "cell_methods_unknown_method", "WARN")

    def test_heuristic_check_status_is_warn_or_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status in (CheckStatus.warn, CheckStatus.error)


# ---------------------------------------------------------------------------
# bad_var_names.nc
# ---------------------------------------------------------------------------


class TestBadVarNames:
    """Dimensions named d0/d1/d2, variables named var1/field_2/X.

    Nothing has units, standard_name, or long_name. No Conventions.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "bad_var_names.nc")

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_missing_conventions_flagged(self, report):
        assert _has_item(report, "Conventions")

    def test_all_vars_missing_standard_or_long_name(self, report):
        assert _has_item(report, "missing_standard_or_long_name")

    def test_all_numeric_vars_missing_units(self, report):
        assert _has_item(report, "missing_units_attr")

    def test_axis_type_cannot_be_inferred(self, report):
        # d0/d1/d2 cannot be matched to any known axis
        notes = report.get("notes", [])
        uninferable = [n for n in notes if "Could not infer CF axis type" in n]
        assert len(uninferable) >= 3

    def test_finding_count_is_high(self, report):
        # Every variable violates at least two rules
        assert len(_all_findings(report)) >= 6

    def test_heuristic_check_status_is_warn_or_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status in (CheckStatus.warn, CheckStatus.error)


# ---------------------------------------------------------------------------
# wrong_dim_order.nc
# ---------------------------------------------------------------------------


class TestWrongDimOrder:
    """Variable 'temp' has dimensions (lon, depth, lat, time).

    CF/COARDS convention requires T, Z, Y, X ordering.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "wrong_dim_order.nc")

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_wrong_dimension_order_flagged(self, report):
        assert _has_item(report, "wrong_dimension_order")

    def test_wrong_dimension_order_severity_is_warn(self, report):
        assert _has_severity(report, "wrong_dimension_order", "WARN")

    def test_wrong_dimension_order_finding_mentions_temp(self, report):
        findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict) and f.get("item") == "wrong_dimension_order"
        ]
        assert any("temp" in str(f.get("message", "")) for f in findings)

    def test_actual_dim_order_is_lon_depth_lat_time(self, ds):
        assert tuple(ds["temp"].dims) == ("lon", "depth", "lat", "time")

    def test_heuristic_check_status_is_warn_or_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status in (CheckStatus.warn, CheckStatus.error)


# ---------------------------------------------------------------------------
# missing_units.nc
# ---------------------------------------------------------------------------


class TestMissingUnits:
    """Temperature and salinity missing units. All coordinates also missing units.

    Time coordinate is numeric with no units string at all.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "missing_units.nc")

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_temperature_missing_units(self, report):
        temp_findings = report.get("variables", {}).get("temperature", [])
        assert any(
            isinstance(f, dict) and f.get("item") == "missing_units_attr"
            for f in temp_findings
        )

    def test_salinity_missing_units(self, report):
        sal_findings = report.get("variables", {}).get("salinity", [])
        assert any(
            isinstance(f, dict) and f.get("item") == "missing_units_attr"
            for f in sal_findings
        )

    def test_time_missing_units_is_error(self, report):
        # Numeric time with no units is an ERROR level finding
        assert _has_severity(report, "coord_attr:units", "ERROR")

    def test_lat_lon_missing_axis_attr(self, report):
        assert _has_item(report, "coord_attr:axis")

    def test_overall_check_status_is_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status == CheckStatus.error

    def test_time_cover_still_runs(self, ds):
        # Checker should degrade gracefully even with no time units
        report = run_time_cover_report(ds, var_name="temperature", time_name="time")
        assert "time_missing" in report


# ---------------------------------------------------------------------------
# missing_lon_slices.nc
# ---------------------------------------------------------------------------


class TestMissingLonSlices:
    """Three persistent NaN longitude bands: 0-30°, 150-180°, and 330-355°.

    Scattered individual NaNs should not be flagged as persistent bands.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "missing_lon_slices.nc")

    @pytest.fixture(scope="class")
    def ocean_report(self, ds):
        return check_ocean_cover(
            ds,
            var_name="chl",
            check_land_ocean_offset=False,
            report_format="python",
        )

    def test_edge_of_map_status_is_fail(self, ocean_report):
        assert ocean_report["edge_of_map"]["status"] == "fail"

    def test_missing_longitude_count_is_large(self, ocean_report):
        # Three bands: 7 + 7 + 6 = 20 missing columns in 0-355° at 5° spacing
        assert ocean_report["edge_of_map"]["missing_longitude_count"] == 20

    def test_first_band_start_at_zero_degrees(self, ocean_report):
        missing = ocean_report["edge_of_map"]["missing_longitudes"]
        assert 0.0 in missing

    def test_last_band_near_dateline(self, ocean_report):
        missing = ocean_report["edge_of_map"]["missing_longitudes"]
        assert any(lon >= 330.0 for lon in missing)

    def test_overall_ocean_summary_is_fail(self, ocean_report):
        assert ocean_report["summary"]["overall_status"] == "fail"

    def test_time_cover_passes_no_missing_slices(self, ds):
        # The NaNs are spatial, not temporal — no time slice should be all-NaN
        report = run_time_cover_report(ds, var_name="chl", time_name="time")
        assert report["time_missing"]["status"] == "pass"


# ---------------------------------------------------------------------------
# bad_time_units.nc
# ---------------------------------------------------------------------------


class TestBadTimeUnits:
    """Four variables each with a different kind of broken time coordinate.

    Must be opened with decode_times=False because xarray cannot decode
    'parsecs since the Big Bang'.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "bad_time_units.nc", decode_times=False)

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_xarray_needs_decode_times_false(self):
        with pytest.raises(ValueError, match="unable to decode time units"):
            xr.open_dataset(DATA / "bad_time_units.nc")

    def test_parsecs_unit_flagged_as_bad_format(self, report):
        # 'parsecs since the Big Bang' doesn't match <unit> since <epoch>
        units_format_findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict)
            and f.get("item") == "coord_attr:units_format"
            and "parsecs" in str(f.get("current", ""))
        ]
        assert units_format_findings

    def test_bare_days_unit_present_in_fixture(self, ds):
        # 'days' without 'since <epoch>' is stored in rel_time.
        # The heuristic only checks time units for coords it can identify as
        # a time axis; rel_time can't be inferred (not named 'time'/'t' and has
        # no standard_name), so the unit lives in the file but isn't caught by
        # the units-format heuristic — that's itself a coverage gap worth noting.
        assert ds["rel_time"].attrs.get("units") == "days"

    def test_martian_calendar_flagged(self, report):
        # calendar='martian' should be flagged (not a valid CF calendar name)
        calendar_findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict) and "calendar" in f.get("item", "")
        ]
        assert calendar_findings, "Expected a calendar-related finding"

    def test_at_least_one_time_unit_violation(self, report):
        # The heuristic flags units for coords it can identify as time axes.
        # 'time' (parsecs) is recognised; rel_time/neg_time/cal_time are not
        # (non-standard names, no standard_name attr), so at least 1 finding.
        units_format_findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict) and f.get("item") == "coord_attr:units_format"
        ]
        assert len(units_format_findings) >= 1

    def test_heuristic_check_status_is_warn_or_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status in (CheckStatus.warn, CheckStatus.error)


# ---------------------------------------------------------------------------
# kitchen_sink.nc
# ---------------------------------------------------------------------------


class TestKitchenSink:
    """Combines every violation: upside-down lat, degrees_west lon, bare variable 'Q',
    wrong dimension order, huge missing longitude band, and 'fortnights' time unit.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        return xr.open_dataset(DATA / "kitchen_sink.nc", decode_times=False)

    @pytest.fixture(scope="class")
    def report(self, ds):
        return _run(ds)

    def test_xarray_needs_decode_times_false(self):
        with pytest.raises(ValueError, match="unable to decode time units"):
            xr.open_dataset(DATA / "kitchen_sink.nc")

    def test_missing_conventions(self, report):
        assert _has_item(report, "Conventions")

    def test_wrong_lon_units_degrees_west(self, report):
        lon_findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict)
            and f.get("item") == "coord_attr:units"
            and f.get("current") == "degrees_west"
        ]
        assert lon_findings

    def test_fortnights_time_unit_flagged(self, report):
        assert _has_item(report, "coord_attr:units_format")
        fortnights_findings = [
            f
            for f in _all_findings(report)
            if isinstance(f, dict)
            and f.get("item") == "coord_attr:units_format"
            and "fortnight" in str(f.get("current", ""))
        ]
        assert fortnights_findings

    def test_variable_Q_missing_units(self, report):
        q_findings = report.get("variables", {}).get("Q", [])
        assert any(
            isinstance(f, dict) and f.get("item") == "missing_units_attr"
            for f in q_findings
        )

    def test_variable_Q_missing_long_and_standard_name(self, report):
        q_findings = report.get("variables", {}).get("Q", [])
        assert any(
            isinstance(f, dict) and f.get("item") == "missing_standard_or_long_name"
            for f in q_findings
        )

    def test_wrong_dimension_order_flagged(self, report):
        assert _has_item(report, "wrong_dimension_order")

    def test_lat_is_descending(self, ds):
        lat = ds["lat"].values
        assert lat[0] > lat[-1], "Latitude should be descending in this fixture"

    def test_ocean_cover_detects_missing_longitude_band(self, ds):
        report = check_ocean_cover(
            ds,
            var_name="Q",
            check_land_ocean_offset=False,
            report_format="python",
        )
        assert report["edge_of_map"]["status"] == "fail"
        assert report["edge_of_map"]["missing_longitude_count"] > 10

    def test_total_finding_count_is_very_high(self, report):
        # A truly pathological file should rack up many findings
        assert len(_all_findings(report)) >= 10

    def test_heuristic_check_status_is_warn_or_error(self, ds):
        result = HeuristicCheck(cf_version="CF-1.12").check(ds)
        assert result.status in (CheckStatus.warn, CheckStatus.error)
