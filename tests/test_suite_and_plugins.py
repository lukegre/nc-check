from __future__ import annotations

import numpy as np
import xarray as xr

import nc_check
from nc_check.models import AtomicCheckResult, CheckStatus
from nc_check.suite import CheckDefinition


def _valid_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": ("lat", [-20.0, 20.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0, 120.0, 240.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )


def test_atomic_check_core_function_handles_exceptions() -> None:
    ds = nc_check.canonicalize_dataset(_valid_dataset())

    def exploding_check(_ds):
        raise RuntimeError("boom")

    result = nc_check.run_atomic_check(
        ds,
        CheckDefinition(name="demo.explode", check=exploding_check),
    )

    assert result.status == CheckStatus.failed
    assert result.name == "demo.explode"
    assert "RuntimeError" in result.info


def test_check_suite_runs_list_of_atomic_checks() -> None:
    ds = nc_check.canonicalize_dataset(_valid_dataset())

    def pass_check(_ds):
        return AtomicCheckResult.passed_result(name="demo.pass", info="ok")

    def skip_check(_ds):
        return AtomicCheckResult.skipped_result(name="demo.skip", info="n/a")

    suite = nc_check.CheckSuite(
        name="demo",
        checks=[
            CheckDefinition(name="demo.pass", check=pass_check),
            CheckDefinition(name="demo.skip", check=skip_check),
        ],
    )

    report = suite.run(ds)

    assert report.summary.checks_run == 2
    assert report.summary.passed == 1
    assert report.summary.skipped == 1
    assert report.summary.failed == 0
    assert report.summary.overall_status == CheckStatus.passed


def test_cf_compliance_plugin_reports_pass_for_valid_dataset() -> None:
    report = nc_check.run_cf_compliance(_valid_dataset())
    payload = report.to_dict()

    assert payload["suite_name"] == "cf_compliance"
    assert payload["summary"]["overall_status"] == "passed"
    assert payload["summary"]["failed"] == 0
    assert len(payload["checks"]) == len(nc_check.cf_check_names())


def test_cf_compliance_plugin_detects_bad_metadata() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
        coords={"time": [0], "lat": [95.0], "lon": [900.0]},
        attrs={"Conventions": "ACDD-1.3"},
    )

    report = nc_check.run_cf_compliance(raw)
    payload = report.to_dict()

    assert payload["summary"]["overall_status"] == "failed"
    failing_names = {
        item["name"] for item in payload["checks"] if item["status"] == "failed"
    }
    assert "cf.conventions" in failing_names
    assert "cf.coordinate_ranges" in failing_names


def test_create_registry_registers_time_cover_checks() -> None:
    registry = nc_check.create_registry(load_entrypoints=False)
    registered_names = set(registry.list_checks())
    assert set(nc_check.time_cover_check_names()).issubset(registered_names)


def test_time_cover_plugin_reports_pass_for_valid_dataset() -> None:
    report = nc_check.run_time_cover(_valid_dataset())
    payload = report.to_dict()

    assert payload["suite_name"] == "time_cover"
    assert payload["summary"]["overall_status"] == "passed"
    assert payload["summary"]["failed"] == 0
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["name"] == "time.missing_slices"
    assert payload["checks"][0]["status"] == "passed"


def test_time_cover_plugin_runs_optional_checks_and_detects_failures() -> None:
    raw = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={
            "time": np.array(
                ["2024-01-01", "2024-01-03", "2024-01-02"], dtype="datetime64[ns]"
            ),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    raw["temp"][1, :, :] = np.nan

    report = nc_check.run_time_cover(
        raw,
        check_time_monotonic=True,
        check_time_regular_spacing=True,
    )
    payload = report.to_dict()

    by_name = {item["name"]: item["status"] for item in payload["checks"]}
    assert by_name["time.missing_slices"] == "failed"
    assert by_name["time.monotonic_increasing"] == "failed"
    assert by_name["time.regular_spacing"] == "failed"
    assert payload["summary"]["overall_status"] == "failed"


def test_time_cover_plugin_var_name_option_limits_scope() -> None:
    raw = xr.Dataset(
        data_vars={
            "temp": (("time", "lat", "lon"), np.ones((2, 1, 1))),
            "salt": (("time", "lat", "lon"), np.ones((2, 1, 1))),
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    raw["temp"][1, :, :] = np.nan

    all_report = nc_check.run_time_cover(raw).to_dict()
    salt_report = nc_check.run_time_cover(raw, var_name="salt").to_dict()

    assert all_report["summary"]["overall_status"] == "failed"
    assert salt_report["summary"]["overall_status"] == "passed"


def test_custom_plugin_can_register_and_run_checks() -> None:
    class NoNegativeLatPlugin:
        name = "custom_lat"

        def register(self, registry):
            def non_negative_lat(ds):
                if float(ds.coords["lat"].min()) < 0:
                    return AtomicCheckResult.failed_result(
                        name="custom.non_negative_lat",
                        info="Latitude contains negative values.",
                    )
                return AtomicCheckResult.passed_result(
                    name="custom.non_negative_lat",
                    info="Latitude is non-negative.",
                )

            registry.register_check(
                name="custom.non_negative_lat",
                check=non_negative_lat,
                plugin=self.name,
            )

    registry = nc_check.create_registry(load_entrypoints=False)
    registry.register_plugin(NoNegativeLatPlugin())

    report = nc_check.run_suite(
        _valid_dataset(),
        suite_name="custom_suite",
        check_names=["custom.non_negative_lat"],
        registry=registry,
    )

    assert report.summary.failed == 1
    assert report.checks[0].name == "custom.non_negative_lat"


def test_report_rendering_is_python_first_then_html() -> None:
    report = nc_check.run_cf_compliance(_valid_dataset())
    payload = nc_check.report_to_dict(report)
    html = nc_check.render_html_report(payload)

    assert payload["summary"]["overall_status"] == "passed"
    assert "<html>" in html
    assert "cf.conventions" in html
