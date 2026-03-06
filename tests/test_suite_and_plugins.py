from __future__ import annotations

import numpy as np
import xarray as xr

import nc_check
import nc_check.plugins.cfchecker_report as cfchecker_report_plugin
from nc_check.models import AtomicCheckResult, CheckStatus
from nc_check.suite import CallableCheck


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

    def exploding_check(_data):
        raise RuntimeError("boom")

    result = nc_check.run_atomic_check(
        ds["temp"],
        CallableCheck(
            name="demo.explode",
            data_scope="data_vars",
            plugin="demo",
            fn=exploding_check,
        ),
    )

    assert result.status == CheckStatus.failed
    assert result.name == "demo.explode"
    assert "RuntimeError" in result.info


def test_check_suite_runs_list_of_atomic_checks() -> None:
    ds = nc_check.canonicalize_dataset(_valid_dataset())

    def pass_check(_data):
        return AtomicCheckResult.passed_result(name="demo.pass", info="ok")

    def skip_check(_data):
        return AtomicCheckResult.skipped_result(name="demo.skip", info="n/a")

    suite = nc_check.CheckSuite(
        name="demo",
        checks=[
            CallableCheck(
                name="demo.pass",
                data_scope="data_vars",
                plugin="demo",
                fn=pass_check,
            ),
            CallableCheck(
                name="demo.skip",
                data_scope="data_vars",
                plugin="demo",
                fn=skip_check,
            ),
        ],
    )

    report = suite.run(ds)

    assert report.summary.checks_run == 2
    assert report.summary.passed == 1
    assert report.summary.skipped == 1
    assert report.summary.failed == 0
    assert report.summary.overall_status == CheckStatus.passed


def test_check_suite_can_apply_single_check_to_all_data_vars() -> None:
    raw = xr.Dataset(
        data_vars={
            "temp": (("time", "lat", "lon"), np.ones((1, 1, 1))),
            "salt": (("time", "lat", "lon"), np.ones((1, 1, 1))),
        },
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    raw["temp"][0, :, :] = np.nan
    ds = nc_check.canonicalize_dataset(raw)

    def no_nan_check(data):
        has_nan = bool(data.isnull().any())
        variable_name = str(data.name)
        if has_nan:
            return AtomicCheckResult.failed_result(
                name="demo.no_nan",
                info=f"{variable_name} contains NaN values.",
            )
        return AtomicCheckResult.passed_result(
            name="demo.no_nan",
            info=f"{variable_name} has no NaN values.",
        )

    suite = nc_check.CheckSuite(
        name="demo",
        checks=[
            CallableCheck(
                name="demo.no_nan",
                data_scope="data_vars",
                plugin="demo",
                fn=no_nan_check,
            )
        ],
    )
    report = suite.run(ds)

    assert report.summary.checks_run == 2
    assert report.summary.failed == 1
    assert report.summary.passed == 1

    by_name = {item.name: item for item in report.checks}
    assert by_name["demo.no_nan[data_vars:temp]"].status == CheckStatus.failed
    assert by_name["demo.no_nan[data_vars:salt]"].status == CheckStatus.passed
    assert by_name["demo.no_nan[data_vars:temp]"].details["scope_item"] == "temp"
    assert by_name["demo.no_nan[data_vars:salt]"].details["scope_item"] == "salt"


def test_per_data_var_check_skips_when_dataset_has_no_data_vars() -> None:
    raw = xr.Dataset(
        coords={
            "time": ("time", [0]),
            "lat": ("lat", [0.0]),
            "lon": ("lon", [0.0]),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    ds = nc_check.canonicalize_dataset(raw)
    called: list[str] = []

    def per_var_check(data):
        called.append(str(data.name))
        return AtomicCheckResult.passed_result(name="demo.never", info="unused")

    suite = nc_check.CheckSuite(
        name="demo",
        checks=[
            CallableCheck(
                name="demo.never",
                data_scope="data_vars",
                plugin="demo",
                fn=per_var_check,
            )
        ],
    )
    report = suite.run(ds)

    assert called == []
    assert report.summary.checks_run == 1
    assert report.summary.skipped == 1
    assert report.summary.overall_status == CheckStatus.skipped
    assert report.checks[0].name == "demo.never"
    assert report.checks[0].details["reason"] == "no_scope_targets"


def test_check_suite_can_scope_over_dims() -> None:
    ds = nc_check.canonicalize_dataset(_valid_dataset())
    seen: list[str] = []

    def dim_check(data):
        seen.append(str(data.name))
        return AtomicCheckResult.passed_result(name="demo.dim_check", info="ok")

    suite = nc_check.CheckSuite(
        name="demo",
        checks=[
            CallableCheck(
                name="demo.dim_check",
                data_scope="dims",
                plugin="demo",
                fn=dim_check,
            )
        ],
    )
    report = suite.run(ds)

    assert set(seen) == {"time", "lat", "lon"}
    assert report.summary.checks_run == 3
    assert report.summary.passed == 3
    assert report.summary.overall_status == CheckStatus.passed


def test_check_suite_runs_in_data_scope_variable_check_order() -> None:
    raw = xr.Dataset(
        data_vars={
            "temp": (("time", "lat", "lon"), np.ones((1, 1, 1))),
            "salt": (("time", "lat", "lon"), np.ones((1, 1, 1))),
        },
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    ds = nc_check.canonicalize_dataset(raw)

    seen: list[tuple[str, str]] = []

    def first_data_var_check(data):
        seen.append(("demo.a", str(data.name)))
        return AtomicCheckResult.passed_result(name="demo.a", info="ok")

    def second_data_var_check(data):
        seen.append(("demo.b", str(data.name)))
        return AtomicCheckResult.passed_result(name="demo.b", info="ok")

    def dim_check(data):
        seen.append(("demo.dim", str(data.name)))
        return AtomicCheckResult.passed_result(name="demo.dim", info="ok")

    suite = nc_check.CheckSuite(
        name="demo_order",
        checks=[
            CallableCheck(
                name="demo.a",
                data_scope="data_vars",
                plugin="demo",
                fn=first_data_var_check,
            ),
            CallableCheck(
                name="demo.dim",
                data_scope="dims",
                plugin="demo",
                fn=dim_check,
            ),
            CallableCheck(
                name="demo.b",
                data_scope="data_vars",
                plugin="demo",
                fn=second_data_var_check,
            ),
        ],
    )

    suite.run(ds)

    assert seen[:4] == [
        ("demo.a", "temp"),
        ("demo.b", "temp"),
        ("demo.a", "salt"),
        ("demo.b", "salt"),
    ]
    assert seen[4:] == [
        ("demo.dim", "time"),
        ("demo.dim", "lat"),
        ("demo.dim", "lon"),
    ]


def test_check_suite_can_scope_over_dataset() -> None:
    ds = nc_check.canonicalize_dataset(_valid_dataset())

    def dataset_check(data):
        if not isinstance(data, xr.Dataset):
            return AtomicCheckResult.failed_result(
                name="demo.dataset",
                info="Expected dataset scope input.",
            )
        return AtomicCheckResult.passed_result(
            name="demo.dataset",
            info="Dataset scope check passed.",
        )

    suite = nc_check.CheckSuite(
        name="demo_dataset_scope",
        checks=[
            CallableCheck(
                name="demo.dataset",
                data_scope="dataset",
                plugin="demo",
                fn=dataset_check,
            )
        ],
    )
    report = suite.run(ds)

    assert report.summary.checks_run == 1
    assert report.summary.passed == 1
    assert report.checks[0].name == "demo.dataset[dataset:dataset]"


def test_suite_report_exposes_results_grouped_by_scope_and_variable() -> None:
    raw = xr.Dataset(
        data_vars={
            "temp": (("time", "lat", "lon"), np.ones((1, 1, 1))),
            "salt": (("time", "lat", "lon"), np.ones((1, 1, 1))),
        },
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "lat": ("lat", [0.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    raw["temp"][0, :, :] = np.nan
    ds = nc_check.canonicalize_dataset(raw)

    def no_nan_check(data):
        variable_name = str(data.name)
        if bool(data.isnull().any()):
            return AtomicCheckResult.failed_result(
                name="demo.no_nan",
                info=f"{variable_name} contains NaN values.",
            )
        return AtomicCheckResult.passed_result(
            name="demo.no_nan",
            info=f"{variable_name} has no NaN values.",
        )

    suite = nc_check.CheckSuite(
        name="demo_results",
        checks=[
            CallableCheck(
                name="demo.no_nan",
                data_scope="data_vars",
                plugin="demo",
                fn=no_nan_check,
            )
        ],
    )
    payload = suite.run(ds).to_dict()

    assert "results" in payload
    assert set(payload["results"]) == {"data_vars"}
    assert set(payload["results"]["data_vars"]) == {"temp", "salt"}
    assert set(payload["results"]["data_vars"]["temp"]) == {"demo.no_nan"}
    assert payload["results"]["data_vars"]["temp"]["demo.no_nan"]["status"] == "failed"
    assert payload["results"]["data_vars"]["salt"]["demo.no_nan"]["status"] == "passed"


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
    assert any(name.startswith("cf.conventions[") for name in failing_names)
    assert any(name.startswith("cf.coordinate_ranges[") for name in failing_names)


def test_create_registry_registers_time_cover_checks() -> None:
    registry = nc_check.create_registry(load_entrypoints=False)
    registered_names = set(registry.list_checks())
    assert set(nc_check.time_cover_check_names()).issubset(registered_names)


def test_create_registry_registers_cfchecker_report_check() -> None:
    registry = nc_check.create_registry(load_entrypoints=False)
    registered_names = set(registry.list_checks())
    assert set(nc_check.cfchecker_report_check_names()).issubset(registered_names)


def test_create_registry_registers_ocean_cover_checks() -> None:
    registry = nc_check.create_registry(load_entrypoints=False)
    registered_names = set(registry.list_checks())
    assert set(nc_check.ocean_check_names()).issubset(registered_names)


def test_time_cover_plugin_reports_pass_for_valid_dataset() -> None:
    report = nc_check.run_time_cover(_valid_dataset())
    payload = report.to_dict()

    assert payload["suite_name"] == "time_cover"
    assert payload["summary"]["overall_status"] == "passed"
    assert payload["summary"]["failed"] == 0
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["name"].startswith("time.missing_slices[")
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

    by_name = {
        item["name"].split("[", 1)[0]: item["status"] for item in payload["checks"]
    }
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


def test_ocean_cover_plugin_runs_from_registry() -> None:
    report = nc_check.run_ocean_cover(_valid_dataset())
    payload = report.to_dict()

    assert payload["suite_name"] == "ocean_cover"
    assert payload["plugin"] == "ocean_cover"
    assert len(payload["checks"]) == len(nc_check.ocean_check_names())

    check_names = {item["name"].split("[", 1)[0] for item in payload["checks"]}
    assert set(nc_check.ocean_check_names()) == check_names


def test_cfchecker_report_plugin_converts_package_output() -> None:
    def fake_run_in_process(_path):
        result = {
            "raw_result": {
                "global": {"ERROR": ["(2.6.1): invalid units"], "WARN": [], "INFO": []},
                "variables": {
                    "temp": {"ERROR": [], "WARN": ["(3): advisory"], "INFO": []}
                },
            },
            "fatal_count": 0,
            "error_count": 1,
            "warning_count": 1,
            "info_count": 0,
        }
        return result, "ERROR: (2.6.1): invalid units\nWARN: (3): advisory"

    original_run_in_process = cfchecker_report_plugin._run_cfchecker_in_process
    cfchecker_report_plugin._run_cfchecker_in_process = fake_run_in_process
    try:
        payload = nc_check.run_cfchecker_report(_valid_dataset()).to_dict()
    finally:
        cfchecker_report_plugin._run_cfchecker_in_process = original_run_in_process

    assert payload["suite_name"] == "cfchecker_report"
    assert payload["plugin"] == "cfchecker_report"
    assert payload["summary"]["checks_run"] == 2
    assert payload["summary"]["overall_status"] == "failed"
    names = {item["name"] for item in payload["checks"]}
    assert "cfchecker.error.2_6_1.1[dataset:dataset]" in names
    assert "cfchecker.warning.3.1[data_vars:temp]" in names

    by_name = {item["name"]: item for item in payload["checks"]}
    assert by_name["cfchecker.error.2_6_1.1[dataset:dataset]"]["status"] == "failed"
    assert by_name["cfchecker.warning.3.1[data_vars:temp]"]["status"] == "skipped"


def test_cfchecker_report_plugin_splits_joined_warning_error_messages() -> None:
    joined = (
        "WARNING: (2.6.1): No 'Conventions' attribute present | "
        "WARNING: (3.1): units attribute should be present | "
        "ERROR: (3.1): Invalid units: Fractional year | "
        "ERROR: (4.4): Invalid units and/or reference time"
    )

    def fake_run_in_process(_path):
        result = {
            "raw_result": {
                "global": {"ERROR": [], "WARN": [joined], "INFO": []},
                "variables": {},
            },
            "fatal_count": 0,
            "error_count": 2,
            "warning_count": 2,
            "info_count": 0,
        }
        return result, joined

    original_run_in_process = cfchecker_report_plugin._run_cfchecker_in_process
    cfchecker_report_plugin._run_cfchecker_in_process = fake_run_in_process
    try:
        payload = nc_check.run_cfchecker_report(_valid_dataset()).to_dict()
    finally:
        cfchecker_report_plugin._run_cfchecker_in_process = original_run_in_process

    assert payload["summary"]["checks_run"] == 4
    assert payload["summary"]["failed"] == 2
    assert payload["summary"]["skipped"] == 2

    checks_by_info = {item["info"]: item for item in payload["checks"]}
    assert checks_by_info["No 'Conventions' attribute present"]["status"] == "skipped"
    assert checks_by_info["units attribute should be present"]["status"] == "skipped"
    assert checks_by_info["Invalid units: Fractional year"]["status"] == "failed"
    assert checks_by_info["Invalid units and/or reference time"]["status"] == "failed"


def test_custom_plugin_can_register_and_run_checks() -> None:
    class NoNegativeLatPlugin:
        name = "custom_lat"

        def register(self, registry):
            def non_negative_lat(data):
                if float(data.coords["lat"].min()) < 0:
                    return AtomicCheckResult.failed_result(
                        name="custom.non_negative_lat",
                        info="Latitude contains negative values.",
                    )
                return AtomicCheckResult.passed_result(
                    name="custom.non_negative_lat",
                    info="Latitude is non-negative.",
                )

            registry.register_check(
                check=CallableCheck(
                    name="custom.non_negative_lat",
                    data_scope="data_vars",
                    plugin=self.name,
                    fn=non_negative_lat,
                )
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
    assert report.checks[0].name.startswith("custom.non_negative_lat[")


def test_report_rendering_is_python_first_then_html() -> None:
    report = nc_check.run_cf_compliance(_valid_dataset())
    payload = nc_check.report_to_dict(report)
    html = nc_check.render_html_report(payload)

    assert payload["summary"]["overall_status"] == "passed"
    assert "<html>" in html
    assert "cf.conventions" in html


def test_report_includes_source_file_name_as_header_subtitle() -> None:
    ds = _valid_dataset()
    ds.attrs["source"] = "/tmp/sample/path/input-file.nc"
    report = nc_check.run_cf_compliance(ds)

    payload = report.to_dict()
    html = report.to_html().data

    assert payload["source_file"] == "input-file.nc"
    assert "<h1>cf_compliance</h1>" in html
    assert "File: input-file.nc" in html
    assert html.index("<h1>cf_compliance</h1>") < html.index("File: input-file.nc")


def test_dataset_accessor_is_registered() -> None:
    ds = _valid_dataset()
    assert hasattr(ds, "check")
    report = ds.check.compliance()
    assert report.suite_name == "cf_compliance"


def test_dataset_accessor_all_combines_selected_reports() -> None:
    ds = _valid_dataset()
    report = ds.check.all(compliance=True, ocean_cover=True, time_cover=True)
    payload = report.to_dict()

    assert payload["suite_name"] == "all_checks"
    assert payload["summary"]["checks_run"] > 0
    assert "data_vars" in payload["results"] or "dataset" in payload["results"]


def test_dataset_accessor_cfchecker_report_matches_api() -> None:
    ds = _valid_dataset()
    direct = nc_check.run_cfchecker_report(ds).to_dict()
    via_accessor = ds.check.cfchecker_report().to_dict()

    assert direct["suite_name"] == via_accessor["suite_name"]
    assert direct["summary"]["checks_run"] == via_accessor["summary"]["checks_run"]


def test_dataset_accessor_writes_html_report_when_fname_provided(tmp_path) -> None:
    ds = _valid_dataset()
    output = tmp_path / "accessor-report.html"

    report = ds.check.compliance(html_report_fname=output)

    assert report.suite_name == "cf_compliance"
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "<html>" in html
