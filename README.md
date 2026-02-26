# nc-check

Plugin-first dataset checks for geospatial `xarray` workflows.

## Design

- Atomic checks are plain Python callables.
- Each check returns one standardized result: `name`, `status` (`skipped`, `passed`, `failed`), and `info`.
- A `CheckSuite` runs a list of atomic checks and returns a Python report.
- Reports are Python-first (`dict`-friendly), with optional HTML rendering.
- Checks operate on a `CanonicalDataset` (subclass of `xarray.Dataset`) where coordinates are always `time`, `lat`, `lon`.

## Install

```bash
uv add nc-check
# or
pip install nc-check
```

## Quickstart

```python
import xarray as xr
import nc_check

raw = xr.Dataset(
    data_vars={"temp": (("t", "latitude", "longitude"), [[[280.0]]])},
    coords={"t": [0], "latitude": [10.0], "longitude": [20.0]},
    attrs={"Conventions": "CF-1.12"},
)

# Canonicalize aliases to time/lat/lon and run the built-in CF plugin suite.
report = nc_check.run_cf_compliance(raw)
print(report.to_dict()["summary"])

# HTML is derived from the Python report.
html = nc_check.render_html_report(report)
nc_check.save_html_report(report, "cf-report.html")
```

## Plugin model

A plugin registers named checks with a `CheckRegistry`.

```python
from nc_check.models import AtomicCheckResult

class MyPlugin:
    name = "my_plugin"

    def register(self, registry):
        def check_no_nan(ds):
            has_nan = bool(ds.to_array().isnull().any())
            if has_nan:
                return AtomicCheckResult.failed_result(
                    name="my.no_nan",
                    info="Dataset contains NaN values.",
                )
            return AtomicCheckResult.passed_result(
                name="my.no_nan",
                info="No NaN values found.",
            )

        registry.register_check(
            name="my.no_nan",
            check=check_no_nan,
            plugin=self.name,
        )
```

Then run it:

```python
registry = nc_check.create_registry(load_entrypoints=False)
registry.register_plugin(MyPlugin())
report = nc_check.run_suite(
    raw,
    suite_name="my_suite",
    check_names=["my.no_nan"],
    registry=registry,
)
```

## Built-in example plugin

`CFCompliancePlugin` is included as a reference implementation with atomic checks:

- `cf.conventions`
- `cf.coordinates_present`
- `cf.latitude_units`
- `cf.longitude_units`
- `cf.time_units`
- `cf.coordinate_ranges`

## Notes

- This version intentionally focuses on library APIs and plugin architecture.
- CLI output interfaces are not part of the current design.
