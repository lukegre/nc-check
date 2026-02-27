# nc-check Docs

`nc-check` helps you validate and prepare `xarray.Dataset` objects for CF-1.12-ready NetCDF output.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[CLI Guide](cli.md){ .md-button }

## Start Here

<div class="grid cards" markdown>

- **Getting Started**

  ---
  Install package dependencies and run your first check in Python or the CLI.

  [Open guide](getting-started.md)

- **Checks and Reports**

  ---
  Learn what each check does and how to inspect HTML/table/python report outputs.

  [Open guide](checks-and-reports.md)

- **Python API**

  ---
  Use accessor methods such as `ds.check.compliance()` and `ds.check.all()`.

  [Open guide](python-api.md)

- **CLI Workflows**

  ---
  Run checks in batch jobs and save reports directly from command line tools.

  [Open guide](cli.md)

- **Build a Check Suite**

  ---
  Register plugins and compose new suites for custom geospatial quality checks.

  [Open guide](add-check-suite.md)

- **Troubleshooting + Dev**

  ---
  Resolve common issues and contribute to plugin/reporting internals.

  [Troubleshooting](troubleshooting.md) | [Development](development.md)

</div>

## Quick Commands

```bash
nc-check input.nc
nc-check all input.nc --save-report
nc-comply input.nc output.nc
```

## Python Quickstart

```python
import xarray as xr
import nc_check

ds = xr.Dataset(
    data_vars={"temp": (("time", "lat", "lon"), [[[280.0]]])},
    coords={"time": [0], "lat": [10.0], "lon": [20.0]},
)

report = ds.check.all(report_format="python")
fixed = ds.check.make_cf_compliant()
```
