# nc-check Docs

`nc-check` helps you validate and prepare `xarray.Dataset` objects for CF-1.12-ready NetCDF output.

## Start Here

- [Getting Started](getting-started.md)
- [CLI Guide](cli.md)
- [Python API Guide](python-api.md)
- [Checks and Reports](checks-and-reports.md)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)

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
