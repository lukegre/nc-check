# nc-check

A Python package (≥3.11) that prepares `xarray.Dataset` objects for **CF-1.12 (Climate and Forecast Metadata Conventions)** compliant NetCDF output. It validates, reports on, and auto-fixes datasets.

## What it does

- **CF compliance checking** — validates datasets against CF conventions using `cfchecker` (if installed) or built-in heuristics as a fallback
- **Ocean/time coverage analysis** — detects missing spatial/temporal data spans
- **Auto-fixing** — non-destructive metadata normalization (`make_cf_compliant()`)
- **CF standard name suggestions** — domain-aware (ocean, atmosphere, land, cryosphere, biogeochemistry)
- **Multi-format reports** — Python dicts, rich terminal tables, or HTML

## Tech stack

- `xarray`, `numpy`, `netcdf4` — data handling
- `pydantic v2` — structured result models (`CheckStatus`, `CheckResult`, `FixResult`, `CheckInfo`)
- `rich` — terminal output
- `cfchecker` (optional) — official CF validator
- `dask` (optional) — lazy evaluation
- `uv` — build/dependency management

## Project structure

```
src/nc_check/
├── accessor.py        # ds.check xarray accessor (registered via @register_dataset_accessor)
├── cli.py             # CLI entry points: nc-check, nc-comply
├── formatting.py      # Report rendering: python / tables / html / auto
├── standard_names.py  # CF standard name suggestions by domain
├── core/
│   ├── check.py       # Abstract Check base class; CheckResult, CheckStatus, CheckInfo, FixResult models
│   ├── compliance.py  # CF compliance engine (cfchecker or heuristic fallback)
│   └── coverage.py    # Coverage gap detection utilities
├── checks/
│   ├── ocean.py       # Ocean grid coverage checks
│   ├── time_cover.py  # Time-slice coverage checks
│   └── heuristic.py   # Built-in heuristic checks (axis guessing, standard name inference)
└── engine/
    ├── suite.py       # Suite/SuiteCheck classes for grouping checks
    ├── runner.py      # Orchestrates multi-suite execution
    ├── registry.py    # Check discovery/registry
    └── defaults.py    # Default configurations
tests/                 # pytest suite; 10+ test files + fixtures in tests/data/
docs/                  # MkDocs documentation
```

## Usage

### Python API

```python
import xarray as xr
import nc_check  # registers ds.check accessor

ds = xr.open_dataset("file.nc")

ds.check.compliance()                    # CF compliance report
ds.check.make_cf_compliant()             # Non-destructive auto-fix
ds.check.ocean_cover()                   # Ocean coverage check
ds.check.time_cover()                    # Time coverage check
ds.check.all(report_format="html")       # Full combined report
```

### CLI

```bash
nc-check input.nc                        # Compliance check (shorthand)
nc-check compliance input.nc
nc-check ocean-cover input.nc
nc-check time-cover input.nc
nc-check all input.nc --save-report      # Saves HTML report
nc-comply input.nc output.nc             # Auto-fix and save
```

## Notable patterns

- **xarray accessor** via `@register_dataset_accessor("check")` — no monkey-patching
- **Dual compliance engine** — `cfchecker` (full) or heuristic (fallback)
- **Abstract `Check` base class** — extensible for custom checks
- **Non-destructive fixes** — metadata normalization only, data unchanged
- **Multi-format reports** — unified output with `report_format` parameter
- **Lazy evaluation** — works with Dask arrays via xarray

## Development

```bash
uv sync --group dev          # Install dependencies
uv run pytest                # Run tests
```
