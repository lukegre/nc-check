# nc-check

A Python package (≥3.11) that prepares `xarray.Dataset` objects for **CF-1.12 (Climate and Forecast Metadata Conventions)** compliant NetCDF output. It validates, reports on, and auto-fixes datasets.

## What it does

- **CF compliance checking** — validates datasets against CF conventions using built-in heuristics (primary path); `cfchecker` is optional and off-table for this project
- **Ocean/time coverage analysis** — detects missing spatial/temporal data spans
- **Auto-fixing** — non-destructive metadata normalization (`make_cf_compliant()`); reports unfixable issues separately
- **CF standard name suggestions** — domain-aware (ocean, atmosphere, land, cryosphere, biogeochemistry)
- **Multi-format reports** — Python dicts, rich terminal tables, or HTML

## Tech stack

- `xarray`, `numpy`, `netcdf4` — data handling
- `pydantic v2` — structured result models (`CheckStatus`, `CheckResult`, `FixResult`, `CheckInfo`)
- `rich` — terminal output
- `cfchecker` (optional) — official CF validator; heuristic engine is the **primary path**
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
│   └── heuristic.py   # PRIMARY check engine — all heuristic checks live here
└── engine/
    ├── suite.py       # Suite/SuiteCheck classes for grouping checks
    ├── runner.py      # Orchestrates multi-suite execution
    ├── registry.py    # Check discovery/registry
    └── defaults.py    # Default configurations
tests/                 # pytest suite; 147 tests, 13 test files + fixtures in tests/data/
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
nc-comply input.nc output.nc             # Auto-fix and save; exits 1 if unfixable issues remain
```

## Notable patterns

- **xarray accessor** via `@register_dataset_accessor("check")` — no monkey-patching
- **Heuristic engine is primary** — `cfchecker` is optional; never assume it's available
- **Abstract `Check` base class** — extensible for custom checks; `HeuristicCheck.fix()` is implemented
- **Non-destructive fixes** — metadata normalization only, data values unchanged
- **`unfixable_items`** — `FixResult.unfixable_items: list[str]` lists issues found but not auto-fixable
- **Multi-format reports** — unified output with `report_format` parameter
- **Lazy evaluation** — works with Dask arrays via xarray
- **`nc-comply` exit codes** — exits 0 when all issues fixed; exits 1 when unfixable items remain (CI-friendly)

## Heuristic checks implemented

All checks use the `_finding()` helper in `heuristic.py` and are integrated into `_heuristic_report()`:

| Check | CF ref | Fixable? |
|-------|--------|----------|
| `Conventions` attribute presence/value | — | Yes (set_global_attr) |
| Coordinate attrs (standard_name, units, axis) | CF §4 | Yes (set_coord_attr) |
| Coordinate uniqueness and monotonicity | — | Yes |
| Lat/lon range validity | CF §4.1–4.2 | Yes |
| Time units format | CF §4.4 | Yes |
| Time `calendar` attribute | CF §4.4.1 | Yes → `calendar="standard"` |
| Bounds variable structure | CF §7.1 | No |
| Cell methods validity | CF §7.4 | No |
| Vertical `positive` attribute | CF §4.3 | No |
| Dimension order (T,Z,Y,X) | COARDS/Ferret | No |
| CMIP6/general recommended global attrs | CMIP6 | No |
| Variable name CF compliance | — | Yes (rename) |
| Variable units/standard_name presence | — | Yes |
| Reference attribute validity (bounds, coordinates, etc.) | — | Yes |

## Key data model

```python
# FixResult — returned by Check.fix()
class FixResult(BaseModel):
    check_id: str
    applied: bool
    info: CheckInfo
    dataset: xr.Dataset          # fixed dataset
    unfixable_items: list[str]   # e.g. ["depth:missing_positive_attr", "wrong_dimension_order"]
```

## Development

```bash
uv sync --group dev          # Install dependencies
uv run pytest                # Run tests (147 tests, ~2s)
uv run pytest -v --tb=short  # Verbose output
```
