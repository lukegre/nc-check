# Development

## Setup

```bash
uv sync --group dev
```

If you want optional CF checker support in development:

```bash
uv sync --group dev --extra cf
```

## Run Tests

```bash
uv run pytest
```

## Build Docs Site

`mkdocs.yml` is included at the repo root for site builds.

Recommended (no global install):

```bash
uv run --with mkdocs-material mkdocs serve
```

Build static site:

```bash
uv run --with mkdocs-material mkdocs build --strict
```

Alternative (global install):

```bash
pip install mkdocs-material
mkdocs serve
```

## Project Layout

- `src/nc_check/contracts/`: typed contracts for suite reports and summaries
- `src/nc_check/engine/registry.py`: check registration model
- `src/nc_check/engine/defaults.py`: default check registrations and summary/detail resolvers
- `src/nc_check/engine/runner.py`: suite runner used by `ds.check.all()`
- `src/nc_check/engine/suite.py`: generic `Suite` / `SuiteCheck` classes for atomic check lists
- `src/nc_check/core/compliance.py`: compliance checks and compliance coercion
- `src/nc_check/core/check.py`: pluggable check and fix abstractions
- `src/nc_check/core/coverage.py`: shared coverage-check helpers
- `src/nc_check/checks/heuristic.py`: heuristic metadata checks
- `src/nc_check/checks/ocean.py`: ocean coverage checks
- `src/nc_check/checks/time_cover.py`: time coverage checks
- `src/nc_check/accessor.py`: `xarray.Dataset.check` accessor API
- `src/nc_check/cli.py`: CLI entrypoints (`nc-check`, `nc-comply`)
- `src/nc_check/formatting.py`: table/html/python report formatting
- `tests/`: test suite

## Local Smoke Check

```bash
uv run python -c "import nc_check, xarray as xr; print('ok')"
```
