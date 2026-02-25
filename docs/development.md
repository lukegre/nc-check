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
uv run --group dev python -m pytest
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

## Add A New Check Suite With A Report

Use `time_cover` as the reference implementation for a suite-style check report.

### 1) Build the suite report in `src/nc_check/checks/`

Create a new module (for example `src/nc_check/checks/my_suite.py`) that:

- defines atomic checks that each return a `dict` with at least a `status`
- wraps those checks with `SuiteCheck`
- runs `Suite(name="my_suite", checks=[...]).run()`
- returns a report dict with:
  - `group`
  - `suite`
  - `checks` (atomic items)
  - `summary` (`checks_run`, `failing_checks`, `warnings_or_skips`, `overall_status`, `overall_ok`)
  - `ok`

If your check can run across many variables, follow the `mode="all_variables"` pattern used by
`ocean_cover` and `time_cover`: return per-variable `reports` plus a flattened top-level `checks`
list for combined summary behavior.

Add a public entrypoint:

- `check_my_suite(..., report_format="auto", report_html_file=None)`

and support the same report contract:

- `python` -> return report `dict`
- `tables` -> print formatted output and return `None`
- `html` -> return HTML string, optionally save via `report_html_file`

### 2) Register the suite in the engine

Update `src/nc_check/engine/defaults.py`:

- add your key to `_DEFAULT_ORDER` so `ds.check.all()` runs it in sequence
- add `_run_my_suite_report(ds, options) -> dict`
- add `_status_from_my_suite_report(report) -> SummaryStatus`
- add `_my_suite_detail(report) -> str`
- register it in `register_default_checks()` with `RegisteredCheck(...)`

Why this matters: `run_suite_checks()` uses the registration to build `groups`, `check_summary`,
and the top-level full-report summary.

### 3) Wire it into the dataset accessor

Update `src/nc_check/accessor.py`:

- import your `check_my_suite`
- add `ds.check.my_suite(...)` method (optional but recommended)
- add toggle + options forwarding in `ds.check.all(...)`
  - `checks_enabled["my_suite"] = ...`
  - `options_by_check["my_suite"] = {...}`

Important: keep `report_format` behavior consistent with existing methods.

### 4) (Optional) Expose it in CLI

If you want a dedicated command:

- add a new subcommand in `src/nc_check/cli.py`
- parse any suite-specific flags
- route to `check_my_suite(...)`
- include it in `_CHECK_MODES`
- update save-report naming help text if needed

### 5) (Optional) Render in full report views

`ds.check.all(report_format="python")` will already include your raw report in
`full_report["reports"]["my_suite"]` once registered.

To render a dedicated section in combined table/HTML outputs, update
`src/nc_check/formatting.py`:

- `print_pretty_full_report(...)` for terminal output
- `_full_report_sections(...)` for HTML output

### 6) Tests to add/update

At minimum:

- suite behavior and summary math
- accessor method wiring and options forwarding
- `ds.check.all()` inclusion/exclusion behavior
- CLI routing (if command added)
- report-format behavior (`python`/`tables`/`html`)

Useful files to mirror:

- `tests/test_suite.py`
- `tests/test_accessor.py`
- `tests/test_cli.py`

### 7) Documentation updates

After implementation, update:

- `docs/checks-and-reports.md` (user-facing check description + report fields)
- `docs/python-api.md` (new accessor method/options)
- `docs/cli.md` (if CLI command was added)

## Local Smoke Check

```bash
uv run python -c "import nc_check, xarray as xr; print('ok')"
```
