# Add A Check Suite And Report

This guide shows how to add a new suite-style check to `nc-check`, with:

- Python API support (`ds.check.<suite>()`)
- optional CLI command (`nc-check <suite>`)
- participation in `ds.check.all()`
- table and HTML reporting
- tests and docs updates

Use `time_cover` and `ocean_cover` as implementation references.

## Before You Start

Read these files first:

- `src/nc_check/checks/time_cover.py`
- `src/nc_check/checks/ocean.py`
- `src/nc_check/engine/defaults.py`
- `src/nc_check/engine/runner.py`
- `src/nc_check/engine/suite.py`
- `src/nc_check/accessor.py`
- `src/nc_check/formatting.py`

## Report Shape Requirements

For best integration with `ds.check.all()`, your suite report should include:

- `group` (string)
- `suite` (string)
- `checks` (list of atomic check items)
- `summary` (`checks_run`, `failing_checks`, `warnings_or_skips`, `overall_status`, `overall_ok`)
- `ok` (boolean)

Why this matters:

- `engine/runner.py` flattens `report["checks"]` into the combined full report.
- If `checks` is missing, it falls back to one synthetic check item (you lose per-atomic detail).

## Example: Add A `value_range` Suite

The example below checks whether variables exceed configured min/max thresholds.

### 1) Create `src/nc_check/checks/value_range.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from ..engine.suite import Suite, SuiteCheck
from ..formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    save_html_report,
    to_yaml_like,
)


class ValueRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    var_name: str | None = None
    min_allowed: float | None = None
    max_allowed: float | None = None


@dataclass(frozen=True)
class ValueRangeCheck:
    var_name: str
    min_allowed: float | None
    max_allowed: float | None

    id: str = "value_range.out_of_bounds"
    name: str = "Out Of Bounds Values"

    def run_report(self, ds: xr.Dataset) -> dict[str, Any]:
        if self.var_name not in ds.data_vars:
            raise ValueError(f"Data variable '{self.var_name}' not found.")

        arr = np.asarray(ds[self.var_name].values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {
                "status": "skipped_no_finite_values",
                "out_of_bounds_count": 0,
                "total_count": int(arr.size),
                "min_observed": None,
                "max_observed": None,
            }

        min_observed = float(np.min(finite))
        max_observed = float(np.max(finite))

        out_of_bounds = np.zeros(finite.shape, dtype=bool)
        if self.min_allowed is not None:
            out_of_bounds |= finite < self.min_allowed
        if self.max_allowed is not None:
            out_of_bounds |= finite > self.max_allowed

        count = int(np.count_nonzero(out_of_bounds))
        status = "fail" if count > 0 else "pass"

        return {
            "status": status,
            "out_of_bounds_count": count,
            "total_count": int(finite.size),
            "min_observed": min_observed,
            "max_observed": max_observed,
            "min_allowed": self.min_allowed,
            "max_allowed": self.max_allowed,
        }


def _single_value_range_report(
    ds: xr.Dataset,
    *,
    var_name: str,
    min_allowed: float | None,
    max_allowed: float | None,
) -> dict[str, Any]:
    check = ValueRangeCheck(
        var_name=var_name,
        min_allowed=min_allowed,
        max_allowed=max_allowed,
    )

    suite = Suite(
        name="value_range",
        checks=[
            SuiteCheck(
                check_id=check.id,
                name=check.name,
                run=lambda: check.run_report(ds),
                detail=lambda result: (
                    f"out_of_bounds={int(result.get('out_of_bounds_count', 0))}"
                ),
            )
        ],
    ).run()

    check_result = suite["checks"][0]["result"] if suite["checks"] else {}
    return {
        "group": suite["group"],
        "suite": suite["suite"],
        "variable": var_name,
        "value_range": check_result,
        "checks": suite["checks"],
        "summary": suite["summary"],
        "ok": suite["ok"],
    }


def run_value_range_report(
    ds: xr.Dataset,
    *,
    config: ValueRangeConfig | None = None,
    var_name: str | None = None,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
) -> dict[str, Any]:
    resolved = config or ValueRangeConfig(
        var_name=var_name,
        min_allowed=min_allowed,
        max_allowed=max_allowed,
    )

    selected_vars = [resolved.var_name] if resolved.var_name else list(ds.data_vars)
    reports: dict[str, dict[str, Any]] = {}
    for item in selected_vars:
        if item is None:
            continue
        reports[item] = _single_value_range_report(
            ds,
            var_name=item,
            min_allowed=resolved.min_allowed,
            max_allowed=resolved.max_allowed,
        )

    if len(reports) == 1:
        return next(iter(reports.values()))

    suite_checks: list[dict[str, Any]] = []
    for variable_name, per_var in reports.items():
        for check_item in per_var.get("checks", []):
            if not isinstance(check_item, dict):
                continue
            flattened = dict(check_item)
            flattened["variable"] = variable_name
            suite_checks.append(flattened)

    suite = Suite.report_from_items("value_range", suite_checks)
    return {
        "group": suite["group"],
        "suite": suite["suite"],
        "mode": "all_variables",
        "checked_variable_count": len(reports),
        "checked_variables": list(reports.keys()),
        "reports": reports,
        "checks": suite["checks"],
        "summary": suite["summary"],
        "ok": suite["ok"],
    }


def check_value_range(
    ds: xr.Dataset,
    *,
    var_name: str | None = None,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    report = run_value_range_report(
        ds,
        config=ValueRangeConfig(
            var_name=var_name,
            min_allowed=min_allowed,
            max_allowed=max_allowed,
        ),
    )

    if resolved_format == "tables":
        print(to_yaml_like(report))
        return None

    if resolved_format == "html":
        html = "<html><body><pre>" + to_yaml_like(report) + "</pre></body></html>"
        save_html_report(html, report_html_file)
        maybe_display_html_report(html)
        return html

    return report
```

Notes:

- The example uses `to_yaml_like` for table/HTML output to keep it short.
- For production parity, follow `time_cover` and add dedicated rich/HTML renderers in `formatting.py`.

### 2) Export the check in `src/nc_check/checks/__init__.py`

```python
from .value_range import check_value_range, run_value_range_report

__all__ = [
    # existing entries...
    "check_value_range",
    "run_value_range_report",
]
```

### 3) Register with the engine (`src/nc_check/engine/defaults.py`)

Add your key to order:

```python
_DEFAULT_ORDER = ("compliance", "ocean_cover", "time_cover", "value_range")
```

Add resolver functions:

```python
def _status_from_value_range_report(report: dict[str, Any]) -> SummaryStatus:
    summary = report.get("summary")
    if isinstance(summary, dict):
        status = str(summary.get("overall_status", "")).lower()
        if status in {"pass", "warn", "fail"}:
            return status  # type: ignore[return-value]
    leaf = report.get("value_range")
    if isinstance(leaf, dict):
        raw = str(leaf.get("status", "")).lower()
        if raw in {"pass", "warn", "fail"}:
            return raw  # type: ignore[return-value]
        if raw.startswith("skip"):
            return "warn"
    return "pass"


def _value_range_detail(report: dict[str, Any]) -> str:
    summary = report.get("summary")
    if isinstance(summary, dict):
        return (
            f"checks={int(summary.get('checks_run', 0))} "
            f"fail={int(summary.get('failing_checks', 0))} "
            f"warn_or_skip={int(summary.get('warnings_or_skips', 0))}"
        )
    leaf = report.get("value_range")
    if isinstance(leaf, dict):
        return f"out_of_bounds={int(leaf.get('out_of_bounds_count', 0))}"
    return "completed"
```

Add runner and registration:

```python
from ..checks.value_range import run_value_range_report


def _run_value_range_report(ds: xr.Dataset, options: dict[str, Any]) -> dict[str, Any]:
    return run_value_range_report(
        ds,
        var_name=options.get("var_name"),
        min_allowed=options.get("min_allowed"),
        max_allowed=options.get("max_allowed"),
    )


register_check(
    RegisteredCheck(
        key="value_range",
        run_report=_run_value_range_report,
        resolve_status=_status_from_value_range_report,
        resolve_detail=_value_range_detail,
    )
)
```

### 4) Add accessor wiring (`src/nc_check/accessor.py`)

Import function:

```python
from .checks.value_range import check_value_range as run_value_range_check
```

Add method:

```python
def value_range(
    self,
    *,
    var_name: str | None = None,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    return run_value_range_check(
        self._ds,
        var_name=var_name,
        min_allowed=min_allowed,
        max_allowed=max_allowed,
        report_format=report_format,
        report_html_file=report_html_file,
    )
```

Add `all()` switches/options:

```python
def all(
    self,
    *,
    # existing toggles...
    value_range: bool = True,
    # existing args...
    min_allowed: float | None = None,
    max_allowed: float | None = None,
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    enabled = {
        "compliance": bool(compliance),
        "ocean_cover": bool(ocean_cover),
        "time_cover": bool(time_cover),
        "value_range": bool(value_range),
    }
    full_report = run_suite_checks(
        self._ds,
        checks_enabled=enabled,
        options_by_check={
            # existing options...
            "value_range": {
                "var_name": var_name,
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
            },
        },
    )
    # existing report_format handling...
```

### 5) Optional CLI command (`src/nc_check/cli.py`)

Add mode:

```python
_CHECK_MODES = {"compliance", "ocean-cover", "time-cover", "value-range", "all"}
```

Parser section:

```python
value_range = subparsers.add_parser(
    "value-range",
    help="Run value-range checks.",
)
_add_shared_options(value_range)
value_range.add_argument("--var-name", default=None)
value_range.add_argument("--min-allowed", type=float, default=None)
value_range.add_argument("--max-allowed", type=float, default=None)
```

Dispatch:

```python
elif mode == "value-range":
    check_value_range(
        ds,
        var_name=getattr(args, "var_name", None),
        min_allowed=getattr(args, "min_allowed", None),
        max_allowed=getattr(args, "max_allowed", None),
        report_format=report_format,
        report_html_file=report_html_file,
    )
```

### 6) Optional full-report rendering (`src/nc_check/formatting.py`)

`ds.check.all(report_format="python")` already includes your raw report under
`report["reports"]["value_range"]`.

To render a first-class section in full output:

- `print_pretty_full_report(...)`: render the new report in terminal mode
- `_full_report_sections(...)`: add a "Value Range" HTML section

Use `ocean_cover` and `time_cover` branches as templates.

### 7) Tests

Add tests in:

- `tests/test_suite.py` for suite summary logic if you add helper behavior
- `tests/test_accessor.py` for:
  - `ds.check.value_range(...)` return shape
  - `ds.check.all(..., value_range=True/False)` wiring
- `tests/test_cli.py` for command routing/flags (if CLI command added)

Example accessor test:

```python
def test_all_python_report_includes_value_range() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time",), [1.0, 2.0, 3.0])},
        coords={"time": [0, 1, 2]},
    )

    report = ds.check.all(
        compliance=False,
        ocean_cover=False,
        time_cover=False,
        value_range=True,
        min_allowed=0.0,
        max_allowed=2.5,
        report_format="python",
    )

    assert report["reports"].keys() == {"value_range"}
    assert report["groups"]["value_range"]["checks_run"] >= 1
```

Example CLI test:

```python
def test_run_check_value_range_mode_routes_to_value_range_checker(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "sample.nc"
    xr.Dataset(
        data_vars={"v": (("time",), [1.0])},
        coords={"time": [0]},
    ).to_netcdf(source)

    seen: dict[str, object] = {}

    def _fake_value_range(
        ds: xr.Dataset,
        *,
        var_name: str | None = None,
        min_allowed: float | None = None,
        max_allowed: float | None = None,
        report_format: str = "tables",
        report_html_file: str | None = None,
    ) -> None:
        seen["var_name"] = var_name
        seen["min_allowed"] = min_allowed
        seen["max_allowed"] = max_allowed
        seen["report_format"] = report_format
        seen["report_html_file"] = report_html_file

    monkeypatch.setattr(cli, "check_value_range", _fake_value_range)

    status = cli.run_check(
        ["value-range", str(source), "--min-allowed", "0", "--max-allowed", "10"]
    )

    assert status == 0
    assert seen["min_allowed"] == 0.0
    assert seen["max_allowed"] == 10.0
    assert seen["report_format"] == "tables"
```

## Common Pitfalls

- Missing `checks` in report:
  - `ds.check.all()` cannot flatten atomic checks and falls back to one synthetic item.
- Inconsistent status words:
  - prefer `pass`, `fail`, `warn`, or `skipped_*` variants for predictable aggregation.
- `report_html_file` used with non-HTML:
  - keep the existing validation pattern (`report_html_file` only when `report_format="html"`).
- Not adding to `_DEFAULT_ORDER`:
  - check is registered but never run by `ds.check.all()`.
- Docs/tests not updated:
  - feature exists but is hard to discover and easy to regress.

## Minimal Checklist

- [ ] new `checks/<suite>.py` module with `run_<suite>_report` and `check_<suite>`
- [ ] exported in `checks/__init__.py` (if intended public)
- [ ] registered in `engine/defaults.py`
- [ ] wired into accessor (`ds.check.<suite>` and `ds.check.all`)
- [ ] optional CLI command added and tested
- [ ] optional pretty full-report rendering
- [ ] docs updated (`checks-and-reports`, `python-api`, `cli`)
- [ ] tests added for report shape + wiring + options forwarding
