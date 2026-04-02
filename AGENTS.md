# Agent Handoff Document — nc-check

> **FOR NEW AGENTS**: Read this entire file before doing anything. It contains full project context, the analysis of what's wrong, and the implementation plan.

---

## Session context

A scientist working on the Global Carbon Budget uses this package to validate and fix NetCDF files before submission. Key constraints established in this session:

- `cfchecker` has heavy/complicated deps — **the heuristic engine is the primary path**, not a fallback
- Fixes are **metadata-only**; no data restructuring unless user explicitly asks
- Primary format: **CMIP6**; files must be **Ferret-readable**
- For unfixable issues: report them clearly; for fixable ones: save output to a new file (already works via `nc-comply`)
- Ocean is the current focus; atmosphere is a future possibility

---

## What the repo does

**nc-check** prepares `xarray.Dataset` objects for CF-1.12 compliant NetCDF output. It validates, reports on, and auto-fixes datasets.

```
src/nc_check/
├── accessor.py        # ds.check xarray accessor
├── cli.py             # nc-check / nc-comply CLI
├── core/
│   ├── check.py       # Abstract Check base; CheckResult, FixResult, CheckStatus (Pydantic v2)
│   ├── compliance.py  # Main engine: check_dataset_compliant(), make_dataset_compliant()
│   └── coverage.py    # Coverage gap utilities
├── checks/
│   ├── heuristic.py   # PRIMARY check engine (cfchecker is optional/off-table)
│   ├── ocean.py       # Ocean grid coverage checks
│   └── time_cover.py  # Time-slice coverage checks
├── engine/
│   ├── suite.py / runner.py / registry.py / defaults.py
├── formatting.py
└── standard_names.py
tests/                 # pytest, 10+ files
```

**Key patterns to reuse:**

| Utility | Location | Notes |
|---------|----------|-------|
| `_finding()` helper | `heuristic.py:115` | Use for all new findings |
| `_axis_guesses()` | `heuristic.py:388` | Infers T/Z/Y/X from dim names/attrs |
| `_references_from_whitespace_list()` | `heuristic.py:287` | Parses space-separated var name lists |
| `_CF_ATTR_CASE_KEYS` | `compliance.py:63` | Dict of known CF attribute names |
| `guess_axis_for_dim()` | `compliance.py` | Called in make_dataset_compliant |
| `CheckInfo`, `FixResult`, `CheckResult` | `core/check.py` | Pydantic models |

Finding structure: `{severity, item, message, current, expected, suggested_fix, extra}`

All findings stored in: `issues["global"]`, `issues["coordinates"][name]`, `issues["variables"][name]`

**Fix pipeline:**
- `make_dataset_compliant()` at `compliance.py:1161` — deep-copies dataset, applies metadata fixes, returns new dataset
- `nc-comply` CLI at `cli.py:318` — calls above then `.to_netcdf(fname_out)`
- `HeuristicCheck` at `heuristic.py:665` — currently has `fixable=False` hardcoded (needs fixing)
- `Check.fix()` at `check.py:89` — currently a stub returning "No fix implemented"

---

## What's wrong (gap analysis)

### Critical gaps vs CF-1.12 + CMIP6 + Ferret

| Gap | Priority | Fixable? |
|-----|----------|----------|
| Cell methods not validated at all (CF §7.4) | High | No — data issue |
| Bounds variable structure not validated (CF §7.1) | High | No — data issue |
| Time `calendar` attribute not checked (CF §4.4.1, CMIP6) | High | Yes → add `calendar="standard"` |
| Vertical `positive` attribute not checked (CF §4.3) | Medium | No — ambiguous direction |
| Dimension order not checked (T,Z,Y,X — Ferret/COARDS) | Medium | No — requires data restructure |
| CMIP6 global attributes not checked | Medium | No |
| `HeuristicCheck.fix()` is a stub — no fixes applied | Critical | — |
| `nc-comply` doesn't report unfixable issues | High | — |

### Ferret-specific status
- `_FillValue` on coordinate variables → **already checked** ✓
- Missing `units` on time/lat/lon/depth → **already checked** ✓
- Non-monotonic coordinates → **already checked** ✓
- `calendar` attribute missing → **NOT yet checked** ✗

---

## Implementation plan

### Chunk 0 — Fix infrastructure (PREREQUISITE)
**File**: `src/nc_check/core/check.py`

- Add `unfixable_items: list[str] = []` field to `FixResult` Pydantic model
- This field will list finding `item` strings detected but not auto-fixable
- No logic changes — just data model addition
- **Test**: `tests/test_check_models.py` — assert `FixResult.unfixable_items` defaults to `[]`

---

### Chunk 1 — Deepen heuristic checks (after Chunk 0)
**File**: `src/nc_check/checks/heuristic.py`

Add using the existing `_finding()` pattern:

**1. Bounds structure** (CF §7.1)
- For each variable with a `bounds` attr: check shape is `(N, 2)`, check values bracket coordinate
- Severity: ERROR for wrong shape, WARN for values not bracketing
- `suggested_fix: None`

**2. Cell methods** (CF §7.4)
- Parse `cell_methods` string, validate method names against CF-allowed list:
  `point`, `sum`, `mean`, `maximum`, `minimum`, `mid_range`, `standard_deviation`,
  `variance`, `mode`, `median`, `maximum_absolute_value`, `minimum_absolute_value`,
  `mean_absolute_value`, `mean_of_upper_decile`
- Validate referenced dimension names exist in the dataset
- Severity: WARN for unknown method, ERROR for non-existent dimension
- `suggested_fix: None`

**3. Time calendar attribute** (CF §4.4.1 + CMIP6)
- For time coordinates: check `calendar` attr is present
- Valid values: `standard`, `gregorian`, `proleptic_gregorian`, `noleap`, `365_day`,
  `all_leap`, `366_day`, `360_day`, `julian`, `none`
- Severity: WARN if missing, ERROR if invalid value
- `suggested_fix: "add_calendar_attr"` (fixable)

**4. Vertical `positive` attribute** (CF §4.3)
- For coords with `axis="Z"` or name in `("depth", "lev", "level", "height", "altitude", "plev")`
- Check `positive` is `"up"` or `"down"`
- Severity: WARN if missing
- `suggested_fix: None`

**5. Dimension order** (COARDS / Ferret)
- For each data variable: if T/Z/Y/X axes present, check order is T→Z→Y→X
- Severity: WARN only
- `suggested_fix: None`

**6. CMIP6 recommended global attributes**
- If `mip_era == "CMIP6"`: check `institution`, `source`, `tracking_id`, `creation_date`, `frequency`, `realm`, `variable_id`
- Otherwise: check `institution`, `source`, `title`, `history`
- Severity: WARN for missing
- `suggested_fix: None`

**Tests** (new `tests/test_heuristic_extended.py`):
- `test_bounds_wrong_shape` → ERROR
- `test_bounds_valid` → no finding
- `test_cell_methods_valid` (`"time: mean"`) → no finding
- `test_cell_methods_bad_method` (`"time: foobar"`) → WARN
- `test_cell_methods_bad_dim` (`"nonexistent_dim: mean"`) → ERROR
- `test_calendar_missing` → WARN
- `test_calendar_invalid_value` → ERROR
- `test_vertical_no_positive` → WARN
- `test_dimension_order_wrong` → WARN
- `test_cmip6_attrs_missing` → WARNs

---

### Chunk 2 — Fix pipeline improvements (parallel with Chunk 1)
**Files**: `src/nc_check/core/compliance.py`, `src/nc_check/checks/heuristic.py`, `src/nc_check/cli.py`

**`make_dataset_compliant()` additions:**
1. If time coord exists and `calendar` missing → add `calendar="standard"`
2. Do NOT fix `positive` or dimension order (add to `unfixable_items` instead)

**`HeuristicCheck.fix()` implementation:**
- Apply calendar fix
- Return `FixResult(applied=True, unfixable_items=[...])` listing items that were found but not auto-fixed

**`nc-comply` CLI update (`cli.py:318-334`):**
- Print summary after saving:
  ```
  Fixed:     3 issues written to output.nc
  Unfixable: 2 issues require manual attention (run nc-check for details)
  ```
- Exit with code 1 if there are unfixable items (so CI pipelines catch it)

**Tests** (new `tests/test_comply_pipeline.py`):
- `test_comply_adds_calendar` → output has `calendar="standard"`
- `test_comply_reports_unfixable` → exit code 1 when unfixable issues present
- `test_comply_roundtrip` → re-run nc-check on output, assert no new errors

---

### Chunk 3 — Test suite (finalise after Chunks 1 & 2)

Priority tests beyond those above:
1. **Fix round-trip** — `make_dataset_compliant()` on known-bad dataset → re-run `_heuristic_report()` → no regressions
2. **Ferret blocker guard** — `_FillValue` on coord → ERROR (explicit named test)
3. **Bounds bracket test** — bounds don't contain coordinate → WARN
4. **CMIP6 detection** — `mip_era="CMIP6"` triggers stricter global attr checks

---

## Execution order

```
Chunk 0  →  Chunk 1 + Chunk 2 (parallel)  →  Chunk 3
```

## Verification

```bash
uv run pytest tests/test_heuristic_extended.py tests/test_comply_pipeline.py -v
uv run pytest                          # full suite — no regressions
nc-check compliance tests/data/sample.nc
nc-comply tests/data/sample.nc /tmp/out.nc && nc-check compliance /tmp/out.nc
```
