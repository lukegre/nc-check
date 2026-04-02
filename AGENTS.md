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
- `nc-comply` CLI at `cli.py:318` — calls above, writes output, reports unfixable count, exits 1 if unfixable
- `HeuristicCheck` at `heuristic.py:1002` — `fixable=True`, `fix()` delegates to `make_dataset_compliant()` and collects `unfixable_items`
- `FixResult` at `check.py:60` — has `unfixable_items: list[str]` field listing finding items that could not be auto-fixed

---

## Status (all implemented)

### CF-1.12 + CMIP6 + Ferret checks

| Check | Implemented | Fixable | Notes |
|-------|-------------|---------|-------|
| Cell methods (CF §7.4) | ✓ `heuristic.py:694` | No — data issue | WARN unknown method, ERROR nonexistent dim |
| Bounds variable structure (CF §7.1) | ✓ `heuristic.py:629` | No — data issue | ERROR wrong shape, WARN values not bracketing |
| Time `calendar` attribute (CF §4.4.1) | ✓ `heuristic.py:765` | Yes → `calendar="standard"` | WARN missing, ERROR invalid value |
| Vertical `positive` attribute (CF §4.3) | ✓ `heuristic.py:808` | No — ambiguous direction | WARN if missing |
| Dimension order T,Z,Y,X (Ferret/COARDS) | ✓ `heuristic.py:848` | No — requires data restructure | WARN only |
| CMIP6 global attributes | ✓ `heuristic.py:903` | No | WARN if missing |
| `HeuristicCheck.fix()` | ✓ `heuristic.py:1002` | — | Applies `make_dataset_compliant()`, returns `unfixable_items` |
| `nc-comply` unfixable reporting | ✓ `cli.py:335` | — | Prints count, exits 1 if any unfixable |

### Ferret-specific status
- `_FillValue` on coordinate variables → **checked + fixed** ✓
- Missing `units` on time/lat/lon/depth → **checked + fixed** ✓
- Non-monotonic coordinates → **checked** ✓
- `calendar` attribute missing → **checked + fixed** ✓
- Dimension order (T,Z,Y,X) → **checked (WARN), not fixable** ✓

---

## Verification

```bash
uv run pytest                          # full suite — 147 tests pass
uv run pytest tests/test_heuristic_extended.py tests/test_comply_pipeline.py -v
nc-check compliance tests/data/sample.nc
nc-comply tests/data/sample.nc /tmp/out.nc && nc-check compliance /tmp/out.nc
```
