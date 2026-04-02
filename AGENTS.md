# Agent Handoff Document — nc-check

> **FOR NEW AGENTS**: Read this entire file before doing anything. It contains full project context, current implementation status, and suggested next work.

---

## Session context

A scientist working on the Global Carbon Budget uses this package to validate and fix NetCDF files before submission. Key constraints established in this session:

- `cfchecker` has heavy/complicated deps — **the heuristic engine is the primary path**, not a fallback
- Fixes are **metadata-only**; no data restructuring unless user explicitly asks
- Primary format: **CMIP6**; files must be **Ferret-readable**
- For unfixable issues: report them clearly; for fixable ones: save output to a new file (via `nc-comply`)
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
tests/                 # pytest, 147 tests, 13 files
```

**Key patterns to reuse:**

| Utility | Location | Notes |
|---------|----------|-------|
| `_finding()` helper | `heuristic.py` | Use for ALL new findings — keyword-only args |
| `_axis_guesses()` | `heuristic.py` | Returns `dict[str, AxisGuess]` — axis_type is "time"/"lat"/"lon" |
| `_references_from_whitespace_list()` | `heuristic.py` | Parses space-separated var name lists |
| `_VALID_CALENDARS` | `heuristic.py` | frozenset of valid CF calendar values |
| `_VERTICAL_NAMES` | `heuristic.py` | frozenset of vertical coord name candidates |
| `_CF_ATTR_CASE_KEYS` | `compliance.py` | Tuple of known CF attribute names for case normalisation |
| `guess_axis_for_dim()` | `heuristic.py` | Infers axis type for a dimension |
| `CheckInfo`, `FixResult`, `CheckResult` | `core/check.py` | Pydantic models |

Finding structure: `{severity, item, message, current, expected, suggested_fix, extra}`

All findings stored in: `issues["global"]`, `issues["coordinates"][name]`, `issues["variables"][name]`

**Fix pipeline (current state — fully implemented):**
- `make_dataset_compliant()` in `compliance.py` — deep-copies dataset, applies metadata fixes, adds `calendar="standard"` to time coords, returns new dataset
- `HeuristicCheck.fix()` in `heuristic.py` — calls `make_dataset_compliant()`, collects unfixable items from report, returns `FixResult` with `unfixable_items` list
- `nc-comply` CLI in `cli.py` — calls `HeuristicCheck.fix()`, writes output, prints summary, exits 1 if unfixable items
- `FixResult.unfixable_items: list[str]` — lists `"scope:item"` strings for issues that need manual attention

**Important implementation notes:**
- For datetime64 time coordinates: `calendar` goes in `coord.encoding`, NOT `coord.attrs` (xarray conflict)
- The `_FillValue` removal in `make_dataset_compliant()` preserves existing encoding — it does NOT wipe it
- `HeuristicCheck.check()` sets `fixable=True`
- Unfixable items = findings where `suggested_fix is None`

---

## CF compliance status

### Implemented checks (heuristic engine)

| Check | CF ref | Severity | Fixable? |
|-------|--------|----------|----------|
| `Conventions` attribute | — | WARN | Yes |
| Coordinate attrs (standard_name, units, axis) | CF §4 | WARN | Yes |
| Coordinate uniqueness + monotonicity | — | WARN | Yes |
| Lat/lon range | CF §4.1–4.2 | ERROR | Yes |
| Time units format | CF §4.4 | ERROR/WARN | Yes |
| Time `calendar` attribute | CF §4.4.1 | WARN/ERROR | Yes → `calendar="standard"` |
| Bounds variable structure | CF §7.1 | ERROR/WARN | No |
| Cell methods validity | CF §7.4 | WARN/ERROR | No |
| Vertical `positive` attribute | CF §4.3 | WARN/ERROR | No |
| Dimension order (T,Z,Y,X) | COARDS/Ferret | WARN | No |
| CMIP6/general recommended global attrs | CMIP6 | WARN | No |
| Variable name CF compliance | — | ERROR | Yes |
| Variable units/standard_name | — | WARN/ERROR | Yes |
| Reference attrs (bounds, coordinates, etc.) | — | ERROR | Yes |

### Ferret-specific status

| Issue | Status |
|-------|--------|
| `_FillValue` on coordinate variables | ✓ checked + fixed |
| Missing `units` on time/lat/lon/depth | ✓ checked |
| Non-monotonic coordinates | ✓ checked |
| `calendar` attribute missing | ✓ checked + fixed |
| Dimension order (T,Z,Y,X) | ✓ checked (unfixable — data restructure needed) |

---

## Suggested next work

The following are not yet implemented. Priority order for the Global Carbon Budget use case:

### Ocean checks (high value)

**1. Missing latitude bands**
- Symmetric with existing `MissingLongitudeBandsCheck` — add `MissingLatitudeBandsCheck` in `ocean.py`
- Same pattern: check for persistent all-NaN latitude slices across time

**2. Physical range check (per standard_name)**
- Add to `ocean.py` or `heuristic.py`
- Known ranges by `standard_name`:

  | standard_name | Warn range | Error range |
  |---|---|---|
  | `sea_water_temperature` | −2 to 35 °C | < −5 or > 50 |
  | `sea_surface_temperature` | −2 to 35 °C | < −5 or > 50 |
  | `sea_water_salinity` | 0 to 42 PSU | < −1 or > 50 |
  | `sea_water_potential_density` | 900 to 1100 kg/m³ | outside 800–1200 |

**3. Depth coordinate units validation**
- Extend the existing `_vertical_positive_findings()` in `heuristic.py`
- `units` should be `m`; WARN for `km`, ERROR if missing on a named depth coord
- Values should be non-negative when `positive="down"`

**4. `grid_mapping` variable completeness**
- If `grid_mapping` attr is set, check referenced variable has `grid_mapping_name`
- If lat/lon are 2D (curvilinear/tripolar), WARN that `grid_mapping` is strongly recommended

### Test gaps

These tests from the original plan were not written; lower priority but would improve coverage:

- **Ferret blocker guard** — explicit test: `_FillValue` on coord → ERROR (named `test_ferret_fillvalue_on_coord`)
- **CMIP6 detection** — `mip_era="CMIP6"` triggers stricter attrs; currently covered by `test_cmip6_attrs_missing` but could be more thorough

---

## Verification

```bash
uv run pytest                          # 147 tests, ~2s — must all pass
uv run pytest tests/test_heuristic_extended.py tests/test_comply_pipeline.py -v
nc-check compliance tests/data/sample.nc
nc-comply tests/data/sample.nc /tmp/out.nc && nc-check compliance /tmp/out.nc
```
