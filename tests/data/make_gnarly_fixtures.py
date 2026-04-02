"""
Generate gnarly NetCDF test fixtures that violate CF conventions in various ways.

Run with:
    uv run python tests/data/make_gnarly_fixtures.py

Each file is designed to trigger specific failure modes in nc-check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

OUT_DIR = Path(__file__).parent


def make_lat_upside_down() -> xr.Dataset:
    """Latitude runs 90→-90 (descending) instead of -90→90.

    Also missing axis/units attributes on coordinates, no Conventions attr.
    """
    lat = np.linspace(90.0, -90.0, 37)   # descending: wrong
    lon = np.linspace(0.0, 357.5, 144)
    time = np.arange(4, dtype=float)

    data = np.random.default_rng(0).standard_normal((4, 37, 144)).astype("float32")

    ds = xr.Dataset(
        data_vars={
            "sst": (
                ("time", "lat", "lon"),
                data,
                {"long_name": "sea surface temperature", "units": "degC"},
            )
        },
        coords={
            "time": ("time", time, {"units": "days since 2000-01-01"}),
            "lat": ("lat", lat),   # deliberately no axis/units/standard_name
            "lon": ("lon", lon),   # deliberately no axis/units/standard_name
        },
        # No Conventions, no institution, no title
    )
    return ds


def make_lon_offset_wrong_labels() -> xr.Dataset:
    """Longitude data runs 180→360 then 0→180 (offset by 180°) but coordinate
    labels claim -180→180.  Units also claim 'degrees_west' (invalid CF unit).
    """
    # Data is physically stored with the Americas in the "wrong" half
    lon_actual = np.concatenate([np.linspace(180.0, 360.0, 73),
                                  np.linspace(0.0, 177.5, 72)])
    # But labelled as if it were a normal -180→180 grid
    lon_labels = np.linspace(-180.0, 177.5, 145)

    lat = np.linspace(-90.0, 90.0, 73)
    time = np.arange(3, dtype=float)

    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 73, 145)).astype("float32")

    ds = xr.Dataset(
        data_vars={
            "precip": (
                ("time", "lat", "lon"),
                data,
                {
                    "long_name": "precipitation",
                    "units": "mm/day",
                    "standard_name": "precipitation_flux",   # wrong standard name
                    "cell_methods": "time: accumulation",    # invalid cell method
                },
            )
        },
        coords={
            "time": ("time", time, {"units": "days since 2000-01-01", "axis": "T"}),
            "lat": (
                "lat",
                lat,
                {
                    "units": "degrees_north",
                    "axis": "Y",
                    "standard_name": "latitude",
                },
            ),
            "lon": (
                "lon",
                lon_labels,   # wrong labels!
                {
                    "units": "degrees_west",  # invalid — CF says degrees_east
                    "axis": "X",
                    "standard_name": "longitude",
                },
            ),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    return ds


def make_bad_var_names() -> xr.Dataset:
    """Variables and dimensions have cryptic/numeric/machine-generated names.

    Dim names: d0, d1, d2 (no recognisable axis).
    Variable names: var1, field_2, X (single letter).
    No standard_name, no long_name, no units anywhere.
    """
    d0 = np.arange(5, dtype=float)    # time-like, but unnamed
    d1 = np.linspace(0.0, 100.0, 10)  # lat-like, but unnamed
    d2 = np.linspace(0.0, 200.0, 20)  # lon-like, but unnamed

    rng = np.random.default_rng(2)
    ds = xr.Dataset(
        data_vars={
            "var1": (("d0", "d1", "d2"), rng.standard_normal((5, 10, 20)).astype("float32")),
            "field_2": (("d0", "d1", "d2"), rng.standard_normal((5, 10, 20)).astype("float32")),
            "X": (("d1", "d2"), rng.standard_normal((10, 20)).astype("float32")),
        },
        coords={
            "d0": d0,
            "d1": d1,
            "d2": d2,
        },
        # Completely bare — no global attrs, no variable attrs
    )
    return ds


def make_wrong_dim_order() -> xr.Dataset:
    """Dimensions in Fortran / wrong order: (lon, lat, time) instead of (time, lat, lon).

    Also includes a depth dimension inserted between lat and time.
    """
    time = np.arange(12, dtype=float)
    lat = np.linspace(-90.0, 90.0, 37)
    lon = np.linspace(0.0, 357.5, 144)
    depth = np.array([0.0, 10.0, 50.0, 200.0, 500.0])  # metres

    rng = np.random.default_rng(3)
    # Dimension order: (lon, depth, lat, time) — completely non-standard
    data = rng.standard_normal((144, 5, 37, 12)).astype("float32")

    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("lon", "depth", "lat", "time"),
                data,
                {"long_name": "ocean temperature", "units": "degC"},
            )
        },
        coords={
            "time": ("time", time, {"units": "days since 1850-01-01", "axis": "T"}),
            "lat": ("lat", lat, {"units": "degrees_north", "axis": "Y", "standard_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "axis": "X", "standard_name": "longitude"}),
            "depth": (
                "depth",
                depth,
                {
                    "units": "m",
                    "positive": "down",     # correct
                    "axis": "Z",
                    "standard_name": "depth",
                },
            ),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    return ds


def make_missing_units() -> xr.Dataset:
    """Multiple variables and coordinates completely missing units.

    - temperature: no units (should be K or degC)
    - salinity: no units (should be 1 or PSU)
    - lat/lon: no units
    - time: no units (numeric axis with no reference)
    Also: lat and lon have no standard_name or axis attributes.
    """
    time = np.arange(6, dtype=float)
    lat = np.linspace(-90.0, 90.0, 37)
    lon = np.linspace(-180.0, 180.0, 73)

    rng = np.random.default_rng(4)
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"),
                (rng.standard_normal((6, 37, 73)) * 10 + 290).astype("float32"),
                {"long_name": "sea water temperature"},   # no units
            ),
            "salinity": (
                ("time", "lat", "lon"),
                (rng.standard_normal((6, 37, 73)) * 0.5 + 35).astype("float32"),
                {"long_name": "sea water salinity"},      # no units
            ),
        },
        coords={
            "time": ("time", time),   # no units, no axis
            "lat": ("lat", lat),      # no units, no axis, no standard_name
            "lon": ("lon", lon),      # no units, no axis, no standard_name
        },
    )
    return ds


def make_missing_lon_slices() -> xr.Dataset:
    """Large persistent gaps in longitude coverage — all NaN across all times and lats.

    Missing bands:
      - 0°–30°   (first band, 7 columns)
      - 150°–180° (middle band, 7 columns)
      - 330°–359° (last band, wraps near dateline, 6 columns)
    """
    lon = np.arange(0.0, 360.0, 5.0)   # 72 longitudes, 5° spacing
    lat = np.linspace(-90.0, 90.0, 37)
    time = np.arange(8, dtype=float)

    rng = np.random.default_rng(5)
    data = rng.standard_normal((8, 37, 72)).astype("float32")

    # Blank out entire longitude bands (all times, all lats)
    missing_bands = [
        (lon >= 0) & (lon <= 30),    # 0°–30°
        (lon >= 150) & (lon <= 180), # 150°–180°
        (lon >= 330),                # 330°–355°
    ]
    for mask in missing_bands:
        data[:, :, mask] = np.nan

    # Add some scattered NaNs that should NOT be flagged as persistent gaps
    data[2, 5, 10] = np.nan
    data[0, 20, 40] = np.nan

    ds = xr.Dataset(
        data_vars={
            "chl": (
                ("time", "lat", "lon"),
                data,
                {
                    "long_name": "chlorophyll-a concentration",
                    "units": "mg m-3",
                    "standard_name": "mass_concentration_of_chlorophyll_in_sea_water",
                },
            )
        },
        coords={
            "time": ("time", time, {"units": "days since 2020-01-01", "axis": "T"}),
            "lat": ("lat", lat, {"units": "degrees_north", "axis": "Y", "standard_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "axis": "X", "standard_name": "longitude"}),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    return ds


def make_bad_time_units() -> xr.Dataset:
    """Time coordinate with completely mangled / nonsensical unit strings.

    - 'var_time': unit is 'parsecs since the Big Bang'  (absurd physical unit)
    - 'rel_time': unit is 'days'  (missing 'since <epoch>' part)
    - 'neg_time': unit is 'hours since -9999-01-01'  (implausible epoch)
    - 'cal_time': calendar attribute set to 'martian'  (invalid calendar name)
    """
    n = 10
    time_vals = np.arange(n, dtype=float)

    ds = xr.Dataset(
        data_vars={
            "sst": (
                ("time",),
                np.ones(n, dtype="float32") * 298.0,
                {"long_name": "sea surface temperature", "units": "K"},
            ),
            "sss": (
                ("rel_time",),
                np.ones(n, dtype="float32") * 35.0,
                {"long_name": "sea surface salinity", "units": "psu"},
            ),
            "ssh": (
                ("neg_time",),
                np.zeros(n, dtype="float32"),
                {"long_name": "sea surface height", "units": "m"},
            ),
            "ice": (
                ("cal_time",),
                np.zeros(n, dtype="float32"),
                {"long_name": "sea ice concentration", "units": "1"},
            ),
        },
        coords={
            "time": (
                "time",
                time_vals,
                {"units": "parsecs since the Big Bang", "axis": "T"},
            ),
            "rel_time": (
                "rel_time",
                time_vals,
                {"units": "days", "axis": "T"},  # missing 'since <epoch>'
            ),
            "neg_time": (
                "neg_time",
                time_vals,
                {"units": "hours since -9999-01-01", "axis": "T"},
            ),
            "cal_time": (
                "cal_time",
                time_vals,
                {
                    "units": "days since 2000-01-01",
                    "calendar": "martian",          # not a valid CF calendar
                    "axis": "T",
                },
            ),
        },
        attrs={"Conventions": "CF-1.12"},
    )
    return ds


def make_kitchen_sink() -> xr.Dataset:
    """Everything wrong at once.

    - latitude descending (upside down)
    - longitude 0→360 but labelled -180→180 AND units 'degrees_west'
    - variable named 'Q' with no standard_name, no long_name, no units
    - dimensions in wrong order: (lon, lat, time)
    - massive missing longitude band (90°–180°)
    - time unit is 'fortnights since 1752-09-14'
    - no Conventions attribute
    - no global title/institution
    """
    n_time, n_lat, n_lon = 6, 37, 72
    lat = np.linspace(90.0, -90.0, n_lat)    # upside-down
    lon = np.linspace(-180.0, 177.5, n_lon)  # labelled -180→180, but data is offset
    time = np.arange(n_time, dtype=float)

    rng = np.random.default_rng(99)
    # Wrong dim order: (lon, lat, time)
    data = rng.standard_normal((n_lon, n_lat, n_time)).astype("float32")

    # Big missing band: lon -90°–0° (should be a quarter of the globe)
    lon_mask = (lon >= -90.0) & (lon <= 0.0)
    data[lon_mask, :, :] = np.nan

    ds = xr.Dataset(
        data_vars={
            "Q": (
                ("lon", "lat", "time"),  # wrong order
                data,
                # no units, no long_name, no standard_name
            )
        },
        coords={
            "time": (
                "time",
                time,
                {"units": "fortnights since 1752-09-14", "axis": "T"},
            ),
            "lat": (
                "lat",
                lat,   # descending
                {
                    "units": "degrees_north",  # correct unit, wrong order
                    "axis": "Y",
                    "standard_name": "latitude",
                },
            ),
            "lon": (
                "lon",
                lon,
                {
                    "units": "degrees_west",   # invalid
                    "axis": "X",
                    "standard_name": "longitude",
                },
            ),
        },
        # No Conventions, no title, no institution
    )
    return ds


def main() -> None:
    fixtures = {
        "lat_upside_down.nc": make_lat_upside_down(),
        "lon_offset_wrong_labels.nc": make_lon_offset_wrong_labels(),
        "bad_var_names.nc": make_bad_var_names(),
        "wrong_dim_order.nc": make_wrong_dim_order(),
        "missing_units.nc": make_missing_units(),
        "missing_lon_slices.nc": make_missing_lon_slices(),
        "bad_time_units.nc": make_bad_time_units(),
        "kitchen_sink.nc": make_kitchen_sink(),
    }

    for name, ds in fixtures.items():
        path = OUT_DIR / name
        ds.to_netcdf(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
