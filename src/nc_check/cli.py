from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import xarray as xr

from . import __version__
from . import accessor as _accessor  # noqa: F401  # register Dataset.check accessor
from . import formatting
from .checks.heuristic import HeuristicCheck
from .checks.ocean import check_ocean_cover
from .checks.time_cover import check_time_cover
from .core import check_dataset_compliant, make_dataset_compliant

_CHECK_MODES = {"compliance", "ocean-cover", "time-cover", "all"}


def _existing_file(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return candidate


def _normalize_check_argv(argv: list[str] | None) -> list[str]:
    """Support `nc-check <file>` as shorthand for `nc-check compliance <file>`.

    Skips leading option flags so that e.g. `nc-check --no-fail input.nc`
    is correctly normalised to `nc-check --no-fail compliance input.nc`.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(raw):
        if not arg.startswith("-"):
            if arg not in _CHECK_MODES:
                return raw[:i] + ["compliance"] + raw[i:]
            break
    return raw


def _build_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nc-check",
        description=(
            "Run NetCDF checks with git-style subcommands.\n"
            "Use `nc-check <command> --help` for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nc-check compliance input.nc\n"
            "  nc-check ocean-cover input.nc\n"
            "  nc-check ocean-cover input.nc --lon-name x --lat-name y --time-name t\n"
            "  nc-check time-cover input.nc\n"
            "  nc-check time-cover input.nc --time-name t\n"
            "  nc-check all input.nc --save-report\n"
            "  nc-check all input.nc --lon-name x --lat-name y --time-name t\n"
            "  nc-check input.nc   # shorthand for `nc-check compliance input.nc`"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"nc-check {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, title="Commands")

    def _add_shared_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "fname", type=_existing_file, help="Input NetCDF file"
        )
        command_parser.add_argument(
            "--save-report",
            action="store_true",
            help=(
                "Save an HTML report next to the input file. Compliance uses "
                "'<input>_report.html'; ocean-cover/time-cover use "
                "'<input>_<command>_report.html'; all uses "
                "'<input>_all_report.html' (single combined report)."
            ),
        )
        command_parser.add_argument(
            "--no-fail",
            action="store_true",
            default=False,
            help=(
                "Exit 0 even when checks find failures. "
                "Useful for CI pipelines that run nc-check for reporting only."
            ),
        )

    compliance = subparsers.add_parser(
        "compliance",
        help="Run CF/Ferret compliance checks.",
    )
    _add_shared_options(compliance)
    compliance.add_argument(
        "--conventions",
        default="cf,ferret",
        help="Comma-separated conventions to check (default: cf,ferret).",
    )
    compliance.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "cfchecker", "heuristic"),
        help="Compliance engine to use (default: auto).",
    )

    ocean_cover = subparsers.add_parser(
        "ocean-cover",
        help="Run ocean-coverage checks.",
    )
    _add_shared_options(ocean_cover)
    ocean_cover.add_argument(
        "--lon-name",
        default=None,
        help="Explicit longitude coordinate name (default: inferred).",
    )
    ocean_cover.add_argument(
        "--lat-name",
        default=None,
        help="Explicit latitude coordinate name (default: inferred).",
    )
    ocean_cover.add_argument(
        "--time-name",
        default="time",
        help="Time coordinate/dimension name for time-aware checks (default: time).",
    )
    ocean_cover.add_argument(
        "--check-lon-0-360",
        action="store_true",
        help="Enable longitude convention check for [0, 360].",
    )
    ocean_cover.add_argument(
        "--check-lon-neg180-180",
        action="store_true",
        help="Enable longitude convention check for [-180, 180].",
    )

    time_cover = subparsers.add_parser(
        "time-cover",
        help="Run time-coverage checks.",
    )
    _add_shared_options(time_cover)
    time_cover.add_argument(
        "--time-name",
        default="time",
        help="Explicit time coordinate/dimension name (default: time).",
    )
    time_cover.add_argument(
        "--check-time-monotonic",
        action="store_true",
        help="Enable monotonic-time-order check.",
    )
    time_cover.add_argument(
        "--check-time-regular-spacing",
        action="store_true",
        help="Enable regular-time-spacing check.",
    )

    check_all = subparsers.add_parser(
        "all",
        help="Run compliance, ocean-cover, and time-cover checks.",
    )
    _add_shared_options(check_all)
    check_all.add_argument(
        "--conventions",
        default="cf,ferret",
        help="Comma-separated conventions to check for compliance (default: cf,ferret).",
    )
    check_all.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "cfchecker", "heuristic"),
        help="Compliance engine to use for the compliance step (default: auto).",
    )
    check_all.add_argument(
        "--lon-name",
        default=None,
        help="Explicit longitude coordinate name for ocean checks (default: inferred).",
    )
    check_all.add_argument(
        "--lat-name",
        default=None,
        help="Explicit latitude coordinate name for ocean checks (default: inferred).",
    )
    check_all.add_argument(
        "--time-name",
        default="time",
        help="Explicit time coordinate/dimension name for ocean/time checks (default: time).",
    )
    check_all.add_argument(
        "--check-lon-0-360",
        action="store_true",
        help="Enable longitude convention check for [0, 360] in ocean checks.",
    )
    check_all.add_argument(
        "--check-lon-neg180-180",
        action="store_true",
        help="Enable longitude convention check for [-180, 180] in ocean checks.",
    )
    check_all.add_argument(
        "--check-time-monotonic",
        action="store_true",
        help="Enable monotonic-time-order check in time checks.",
    )
    check_all.add_argument(
        "--check-time-regular-spacing",
        action="store_true",
        help="Enable regular-time-spacing check in time checks.",
    )

    return parser


def _report_failed(mode: str, report: Any) -> bool:
    """Return True if the report indicates one or more check failures."""
    if not isinstance(report, dict):
        return False
    if mode == "compliance":
        counts = report.get("counts") or {}
        return int(counts.get("fatal", 0) or 0) + int(counts.get("error", 0) or 0) > 0
    if mode in ("ocean-cover", "time-cover"):
        return not bool(report.get("ok", True))
    if mode == "all":
        return not bool((report.get("summary") or {}).get("overall_ok", True))
    return False


def _render_check_report(
    mode: str,
    report: dict[str, Any],
    report_format: str,
    report_html_file: Path | None,
) -> None:
    """Render a collected report dict in the requested format."""
    if report_format == "tables":
        if mode == "compliance":
            formatting.print_pretty_report(report)
        elif mode == "ocean-cover":
            if report.get("mode") == "all_variables":
                formatting.print_pretty_ocean_reports(list(report["reports"].values()))
            else:
                formatting.print_pretty_ocean_report(report)
        elif mode == "time-cover":
            if report.get("mode") == "all_variables":
                formatting.print_pretty_time_cover_reports(
                    list(report["reports"].values())
                )
            else:
                formatting.print_pretty_time_cover_report(report)
        elif mode == "all":
            formatting.print_pretty_full_report(report)
    elif report_format == "html":
        if mode == "compliance":
            html = formatting.render_pretty_report_html(report)
        elif mode == "ocean-cover":
            if report.get("mode") == "all_variables":
                html = formatting.render_pretty_ocean_reports_html(
                    list(report["reports"].values())
                )
            else:
                html = formatting.render_pretty_ocean_report_html(report)
        elif mode == "time-cover":
            if report.get("mode") == "all_variables":
                html = formatting.render_pretty_time_cover_reports_html(
                    list(report["reports"].values())
                )
            else:
                html = formatting.render_pretty_time_cover_report_html(report)
        elif mode == "all":
            html = formatting.render_pretty_full_report_html(report)
        else:
            return
        formatting.save_html_report(html, report_html_file)
        formatting.maybe_display_html_report(html)


def run_check(argv: list[str] | None = None) -> int:
    parser = _build_check_parser()
    args = parser.parse_args(_normalize_check_argv(argv))

    mode = str(args.command)
    input_file: Path = args.fname
    no_fail: bool = bool(getattr(args, "no_fail", False))

    report_format = "html" if args.save_report else "tables"
    report_html_file = (
        _default_report_html_path(input_file, mode) if args.save_report else None
    )
    conventions = getattr(args, "conventions", "cf,ferret")
    engine = getattr(args, "engine", "auto")
    lon_name = getattr(args, "lon_name", None)
    lat_name = getattr(args, "lat_name", None)
    time_name = getattr(args, "time_name", "time")
    check_lon_0_360 = bool(getattr(args, "check_lon_0_360", False))
    check_lon_neg180_180 = bool(getattr(args, "check_lon_neg180_180", False))
    check_time_monotonic = bool(getattr(args, "check_time_monotonic", False))
    check_time_regular_spacing = bool(
        getattr(args, "check_time_regular_spacing", False)
    )

    try:
        with xr.open_dataset(input_file, chunks={}) as ds:
            if mode == "compliance":
                result = check_dataset_compliant(
                    ds,
                    conventions=conventions,
                    engine=engine,
                    report_format="python",
                )
                report: dict[str, Any] = result if isinstance(result, dict) else {}
            elif mode == "ocean-cover":
                result = check_ocean_cover(
                    ds,
                    lon_name=lon_name,
                    lat_name=lat_name,
                    time_name=time_name,
                    check_lon_0_360=check_lon_0_360,
                    check_lon_neg180_180=check_lon_neg180_180,
                    report_format="python",
                )
                report = result if isinstance(result, dict) else {}
            elif mode == "time-cover":
                result = check_time_cover(
                    ds,
                    time_name=time_name,
                    check_time_monotonic=check_time_monotonic,
                    check_time_regular_spacing=check_time_regular_spacing,
                    report_format="python",
                )
                report = result if isinstance(result, dict) else {}
            elif mode == "all":
                report = _run_all_checks(
                    ds,
                    conventions=conventions,
                    engine=engine,
                    lon_name=lon_name,
                    lat_name=lat_name,
                    time_name=time_name,
                    check_lon_0_360=check_lon_0_360,
                    check_lon_neg180_180=check_lon_neg180_180,
                    check_time_monotonic=check_time_monotonic,
                    check_time_regular_spacing=check_time_regular_spacing,
                )
            else:
                parser.error(f"Unsupported mode: {mode}")
                return 1

            _render_check_report(mode, report, report_format, report_html_file)

    except Exception as exc:
        print(f"nc-check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not no_fail and _report_failed(mode, report):
        return 1
    return 0


def _default_report_html_path(input_file: Path, mode: str = "compliance") -> Path:
    name = input_file.name
    if name.lower().endswith(".nc"):
        stem = name[:-3]
    else:
        stem = input_file.stem

    if mode == "compliance":
        suffix = "_report"
    else:
        suffix = f"_{mode.replace('-', '_')}_report"
    report_name = f"{stem}{suffix}.html"
    return input_file.with_name(report_name)


def _run_all_checks(
    ds: xr.Dataset,
    *,
    conventions: str | list[str] | tuple[str, ...] | None,
    engine: str,
    lon_name: str | None,
    lat_name: str | None,
    time_name: str | None,
    check_lon_0_360: bool,
    check_lon_neg180_180: bool,
    check_time_monotonic: bool,
    check_time_regular_spacing: bool,
) -> dict[str, Any]:
    result = ds.check.all(
        conventions=conventions,
        engine=engine,
        lon_name=lon_name,
        lat_name=lat_name,
        time_name=time_name,
        check_lon_0_360=check_lon_0_360,
        check_lon_neg180_180=check_lon_neg180_180,
        check_time_monotonic=check_time_monotonic,
        check_time_regular_spacing=check_time_regular_spacing,
        report_format="python",
    )
    return result if isinstance(result, dict) else {}


def run_comply(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nc-comply",
        description="Apply safe CF compliance fixes and write a new NetCDF file.",
    )
    parser.add_argument("fname_in", type=_existing_file, help="Input NetCDF file")
    parser.add_argument("fname_out", type=Path, help="Output NetCDF file")
    args = parser.parse_args(argv)

    try:
        with xr.open_dataset(args.fname_in, chunks={}) as ds:
            check = HeuristicCheck(cf_version="CF-1.12")
            check_result = check.check(ds)
            fix_result = check.fix(ds, result=check_result)
            fix_result.dataset.to_netcdf(args.fname_out)

            n_unfixable = len(fix_result.unfixable_items)
            print(f"Fixed:     issues written to {args.fname_out}")
            if n_unfixable:
                print(
                    f"Unfixable: {n_unfixable} issue(s) require manual attention "
                    f"(run nc-check for details)"
                )
                return 1
    except Exception as exc:
        print(f"nc-comply: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main_check() -> None:
    raise SystemExit(run_check())


def main_comply() -> None:
    raise SystemExit(run_comply())
