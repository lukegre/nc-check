from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import xarray as xr

from .core import (
    CF_STANDARD_NAME_TABLE_URL,
    ComplianceEngine,
    StandardNameDomain,
    check_dataset_compliant,
    make_dataset_compliant,
)
from .formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    print_pretty_full_report,
    render_pretty_full_report_html,
    save_html_report,
)
from .engine.runner import run_suite_checks
from .checks.ocean import (
    check_ocean_cover as run_ocean_cover_check,
)
from .checks.time_cover import check_time_cover as run_time_cover_check

_WRAPS_ASSIGNED = ("__module__", "__name__", "__qualname__", "__annotations__")


@xr.register_dataset_accessor("check")
class CFCoercerAccessor:
    """Dataset-level API exposed as ``ds.check`` on any ``xarray.Dataset``.

    This accessor groups CF metadata checks and safe auto-fixes:
    - ``compliance()`` for CF/Ferret convention validation.
    - ``make_cf_compliant()`` for non-destructive metadata normalization.
    - ``ocean_cover()`` and ``time_cover()`` for coverage-oriented QA checks.
    - ``all()`` to run several checks and return one combined report.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._ds = xarray_obj

    @wraps(check_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def compliance(
        self,
        *,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: StandardNameDomain | None = None,
        fallback_to_heuristic: bool = True,
        engine: ComplianceEngine = "auto",
        conventions: str | list[str] | tuple[str, ...] | None = None,
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Run CF/Ferret compliance checks for this dataset.

        Parameters
        ----------
        cf_version
            CF version passed to the checker (for example ``"1.12"``).
        standard_name_table_xml, cf_area_types_xml, cf_region_names_xml
            Optional local/remote XML resources used by ``cfchecker``.
        cache_tables
            Reuse downloaded checker tables between runs.
        domain
            ``Literal["ocean", "atmosphere", "land", "cryosphere",
            "biogeochemistry"] | None`` to bias standard-name suggestions.
        fallback_to_heuristic
            If ``True``, use built-in heuristic checks when ``cfchecker`` cannot run.
        engine
            ``Literal["auto", "cfchecker", "heuristic"]``.
        conventions
            Convention set to enforce (``"cf"``, ``"ferret"``, or both).
        report_format
            ``Literal["auto", "python", "tables", "html"]``.
        report_html_file
            Output file path when ``report_format="html"``.

        Returns
        -------
        dict | str | None
            A Python report dictionary for ``"python"``, HTML for ``"html"``,
            or ``None`` for ``"tables"`` (printed output).
        """
        return check_dataset_compliant(
            self._ds,
            cf_version=cf_version,
            standard_name_table_xml=standard_name_table_xml,
            cf_area_types_xml=cf_area_types_xml,
            cf_region_names_xml=cf_region_names_xml,
            cache_tables=cache_tables,
            domain=domain,
            fallback_to_heuristic=fallback_to_heuristic,
            engine=engine,
            conventions=conventions,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    @wraps(make_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def make_cf_compliant(self) -> xr.Dataset:
        """Return a copied dataset with safe CF-1.12 metadata fixes applied.

        This method is non-destructive: the original dataset is unchanged.
        """
        return make_dataset_compliant(self._ds)

    @wraps(run_ocean_cover_check, assigned=_WRAPS_ASSIGNED)
    def ocean_cover(
        self,
        *,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Run ocean-coverage sanity checks for gridded variables.

        Parameters
        ----------
        var_name
            Optional variable to check. If omitted, all compatible variables are used.
        lon_name, lat_name, time_name
            Coordinate names to use. Longitude/latitude can be inferred.
        check_edge_of_map
            Detect persistent missing longitude bands at map edges.
        check_land_ocean_offset
            Compare expected land/ocean reference points on global grids.
        report_format
            ``"python"``, ``"tables"``, ``"html"``, or ``"auto"``.
        report_html_file
            Output file path when ``report_format="html"``.

        Returns
        -------
        dict | str | None
            A Python report dictionary for ``"python"``, HTML for ``"html"``,
            or ``None`` for ``"tables"`` (printed output).
        """
        return run_ocean_cover_check(
            self._ds,
            var_name=var_name,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            check_edge_of_map=check_edge_of_map,
            check_land_ocean_offset=check_land_ocean_offset,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    @wraps(run_time_cover_check, assigned=_WRAPS_ASSIGNED)
    def time_cover(
        self,
        *,
        var_name: str | None = None,
        time_name: str | None = "time",
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Run time-coverage checks and report missing time-slice ranges.

        Parameters
        ----------
        var_name
            Optional variable to check. If omitted, checks all data variables.
        time_name
            Preferred name of the time dimension/coordinate.
        report_format
            ``"python"``, ``"tables"``, ``"html"``, or ``"auto"``.
        report_html_file
            Output file path when ``report_format="html"``.

        Returns
        -------
        dict | str | None
            A Python report dictionary for ``"python"``, HTML for ``"html"``,
            or ``None`` for ``"tables"`` (printed output).
        """
        return run_time_cover_check(
            self._ds,
            var_name=var_name,
            time_name=time_name,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    def all(
        self,
        *,
        compliance: bool = True,
        ocean_cover: bool = True,
        time_cover: bool = True,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: StandardNameDomain | None = None,
        fallback_to_heuristic: bool = True,
        engine: ComplianceEngine = "auto",
        conventions: str | list[str] | tuple[str, ...] | None = None,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Run selected checks and return one combined report.

        Parameters
        ----------
        compliance, ocean_cover, time_cover
            Enable/disable each check family. At least one must be ``True``.
        cf_version, standard_name_table_xml, cf_area_types_xml, cf_region_names_xml
            Forwarded to :meth:`compliance` when enabled.
        cache_tables, domain, fallback_to_heuristic, engine, conventions
            Forwarded to :meth:`compliance` when enabled.
        var_name, lon_name, lat_name, time_name
            Forwarded to coverage checks when enabled.
        check_edge_of_map, check_land_ocean_offset
            Forwarded to :meth:`ocean_cover` when enabled.
        report_format
            ``"python"``, ``"tables"``, ``"html"``, or ``"auto"``.
        report_html_file
            Output file path when ``report_format="html"``.

        Returns
        -------
        dict | str | None
            Combined report as a dictionary, rendered HTML, or ``None`` for tables.
        """
        resolved_format = normalize_report_format(report_format)
        if report_html_file is not None and resolved_format != "html":
            raise ValueError(
                "`report_html_file` is only valid when report_format='html'."
            )

        enabled = {
            "compliance": bool(compliance),
            "ocean_cover": bool(ocean_cover),
            "time_cover": bool(time_cover),
        }
        full_report = run_suite_checks(
            self._ds,
            checks_enabled=enabled,
            options_by_check={
                "compliance": {
                    "cf_version": cf_version,
                    "standard_name_table_xml": standard_name_table_xml,
                    "cf_area_types_xml": cf_area_types_xml,
                    "cf_region_names_xml": cf_region_names_xml,
                    "cache_tables": cache_tables,
                    "domain": domain,
                    "fallback_to_heuristic": fallback_to_heuristic,
                    "engine": engine,
                    "conventions": conventions,
                },
                "ocean_cover": {
                    "var_name": var_name,
                    "lon_name": lon_name,
                    "lat_name": lat_name,
                    "time_name": time_name,
                    "check_edge_of_map": check_edge_of_map,
                    "check_land_ocean_offset": check_land_ocean_offset,
                },
                "time_cover": {
                    "var_name": var_name,
                    "time_name": time_name,
                },
            },
        )

        if resolved_format == "python":
            return full_report
        if resolved_format == "tables":
            print_pretty_full_report(full_report)
            return None

        html_report = render_pretty_full_report_html(full_report)
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report
