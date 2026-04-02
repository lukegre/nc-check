"""Plugin-based dataset check framework for canonical time/lat/lon xarray datasets."""

from .settings import config
from .dataset import CanonicalDataset
from .models import AtomicCheckResult, CheckStatus, SuiteReport
from .suite import (
    CheckSuite,
    CallableCheck,
    CallableFixCheck,
    FixableCheck,
    run_atomic_check,
)
from .plugins import (
    cf_compliance_suite,
    ocean_check_suite,
    time_cover_suite,
    gcb_ocean_dataprod_suite,
    cfchecker_report_suite,
)
from .reporting import render_html_report, report_to_dict, save_html_report
from . import accessor, suite  # noqa: F401


__all__ = [
    "CanonicalDataset",
    "AtomicCheckResult",
    "CheckStatus",
    "SuiteReport",
    "CheckSuite",
    "CallableCheck",
    "CallableFixCheck",
    "FixableCheck",
    "run_atomic_check",
    "cf_compliance_suite",
    "ocean_check_suite",
    "time_cover_suite",
    "render_html_report",
    "report_to_dict",
    "gcb_ocean_dataprod_suite",
    "cfchecker_report_suite",
    "save_html_report",
    "config",
    "suite",
]
