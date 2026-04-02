"""Utilities to coerce xarray datasets toward CF-1.12 compliance."""

# Legacy API (backwards-compatible)
from .accessor import CFCoercerAccessor
from .core import (
    CF_STANDARD_NAME_TABLE_URL,
    CF_VERSION,
    ComplianceEngine,
    check_dataset_compliant,
    make_dataset_compliant,
)
from .checks.ocean import check_ocean_cover
from .checks.time_cover import check_time_cover

# New plugin-based API
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
from . import suite  # noqa: F401

__all__ = [
    # Legacy exports
    "CFCoercerAccessor",
    "CF_STANDARD_NAME_TABLE_URL",
    "CF_VERSION",
    "ComplianceEngine",
    "check_dataset_compliant",
    "check_ocean_cover",
    "check_time_cover",
    "make_dataset_compliant",
    # New plugin-based exports
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
    "gcb_ocean_dataprod_suite",
    "cfchecker_report_suite",
    "render_html_report",
    "report_to_dict",
    "save_html_report",
    "config",
    "suite",
]
