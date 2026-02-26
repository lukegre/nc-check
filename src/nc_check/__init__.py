"""Plugin-based dataset check framework for canonical time/lat/lon xarray datasets."""

from .api import canonicalize_dataset, create_registry, run_cf_compliance, run_suite
from .dataset import CanonicalDataset
from .models import AtomicCheckResult, CheckStatus, SuiteReport
from .plugins import CFCompliancePlugin, CheckRegistry, cf_check_names
from .reporting import render_html_report, report_to_dict, save_html_report
from .suite import CheckDefinition, CheckSuite, run_atomic_check

__all__ = [
    "AtomicCheckResult",
    "CFCompliancePlugin",
    "CanonicalDataset",
    "CheckDefinition",
    "CheckRegistry",
    "CheckStatus",
    "CheckSuite",
    "SuiteReport",
    "canonicalize_dataset",
    "cf_check_names",
    "create_registry",
    "render_html_report",
    "report_to_dict",
    "run_atomic_check",
    "run_cf_compliance",
    "run_suite",
    "save_html_report",
]
