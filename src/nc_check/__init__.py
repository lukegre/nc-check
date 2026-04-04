"""Utilities to coerce xarray datasets toward CF-1.12 compliance."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__: str = _pkg_version("nc-check")
except _PackageNotFoundError:
    __version__ = "unknown"

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

__all__ = [
    "__version__",
    "CFCoercerAccessor",
    "CF_STANDARD_NAME_TABLE_URL",
    "CF_VERSION",
    "ComplianceEngine",
    "check_dataset_compliant",
    "check_ocean_cover",
    "check_time_cover",
    "make_dataset_compliant",
]
