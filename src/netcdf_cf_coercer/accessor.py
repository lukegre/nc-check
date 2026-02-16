from __future__ import annotations

from typing import Any

import xarray as xr

from .core import check_dataset_compliant, make_dataset_compliant


@xr.register_dataset_accessor("cf")
class CFCoercerAccessor:
    """Dataset-level CF helpers.

    Methods:
    - ``check_compliant()``: inspect CF-1.12 metadata issues.
    - ``make_compliant()``: return dataset with safe, automatic fixes applied.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._ds = xarray_obj

    def check_compliant(self, **kwargs: Any) -> dict[str, Any]:
        """Return a dictionary describing non-compliant or likely non-compliant items."""
        return check_dataset_compliant(self._ds, **kwargs)

    def make_compliant(self) -> xr.Dataset:
        """Return a copy of the dataset with safe CF-1.12 compliance fixes."""
        return make_dataset_compliant(self._ds)
