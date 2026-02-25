import pytest
import xarray as xr
from pydantic import ValidationError

from nc_check.core import Check, CheckInfo, CheckResult, CheckStatus


def test_check_result_requires_info_message_for_warn_error_fatal() -> None:
    with pytest.raises(ValidationError, match="info.message is required"):
        CheckResult(check_id="custom.warn", status="warn", info={"message": ""})


def test_check_result_as_report_dict_has_status_and_info() -> None:
    result = CheckResult(
        check_id="custom.missing_units",
        status="error",
        info={"message": "Variable is missing units", "code": "CUSTOM001"},
    )

    report_item = result.as_report_dict()

    assert report_item["status"] == "error"
    assert report_item["info"]["message"] == "Variable is missing units"
    assert report_item["info"]["code"] == "CUSTOM001"


class _WarnCheck(Check):
    id = "custom.warn"
    description = "Warn for testing"

    def check(self, ds: xr.Dataset) -> CheckResult:
        return CheckResult(
            check_id=self.id,
            status=CheckStatus.warn,
            info=CheckInfo(message="warn"),
            fixable=True,
        )


def test_default_fix_returns_info_and_dataset() -> None:
    ds = xr.Dataset(data_vars={"v": (("x",), [1.0, 2.0])})
    check = _WarnCheck()

    result = check.check(ds)
    fix_result = check.fix(ds, result=result)
    info, fixed_ds = fix_result.as_tuple()

    assert fix_result.applied is False
    assert info["message"] == "No fix implemented for this check."
    assert fixed_ds.identical(ds)
