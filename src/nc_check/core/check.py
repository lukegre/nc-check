from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CheckStatus(str, Enum):
    passed = "passed"
    skipped = "skipped"
    warn = "warn"
    error = "error"
    fatal = "fatal"


class CheckInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    suggested_fix: str | None = None
    code: str | None = None
    exception: str | None = None


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1)
    status: CheckStatus
    info: CheckInfo = Field(default_factory=CheckInfo)
    fixable: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> CheckResult:
        actionable_statuses = {CheckStatus.warn, CheckStatus.error, CheckStatus.fatal}
        if self.status in actionable_statuses and not self.info.message.strip():
            raise ValueError("info.message is required when status is warn/error/fatal")
        if self.status == CheckStatus.passed and self.info.exception is not None:
            raise ValueError("passed checks cannot include info.exception")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "info": self.info.model_dump(mode="json", exclude_none=True),
        }


class FixResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    check_id: str = Field(min_length=1)
    applied: bool
    info: CheckInfo
    dataset: xr.Dataset = Field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "applied": self.applied,
            "info": self.info.model_dump(mode="json", exclude_none=True),
            "dataset": self.dataset,
        }

    def as_tuple(self) -> tuple[dict[str, Any], xr.Dataset]:
        return self.info.model_dump(mode="json", exclude_none=True), self.dataset


class Check(ABC):
    id: str
    description: str
    tags: tuple[str, ...] = ()

    @abstractmethod
    def check(self, ds: xr.Dataset) -> CheckResult:
        raise NotImplementedError

    def fix(self, ds: xr.Dataset, result: CheckResult | None = None) -> FixResult:
        return FixResult(
            check_id=self.id,
            applied=False,
            info=CheckInfo(message="No fix implemented for this check."),
            dataset=ds,
        )

    def run(
        self, ds: xr.Dataset, *, apply_fix: bool = False
    ) -> tuple[CheckResult, xr.Dataset, FixResult | None]:
        result = self.check(ds)
        if not apply_fix:
            return result, ds, None
        if result.status not in {CheckStatus.warn, CheckStatus.error}:
            return result, ds, None
        fix_result = self.fix(ds.copy(deep=True), result=result)
        return result, fix_result.dataset, fix_result
