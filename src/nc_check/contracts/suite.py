from __future__ import annotations

from typing import Any, Literal, TypedDict

SummaryStatus = Literal["pass", "fail", "warn"]


class CheckSummaryItem(TypedDict):
    check: str
    status: SummaryStatus
    detail: str


class SuiteReport(TypedDict):
    checks_enabled: dict[str, bool]
    check_summary: list[CheckSummaryItem]
    reports: dict[str, dict[str, Any]]
    summary: dict[str, Any]
