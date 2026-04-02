from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

SummaryStatus = Literal["pass", "fail", "warn"]


class CheckSummaryItem(TypedDict):
    check: str
    status: SummaryStatus
    detail: str


class AtomicCheckItem(TypedDict):
    id: str
    name: str
    group: str
    status: str
    detail: str
    result: NotRequired[dict[str, Any]]
    variable: NotRequired[str]


class GroupSummaryItem(TypedDict):
    checks_run: int
    failing_checks: int
    warnings_or_skips: int
    status: SummaryStatus
    overall_ok: bool


class SuiteReport(TypedDict):
    checks_enabled: dict[str, bool]
    checks: list[AtomicCheckItem]
    groups: dict[str, GroupSummaryItem]
    reports: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    check_summary: NotRequired[list[CheckSummaryItem]]
