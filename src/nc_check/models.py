from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(str, Enum):
    skipped = "skipped"
    passed = "passed"
    failed = "failed"


@dataclass(frozen=True)
class AtomicCheckResult:
    name: str
    status: CheckStatus
    info: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def skipped_result(
        cls,
        *,
        name: str,
        info: str,
        details: dict[str, Any] | None = None,
    ) -> AtomicCheckResult:
        return cls(
            name=name,
            status=CheckStatus.skipped,
            info=info,
            details={} if details is None else dict(details),
        )

    @classmethod
    def passed_result(
        cls,
        *,
        name: str,
        info: str,
        details: dict[str, Any] | None = None,
    ) -> AtomicCheckResult:
        return cls(
            name=name,
            status=CheckStatus.passed,
            info=info,
            details={} if details is None else dict(details),
        )

    @classmethod
    def failed_result(
        cls,
        *,
        name: str,
        info: str,
        details: dict[str, Any] | None = None,
    ) -> AtomicCheckResult:
        return cls(
            name=name,
            status=CheckStatus.failed,
            info=info,
            details={} if details is None else dict(details),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "info": self.info,
            "details": self.details,
        }


@dataclass(frozen=True)
class SuiteSummary:
    checks_run: int
    passed: int
    skipped: int
    failed: int
    overall_status: CheckStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks_run": self.checks_run,
            "passed": self.passed,
            "skipped": self.skipped,
            "failed": self.failed,
            "overall_status": self.overall_status.value,
        }


@dataclass(frozen=True)
class SuiteReport:
    suite_name: str
    plugin: str | None
    checks: list[AtomicCheckResult]
    summary: SuiteSummary
    results: dict[str, dict[str, dict[str, AtomicCheckResult]]] = field(
        default_factory=dict
    )
    dataset_html: str | None = None

    def to_dict(self) -> dict[str, Any]:
        structured = {
            data_scope: {
                variable: {
                    check_name: check_result.to_dict()
                    for check_name, check_result in checks_by_name.items()
                }
                for variable, checks_by_name in variables.items()
            }
            for data_scope, variables in self.results.items()
        }
        return {
            "suite_name": self.suite_name,
            "plugin": self.plugin,
            "checks": [item.to_dict() for item in self.checks],
            "results": structured,
            "dataset_html": self.dataset_html,
            "summary": self.summary.to_dict(),
        }
