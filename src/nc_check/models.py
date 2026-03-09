from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from typing import Any

from IPython.display import HTML


class CheckStatus(str, Enum):
    skipped = "skipped"
    passed = "passed"
    warning = "warning"
    failed = "failed"
    fatal = "fatal"


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

    @classmethod
    def warn_result(
        cls,
        *,
        name: str,
        info: str,
        details: dict[str, Any] | None = None,
    ) -> AtomicCheckResult:
        return cls(
            name=name,
            status=CheckStatus.warning,
            info=info,
            details={} if details is None else dict(details),
        )

    @classmethod
    def fatal_result(
        cls,
        *,
        name: str,
        info: str,
        details: dict[str, Any] | None = None,
    ) -> AtomicCheckResult:
        return cls(
            name=name,
            status=CheckStatus.fatal,
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
    warnings: int
    failed: int
    fatal: int
    overall_status: CheckStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks_run": self.checks_run,
            "passed": self.passed,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "failed": self.failed,
            "fatal": self.fatal,
            "overall_status": self.overall_status.value,
        }


@dataclass(frozen=True)
class SuiteReport:
    suite_name: str
    checks: list[AtomicCheckResult]
    summary: SuiteSummary
    results: dict[str, dict[str, dict[str, AtomicCheckResult]]] = field(
        default_factory=dict
    )
    source_file: str | None = None
    _dataset_html: str | None = None

    def to_json(self, report_fname: str | PathLike[str] | None = None) -> None | str:
        import json

        json_str = json.dumps(self.to_dict(), indent=2)
        if report_fname:
            with open(report_fname, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def to_html(self, report_fname: str | PathLike[str] | None = None) -> None | HTML:
        from .reporting import render_html_report

        html = render_html_report(self)
        if report_fname:
            with open(report_fname, "w", encoding="utf-8") as f:
                f.write(html)
        else:
            return HTML(html)

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
            "checks": [item.to_dict() for item in self.checks],
            "results": structured,
            "summary": self.summary.to_dict(),
            "source_file": self.source_file,
        }

    def __repr__(self) -> str:
        return f"SuiteReport(suite_name={self.suite_name}, checks={len(self.checks)}, summary={self.summary})"
