from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .dataset import CanonicalDataset
from .models import AtomicCheckResult, CheckStatus, SuiteReport, SuiteSummary

AtomicCheck = Callable[[CanonicalDataset], AtomicCheckResult]


@dataclass(frozen=True)
class CheckDefinition:
    name: str
    check: AtomicCheck
    plugin: str = "local"


def run_atomic_check(
    dataset: CanonicalDataset, definition: CheckDefinition
) -> AtomicCheckResult:
    try:
        result = definition.check(dataset)
    except Exception as exc:  # defensive wrapper for third-party checks
        return AtomicCheckResult.failed_result(
            name=definition.name,
            info=f"Check raised {type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__},
        )

    if not isinstance(result, AtomicCheckResult):
        return AtomicCheckResult.failed_result(
            name=definition.name,
            info="Check returned an invalid result type.",
            details={"expected": "AtomicCheckResult", "actual": type(result).__name__},
        )

    if result.name != definition.name:
        return AtomicCheckResult(
            name=definition.name,
            status=result.status,
            info=result.info,
            details={**result.details, "reported_name": result.name},
        )

    return result


class CheckSuite:
    def __init__(
        self,
        *,
        name: str,
        checks: Iterable[CheckDefinition],
        plugin: str | None = None,
    ) -> None:
        self.name = name
        self.checks = list(checks)
        self.plugin = plugin

    def run(self, dataset: CanonicalDataset) -> SuiteReport:
        results = [run_atomic_check(dataset, definition) for definition in self.checks]
        summary = _summary_from_results(results)
        return SuiteReport(
            suite_name=self.name,
            plugin=self.plugin,
            checks=results,
            summary=summary,
        )


def _summary_from_results(results: list[AtomicCheckResult]) -> SuiteSummary:
    passed = sum(1 for result in results if result.status == CheckStatus.passed)
    skipped = sum(1 for result in results if result.status == CheckStatus.skipped)
    failed = sum(1 for result in results if result.status == CheckStatus.failed)

    if failed > 0:
        overall = CheckStatus.failed
    elif passed > 0:
        overall = CheckStatus.passed
    else:
        overall = CheckStatus.skipped

    return SuiteSummary(
        checks_run=len(results),
        passed=passed,
        skipped=skipped,
        failed=failed,
        overall_status=overall,
    )
