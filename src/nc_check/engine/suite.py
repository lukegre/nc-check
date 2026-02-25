from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


RunAtomicCheck = Callable[[], dict[str, Any]]
CheckDetailFormatter = Callable[[dict[str, Any]], str]


def _status_kind(status: Any) -> str:
    normalized = str(status).strip().lower()
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return "fail"
    if normalized in {"warn", "warning"}:
        return "warn"
    if normalized in {"skip", "skipped"} or normalized.startswith("skip"):
        return "skip"
    return "pass"


def _summary_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = [_status_kind(item.get("status")) for item in checks]
    failing_checks = sum(1 for kind in kinds if kind == "fail")
    warning_checks = sum(1 for kind in kinds if kind == "warn")
    warnings_or_skips = sum(1 for kind in kinds if kind in {"warn", "skip"})
    if failing_checks > 0:
        overall_status = "fail"
    elif warning_checks > 0:
        overall_status = "warn"
    else:
        overall_status = "pass"
    return {
        "checks_run": len(checks),
        "failing_checks": failing_checks,
        "warnings_or_skips": warnings_or_skips,
        "overall_status": overall_status,
        "overall_ok": overall_status == "pass",
    }


@dataclass(frozen=True)
class SuiteCheck:
    check_id: str
    name: str
    run: RunAtomicCheck
    detail: CheckDetailFormatter | None = None

    def run_once(self) -> dict[str, Any]:
        result = self.run()
        detail = self.detail(result) if self.detail is not None else ""
        return {
            "id": self.check_id,
            "name": self.name,
            "status": str(result.get("status", "unknown")),
            "detail": detail,
            "result": result,
        }


class Suite:
    def __init__(self, name: str, checks: list[SuiteCheck]) -> None:
        self.name = name
        self.checks = checks

    def run(self) -> dict[str, Any]:
        items = [check.run_once() for check in self.checks]
        return self.report_from_items(self.name, items)

    @staticmethod
    def report_from_items(name: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
        summary = _summary_from_checks(checks)
        return {
            "suite": name,
            "checks": checks,
            "summary": summary,
            "ok": bool(summary["overall_ok"]),
        }
