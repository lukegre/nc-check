from __future__ import annotations

from nc_check.engine.suite import Suite, SuiteCheck


def test_suite_runs_list_of_atomic_checks() -> None:
    suite = Suite(
        name="example_suite",
        checks=[
            SuiteCheck(
                check_id="example.pass",
                name="Pass Check",
                run=lambda: {"status": "pass"},
                detail=lambda result: f"status={result['status']}",
            ),
            SuiteCheck(
                check_id="example.skip",
                name="Skip Check",
                run=lambda: {"status": "skipped_no_data"},
                detail=lambda result: f"status={result['status']}",
            ),
            SuiteCheck(
                check_id="example.fail",
                name="Fail Check",
                run=lambda: {"status": "fail", "count": 2},
                detail=lambda result: f"count={result['count']}",
            ),
        ],
    )

    report = suite.run()

    assert report["suite"] == "example_suite"
    assert report["summary"]["checks_run"] == 3
    assert report["summary"]["failing_checks"] == 1
    assert report["summary"]["warnings_or_skips"] == 1
    assert report["summary"]["overall_status"] == "fail"
    assert report["ok"] is False
    assert [item["id"] for item in report["checks"]] == [
        "example.pass",
        "example.skip",
        "example.fail",
    ]
