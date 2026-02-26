from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .models import SuiteReport


def report_to_dict(report: SuiteReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, SuiteReport):
        return report.to_dict()
    return report


def render_html_report(report: SuiteReport | dict[str, Any]) -> str:
    payload = report_to_dict(report)
    suite_name = escape(str(payload.get("suite_name", "suite")))
    plugin = payload.get("plugin")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []

    header_rows = [
        ("Overall status", str(summary.get("overall_status", "unknown"))),
        ("Checks run", str(summary.get("checks_run", 0))),
        ("Passed", str(summary.get("passed", 0))),
        ("Skipped", str(summary.get("skipped", 0))),
        ("Failed", str(summary.get("failed", 0))),
    ]
    if plugin is not None:
        header_rows.append(("Plugin", str(plugin)))

    summary_html = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"
        for key, value in header_rows
    )

    check_rows: list[str] = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = escape(str(item.get("name", "")))
        status = escape(str(item.get("status", "")))
        info = escape(str(item.get("info", "")))
        details = item.get("details")
        if isinstance(details, dict) and details:
            details_text = "<br>".join(
                f"<code>{escape(str(k))}</code>: {escape(str(v))}"
                for k, v in sorted(details.items())
            )
        else:
            details_text = ""

        check_rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{status}</td>"
            f"<td>{info}</td>"
            f"<td>{details_text}</td>"
            "</tr>"
        )

    check_rows_html = "".join(check_rows)

    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>nc-check report</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem;color:#111;}"
        "h1{margin:0 0 1rem 0;}"
        "table{border-collapse:collapse;width:100%;margin:0 0 1.5rem 0;}"
        "th,td{border:1px solid #d0d7de;padding:.5rem;vertical-align:top;text-align:left;}"
        "thead th{background:#f6f8fa;}"
        "code{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;}"
        "</style></head><body>"
        f"<h1>Suite: {suite_name}</h1>"
        "<h2>Summary</h2>"
        f"<table>{summary_html}</table>"
        "<h2>Checks</h2>"
        "<table><thead><tr><th>Name</th><th>Status</th><th>Info</th><th>Details</th></tr></thead>"
        f"<tbody>{check_rows_html}</tbody></table>"
        "</body></html>"
    )


def save_html_report(
    report: SuiteReport | dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html_report(report), encoding="utf-8")
    return output
