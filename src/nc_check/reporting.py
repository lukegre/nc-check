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
    def _status_class(value: str) -> str:
        lookup = {
            "passed": "status-passed",
            "failed": "status-failed",
            "skipped": "status-skipped",
        }
        return lookup.get(value.strip().lower(), "status-unknown")

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

    summary_rows: list[str] = []
    for key, value in header_rows:
        value_text = str(value)
        if key == "Overall status":
            value_html = (
                f"<span class='status-badge {_status_class(value_text)}'>"
                f"{escape(value_text)}</span>"
            )
        else:
            value_html = f"<span class='summary-value'>{escape(value_text)}</span>"
        summary_rows.append(
            f"<tr><th scope='row'>{escape(key)}</th><td>{value_html}</td></tr>"
        )
    summary_html = "".join(summary_rows)

    check_rows: list[str] = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = escape(str(item.get("name", "")))
        raw_status = str(item.get("status", ""))
        status = escape(raw_status)
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
            f"<td><span class='status-badge {_status_class(raw_status)}'>{status}</span></td>"
            f"<td>{info}</td>"
            f"<td>{details_text}</td>"
            "</tr>"
        )

    check_rows_html = "".join(check_rows) or (
        "<tr><td colspan='4' class='empty-checks'>No checks were included.</td></tr>"
    )

    plugin_html = (
        f"<p class='meta'>Plugin: <strong>{escape(str(plugin))}</strong></p>"
        if plugin is not None
        else ""
    )

    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>nc-check report</title>"
        "<style>"
        ":root{"
        "--bg:#eaf2f8;"
        "--text:#123047;"
        "--muted:#4f677a;"
        "--surface:#ffffff;"
        "--line:#d7e1ea;"
        "--accent:#0f6ca6;"
        "--pass-bg:#e4f7eb;"
        "--pass-text:#0d6331;"
        "--fail-bg:#fde8e8;"
        "--fail-text:#9a1f1f;"
        "--skip-bg:#fff4db;"
        "--skip-text:#8a5a00;"
        "--unknown-bg:#e6eef4;"
        "--unknown-text:#36536a;"
        "}"
        "*{box-sizing:border-box;}"
        "body{"
        "margin:0;"
        "padding:2rem 1.25rem 3rem;"
        "color:var(--text);"
        'font-family:"Trebuchet MS","Segoe UI",Tahoma,sans-serif;'
        "background:"
        "radial-gradient(circle at top right,#d7ebf8 0,transparent 35%),"
        "radial-gradient(circle at bottom left,#d8efe8 0,transparent 40%),"
        "var(--bg);"
        "line-height:1.45;"
        "}"
        ".report{max-width:1024px;margin:0 auto;display:grid;gap:1rem;}"
        ".panel{"
        "background:var(--surface);"
        "border:1px solid var(--line);"
        "border-radius:14px;"
        "box-shadow:0 8px 22px rgba(18,48,71,.08);"
        "overflow:hidden;"
        "}"
        ".header{padding:1.35rem 1.45rem 1.2rem;}"
        "h1{font-size:1.8rem;margin:0 0 .35rem;letter-spacing:.2px;}"
        ".meta{margin:0;color:var(--muted);}"
        ".section-title{padding:1rem 1.25rem;border-bottom:1px solid var(--line);"
        "font-size:1.05rem;font-weight:700;background:#f8fbfd;}"
        "table{width:100%;border-collapse:separate;border-spacing:0;}"
        "th,td{padding:.7rem .8rem;border-bottom:1px solid var(--line);"
        "text-align:left;vertical-align:top;}"
        "tbody tr:last-child td,tbody tr:last-child th{border-bottom:0;}"
        ".summary-table th{width:38%;font-weight:600;color:var(--muted);}"
        ".checks-table thead th{background:#f8fbfd;font-weight:700;}"
        ".checks-table tbody tr:nth-child(2n){background:#fbfdff;}"
        ".checks-table tbody tr:hover{background:#f2f8fd;}"
        ".summary-value{font-weight:600;}"
        ".status-badge{"
        "display:inline-flex;align-items:center;justify-content:center;"
        "padding:.2rem .55rem;border-radius:999px;font-size:.8rem;"
        "font-weight:700;text-transform:capitalize;white-space:nowrap;"
        "}"
        ".status-passed{background:var(--pass-bg);color:var(--pass-text);}"
        ".status-failed{background:var(--fail-bg);color:var(--fail-text);}"
        ".status-skipped{background:var(--skip-bg);color:var(--skip-text);}"
        ".status-unknown{background:var(--unknown-bg);color:var(--unknown-text);}"
        "code{"
        "background:#edf4fa;color:#204761;border:1px solid #d8e5f0;"
        "padding:.08rem .35rem;border-radius:5px;font-size:.9em;"
        "}"
        ".empty-checks{text-align:center;color:var(--muted);padding:1rem;}"
        "@media (max-width:800px){"
        "body{padding:1rem .65rem 1.5rem;}"
        "h1{font-size:1.4rem;}"
        "th,td{padding:.6rem .55rem;font-size:.94rem;}"
        ".summary-table th{width:45%;}"
        "}"
        "</style></head><body>"
        "<main class='report'>"
        "<section class='panel header'>"
        f"<h1>Suite: {suite_name}</h1>"
        f"{plugin_html}"
        "</section>"
        "<section class='panel'>"
        "<div class='section-title'>Summary</div>"
        f"<table class='summary-table'>{summary_html}</table>"
        "</section>"
        "<section class='panel'>"
        "<div class='section-title'>Checks</div>"
        "<table class='checks-table'><thead>"
        "<tr><th>Name</th><th>Status</th><th>Info</th><th>Details</th></tr>"
        "</thead>"
        f"<tbody>{check_rows_html}</tbody></table>"
        "</section>"
        "</main>"
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
