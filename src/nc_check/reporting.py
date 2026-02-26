from __future__ import annotations

from html import escape
from pathlib import Path
import re
from typing import Any

from .models import SuiteReport

_SCOPED_NAME_PATTERN = re.compile(
    r"^(?P<check>.+)\[(?P<scope>[^:\]]+):(?P<item>[^\]]+)\]$"
)
_INTERNAL_DETAIL_KEYS = {"data_scope", "scope_item", "reported_name"}

_REPORT_STYLES = (
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
    "h1{font-size:1.8rem;margin:0 0 .35rem;letter-spacing:.2px;text-align:left;}"
    ".meta{margin:0;color:var(--muted);}"
    ".section-title{padding:1rem 1.25rem;border-bottom:1px solid var(--line);"
    "font-size:1.05rem;font-weight:700;background:#f8fbfd;text-align:left;}"
    ".dataset-content{padding:.75rem .95rem;overflow:auto;}"
    ".dataset-content *{text-align:left;}"
    ".dataset-group>summary{"
    "cursor:pointer;padding:1rem 1.25rem;border-bottom:1px solid var(--line);"
    "font-size:1.05rem;font-weight:700;background:#f8fbfd;text-align:left;list-style:none;}"
    ".dataset-group>summary::-webkit-details-marker{display:none;}"
    ".dataset-group>summary::before{"
    "content:'▸';display:inline-block;transform:rotate(0deg);"
    "transition:transform .12s ease;color:var(--accent);margin-right:.45rem;}"
    ".dataset-group[open]>summary::before{transform:rotate(90deg);}"
    "table{width:100%;border-collapse:separate;border-spacing:0;}"
    "th,td{padding:.7rem .8rem;border-bottom:1px solid var(--line);"
    "text-align:left;vertical-align:top;}"
    "tbody tr:last-child td,tbody tr:last-child th{border-bottom:0;}"
    ".summary-table th{width:38%;font-weight:600;color:var(--muted);}"
    ".checks-table thead th{background:#f8fbfd;font-weight:700;}"
    ".checks-table tbody tr:nth-child(2n){background:#fbfdff;}"
    ".checks-table tbody tr:hover{background:#f2f8fd;}"
    ".grouped-checks-table th,.grouped-checks-table td{font-size:.92rem;}"
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
    ".scope-group{margin:0;}"
    ".var-group{margin:.6rem 0;border:1px solid var(--line);"
    "border-radius:10px;background:#ffffff;overflow:hidden;}"
    ".scope-group>summary,.var-group>summary{"
    "cursor:pointer;display:flex;justify-content:space-between;gap:.8rem;"
    "align-items:center;font-weight:700;list-style:none;}"
    ".scope-group>summary{"
    "padding:1rem 1.25rem;"
    "background:#f8fbfd;"
    "border-bottom:1px solid var(--line);"
    "}"
    ".var-group>summary{padding:.72rem .85rem;background:#f8fbfd;}"
    ".scope-group>summary::-webkit-details-marker,"
    ".var-group>summary::-webkit-details-marker{display:none;}"
    ".scope-group>summary::before,.var-group>summary::before{"
    "content:'▸';display:inline-block;transform:rotate(0deg);"
    "transition:transform .12s ease;color:var(--accent);margin-right:.4rem;}"
    ".scope-group[open]>summary::before,.var-group[open]>summary::before{"
    "transform:rotate(90deg);}"
    ".group-title{display:inline-flex;align-items:center;}"
    ".group-stats{font-weight:600;color:var(--muted);font-size:.88rem;"
    "margin-left:auto;display:inline-flex;align-items:center;gap:.45rem;}"
    ".group-stats .status-badge{margin-right:.45rem;}"
    ".count-summary{white-space:nowrap;}"
    ".scope-group>.group-content{padding:.65rem .85rem .85rem;}"
    ".var-group>.group-content{padding:.15rem .2rem .5rem;}"
    ".empty-checks{text-align:center;color:var(--muted);padding:1rem;}"
    "@media (max-width:800px){"
    "body{padding:1rem .65rem 1.5rem;}"
    "h1{font-size:1.4rem;}"
    "th,td{padding:.6rem .55rem;font-size:.94rem;}"
    ".summary-table th{width:45%;}"
    "}"
)


def report_to_dict(report: SuiteReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, SuiteReport):
        return report.to_dict()
    return report


def _status_class(value: str) -> str:
    lookup = {
        "passed": "status-passed",
        "failed": "status-failed",
        "skipped": "status-skipped",
    }
    return lookup.get(value.strip().lower(), "status-unknown")


def _status_counts(items: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    passed = 0
    failed = 0
    skipped = 0
    for item in items:
        status = str(item.get("status", "")).strip().lower()
        if status == "passed":
            passed += 1
        elif status == "failed":
            failed += 1
        elif status == "skipped":
            skipped += 1
    return len(items), passed, failed, skipped


def _overall_status(total: int, passed: int, failed: int, skipped: int) -> str:
    if failed > 0:
        return "failed"
    if passed > 0:
        return "passed"
    if skipped > 0 or total > 0:
        return "skipped"
    return "unknown"


def _parse_scoped_name(name: str) -> tuple[str, str, str] | None:
    match = _SCOPED_NAME_PATTERN.match(name)
    if match is None:
        return None
    return (
        str(match.group("check")),
        str(match.group("scope")),
        str(match.group("item")),
    )


def _scope_label(scope_name: str) -> str:
    labels = {
        "data_vars": "Data Variables",
        "coords": "Coordinates",
        "dims": "Dimensions",
    }
    return labels.get(scope_name, scope_name)


def _details_html(item: dict[str, Any]) -> str:
    details = item.get("details")
    if not isinstance(details, dict) or not details:
        return ""
    filtered_details = {
        key: value
        for key, value in details.items()
        if str(key) not in _INTERNAL_DETAIL_KEYS
    }
    if not filtered_details:
        return ""
    return "<br>".join(
        f"<code>{escape(str(key))}</code>: {escape(str(value))}"
        for key, value in sorted(filtered_details.items())
    )


def _grouped_checks(
    checks: list[Any], nested_results: dict[str, Any]
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    if nested_results:
        for raw_scope, raw_variables in nested_results.items():
            if not isinstance(raw_variables, dict):
                continue
            scope_name = str(raw_scope)
            scope_bucket = grouped.setdefault(scope_name, {})
            for raw_variable, raw_checks in raw_variables.items():
                if not isinstance(raw_checks, dict):
                    continue
                variable_name = str(raw_variable)
                variable_bucket = scope_bucket.setdefault(variable_name, {})
                for raw_check_name, raw_check in raw_checks.items():
                    if not isinstance(raw_check, dict):
                        continue
                    variable_bucket[str(raw_check_name)] = raw_check
        if grouped:
            return grouped

    for raw_item in checks:
        if not isinstance(raw_item, dict):
            continue

        details = raw_item.get("details")
        details_map = details if isinstance(details, dict) else {}
        check_name = str(raw_item.get("name", ""))
        parsed = _parse_scoped_name(check_name)

        if parsed is not None:
            base_check_name, data_scope, variable_name = parsed
        else:
            base_check_name = check_name
            data_scope = str(details_map.get("data_scope", "unknown"))
            variable_name = str(details_map.get("scope_item", "dataset"))

        scope_bucket = grouped.setdefault(data_scope, {})
        variable_bucket = scope_bucket.setdefault(variable_name, {})
        variable_bucket[base_check_name] = raw_item

    return grouped


def _render_check_rows(checks_by_name: dict[str, dict[str, Any]]) -> str:
    rows: list[str] = []
    for check_name in sorted(checks_by_name):
        item = checks_by_name[check_name]
        if not isinstance(item, dict):
            continue
        raw_status = str(item.get("status", ""))
        status = escape(raw_status)
        info = escape(str(item.get("info", "")))
        details_text = _details_html(item)
        rows.append(
            "<tr>"
            f"<td>{escape(check_name)}</td>"
            f"<td><span class='status-badge {_status_class(raw_status)}'>{status}</span></td>"
            f"<td>{info}</td>"
            f"<td>{details_text}</td>"
            "</tr>"
        )

    return "".join(rows) or (
        "<tr><td colspan='4' class='empty-checks'>No checks were included.</td></tr>"
    )


def _render_variable_section(
    variable_name: str, checks_by_name: dict[str, dict[str, Any]]
) -> str:
    check_items = [item for item in checks_by_name.values() if isinstance(item, dict)]
    var_total, var_passed, var_failed, var_skipped = _status_counts(check_items)
    var_status = _overall_status(var_total, var_passed, var_failed, var_skipped)
    check_rows_html = _render_check_rows(checks_by_name)

    return (
        "<details class='var-group'>"
        "<summary>"
        f"<span class='group-title'>{escape(variable_name)}</span>"
        "<span class='group-stats'>"
        f"<span class='status-badge {_status_class(var_status)}'>{escape(var_status)}</span>"
        f"<span class='count-summary'>checks={var_total} | pass={var_passed} | fail={var_failed} | skip={var_skipped}</span>"
        "</span>"
        "</summary>"
        "<div class='group-content'>"
        "<table class='checks-table grouped-checks-table'>"
        "<thead><tr><th>Check</th><th>Status</th><th>Info</th><th>Details</th></tr></thead>"
        f"<tbody>{check_rows_html}</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


def _render_scope_section(
    data_scope: str, variables: dict[str, dict[str, dict[str, Any]]]
) -> str:
    variable_sections: list[str] = []
    scope_items_for_counts: list[dict[str, Any]] = []

    for per_variable in variables.values():
        scope_items_for_counts.extend(
            item for item in per_variable.values() if isinstance(item, dict)
        )

    for variable_name in sorted(variables):
        variable_sections.append(
            _render_variable_section(variable_name, variables[variable_name])
        )

    scope_total, scope_passed, scope_failed, scope_skipped = _status_counts(
        scope_items_for_counts
    )
    scope_status = _overall_status(
        scope_total, scope_passed, scope_failed, scope_skipped
    )

    return (
        "<details class='scope-group' open>"
        "<summary>"
        f"<span class='group-title'>{escape(_scope_label(data_scope))}</span>"
        "<span class='group-stats'>"
        f"<span class='status-badge {_status_class(scope_status)}'>{escape(scope_status)}</span>"
        f"<span class='count-summary'>checks={scope_total} | pass={scope_passed} | fail={scope_failed} | skip={scope_skipped}</span>"
        "</span>"
        "</summary>"
        "<div class='group-content'>"
        f"{''.join(variable_sections)}"
        "</div>"
        "</details>"
    )


def _render_grouped_sections(
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> str:
    scope_order = {"data_vars": 0, "coords": 1, "dims": 2}
    scope_sections: list[str] = []

    for data_scope in sorted(grouped, key=lambda key: (scope_order.get(key, 99), key)):
        scope_sections.append(_render_scope_section(data_scope, grouped[data_scope]))

    return "".join(scope_sections) or (
        "<div class='empty-checks'>No checks were included.</div>"
    )


def _render_dataset_section(dataset_html: str | None) -> str:
    if not dataset_html or not dataset_html.strip():
        return ""
    return (
        "<section class='panel dataset-panel'>"
        "<details class='dataset-group'>"
        "<summary>Dataset Preview</summary>"
        f"<div class='dataset-content'>{dataset_html}</div>"
        "</details>"
        "</section>"
    )


def _render_summary_rows(summary: dict[str, Any], plugin: Any) -> str:
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
    return "".join(summary_rows)


def _render_plugin_meta(plugin: Any) -> str:
    if plugin is None:
        return ""
    return f"<p class='meta'>Plugin: <strong>{escape(str(plugin))}</strong></p>"


def render_html_report(report: SuiteReport | dict[str, Any]) -> str:
    payload = report_to_dict(report)
    suite_name = escape(str(payload.get("suite_name", "suite")))
    plugin = payload.get("plugin")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    nested_results = (
        payload.get("results") if isinstance(payload.get("results"), dict) else {}
    )
    raw_dataset_html = payload.get("dataset_html")
    dataset_html = raw_dataset_html if isinstance(raw_dataset_html, str) else None

    grouped_html = _render_grouped_sections(_grouped_checks(checks, nested_results))
    dataset_section_html = _render_dataset_section(dataset_html)
    summary_html = _render_summary_rows(summary, plugin)
    plugin_html = _render_plugin_meta(plugin)

    return (
        "<!doctype html>"
        "<html>"
        "<head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>nc-check report</title>"
        f"<style>{_REPORT_STYLES}</style>"
        "</head>"
        "<body>"
        "<main class='report'>"
        "<section class='panel header'>"
        f"<h1>Suite: {suite_name}</h1>"
        f"{plugin_html}"
        "</section>"
        f"{dataset_section_html}"
        "<section class='panel'>"
        "<div class='section-title'>Summary</div>"
        f"<table class='summary-table'><tbody>{summary_html}</tbody></table>"
        "</section>"
        "<section class='panel'>"
        f"{grouped_html}"
        "</section>"
        "</main>"
        "</body>"
        "</html>"
    )


def save_html_report(
    report: SuiteReport | dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html_report(report), encoding="utf-8")
    return output
