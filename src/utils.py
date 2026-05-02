from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
RESULT_LINE_RE = re.compile(
    r"(best valid|test result)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
RUN_FILENAME_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9_]+)"
    r"-(?P<dataset>.+)"
    r"-(?P<timestamp>[A-Z][a-z]{2}-\d{1,2}-\d{4}_\d{2}-\d{2}-\d{2})"
    r"-(?P<hash>[0-9a-f]{6})"
    r"\.log$"
)


def parse_recbole_metrics(log_path: str | Path) -> dict[str, dict[str, float]]:
    """
    Parse RecBole metrics from a log file.

    Returns a dict with up to two keys:
    - ``best_valid``: metrics printed by ``best valid``
    - ``test_result``: metrics printed by ``test result``
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    return parse_recbole_metrics_text(text)


def parse_recbole_metrics_text(text: str) -> dict[str, dict[str, float]]:
    """Parse RecBole metrics from raw log text.

    Handles both ``{...}`` dict literals and ``OrderedDict([...])`` wrappers
    (RecBole emits the latter in practice).
    """
    results: dict[str, dict[str, float]] = {}

    for raw_line in text.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line).rstrip()
        match = RESULT_LINE_RE.search(line)
        if not match:
            continue

        label, payload = match.groups()
        parsed = _eval_metric_payload(payload)
        if parsed is None:
            continue

        key = "best_valid" if label.lower() == "best valid" else "test_result"
        results[key] = _coerce_metric_dict(parsed)

    return results


def _eval_metric_payload(payload: str) -> dict[str, Any] | None:
    """Evaluate a metric payload, accepting dict literal or OrderedDict([...])."""
    payload = payload.strip()
    # Strip OrderedDict(...) wrapper if present.
    if payload.startswith("OrderedDict(") and payload.endswith(")"):
        payload = payload[len("OrderedDict("):-1]
    try:
        value = ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, list):  # list of (key, value) tuples
        try:
            value = dict(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    return value


def _coerce_metric_dict(data: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def parse_run_filename(path: str | Path) -> dict[str, str] | None:
    """Split a RecBole-style log filename into model / dataset / timestamp / hash.

    Filename pattern:
        {MODEL}-{DATASET}-{Mon}-{DD}-{YYYY}_{HH}-{MM}-{SS}-{HASH}.log

    Returns None if the filename does not match.
    """
    name = Path(path).name
    m = RUN_FILENAME_RE.match(name)
    if not m:
        return None
    return m.groupdict()


def collect_results(
    log_root: str | Path,
    include_valid: bool = False,
) -> tuple["pd.DataFrame", list[Path]]:
    """Walk *log_root* recursively, parse every ``.log`` file, return a wide DataFrame.

    Each row is one run. Columns:
        model, dataset, timestamp, hash, log_path,
        <metric>... (test result),
        valid_<metric>... (only if include_valid=True)

    The second return value is the list of files that failed to parse
    (filename did not match, or no test_result line found).
    """
    import pandas as pd  # local import keeps src/utils.py light at import time

    rows: list[dict[str, Any]] = []
    failures: list[Path] = []

    for log_path in sorted(Path(log_root).rglob("*.log")):
        meta = parse_run_filename(log_path)
        if meta is None:
            failures.append(log_path)
            continue

        metrics = parse_recbole_metrics(log_path)
        test = metrics.get("test_result")
        if not test:
            failures.append(log_path)
            continue

        row: dict[str, Any] = {**meta, "log_path": str(log_path), **test}
        if include_valid:
            for k, v in (metrics.get("best_valid") or {}).items():
                row[f"valid_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, failures
