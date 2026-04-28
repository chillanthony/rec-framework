from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
RESULT_LINE_RE = re.compile(
    r"(best valid|test result)\s*:\s*(\{.*\})",
    re.IGNORECASE,
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
    """Parse RecBole metrics from raw log text."""
    results: dict[str, dict[str, float]] = {}

    for raw_line in text.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        match = RESULT_LINE_RE.search(line)
        if not match:
            continue

        label, payload = match.groups()
        key = "best_valid" if label.lower() == "best valid" else "test_result"
        results[key] = _coerce_metric_dict(ast.literal_eval(payload))

    return results


def _coerce_metric_dict(data: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics
