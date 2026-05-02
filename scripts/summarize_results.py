"""
summarize_results.py — Aggregate RecBole experiment logs into a Markdown table.

Walks ``log/`` recursively, parses every ``.log`` file via
``src.utils.collect_results``, and writes a compact comparison table containing
only HR@10, HR@20, NDCG@10, NDCG@20 from each run's test result.

Usage
-----
# Scan all logs, write to results/summary.md
python scripts/summarize_results.py

# Custom log root and output path
python scripts/summarize_results.py --log_root log/ --out results/summary.md
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import collect_results


# Test metrics kept in the markdown table, in display order.
# RecBole uses "hit@K" for what the literature calls Hit Ratio (HR).
MD_METRICS: list[tuple[str, str]] = [
    ("hit@10", "HR@10"),
    ("hit@20", "HR@20"),
    ("ndcg@10", "NDCG@10"),
    ("ndcg@20", "NDCG@20"),
]
MD_ID_COLS: list[str] = ["model", "dataset", "timestamp", "hash"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate RecBole logs into a Markdown summary table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--log_root", default="log/",
        help="Directory to scan recursively for .log files (default: log/).",
    )
    p.add_argument(
        "--out", default="results/summary.md",
        help="Output Markdown path (default: results/summary.md).",
    )
    return p


def write_markdown(df, md_path: Path) -> int:
    """Write a compact MD table with only HR@{10,20} and NDCG@{10,20} test metrics.

    Returns the number of data rows written.
    """
    id_cols = [c for c in MD_ID_COLS if c in df.columns]
    metric_cols = [src for src, _ in MD_METRICS if src in df.columns]
    if not metric_cols:
        print("[summarize] no HR/NDCG @ 10|20 columns found; nothing to write.")
        return 0

    sub = df[id_cols + metric_cols].copy()
    rename = {src: disp for src, disp in MD_METRICS if src in sub.columns}
    sub = sub.rename(columns=rename)
    display_cols = list(sub.columns)

    header = "| " + " | ".join(display_cols) + " |"
    sep = "| " + " | ".join("---" for _ in display_cols) + " |"

    lines = [header, sep]
    metric_display_set = set(rename.values())
    for _, row in sub.iterrows():
        cells = []
        for col in display_cols:
            val = row[col]
            if col in metric_display_set:
                cells.append(f"{val:.4f}" if val == val else "")  # NaN → blank
            else:
                cells.append("" if val is None or val != val else str(val))
        lines.append("| " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(sub)


def main():
    args = build_parser().parse_args()

    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    df, failures = collect_results(log_root)

    if not df.empty:
        sort_keys = [c for c in ("model", "dataset", "timestamp") if c in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_markdown(df, out_path) if not df.empty else 0
    print(f"[summarize] wrote {n} rows to {out_path.relative_to(PROJECT_ROOT)}")

    if failures:
        fail_path = out_path.parent / "parse_failures.txt"
        fail_path.write_text("\n".join(str(p) for p in failures) + "\n", encoding="utf-8")
        print(
            f"[summarize] {len(failures)} files failed to parse, "
            f"listed in {fail_path.relative_to(PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    main()
