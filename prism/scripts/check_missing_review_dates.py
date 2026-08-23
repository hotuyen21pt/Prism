#!/usr/bin/env python3
"""List records without a review_date in the train/dev/test JSONL files."""

import argparse
import json
from pathlib import Path


def find_missing_dates(path: Path):
    missing = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("review_date"):
                missing.append(
                    {
                        "line": line_number,
                        "instance_id": row.get("instance_id"),
                        "source_review_id": row.get("source_review_id"),
                        "text": row.get("text", ""),
                    }
                )
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Find JSONL records that do not have a review_date."
    )
    default_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument("--data-dir", type=Path, default=default_dir)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev.jsonl", "test.jsonl", "train.jsonl"],
        help="JSONL split files to check.",
    )
    args = parser.parse_args()

    total_missing = 0
    for split_name in args.splits:
        path = args.data_dir / split_name
        missing = find_missing_dates(path)
        total_missing += len(missing)
        print(f"{split_name}: {len(missing):,} record(s) without review_date")
        for record in missing:
            print(
                f"  line {record['line']}: "
                f"{record['instance_id']} | "
                f"{record['source_review_id']} | {record['text']}"
            )

    print(f"Total: {total_missing:,} record(s) without review_date")
    return 1 if total_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())