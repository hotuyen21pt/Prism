#!/usr/bin/env python3
"""Add review_date from the labeled review file to train/dev/test JSONL files."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def build_date_index(path: Path):
    dates = {}
    duplicate_ids = set()
    for line_number, row in read_jsonl(path):
        review_id = row.get("source_review_id")
        review_date = row.get("review_date")
        if not review_id:
            raise ValueError(f"{path}:{line_number} has no source_review_id")
        if review_id in dates and dates[review_id] != review_date:
            duplicate_ids.add(review_id)
        dates[review_id] = review_date

    if duplicate_ids:
        ids = ", ".join(sorted(duplicate_ids))
        raise ValueError(f"Conflicting review_date values for: {ids}")
    return dates


def map_file(path: Path, dates, dry_run: bool, strict: bool):
    rows = []
    missing = []
    invalid_instances = []
    changed = 0

    for line_number, row in read_jsonl(path):
        review_id = row.get("source_review_id")
        instance_id = row.get("instance_id") or ""
        if review_id not in dates or not dates[review_id]:
            missing.append((line_number, review_id))
        elif review_id not in instance_id:
            invalid_instances.append((line_number, review_id, instance_id))
        elif row.get("review_date") != dates[review_id]:
            changed += 1
        row["review_date"] = dates.get(review_id)
        rows.append(row)

    if not dry_run and not invalid_instances and not (strict and missing):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)

    return changed, missing, invalid_instances


def main():
    parser = argparse.ArgumentParser(
        description="Map review_date into train/dev/test using source_review_id."
    )
    default_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument("--data-dir", type=Path, default=default_dir)
    parser.add_argument("--labeled", default="hotel_absa_labeled.jsonl")
    parser.add_argument(
        "--splits", nargs="+", default=["train.jsonl", "dev.jsonl", "test.jsonl"]
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without rewriting files."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Do not write files when a date is missing."
    )
    args = parser.parse_args()

    labeled_path = args.data_dir / args.labeled
    dates = build_date_index(labeled_path)
    print(f"Loaded {len(dates):,} labeled reviews from {labeled_path}")

    total_changed = 0
    failed = False
    for split_name in args.splits:
        path = args.data_dir / split_name
        changed, missing, invalid_instances = map_file(
            path, dates, args.dry_run, args.strict
        )
        total_changed += changed
        print(f"{split_name}: {changed:,} review_date values {'would be ' if args.dry_run else ''}updated")
        if missing:
            failed = True
            print(f"  missing source review/date: {len(missing):,}")
            print(f"  examples: {missing[:3]}")
        if invalid_instances:
            failed = True
            print(f"  instance_id does not contain source_review_id: {len(invalid_instances):,}")
            print(f"  examples: {invalid_instances[:3]}")

    print(f"Total updated: {total_changed:,}")
    if failed and args.strict:
        raise SystemExit("Mapping was not written because validation failed.")


if __name__ == "__main__":
    main()