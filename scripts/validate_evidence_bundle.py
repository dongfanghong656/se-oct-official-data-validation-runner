#!/usr/bin/env python3
"""Fail closed unless a scientific CI run produced parseable, nonempty evidence."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_numbers(value: Any, path: str = "root"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_numbers(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, float(value)


def lookup(data: Any, dotted_path: str) -> tuple[bool, Any]:
    current = data
    for component in dotted_path.split("."):
        if isinstance(current, dict) and component in current:
            current = current[component]
        else:
            return False, None
    return True, current


def csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-key", action="append", default=[])
    parser.add_argument("--required-file", type=Path, action="append", default=[])
    parser.add_argument("--required-csv", type=Path, action="append", default=[])
    parser.add_argument("--min-bytes", type=int, default=100)
    parser.add_argument("--min-csv-rows", type=int, default=1)
    parser.add_argument("--allow-nonfinite", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    data: Any = None
    if not args.metrics.exists():
        failures.append("metrics_missing")
    elif args.metrics.stat().st_size < args.min_bytes:
        failures.append(f"metrics_too_small:{args.metrics.stat().st_size}")
    else:
        try:
            data = json.loads(args.metrics.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"metrics_parse_error:{type(exc).__name__}:{exc}")

    if isinstance(data, dict):
        for key in args.required_key:
            exists, value = lookup(data, key)
            if not exists:
                failures.append(f"required_key_missing:{key}")
            elif value in (None, [], {}, ""):
                failures.append(f"required_key_empty:{key}")
        if not args.allow_nonfinite:
            nonfinite = [path for path, number in walk_numbers(data) if not math.isfinite(number)]
            if nonfinite:
                failures.append("nonfinite_values:" + ",".join(nonfinite[:20]))

    required_files = [*args.required_file, *args.required_csv]
    file_records: list[dict[str, Any]] = []
    for path in required_files:
        if not path.exists():
            failures.append(f"required_file_missing:{path}")
            file_records.append({"path": str(path), "exists": False})
            continue
        size = path.stat().st_size
        record: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "size": size,
            "sha256": sha256(path),
        }
        if size == 0:
            failures.append(f"required_file_empty:{path}")
        if path in args.required_csv:
            try:
                rows = csv_data_rows(path)
                record["data_rows"] = rows
                if rows < args.min_csv_rows:
                    failures.append(f"csv_too_few_rows:{path}:{rows}")
            except Exception as exc:
                failures.append(f"csv_parse_error:{path}:{type(exc).__name__}:{exc}")
        file_records.append(record)

    status = "VERIFIED" if not failures else "FAILED"
    record = {
        "schema": "scientific-evidence-gate-v2",
        "status": status,
        "metrics_path": str(args.metrics),
        "metrics_sha256": sha256(args.metrics) if args.metrics.exists() else None,
        "metrics_size": args.metrics.stat().st_size if args.metrics.exists() else None,
        "required_keys": args.required_key,
        "required_files": file_records,
        "failures": failures,
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rule": "A commit name, workflow trigger, provenance-only file, or successful producer command without parseable metrics is not execution evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if status == "VERIFIED" else 2)


if __name__ == "__main__":
    main()
