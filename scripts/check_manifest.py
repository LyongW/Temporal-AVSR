#!/usr/bin/env python3
"""Validate a Temporal-AVSR TSV/WRD pair without loading ML dependencies."""

import argparse
import math
from pathlib import Path


def validate(tsv: Path, labels: Path, check_files: bool = True) -> int:
    count = 0
    with tsv.open(encoding="utf-8") as source:
        header = source.readline().strip()
        if not header:
            raise ValueError("The first TSV line must specify the media root")
        root = Path(header).expanduser()
        if not root.is_absolute():
            root = tsv.resolve().parent / root
        for line_number, line in enumerate(source, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) not in (6, 7):
                raise ValueError(f"Line {line_number}: expected 6 or 7 tab-separated fields")
            frames, samples = int(fields[-3]), int(fields[-2])
            rate = float(fields[-1])
            if frames <= 0 or samples <= 0 or not math.isfinite(rate) or rate < 0:
                raise ValueError(f"Line {line_number}: invalid length or speech rate")
            if check_files:
                for name in fields[1:3]:
                    path = root / name
                    if not path.is_file():
                        raise FileNotFoundError(f"Line {line_number}: {path}")
            count += 1
    with labels.open(encoding="utf-8") as source:
        label_count = sum(1 for _ in source)
    if count == 0 or label_count != count:
        raise ValueError(f"TSV has {count} samples; WRD has {label_count} labels")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--skip-media-check", action="store_true",
                        help="Check format and label counts only")
    args = parser.parse_args()
    try:
        count = validate(args.tsv, args.labels or args.tsv.with_suffix(".wrd"),
                         not args.skip_media_check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Invalid manifest: {error}\n")
    print(f"OK: {count} samples")


if __name__ == "__main__":
    main()
