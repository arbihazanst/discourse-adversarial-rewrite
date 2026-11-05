#!/usr/bin/env python3
"""Merge prediction logs atomically, preserving all attempts and run identities."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

from benchmark_common import COLUMNS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    for source in args.inputs:
        if not source.is_file():
            parser.error(f'Missing predictions file: {source}')
        if source.resolve() == args.out.resolve() or (args.out.exists() and source.samefile(args.out)):
            parser.error('--out must not refer to an input file (including links).')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=args.out.parent, delete=False) as dest:
            temporary = Path(dest.name)
            writer = csv.DictWriter(dest, fieldnames=COLUMNS)
            writer.writeheader()
            for source in args.inputs:
                with source.open(encoding='utf-8-sig', newline='') as file:
                    reader = csv.DictReader(file)
                    if reader.fieldnames != COLUMNS:
                        parser.error(f'{source}: incompatible/legacy prediction schema')
                    count = 0
                    for row in reader:
                        if None in row or any(v is None for v in row.values()) or not row['run_id']:
                            parser.error(f'{source}: malformed row or missing run_id')
                        writer.writerow(row)
                        count += 1
                print(f'{source}: {count} records')
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(temporary, args.out)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Merged: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
