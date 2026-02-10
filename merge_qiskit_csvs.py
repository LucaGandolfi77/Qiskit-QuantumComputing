#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

Q_DIR = Path('QISKIT')
OUT = Q_DIR / 'all_qiskit_lp.csv'


def merge_all(qdir: Path, out: Path) -> int:
    files = sorted(qdir.glob('*.csv'))
    if not files:
        print('No CSV files found in', qdir)
        return 1
    with out.open('w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['notebook', 'section', 'content'])
        for f in files:
            try:
                with f.open('r', encoding='utf-8', newline='') as fin:
                    reader = csv.reader(fin)
                    header = next(reader, None)
                    # Expect header like ['section','content'] or ['section','content'] with notebook column absent
                    for row in reader:
                        if not row:
                            continue
                        if len(row) == 2:
                            section, content = row
                        elif len(row) >= 3:
                            # possible extra columns; take first two as section/content
                            section, content = row[0], row[1]
                        else:
                            # single column -> treat as content
                            section = ''
                            content = row[0]
                        writer.writerow([f.name, section, content])
            except Exception as e:
                print(f'Failed to read {f}: {e}')
    print(f'Wrote merged CSV: {out} (from {len(files)} files)')
    return 0


if __name__ == '__main__':
    q = Path(sys.argv[1]) if len(sys.argv) > 1 else Q_DIR
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    raise SystemExit(merge_all(q, o))
