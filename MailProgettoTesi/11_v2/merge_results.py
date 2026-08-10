#!/usr/bin/env python3
"""
merge_results.py
----------------
Merge all q_*_results.json files into a single JSON and CSV summary.

Usage:
    python merge_results.py                        # scans current directory
    python merge_results.py --dir batch_runs       # scans a specific folder
    python merge_results.py --dir batch_runs --out merged_results.json
    python merge_results.py --recursive            # scans subdirectories too
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Merge q_*_results.json files")
    parser.add_argument("--dir",       default=".",          help="Directory to scan (default: current)")
    parser.add_argument("--out",       default="",           help="Output JSON filename (default: merged_TIMESTAMP.json)")
    parser.add_argument("--csv",       default="",           help="Output CSV  filename (default: merged_TIMESTAMP.csv)")
    parser.add_argument("--pattern",   default="*results.json", help="Glob pattern (default: *results.json)")
    parser.add_argument("--recursive", action="store_true",  help="Scan subdirectories recursively")
    parser.add_argument("--sort",      default="n_vms",
                        choices=["n_vms", "n_servers", "timestamp", "classic_time", "quantum_time"],
                        help="Sort merged results by this field (default: n_vms)")
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Validate minimal expected structure
        if "classic" not in data or "quantum" not in data:
            print(f"  [SKIP] {path.name} — missing 'classic' or 'quantum' keys")
            return None
        data["_source_file"] = path.name
        return data
    except json.JSONDecodeError as e:
        print(f"  [ERROR] {path.name} — JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {path.name} — {e}")
        return None


def flatten_for_csv(entry: dict) -> dict:
    """Extract the most useful fields into a flat dict for CSV export."""
    meta = entry.get("meta", {})
    classic = entry.get("classic", {})
    quantum = entry.get("quantum", {})
    c_time  = classic.get("duration_seconds", None)
    q_time  = quantum.get("duration_seconds", None)
    speedup = round(c_time / q_time, 3) if (c_time and q_time and q_time > 0) else None
    c_obj   = classic.get("objective", None)
    q_obj   = quantum.get("objective", None)

    return {
        "source_file":       entry.get("_source_file", ""),
        "n_servers":         meta.get("n_servers", entry.get("n_servers", "")),
        "n_vms":             meta.get("n_vms",     entry.get("n_vms", "")),
        "require_all_on":    meta.get("require_all_on", ""),
        "classic_objective": round(c_obj, 6) if c_obj is not None else "",
        "classic_status":    classic.get("status", ""),
        "classic_time_s":    round(c_time, 3) if c_time is not None else "",
        "quantum_objective": round(q_obj, 6) if q_obj is not None else "",
        "quantum_status":    quantum.get("status", ""),
        "quantum_time_s":    round(q_time, 3) if q_time is not None else "",
        "speedup_x":         speedup,
        "best_solver":       ("quantum" if (q_obj is not None and c_obj is not None and q_obj < c_obj)
                              else "classical") if (q_obj is not None and c_obj is not None) else "",
        "n_classic_residuals": len(classic.get("residuals", [])),
        "n_quantum_residuals": len(quantum.get("residuals", [])),
    }


def sort_key(entry: dict, field: str):
    meta = entry.get("meta", {})
    mapping = {
        "n_vms":        lambda e: meta.get("n_vms", 0),
        "n_servers":    lambda e: meta.get("n_servers", 0),
        "timestamp":    lambda e: e.get("_source_file", ""),
        "classic_time": lambda e: e.get("classic", {}).get("duration_seconds", 0),
        "quantum_time": lambda e: e.get("quantum", {}).get("duration_seconds", 0),
    }
    return mapping.get(field, lambda e: 0)(entry)


def main():
    args = parse_args()
    scan_dir = Path(args.dir)

    if not scan_dir.exists():
        print(f"[ERROR] Directory not found: {scan_dir}")
        sys.exit(1)

    # ── Find JSON files ──────────────────────────────────────
    glob_fn = scan_dir.rglob if args.recursive else scan_dir.glob
    files   = sorted(glob_fn(args.pattern))

    if not files:
        print(f"[INFO] No files matching '{args.pattern}' found in {scan_dir}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) matching '{args.pattern}' in {scan_dir}")

    # ── Load & validate ──────────────────────────────────────
    loaded = []
    for f in files:
        print(f"  Loading: {f.name}")
        entry = load_json(f)
        if entry:
            loaded.append(entry)

    if not loaded:
        print("[ERROR] No valid JSON files loaded. Aborting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded: {len(loaded)} / {len(files)} files")

    # ── Sort ─────────────────────────────────────────────────
    loaded.sort(key=lambda e: sort_key(e, args.sort))

    # ── Build merged output ──────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    merged = {
        "merge_timestamp": ts,
        "source_directory": str(scan_dir.resolve()),
        "total_runs": len(loaded),
        "sort_by": args.sort,
        "runs": loaded,
    }

    # ── Save JSON ────────────────────────────────────────────
    out_json = Path(args.out) if args.out else scan_dir / f"merged_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nMerged JSON saved → {out_json}")

    # ── Save CSV ─────────────────────────────────────────────
    out_csv = Path(args.csv) if args.csv else scan_dir / f"merged_{ts}.csv"
    flat_rows = [flatten_for_csv(e) for e in loaded]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else []

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Merged CSV  saved → {out_csv}")

    # ── Print summary table ──────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Instance':<14} {'Classic Obj':>12} {'Classic t':>10} "
          f"{'Quantum Obj':>12} {'Quantum t':>10} {'Speedup':>8}")
    print("-"*72)
    for row in flat_rows:
        inst = f"{row['n_servers']}s×{row['n_vms']}vm"
        print(f"{inst:<14} {str(row['classic_objective']):>12} {str(row['classic_time_s']):>9}s "
              f"{str(row['quantum_objective']):>12} {str(row['quantum_time_s']):>9}s "
              f"{str(row['speedup_x']):>7}×")
    print("="*72)
    print(f"\nDone. {len(loaded)} runs merged.")


if __name__ == "__main__":
    main()
