import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch runner for qiskit.py")
parser.add_argument(
    "--fast",
    action="store_true",
    help="Pass --fast to each qiskit.py run (lower precision, much faster)"
)
args = parser.parse_args()

# ── Paths ────────────────────────────────────────────────────
script_dir  = Path(__file__).parent
script_path = script_dir / "qiskit_opt.py"

if not script_path.exists():
    print(f"File not found: {script_path}")
    sys.exit(1)

out_dir = script_dir / "batch_runs"
out_dir.mkdir(exist_ok=True)

# ── Batch loop ───────────────────────────────────────────────
for n_servers in range(1, 7):
    for n_vms in range(1, 7):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "fast" if args.fast else "full"
        log_file = out_dir / f"run_s{n_servers}_v{n_vms}_{mode_tag}_{ts}.log"

        cmd = [
            sys.executable, str(script_path),
            "--n_servers", str(n_servers),
            "--n_vms",     str(n_vms),
        ]
        if args.fast:
            cmd.append("--fast")         # forwarded to qiskit.py

        print(f"Running [{mode_tag}]: servers={n_servers}, vms={n_vms} -> log: {log_file.name}")

        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(f"Command: {' '.join(cmd)}\n\n")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                fh.write(proc.stdout)
                if proc.returncode == 0:
                    print(f"Completed: s={n_servers} v={n_vms}")
                else:
                    print(f"Error (code {proc.returncode}) for s={n_servers} v={n_vms}; see {log_file.name}")
            except Exception as e:
                fh.write(str(e))
                print(f"Exception during execution: {e}")

print(f"Batch completed [{mode_tag}]. Logs saved in:", out_dir)