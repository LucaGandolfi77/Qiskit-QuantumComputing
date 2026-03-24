import sys
import subprocess
import argparse
import math
import random
from pathlib import Path
from datetime import datetime

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch runner for qiskit_opt.py")
parser.add_argument("--fast", action="store_true",
                    help="Pass --fast to each qiskit_opt.py run")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
args = parser.parse_args()

random.seed(args.seed)

# ── Paths ────────────────────────────────────────────────────
script_dir  = Path(__file__).parent
script_path = script_dir / "qiskit_opt.py"

if not script_path.exists():
    print(f"File not found: {script_path}")
    sys.exit(1)

out_dir = script_dir / "batch_runs"
out_dir.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────
def default_capacities(n_servers: int) -> list[int]:
    """Same default as qiskit_opt.py: 11 for first 3, 10 for the rest."""
    return [11 if i < 3 else 10 for i in range(n_servers)]

def safe_vm_alloc(capacities: list[int], n_vms: int, margin: float = 1.25) -> list[int]:
    """
    Compute vm_allocation_limits that are guaranteed to satisfy both:
      - first3vm:    sum(lim[:min(3,n_vms)]) >= sum(cap[:min(3,n_servers)])
      - server_load: sum(lim)                >= sum(cap) - n_servers

    Formula: per-VM minimum = max of the two constraints above,
             then add 25% safety margin + small random perturbation.
    """
    n_s       = len(capacities)
    n_fv      = min(3, n_vms)
    n_fs      = min(3, n_s)

    need_f3   = sum(capacities[:n_fs])          # first3vm requirement
    need_load = max(sum(capacities) - n_s, 0)   # server_load requirement

    min_per_vm = max(
        math.ceil(need_f3   / n_fv),            # distribute first3vm budget
        math.ceil(need_load / n_vms),           # distribute server_load budget
    )
    base = math.ceil(min_per_vm * margin)

    # Small random jitter (+0..+3) so not all VMs have identical limits
    return [base + random.randint(0, 3) for _ in range(n_vms)]

# ── Batch loop ───────────────────────────────────────────────
mode_tag = "fast" if args.fast else "full"

for n_servers in range(1, 7):
    for n_vms in range(1, 7):

        caps = default_capacities(n_servers)
        lims = safe_vm_alloc(caps, n_vms)

        # Sanity check before even launching the subprocess
        n_fv      = min(3, n_vms)
        n_fs      = min(3, n_servers)
        need_f3   = sum(caps[:n_fs])
        need_load = sum(caps) - n_servers
        ok_f3     = sum(sorted(lims, reverse=True)[:n_fv]) >= need_f3
        ok_load   = sum(lims) >= need_load
        if not (ok_f3 and ok_load):
            print(f"[SKIP] s={n_servers} v={n_vms}: computed limits still infeasible "
                  f"(f3={ok_f3}, load={ok_load}) — check safe_vm_alloc()")
            continue

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = out_dir / f"run_s{n_servers}_v{n_vms}_{mode_tag}_{ts}.log"

        cmd = [
            sys.executable, str(script_path),
            "--n_servers",            str(n_servers),
            "--n_vms",                str(n_vms),
            # ── pass computed values explicitly ──────────────
            "--capacities",           ",".join(str(c) for c in caps),
            "--vm_allocation_limits", ",".join(str(l) for l in lims),
            "--pi_list",              ",".join(["1.0"] * n_servers),
            "--pd_list",              ",".join(["1.0"] * n_servers),
        ]
        if args.fast:
            cmd.append("--fast")

        print(
            f"[{mode_tag}] s={n_servers} v={n_vms} | "
            f"caps={caps} | lims={lims} | "
            f"sum(lims)={sum(lims)} >= need={need_load} | "
            f"-> {log_file.name}"
        )

        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(f"Command: {' '.join(cmd)}\n\n")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    #encoding="utf-8",   # force UTF-8 in subprocess output
                    check=False,
                )
                fh.write(proc.stdout or "")
                if proc.returncode == 0:
                    print(f"  -> Completed: s={n_servers} v={n_vms}")
                else:
                    print(f"  -> Error (code {proc.returncode}); see {log_file.name}")
            except Exception as e:
                fh.write(str(e))
                print(f"  -> Exception: {e}")

print(f"\nBatch completed [{mode_tag}]. Logs in: {out_dir}")