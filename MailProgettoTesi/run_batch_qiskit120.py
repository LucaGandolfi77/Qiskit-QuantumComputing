import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Cartella dello script target
script_dir = Path(__file__).parent
script_path = script_dir / "qiskit120_fixed.py"

if not script_path.exists():
    print(f"File non trovato: {script_path}")
    sys.exit(1)

out_dir = script_dir / "batch_runs"
out_dir.mkdir(exist_ok=True)

for n_servers in range(1, 7):
    for n_vms in range(1, 7):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = out_dir / f"run_s{n_servers}_v{n_vms}_{ts}.log"
        cmd = [sys.executable, str(script_path), "--n_servers", str(n_servers), "--n_vms", str(n_vms)]
        print(f"Eseguo: servers={n_servers}, vms={n_vms} -> log: {log_file.name}")
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(f"Command: {' '.join(cmd)}\n\n")
            try:
                proc = subprocess.run(cmd, cwd=script_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
                fh.write(proc.stdout)
                if proc.returncode == 0:
                    print(f"Completato: s={n_servers} v={n_vms}")
                else:
                    print(f"Errore (code {proc.returncode}) per s={n_servers} v={n_vms}; vedi {log_file.name}")
            except Exception as e:
                fh.write(str(e))
                print(f"Eccezione durante l'esecuzione: {e}")

print("Batch completato. Log salvati in:", out_dir)
