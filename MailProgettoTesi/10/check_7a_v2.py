import json

with open('/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/7a/merged_20260325_141122.json') as f:
    data = json.load(f)

total = data['total_runs']
fc = sum(1 for r in data['runs'] if r['classic']['status'] in ('FEASIBLE','SUCCESS'))
fq = sum(1 for r in data['runs'] if r['quantum']['status'] in ('FEASIBLE','SUCCESS'))
ic = sum(1 for r in data['runs'] if r['classic']['status'] == 'INFEASIBLE')
iq = sum(1 for r in data['runs'] if r['quantum']['status'] == 'INFEASIBLE')
print(f"Total: {total}  Classic: {fc} OK, {ic} INFEASIBLE  Quantum: {fq} OK, {iq} INFEASIBLE\n")

# Show first run LP to check ADMM_SLACK value
print("=== First run LP ===")
print(data['runs'][0]['qp_lp'])

# Show all infeasible runs with detailed violations
for r in data['runs']:
    m = r['meta']
    nv, ns = m['n_vms'], m['n_servers']
    for label in ['classic', 'quantum']:
        if r[label]['status'] == 'INFEASIBLE':
            x = r[label]['x']
            u_vals = x[ns + nv*ns:]
            v_vals = x[ns : ns + nv*ns]
            caps = r['input']['capacities']
            lims = r['input']['vm_allocation_limits']

            viols = []
            for i in range(ns):
                load = sum(v_vals[j*ns + i] for j in range(nv))
                needed = caps[i] - 1
                if load < needed - 1e-6:
                    viols.append(f"load_s{i}:{load:.8f}<{needed}")

            for j in range(nv):
                alloc = sum(v_vals[j*ns + i] for i in range(ns))
                lim = lims[j] if j < len(lims) else lims[-1]
                if alloc > lim + 1e-6:
                    viols.append(f"alloc_v{j}:{alloc:.8f}>{lim}")

            for j, u in enumerate(u_vals):
                if u < 0.99 - 1e-6:
                    viols.append(f"u{j}={u:.8f}<0.99")

            for i in range(ns):
                if abs(x[i] - 1.0) > 1e-6:
                    viols.append(f"s{i}={x[i]:.6f}!=1")

            if not viols:
                viols.append("NO OBVIOUS VIOLATION (check quadratic constraints)")

            print(f"S={ns} V={nv} {label:7s} INFEASIBLE | obj={r[label]['objective']:.4f} | {', '.join(viols)}")
