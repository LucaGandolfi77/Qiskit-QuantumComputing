import json

with open('/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/7a/merged_20260325_115018.json') as f:
    data = json.load(f)

print(f"Total runs: {data['total_runs']}")
feasible_c = sum(1 for r in data['runs'] if r['classic']['status'] == 'FEASIBLE')
feasible_q = sum(1 for r in data['runs'] if r['quantum']['status'] == 'FEASIBLE')
infeasible_c = sum(1 for r in data['runs'] if r['classic']['status'] == 'INFEASIBLE')
infeasible_q = sum(1 for r in data['runs'] if r['quantum']['status'] == 'INFEASIBLE')
print(f"Classic:  {feasible_c} FEASIBLE, {infeasible_c} INFEASIBLE")
print(f"Quantum:  {feasible_q} FEASIBLE, {infeasible_q} INFEASIBLE")
print()

# Check the LP to understand the actual bound values
print("=== First run LP ===")
print(data['runs'][0]['qp_lp'])
print()

# Show only infeasible runs with violation details
for r in data['runs']:
    m = r['meta']
    nv = m['n_vms']
    ns = m['n_servers']
    caps = r['input']['capacities']
    lims = r['input']['vm_allocation_limits']

    # Parse actual bounds from LP string to check against
    lp = r['qp_lp']

    for label in ['classic', 'quantum']:
        if r[label]['status'] != 'INFEASIBLE':
            continue
        x = r[label]['x']
        u_vals = x[ns + nv*ns:]
        v_vals = x[ns : ns + nv*ns]
        s_vals = x[:ns]
        violations = []

        # server_load: check vs actual RHS in constraint
        for i in range(ns):
            load = sum(v_vals[j*ns + i] for j in range(nv))
            needed = caps[i] - 1
            if load < needed - 1e-6:
                violations.append(f"load_s{i}:{load:.6f}<{needed}")

        # vm_alloc
        for j in range(nv):
            alloc = sum(v_vals[j*ns + i] for i in range(ns))
            lim = lims[j] if j < len(lims) else lims[-1]
            if alloc > lim + 1e-6:
                violations.append(f"alloc_v{j}:{alloc:.6f}>{lim}")

        # u_vars below stated lb
        for j, u in enumerate(u_vals):
            violations.append(f"u{j}={u:.8f}")

        # s_vars
        for i in range(ns):
            if abs(s_vals[i] - 1.0) > 1e-6:
                violations.append(f"s{i}={s_vals[i]:.4f}!=1")

        print(f"S={ns} V={nv} {label:7s} INFEASIBLE | {', '.join(violations)}")
