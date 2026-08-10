import json

with open('/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/7a/merged_20260325_115018.json') as f:
    data = json.load(f)

# Detailed check of one infeasible run
for r in data['runs']:
    if r['classic']['status'] != 'INFEASIBLE':
        continue
    m = r['meta']
    nv = m['n_vms']
    ns = m['n_servers']
    caps = r['input']['capacities']
    lims = r['input']['vm_allocation_limits']

    print(f"\n{'='*60}")
    print(f"S={ns} V={nv}")
    print(f"LP:\n{r['qp_lp']}")

    for label in ['classic']:
        x = r[label]['x']
        print(f"\n--- {label} solution x = {x}")
        print(f"  s_vars = {x[:ns]}")
        v_vals = x[ns : ns + nv*ns]
        print(f"  v_vars = {v_vals}")
        u_vals = x[ns + nv*ns:]
        print(f"  u_vars = {u_vals}")

        # Check ALL constraints from LP
        print(f"\n  Constraint checks:")

        # server_load
        for i in range(ns):
            load = sum(v_vals[j*ns + i] for j in range(nv))
            rhs = caps[i] - 1
            ok = "OK" if load >= rhs - 1e-6 else f"VIOLATED ({load:.10f} < {rhs})"
            print(f"    server_load_{i}: sum_v={load:.10f} >= {rhs} -> {ok}")

        # server_on
        for i in range(ns):
            ok = "OK" if abs(x[i] - 1.0) <= 1e-6 else f"VIOLATED ({x[i]})"
            print(f"    server_on_{i}: s{i}={x[i]:.10f} == 1 -> {ok}")

        # vm_alloc
        for j in range(nv):
            alloc = sum(v_vals[j*ns + i] for i in range(ns))
            lim = lims[j] if j < len(lims) else lims[-1]
            ok = "OK" if alloc <= lim + 1e-6 else f"VIOLATED ({alloc:.10f} > {lim})"
            print(f"    vm_alloc_{j}: sum={alloc:.10f} <= {lim} -> {ok}")

        # min_cpu_vm
        for j in range(len(u_vals)):
            ok = "OK" if u_vals[j] >= 1.0 - 1e-6 else f"VIOLATED ({u_vals[j]:.10f} < 1.0)"
            print(f"    min_cpu_vm_{j}: u{j}={u_vals[j]:.10f} >= 1.0 -> {ok}")

        # bounds
        for j in range(len(v_vals)):
            if v_vals[j] < -1e-6:
                print(f"    BOUND VIOLATED: v[{j}] = {v_vals[j]:.10f} < 0")
        for j in range(len(u_vals)):
            if u_vals[j] < 1.0 - 1e-6:
                print(f"    BOUND VIOLATED: u[{j}] = {u_vals[j]:.10f} < 1.0")

        print(f"    ALL EXPLICIT CHECKS PASSED? No violations found above means qiskit has stricter internal check")

    # Only show first 3 infeasible runs
    break
