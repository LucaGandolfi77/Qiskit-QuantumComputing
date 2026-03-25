import json, glob, numpy as np

for f in sorted(glob.glob('q_*_results.json')):
    d = json.load(open(f))
    S = d['meta']['n_servers']
    V = d['meta']['n_vms']
    status = d['classic']['status']
    if status != 'INFEASIBLE':
        continue
    x = np.array(d['classic']['x'])
    s_vals = x[:S]
    v_vals = x[S:S + S*V]
    u_vals = x[S + S*V:S + S*V + V]

    violations = []

    # cons_server_load: sum_j v[(j,i)] >= cap[i]-1
    caps = d['input']['capacities']
    for i in range(S):
        load = sum(v_vals[j*S + i] for j in range(V))
        rhs = caps[i] - 1
        if load < rhs:
            violations.append(f'  server_load_{i}: {load:.8f} < {rhs} (gap={rhs-load:.2e})')

    # cons_vm_alloc: sum_i v[(j,i)] <= limit
    limits = d['input']['vm_allocation_limits']
    for j in range(V):
        lim = limits[j] if j < len(limits) else limits[-1]
        alloc = sum(v_vals[j*S + i] for i in range(S))
        if alloc > lim:
            violations.append(f'  vm_alloc_{j}: {alloc:.8f} > {lim} (gap={alloc-lim:.2e})')

    # cons_min_cpu: u[j] >= min_cpu
    mcpu = d['input']['min_cpu_per_vm']
    for j in range(V):
        if u_vals[j] < mcpu:
            violations.append(f'  min_cpu_{j}: {u_vals[j]:.8f} < {mcpu} (gap={mcpu-u_vals[j]:.2e})')

    if violations:
        print(f'S={S} V={V}: INFEASIBLE')
        for v in violations:
            print(v)
    else:
        print(f'S={S} V={V}: INFEASIBLE but NO violations found (constraint check may be wrong)')
