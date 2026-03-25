import json, sys

with open('merged_20260325_114223.json') as f:
    data = json.load(f)

print(f"Total runs: {data['total_runs']}\n")

for r in data['runs']:
    m = r['meta']
    cs = r['classic']['status']
    cx = r['classic']['x']
    nv = m['n_vms']
    ns = m['n_servers']
    u_classic = cx[ns + nv*ns:]
    v_classic = cx[ns : ns + nv*ns]
    caps = r['input']['capacities']
    lims = r['input']['vm_allocation_limits']

    violations = []

    # server_load: sum_j v[j,i] >= cap[i] - 1
    for i in range(ns):
        load = sum(v_classic[j*ns + i] for j in range(nv))
        needed = caps[i] - 1
        if load < needed - 1e-6:
            violations.append(f"load_s{i}: {load:.6f} < {needed}")

    # vm_alloc: sum_i v[j,i] <= lim[j]
    for j in range(nv):
        alloc = sum(v_classic[j*ns + i] for i in range(ns))
        lim = lims[j] if j < len(lims) else lims[-1]
        if alloc > lim + 1e-6:
            violations.append(f"alloc_v{j}: {alloc:.6f} > {lim}")

    # u_vars >= 0.999
    for j, u in enumerate(u_classic):
        if u < 0.999 - 1e-6:
            violations.append(f"u{j}={u:.8f} < 0.999")

    # server_on: s[i] == 1
    for i in range(ns):
        if abs(cx[i] - 1.0) > 1e-6:
            violations.append(f"s{i}={cx[i]:.6f} != 1")

    viol_str = ", ".join(violations) if violations else "NONE FOUND"
    print(f"S={ns} V={nv} | {cs:11s} | violations: {viol_str}")
