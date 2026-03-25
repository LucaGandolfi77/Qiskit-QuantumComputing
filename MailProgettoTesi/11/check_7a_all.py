import json

with open('/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/7a/merged_20260325_115018.json') as f:
    data = json.load(f)

print(f"Total runs: {data['total_runs']}\n")

# Show ALL runs with status
for r in data['runs']:
    m = r['meta']
    cs = r['classic']['status']
    qs = r['quantum']['status']
    co = r['classic']['objective']
    qo = r['quantum']['objective']
    nv = m['n_vms']
    ns = m['n_servers']
    x = r['classic']['x']
    v_vals = x[ns : ns + nv*ns]

    # Check if all constraints satisfied at 1e-5
    all_ok = True
    caps = r['input']['capacities']
    lims = r['input']['vm_allocation_limits']
    for i in range(ns):
        load = sum(v_vals[j*ns + i] for j in range(nv))
        if load < caps[i] - 1 - 1e-5:
            all_ok = False

    print(f"S={ns} V={nv} | C:{cs:11s} Q:{qs:11s} | obj_c={co:10.4f} obj_q={qo:10.4f} | my_check={'OK' if all_ok else 'FAIL'}")
