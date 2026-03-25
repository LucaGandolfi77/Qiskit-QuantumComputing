import json, numpy as np, glob

for f in sorted(glob.glob('q_20260325_1522*_results.json') + glob.glob('q_20260325_1523*_results.json')):
    d = json.load(open(f))
    if d['classic']['status'] != 'INFEASIBLE':
        continue
    S = d['meta']['n_servers']
    V = d['meta']['n_vms']
    x = np.array(d['classic']['x'])
print(f'S={S} V={V} status={d["classic"]["status"]}')
s_vals = x[:S]
v_vals = x[S:S+S*V]
u_vals = x[S+S*V:]
print('s:', s_vals)
print('v:', v_vals)
print('u:', u_vals)
caps = d['input']['capacities']
limits = d['input']['vm_allocation_limits']
mcpu = d['input']['min_cpu_per_vm']
print(f'caps={caps}, limits={limits}, mcpu={mcpu}')
for i in range(S):
    load = sum(v_vals[j*S+i] for j in range(V))
    rhs = caps[i]-1
    mark = '!!!' if load < rhs else 'ok'
    print(f'  server_load_{i}: {load:.12f} >= {rhs} {mark}')
for j in range(V):
    lim = limits[j] if j < len(limits) else limits[-1]
    alloc = sum(v_vals[j*S+i] for i in range(S))
    mark = '!!!' if alloc > lim else 'ok'
    print(f'  vm_alloc_{j}: {alloc:.12f} <= {lim} {mark}')
for j in range(V):
    mark = '!!!' if u_vals[j] < mcpu else 'ok'
    print(f'  u_{j}: {u_vals[j]:.12f} >= {mcpu} {mark}')
    print()
