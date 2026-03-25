import json, glob, numpy as np

# Find all new batch files (15:27+ timestamp)
files = sorted(glob.glob('q_20260325_1527*_results.json') + 
               glob.glob('q_20260325_1528*_results.json') +
               glob.glob('q_20260325_1529*_results.json') +
               glob.glob('q_20260325_153*_results.json') +
               glob.glob('q_20260325_154*_results.json') +
               glob.glob('q_20260325_155*_results.json'))

fail_count = 0
ok_count = 0
for f in files:
    d = json.load(open(f))
    S = d['meta']['n_servers']
    V = d['meta']['n_vms']
    cs = d['classic']['status']
    qs = d['quantum']['status']
    if cs == 'INFEASIBLE' or qs == 'INFEASIBLE':
        fail_count += 1
        x = np.array(d['classic']['x'])
        caps = d['input']['capacities']
        limits = d['input']['vm_allocation_limits']
        mcpu = d['input']['min_cpu_per_vm']
        s_vals = x[:S]
        v_vals = x[S:S+S*V]
        u_vals = x[S+S*V:S+S*V+V]
        print(f'S={S} V={V}: classic={cs} quantum={qs}')
        for i in range(S):
            load = sum(v_vals[j*S+i] for j in range(V))
            rhs = caps[i] - 1
            if load < rhs:
                print(f'  server_load_{i}: {load:.15f} < {rhs} (gap={rhs-load:.2e})')
        for j in range(V):
            lim = limits[j] if j < len(limits) else limits[-1]
            alloc = sum(v_vals[j*S+i] for i in range(S))
            if alloc > lim:
                print(f'  vm_alloc_{j}: {alloc:.15f} > {lim} (gap={alloc-lim:.2e})')
        for j in range(V):
            if u_vals[j] < mcpu:
                print(f'  min_cpu_{j}: {u_vals[j]:.15f} < {mcpu} (gap={mcpu-u_vals[j]:.2e})')
        # Also check if any variable out of bounds
        print(f'  v range: [{v_vals.min():.8f}, {v_vals.max():.8f}]')
        print(f'  u range: [{u_vals.min():.8f}, {u_vals.max():.8f}]')
        print()
    else:
        ok_count += 1

print(f'\nTotal: {ok_count} SUCCESS, {fail_count} INFEASIBLE out of {ok_count+fail_count} files')
