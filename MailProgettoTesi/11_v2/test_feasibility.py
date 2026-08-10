"""Reproduce S=5 V=1 case to check what qiskit's is_feasible rejects."""
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp

n_servers = 5
n_vms = 1
capacities = [11, 11, 11, 10, 10]
pi_list = [1.0]*5
pd_list = [1.0]*5
vm_allocation_limits = [61]
min_cpu_per_vm = 1.0

mdl = Model("test")
s_vars = [mdl.binary_var(name=f"si{i}") for i in range(n_servers)]
v_vars = {}
for j in range(n_vms):
    for i in range(n_servers):
        v_vars[(j, i)] = mdl.continuous_var(lb=0.0, name=f"vj{j}i{i}")
u_vars = [mdl.continuous_var(lb=min_cpu_per_vm, name=f"uj{j}") for j in range(n_vms)]

obj = mdl.sum(
    pi_list[i] * s_vars[i]
    + pd_list[i] * mdl.sum(u_vars[j] * v_vars[(j, i)] for j in range(n_vms))
    for i in range(n_servers)
)
mdl.minimize(obj)

for i in range(n_servers):
    mdl.add_constraint(
        mdl.sum(v_vars[(j, i)] for j in range(n_vms)) >= capacities[i] - 1,
        f"cons_server_load_{i}")

for i in range(n_servers):
    mdl.add_constraint(s_vars[i] == 1, f"cons_server_on_{i}")

for j in range(n_vms):
    mdl.add_constraint(
        mdl.sum(v_vars[(j, i)] for i in range(n_servers)) <= vm_allocation_limits[j],
        f"cons_vm_alloc_{j}")

for j in range(n_vms):
    mdl.add_constraint(u_vars[j] >= min_cpu_per_vm, f"cons_min_cpu_vm_{j}")

qp = from_docplex_mp(mdl)

# The "perfect" solution from ADMM
x = [1.0, 1.0, 1.0, 1.0, 1.0,
     9.999999999999996, 9.999999999999996, 9.999999999999996,
     8.999999999999996, 8.999999999999996,
     1.0000000000000002]

print("Variables in QP:")
for i, v in enumerate(qp.variables):
    print(f"  [{i}] {v.name} type={v.vartype} lb={v.lowerbound} ub={v.upperbound} -> x={x[i]}")

print("\nConstraints in QP:")
for c in qp.linear_constraints:
    print(f"  {c.name}: sense={c.sense}, rhs={c.rhs}")

print("\nis_feasible:", qp.is_feasible(x))
feasible, violated_vars, violated_constraints = qp.get_feasibility_info(x)
print(f"feasible={feasible}")
print(f"violated_vars={[(v.name, x[qp.variables.index(v)]) for v in violated_vars]}")
print(f"violated_constraints={[(c.name,) for c in violated_constraints]}")
