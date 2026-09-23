# type: ignore
mdl = Model("server_vm_allocation")

# Binary variables
s = mdl.binary_var_list(n_servers)

# Continuous variables
v = [[mdl.continuous_var(lb=0)
      for i in range(n_servers)]
      for j in range(n_vms)]

u = mdl.continuous_var_list(
        n_vms, lb=min_cpu_per_vm)



qp = from_docplex_mp(mdl)

admm = ADMMOptimizer(
    qubo_optimizer=MinimumEigenOptimizer(
        NumPyMinimumEigensolver()
    ),
    continuous_optimizer=CobylaOptimizer(),
    params=admm_params
)

qubo_solver = MinimumEigenOptimizer(
    NumPyMinimumEigensolver()
)

continuous_solver = CobylaOptimizer()


qaoa = QAOA(
    sampler=StatevectorSampler(),
    optimizer=COBYLA(maxiter=300),
    reps=3
)


qubo_solver = MinimumEigenOptimizer(qaoa)



need_load = sum(capacities) - n_servers
have_load = sum(vm_allocation_limits)

if have_load < need_load:
    raise ValueError("Infeasible instance")


for _ in range(20):
    # identify LE-tight variables
    le_tight = set()

    # repair GE violations
    candidates = [k for k in coeffs
                  if k not in le_tight]
    
    # repair LE violations
    ...

    x = np.clip(x, lb, ub)


need_ load = sum(capacities) - n_servers.
have_load = sum(vm_allocation_limits)

if have_load < need_load:
    ABORT - infeasible instance

