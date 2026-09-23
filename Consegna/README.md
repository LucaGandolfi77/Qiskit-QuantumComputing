Studio presentazione


# Quantum-Classical Hybrid Optimization of Server-VM Allocation
### A Comparative Study Using DOcplex, ADMM, and QAOA
**Presenter:** [Luca Gandolfi]  
**Venue:** [University / Engineering Conference]  
**Date:** [10/09/2026]
---
## Agenda
1. Problem Overview — what we are trying to solve and why it matters
2. Model Design — variables, objective, and constraints
3. Classical Approach — ADMM with exact eigensolver and COBYLA
4. Quantum Approach — ADMM with QAOA
5. Feasibility and Post-processing — pre-checks and the `snap_to_feasible` routine
6. Results — 36-run benchmark across all configuration sizes
7. Strengths and Limitations — honest assessment of the framework
8. Future Work — where this research can go next
9. Conclusion — key takeaways
---
## 1. Problem Overview
### The setting
Modern data centers face a continuous allocation challenge: given **N physical servers**
and **M virtual machines (VMs)**, how do you distribute workloads to minimize total operating cost while satisfying all capacity and service-level constraints?
This is not a toy problem. It underpins cloud computing billing, energy efficiency in hyperscale infrastructure, and real-time auto-scaling decisions.
### Why it is hard
The difficulty arises from the *mixed nature* of the decisions involved:
- **Which servers to activate?** — this is a discrete (binary) decision.
- **How much of each VM to allocate to each server?** — this is a continuous decision.
- **How much CPU to guarantee each VM?** — another continuous decision with a hard lower bound.
Problems combining binary and continuous variables are known as
**Mixed-Integer Programming (MIP)** problems. MIP is NP-hard in general.
Classical solvers handle small instances well but struggle to scale.
This is precisely where quantum computing promises a potential future advantage.
### Research question
> Can a quantum-inspired solver (QAOA embedded in ADMM) find the same optimal solutions
> as a classical exact solver, and under what conditions might it become competitive?
---
## 2. Model Design
### 2.1 Decision Variables
The model defines three families of variables:
| Variable | Type | Meaning |
|---|---|---|
| `sᵢ` ∈ {0, 1} | Binary | Server *i* is ON (1) or OFF (0) |
| `vⱼᵢ` ≥ 0 | Continuous | Allocation of VM *j* to server *i* |
| `uⱼ` ≥ `min_cpu` | Continuous | CPU units guaranteed to VM *j* |
With N=6 servers and M=6 VMs, the model has:
- 6 binary variables
- 36 continuous allocation variables
- 6 continuous CPU variables
- **48 decision variables total**
### 2.2 Objective Function
The cost to minimize combines two terms:
```
minimize  Σᵢ [ πᵢ · sᵢ  +  ρᵢ · Σⱼ ( uⱼ · vⱼᵢ ) ]
```
- **πᵢ** — fixed activation cost for server *i* (paid whenever the server is ON)
- **ρᵢ** — dynamic cost multiplier (cost per unit of CPU × allocation on server *i*)
This structure captures both the base infrastructure cost and the variable utilization
cost in a single objective. In the experiments reported here, uniform costs (π=ρ=1.0)
are used to isolate the structural behaviour of the optimizer.
> **Design note:** the interaction term `uⱼ · vⱼᵢ` is *bilinear* (a product of two
> decision variables), making this a **Quadratic Program (QP)**. This is key to why
> Qiskit's QUBO-based quantum solver can engage with it.
### 2.3 Constraints
Four constraint families are enforced:
**① Server load:** Each server must carry at least its capacity minus one unit of load.
```
Σⱼ vⱼᵢ  ≥  capᵢ − 1     for each server i
```
This ensures servers are not idle. Capacities in the benchmark range from 10 to 11 units.
**② Server activation:** When `require_all_on=True`, all servers are forced active.
```
sᵢ = 1     for each server i
```
This constraint, while restrictive, models a scenario where SLAs mandate full infrastructure
availability — common in enterprise or regulated environments.
**③ VM allocation limits:** Each VM has a total allocation budget across all servers.
```
Σᵢ vⱼᵢ  ≤  limⱼ     for each VM j
```
This prevents any single VM from monopolizing server capacity.
**④ Minimum CPU:** Each VM must receive a guaranteed CPU floor.
```
uⱼ  ≥  min_cpu_per_vm     (default: 1.0 units)
```
### 2.4 The Quadratic Program
Once the DOcplex model is built, it is translated directly into a
`QuadraticProgram` via `from_docplex_mp(mdl)`. This is the canonical
representation that Qiskit's ADMM optimizer operates on. The LP export
of this program is also saved to the output JSON for full reproducibility.
---
## 3. Classical Approach
### 3.1 The ADMM Framework
**ADMM** — Alternating Direction Method of Multipliers — is a decomposition strategy
that splits a complex mixed-integer problem into two simpler, alternating subproblems:
```
┌─────────────────────────────────────────────────────┐
│                  ADMM Iteration Loop                │
│                                                     │
│   1.  Binary subproblem  → solve for s (QUBO)       │
│   2.  Continuous subproblem → solve for v, u (QP)   │
│   3.  Update dual variables (Lagrangian multipliers) │
│   4.  Check residual convergence → stop if < tol    │
└─────────────────────────────────────────────────────┘
```
ADMM is particularly effective here because the binary and continuous parts
of the problem have very different structures — and we can choose the best
solver for each independently.
### 3.2 Solvers Used
| Subproblem | Solver | Role |
|---|---|---|
| Binary (QUBO) | `NumPyMinimumEigensolver` | Exact diagonalization — finds the global minimum |
| Continuous (QP) | `CobylaOptimizer` | Gradient-free classical optimizer for the remaining QP |
### 3.3 ADMM Parameters
```python
ADMMParameters(
    rho_initial = 100,    # Penalty term for constraint violations
    beta        = 1000,   # Quadratic regularization weight
    factor_c    = 900,    # Augmented Lagrangian scaling factor
    maxiter     = 100,    # Maximum ADMM iterations
    three_block = True,   # Separate u-block for continuous CPU variables
    tol         = 1e-4,   # Convergence tolerance (aligned to feasibility threshold)
)
```
> **Note on `tol=1e-4`:** This tolerance was deliberately aligned with the feasibility
> check threshold used by Qiskit's `is_feasible()`. A tighter tolerance caused false
> INFEASIBLE labels at convergence boundaries — this is a subtle but important
> engineering choice.
### 3.4 Classical Performance Summary
Across 36 benchmark runs (N × M configurations from 1×1 to 6×6):
| Metric | Value |
|---|---|
| All runs status | ✅ SUCCESS (36/36) |
| Fastest run (1×1) | 0.08 s |
| Slowest run (4×6) | 15.08 s |
| Mean runtime | 2.29 s |
---
## 4. Quantum Approach
### 4.1 QAOA — The Algorithm
**QAOA** (Quantum Approximate Optimization Algorithm) is a variational quantum algorithm
designed for combinatorial optimization. It applies a sequence of parameterized quantum
gates to encode the QUBO problem into a quantum circuit:
```
|ψ(γ, β)⟩ = [ U_mixer(β_p) · U_problem(γ_p) ]^p · |+⟩^n
```
- **U_problem** encodes the cost function via phase rotations
- **U_mixer** applies transverse field rotations to explore the solution space
- **p (reps)** is the circuit depth — higher p gives better approximations at higher cost
- **γ, β** are variational parameters optimized classically
The expectation value ⟨ψ(γ,β)| Ĥ_cost |ψ(γ,β)⟩ is minimized to find the best
bitstring — the approximate solution to the QUBO.
### 4.2 Configuration
```python
QAOA(
    sampler   = StatevectorSampler(),   # Exact statevector simulation
    optimizer = COBYLA(maxiter=300),    # Classical outer optimization loop
    reps      = 3,                      # Circuit depth p=3
)
```
The `StatevectorSampler` runs an exact quantum simulation on classical hardware.
This is ideal for validation and benchmarking, though it does not scale to large
qubit counts.
### 4.3 How QAOA Plugs into ADMM
QAOA replaces the `NumPyMinimumEigensolver` as the QUBO solver:
```
ADMM iteration
  └─> QUBO subproblem
        └─> MinimumEigenOptimizer( QAOA )
              └─> quantum circuit executes
              └─> COBYLA adjusts γ, β
              └─> returns approximate binary solution
```
Everything else — the continuous QP solved by COBYLA, the dual updates, the
convergence check — remains identical to the classical pipeline. This makes the
comparison clean and architecturally fair.
### 4.4 Quantum Performance Summary
| Metric | Value |
|---|---|
| All runs status | ✅ SUCCESS (36/36) |
| Fastest run (1×1) | 1.26 s |
| Slowest run (6×3) | 70.79 s |
| Mean runtime | 11.50 s |
---
## 5. Feasibility and Post-processing
### 5.1 Pre-run Feasibility Check
Before any solver is invoked, the code performs a structural feasibility check:
```python
need_load = sum(capacities) - n_servers
have_load = sum(vm_allocation_limits)
if have_load < need_load:
    ABORT — problem is structurally infeasible
```
**Why this matters:** If the VM allocation budgets collectively cannot cover the
minimum server loads required by the constraints, no solver — classical or quantum —
can find a feasible solution. Running an optimizer on an infeasible problem wastes
time and produces misleading output. This check costs microseconds and prevents
exactly that scenario.
The `safe_vm_alloc()` function generates allocation limits that pass this check
automatically when none are provided, adding a 25% margin for robustness.
### 5.2 The `snap_to_feasible` Correction Step
ADMM is an iterative, approximate algorithm. Its solutions often satisfy constraints
to within a small numerical tolerance (e.g., 10⁻⁴) but not exactly — and Qiskit's
`is_feasible()` method applies **strict zero-tolerance** checks.
Without correction, a near-optimal, near-feasible solution would be flagged as
`INFEASIBLE`, even though it is essentially correct. The `snap_to_feasible` function
addresses this with a targeted post-processing pass:
```
1. Clip all variables to their declared [lb, ub] bounds
2. For up to 20 iterations:
   a. Find GE (≥) constraint violations → increase eligible continuous vars
   b. Find LE (≤) constraint violations → decrease eligible continuous vars
   c. Prioritise variables not already at a ceiling to avoid oscillation
   d. Re-clip after each adjustment
3. Recompute feasibility status and objective value
```
> **Key engineering insight:** The routine preferentially adjusts variables
> *not already pressed against an upper bound*. Without this, fixing one
> constraint could break another (oscillation). The ceiling-aware selection
> ensures monotonic convergence.
After snapping:
- `result._status` is set to `SUCCESS` if `qp.is_feasible(x)` passes
- `result._fval` is recomputed from the corrected variable values
- The objective is perturbed by at most ~10⁻³ from the ADMM value
This is not "cheating" — it is a standard technique in MIP post-processing,
equivalent to rounding and local repair.
---
## 6. Results
### 6.1 Benchmark Setup
- **36 runs** across all (N, M) configurations: N ∈ {1…6} servers, M ∈ {1…6} VMs
- All runs: `require_all_on=True`, `min_cpu_per_vm=1.0`, uniform costs (π=ρ=1.0)
- Server capacities: 11 units for the first three servers, 10 for the remainder
- VM allocation limits: auto-generated with a 25% safety margin
- Results serialized to JSON and CSV for reproducibility
### 6.2 Objective Values
Both approaches converge to the same objective value across all 36 configurations.
The pattern is remarkably clean:
| # Servers | Objective (any # VMs) |
|---|---|
| 1 | ≈ 11.00 |
| 2 | ≈ 22.00 |
| 3 | ≈ 33.00 |
| 4 | ≈ 43.00 |
| 5 | ≈ 53.00 |
| 6 | ≈ 63.00 |
> This linear scaling confirms that, under uniform costs and the `require_all_on`
> constraint, the objective is dominated by server capacity — the solver correctly
> minimizes the utilization terms to their lower bounds.
**Both classical and quantum find the same optimum in every single run.**
The maximum deviation between the two approaches across all 36 runs is less than 10⁻⁶.
### 6.3 Runtime Comparison
```
Configuration   Classical   Quantum     Quantum Overhead
─────────────────────────────────────────────────────────
1×1             0.08 s       1.26 s      15.7×  slower
2×1             0.17 s       1.64 s       9.7×  slower
3×1             0.21 s       2.23 s      10.4×  slower
4×1             0.29 s       3.21 s      11.2×  slower
5×1             0.41 s       6.11 s      15.1×  slower
6×1             0.51 s      24.16 s      47.3×  slower
─────────────────────────────────────────────────────────
3×3             0.74 s       2.77 s       3.7×  slower
4×4             1.55 s       5.21 s       3.4×  slower
5×5             2.98 s      10.04 s       3.4×  slower
6×6            11.23 s      53.36 s       4.8×  slower
─────────────────────────────────────────────────────────
Mean            2.29 s      11.50 s       ~5×   slower
```
**Key observation:** The quantum overhead is not constant. It is largest when
the QUBO subproblem has few binary variables (small N) because QAOA's
circuit initialization and COBYLA convergence carry a fixed per-call overhead
that dwarfs the binary optimization itself. As problem size grows (more servers,
more VMs), the ratio improves slightly — but the classical solver remains
faster across the entire tested range.
### 6.4 Outputs Generated
For each run, the framework saves:
| Output | Format | Contents |
|---|---|---|
| `q_<timestamp>_results.json` | JSON | Full run metadata, input parameters, LP formulation, solution vectors, residuals, timings |
| `q_<timestamp>_results.png` | PNG | 2×2 grid: ADMM residual curves + solution bar charts for both approaches |
| `merged_<timestamp>.json` | JSON | Merged results from all 36 runs |
| `merged_<timestamp>.csv` | CSV | Tabular summary with objective, status, timings, speedup, and residual counts |
The JSON output includes the full LP string of the QuadraticProgram, ensuring
that any run can be independently verified or reproduced.
---
## 7. Strengths and Limitations
### Strengths
**✅ Architecturally clean comparison**  
Both solvers operate within the same ADMM framework, differing only in
the QUBO subproblem solver. This makes the comparison fair and controlled.
**✅ Robust feasibility handling**  
The combination of a pre-run structural check and the `snap_to_feasible`
post-processor ensures that solver output is always interpretable, regardless
of ADMM's convergence path.
**✅ Full reproducibility**  
Every parameter, cost coefficient, constraint, and solution vector is
serialized to JSON. Any run can be re-verified without re-executing the optimizer.
**✅ Modular and extensible**  
Adding a new solver (e.g., VQE, simulated annealing) requires only swapping
the `qubo_optimizer` argument in `ADMMOptimizer`.
**✅ Configurable via CLI**  
All problem parameters (server count, VM count, costs, capacities) can be
overridden at the command line without modifying the source code.
---
### Limitations
**⚠️ Problem size cap (MAX_N = 7)**  
QAOA's circuit depth grows with the number of binary variables (servers).
Beyond 7 servers, the statevector simulation becomes computationally prohibitive
on classical hardware.
**⚠️ Simulation, not real quantum hardware**  
All quantum runs use `StatevectorSampler` — exact simulation. Real quantum
hardware introduces gate errors, decoherence, and shot noise. Results would
differ materially.
**⚠️ Uniform cost coefficients**  
All experiments use π=ρ=1.0 for all servers. This simplifies analysis but
may not surface interesting trade-offs between the two approaches. Heterogeneous
cost landscapes may produce different relative results.
**⚠️ Quantum is consistently slower here**  
In this simulation regime, QAOA offers no computational advantage over the
exact eigensolver. The quantum overhead is real and significant (2×–47×).
This is expected: quantum advantage is a hardware, not a simulation, phenomenon.
**⚠️ `snap_to_feasible` modifies results**  
The correction step improves feasibility compliance but introduces a small
perturbation to the objective value. In some runs the corrected objective
differs from the raw ADMM objective by up to ~10⁻³. This is negligible
in practice but should be disclosed in rigorous benchmarking.
---
## 8. Future Work
### Near-term
- **Heterogeneous cost parameters** — introduce varying π and ρ per server
  to model real pricing structures (e.g., on-demand vs. reserved instances)
  and stress-test whether QAOA's landscape navigation differs from exact search.
- **Warm starting QAOA** — initialize variational parameters γ, β from the
  classical solution to reduce COBYLA iterations and improve circuit convergence.
- **Larger problems via circuit cutting** — use Qiskit's circuit cutting
  utilities to distribute larger QUBO subproblems across multiple smaller
  quantum circuits, extending beyond the MAX_N=7 limit.
### Medium-term
- **Real quantum hardware execution** — run on IBM Quantum or similar backends
  to measure the impact of noise, decoherence, and readout errors on solution quality.
- **Alternative quantum algorithms** — replace QAOA with VQE (Variational Quantum
  Eigensolver) or quantum annealing formulations, and compare convergence profiles.
- **Dynamic constraint generation** — implement lazy constraint addition to
  handle large VM counts without encoding all constraints at once.
### Long-term
- **Online / streaming allocation** — extend the model to handle workloads
  arriving over time, turning the static QP into a dynamic optimization problem.
- **Multi-objective formulation** — add energy consumption as a second objective,
  creating a Pareto front between cost and carbon footprint.
- **Integration with real infrastructure APIs** — connect the optimizer to a
  Kubernetes or OpenStack scheduler to demonstrate end-to-end deployment.
---
## 9. Conclusion
### Summary
This project demonstrates a complete, working framework for solving a
server-VM allocation problem using both classical and quantum-inspired approaches —
built on DOcplex, Qiskit Algorithms, and the ADMM decomposition strategy.
The key results across 36 benchmark runs are unambiguous:
| Metric | Outcome |
|---|---|
| Solution quality | Both approaches find the same optimal value in all 36 cases |
| Classical speed | Mean 2.3 s, range 0.08 s – 15.1 s |
| Quantum speed | Mean 11.5 s, range 1.3 s – 70.8 s |
| Quantum overhead | 2×–47× slower in simulation |
| Feasibility | 100% success rate with pre-check + snap post-processor |
### Key Takeaways
1. **ADMM is a powerful decomposition strategy** for mixed-integer problems.
   It cleanly separates the hard binary decisions from the tractable continuous ones.
2. **QAOA can match exact solvers in solution quality** on problems of this scale —
   but not yet in speed, particularly in simulation.
3. **Engineering robustness matters as much as algorithmic elegance.** The
   feasibility pre-check and the `snap_to_feasible` routine were not afterthoughts —
   they are what makes the framework reliable in practice.
4. **This is a benchmark, not a deployment.** The value of this work lies in
   establishing a reproducible comparison baseline. The quantum advantage story
   will need real hardware to advance beyond this point.
5. **The framework is ready to grow.** Every component is modular, every result
   is serialized, and the problem formulation is extensible. The next step —
   heterogeneous costs, larger circuits, real hardware — is within reach.
---
> *"Optimization is not just about finding a minimum. It is about understanding
> the shape of the problem well enough to know which tools deserve to hold it."*
---
**Thank you — Questions welcome.**
---
### Appendix A — Software Stack
| Library | Role |
|---|---|
| `docplex` | Algebraic modeling of the MIP |
| `qiskit` | Quantum circuit representation |
| `qiskit-aer` | High-performance quantum simulator |
| `qiskit-algorithms` | QAOA, NumPyMinimumEigensolver |
| `qiskit-optimization` | ADMMOptimizer, QuadraticProgram, translators |
| `matplotlib` | Residual curves and solution bar charts |
| `numpy` | Numerical operations in `snap_to_feasible` |
| `argparse` | CLI parameter configuration |
### Appendix B — ADMM Convergence Notes
ADMM convergence is tracked via the *residual* — the norm of the primal constraint
violation at each iteration. In the benchmark data, most configurations converge
in a single ADMM iteration (residual drops to 0.0 immediately), while more
complex instances (e.g., 4×6) require up to 5 iterations. This is consistent
with the theoretical behaviour of ADMM on well-conditioned quadratic programs
with strongly convex continuous subproblems.
### Appendix C — Variable Naming Convention
| Code name | Mathematical symbol | Description |
|---|---|---|
| `si{i}` | sᵢ | Binary: server *i* active |
| `vj{j}i{i}` | vⱼᵢ | Continuous: VM *j* load on server *i* |
| `uj{j}` | uⱼ | Continuous: CPU for VM *j* |
| `pi_list[i]` | πᵢ | Fixed server activation cost |
| `pd_list[i]` | ρᵢ | Dynamic utilization cost multiplier |
| `capacities[i]` | capᵢ | Server capacity (load units) |
| `vm_allocation_limits[j]` | limⱼ | Maximum total allocation for VM *j* |






___________________________________________________________________________________________________


Buongiorno professore,
Sono Luca Gandolfi, studente lavoratore del corso di High Performance Computing. 

Mi scuso per il ritardo nella risposta: ci eravamo sentiti lo scorso semestre per le modalità d’esame. 

Ho già passato lo scritto in data xx, con votazione xx.
Quest’estate ho sviluppato il progetto che le invio in allegato con uno zip che include: 
- assignement.pdf con l’assegnamento consegnatomi
- File README.md con tutte le istruzioni per eseguire il progetto
- File .py e .bash dei sorgenti del progetto
- Un pdf con le mie conclusioni

Rimango a disposizione per qualsiasi dubbio sul progetto: essendo lavoratore, sono disponibile anche per un eventuale colloquio in presenza. In quel caso, le chiederei se possibile di saperlo con qualche giorno di anticipo, così da poter chiedere le ore di permesso per recarmi in università.

Rimango in attesa di una sua risposta.

La ringrazio per l’attenzione. 
Cordiali saluti,
Luca Gandolfi


____________________________________________________________________________________________________


REPORT-02.md


# Assegnamento - Qiskit - QAOA - Luca Gandolfi
## Descrizione generale
La presente cartella documenta uno studio sperimentale di ottimizzazione combinatoria sviluppato con Qiskit 1.x, finalizzato al confronto tra una procedura di risoluzione classica e una procedura quantistica simulata applicate al problema di allocazione del carico tra server fisici e macchine virtuali.
L'interesse del caso di studio risiede nella possibilita di confrontare due varianti della stessa pipeline:
- una configurazione classica basata su `ADMM + NumPyMinimumEigensolver + COBYLA`
- una configurazione quantistica simulata basata su `ADMM + QAOA + COBYLA`
Le istanze considerate coprono il dominio `N, M in {1, ..., 6}`, dove `N` rappresenta il numero di server e `M` il numero di macchine virtuali.
## Inquadramento del problema
Il modello implementato in [qiskit_opt.py] descrive un problema di minimizzazione dei costi in ambito data center. Le variabili decisionali sono le seguenti:
- variabili binarie `s_i`, che rappresentano lo stato di attivazione del server `i`
- variabili continue `v_{j,i}`, che rappresentano la quota di carico della VM `j` allocata sul server `i`
- variabili continue `u_j`, che rappresentano il livello minimo di CPU associato alla VM `j`
La funzione obiettivo combina due contributi:
- un costo fisso di attivazione dei server
- un costo variabile associato all'utilizzo delle risorse computazionali
I vincoli principali del modello impongono che:
- ciascun server riceva un livello minimo di carico pari a `capacities[i] - 1`
- ciascuna VM rispetti il proprio limite massimo di allocazione
- ciascuna VM soddisfi un livello minimo di CPU pari a `min_cpu_per_vm`
- tutti i server risultino attivi quando il parametro `require_all_on` e impostato a vero
Il problema viene definito in DOcplex e successivamente tradotto in `QuadraticProgram`, in modo da poter essere trattato tramite gli strumenti di `qiskit-optimization`.
## Impostazione metodologica
L'approccio adottato si basa su una decomposizione ADMM, utilizzata per separare la parte discreta del problema dalla parte continua. In questo schema:
- il sottoproblema combinatorio viene trattato come QUBO
- il sottoproblema continuo viene risolto tramite COBYLA
- il confronto tra le due pipeline dipende esclusivamente dal metodo scelto per il sottoproblema QUBO
Le due configurazioni sperimentali sono quindi:
- soluzione classica, ottenuta con `NumPyMinimumEigensolver`
- soluzione quantistica simulata, ottenuta con `QAOA` e `StatevectorSampler`
Un elemento metodologico rilevante e' la funzione `snap_to_feasible()`, introdotta per correggere lievi scostamenti numerici prodotti da ADMM. Tale passaggio e' necessario per evitare che soluzioni praticamente ammissibili vengano classificate come `INFEASIBLE` a causa della verifica rigorosa di fattibilita' eseguita da Qiskit.
Prima della risoluzione viene inoltre eseguito un controllo strutturale di fattibilita', basato sulla condizione:
```text
sum(vm_allocation_limits) >= sum(capacities) - n_servers
```
Se tale condizione non risulta soddisfatta, l'istanza viene scartata prima dell'avvio del solver in ottica di risparmio computazionale.
## Organizzazione della cartella
La cartella e' organizzata come una pipeline sperimentale completa, dalla definizione del modello alla produzione dei materiali finali di sintesi.
### 1. Script principale
[qiskit_opt.py] costituisce il nucleo dell'esperimento. Lo script:
- costruisce il modello DOcplex
- converte il modello in `QuadraticProgram`
- esegue la risoluzione classica
- esegue la risoluzione quantistica simulata
- misura i tempi di esecuzione
- salva un file JSON con parametri, modello LP, risultati e residuali
- salva un'immagine PNG con il confronto grafico tra le due soluzioni
Esempio di esecuzione di una singola istanza:
```bash
python qiskit_opt.py \
  --n_servers 5 \
  --n_vms 5 \
  --require_all_on 1 \
  --min_cpu_per_vm 1.0 \
  --capacities 11,11,11,10,10 \
  --vm_allocation_limits 15,15,15,15,15 \
  --pi_list 1,1,1,1,1 \
  --pd_list 1,1,1,1,1
```
L'opzione `--fast` consente di eseguire test rapidi riducendo il costo computazionale della configurazione QAOA. Grazie a questa opzione, i tempi di esecuzione si riducono da ore intere a poche decine di minuti
### 2. Esecuzione batch
[run_batch_qiskit.py] automatizza l'esecuzione della griglia completa di istanze `1..6 x 1..6`. Lo script:
- genera i valori di default per `capacities`
- costruisce limiti `vm_allocation_limits` con margine di sicurezza
- richiama [qiskit_opt.py] come processo separato
- salva i log testuali nella directory `batch_runs/`
Esempi di utilizzo:
```bash
python run_batch_qiskit.py
python run_batch_qiskit.py --fast
python run_batch_qiskit.py --seed 42
```
### 3. Aggregazione dei risultati
[merge_results.py] esegue il merge dei file `*results.json` e produce:
- un file JSON aggregato, contenente tutte le run
- un file CSV piatto, utile per analisi comparative, grafici o successiva reportistica
Tra i campi piu rilevanti del CSV si segnalano:
- `classic_objective` e `quantum_objective`
- `classic_time_s` e `quantum_time_s`
- `speedup_x`, definito come `classic_time / quantum_time`
- `best_solver`
Esempi di utilizzo:
```bash
python merge_results.py
python merge_results.py --dir batch_runs --recursive
python merge_results.py --sort quantum_time
```
### 4. Raggruppamento risultati nella presentazione finale
La presentazione include:
- inquadramento del problema
- formulazione e approccio ADMM
- discussione della procedura `snap_to_feasible()`
- grafici comparativi su tempi, speedup e scalabilita'

## Evidenze sperimentali principali
Dall'analisi del materiale gia salvato nella cartella emergono i seguenti risultati:
- il file di merge aggregato raccoglie 36 istanze, corrispondenti alla griglia completa `6 x 6`
- tutte le istanze risultano `SUCCESS` sia nella pipeline classica sia nella pipeline quantistica simulata
- nel CSV aggregato il solver indicato come migliore in termini di tempo risulta sempre `classical`
- il valore `speedup_x = classic_time / quantum_time` rimane sempre minore di 1
Ne consegue che, nel dataset attualmente disponibile, la configurazione quantistica simulata non mostra un vantaggio prestazionale rispetto alla controparte classica. Il contributo principale di questa cartella e quindi metodologico: essa fornisce una pipeline coerente e riproducibile per il confronto tra due strategie di soluzione sulla medesima formulazione di ottimizzazione.
## Dipendenze software
Le dipendenze principali impiegate dagli script sono:
- `qiskit`
- `qiskit-aer`
- `qiskit-algorithms`
- `qiskit-optimization`
- `docplex`
- `cplex`
- `numpy`
- `matplotlib`
- `python-pptx`
Lo script principale tenta l'installazione automatica di alcuni pacchetti via `pip`. Per una sperimentazione controllata e riproducibile e tuttavia preferibile predisporre l'ambiente software in anticipo.
## Procedura di riproduzione
Per riprodurre il flusso sperimentale si puo procedere come segue.
1. Eseguire una singola istanza di test:
```bash
python qiskit_opt.py --n_servers 3 --n_vms 2
```
2. Eseguire il batch completo:
```bash
python run_batch_qiskit.py
```
3. Aggregare i risultati:
```bash
python merge_results.py
```

## Limiti del caso di studio
L'interpretazione dei risultati richiede alcune cautele:
- la componente quantistica e simulata tramite statevector e non corrisponde a un'esecuzione su hardware quantistico reale
- il problema contiene una parte continua risolta con COBYLA, pertanto il confronto non isola esclusivamente la componente combinatoria
## Sintesi conclusiva
Il progetto presentato puo' essere considerato come un caso di studio completo, strutturato per:
- definire formalmente un problema di allocazione server/VM
- confrontare una soluzione classica e una soluzione quantistica simulata all'interno della stessa architettura ADMM
- eseguire campagne batch su piu istanze
- aggregare e analizzare i risultati in formato JSON e CSV

---
