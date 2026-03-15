# Quantum Optimization — Allocazione VM a Server Fisici

## Descrizione del problema

Questo progetto risolve il problema di **allocazione di macchine virtuali (VM) a server fisici**
per **minimizzare il consumo energetico** dell'infrastruttura, senza sovraccaricare i server.

### Formulazione matematica

$$
\min \sum_{i=1}^{M} \left[ P^I_i \cdot s_i + P^D_i \cdot l_i \right]
$$

Soggetto a:

$$
l_i = \sum_{j=1}^{N} u_j \cdot v_{ji}, \quad \forall i \in [1, M] \quad \text{(definizione carico)}
$$

$$
l_i \leq C_i \cdot s_i, \quad \forall i \in [1, M] \quad \text{(capacità server)}
$$

$$
\sum_{i=1}^{M} v_{ji} = 1, \quad \forall j \in [1, N] \quad \text{(ogni VM assegnata a 1 server)}
$$

Dove:
- $s_i \in \{0,1\}$ — server $i$ acceso/spento
- $v_{ji} \in \{0,1\}$ — VM $j$ assegnata al server $i$
- $l_i \in [0, C_i]$ — carico CPU continuo sul server $i$ (variabile continua)
- $P^I_i$ — consumo idle del server $i$
- $P^D_i$ — consumo dinamico per unità di carico del server $i$
- $C_i$ — capacità CPU massima del server $i$
- $u_j$ — carico CPU della VM $j$

Le variabili continue $l_i$ sono essenziali per la decomposizione ADMM:
l'ADMM separa il problema in un sottoproblema **QUBO** (variabili binarie $s_i, v_{ji}$)
e un sottoproblema **convesso** (variabili continue $l_i$).

### Parametri di default (M=2, N=3)

| Parametro | Valori |
|-----------|--------|
| $P^I$ (idle) | [100, 120] W |
| $P^D$ (dinamico) | [50, 60] W |
| $C$ (capacità) | [4, 5] |
| $u$ (carico VM) | [2, 1, 3] |

## Flusso di risoluzione

```
QuadraticProgram (QP)
    │
    ▼
QUBO (Quadratic Unconstrained Binary Optimization)
    │  ← penalizzazione dei vincoli nella funzione obiettivo
    │  ← introduzione variabili slack per vincoli ≤
    ▼
Hamiltoniano di Ising
    │  ← mappatura x_i = (1 - Z_i) / 2
    ▼
ADMM (Alternating Direction Method of Multipliers)
    ├── QUBO sub-problem  ← risolto con QAOA (quantistico) o NumPy (classico)
    ├── Continuous sub-problem  ← risolto con solver convesso
    └── Aggiornamento moltiplicatori duali
```

### Decomposizione ADMM

L'ADMM decompone il problema misto-intero in:

1. **QUBO sub-problem** (variabili binarie $x_0$):
   Minimizza $f(x_0) + u^T x_0 + \frac{\rho}{2} \|x_0 - z\|^2$ con $x_0 \in \{0,1\}^n$

2. **Continuous sub-problem** (variabili di consenso $z$):
   Minimizza $g(z) - u^T z + \frac{\rho}{2} \|x_0 - z\|^2$ soggetto ai vincoli lineari

3. **Aggiornamento duale**: $u \leftarrow u + \rho(x_0 - z)$

## Dimensioni del problema (M=2, N=3)

| Grandezza | Valore |
|-----------|--------|
| Variabili binarie (s + v) | 8 (2 + 3×2) |
| Variabili continue (l) | 2 |
| Variabili originali totali | 10 |
| Variabili QUBO (binarie + slack) | 14 |
| Variabili slack | 6 |
| Qubit Ising | 14 |
| Vincoli uguaglianza | 5 (2 load_def + 3 assign) |
| Vincoli disuguaglianza | 2 (capacità server) |

## Installazione

```bash
# Creare un ambiente virtuale (consigliato)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# Installare le dipendenze
pip install -r requirements.txt
```

## Esecuzione

```bash
# Eseguire il flusso completo
python main.py

# Eseguire singoli moduli per test
python problem_formulation.py   # solo formulazione e analisi QUBO
python admm_solver.py           # solo risoluzione ADMM
python inspect_subproblems.py   # solo ispezione sottoproblemi
python results_analysis.py      # solo analisi risultati
```

## Struttura del progetto

```
QAOA/
├── requirements.txt           # dipendenze Python
├── README.md                  # questo file
├── main.py                    # entry point, esegue tutto il flusso
├── problem_formulation.py     # definizione del QuadraticProgram
├── admm_solver.py             # configurazione e lancio ADMMOptimizer
├── inspect_subproblems.py     # ispezione sottoproblemi QUBO e convex
├── results_analysis.py        # visualizzazione risultati e convergenza
└── convergence.png            # plot generato (dopo esecuzione)
```

## Dipendenze principali

- **qiskit >= 1.0** — framework di quantum computing
- **qiskit-optimization 0.7.0** — modulo di ottimizzazione (ADMM, QUBO, converters)
- **qiskit-algorithms** — algoritmi quantistici (NumPyMinimumEigensolver)
- **docplex** — modellazione matematica IBM (opzionale, con fallback)
- **cvxpy** — solver per problemi convessi
- **matplotlib** — visualizzazione

## Riferimento

> Gambella C., Simonetto A.,
> *"Multi-block ADMM Heuristics for Mixed-Binary Optimization
>  on Classical and Quantum Computers"*,
> IEEE Transactions on Quantum Engineering (TQE), 2020.
> DOI: 10.1109/TQE.2020.2994748
