REPORT TECNICO — SCALING STUDY: QAOA VM ALLOCATION
Versione: 2.0 · Data run scaling: 16 marzo 2026, ore 09:28 · Configurazioni testate: 20

SEZIONE 1 — PANORAMICA DEL MODULO DI SCALING [PPT]
Questi 4 nuovi file costituiscono il modulo di scaling study del progetto: estendono l'esperimento singolo (M=2, N=3) a una griglia completa di istanze con M ∈ {1,2,3,4} server e N ∈ {1,2,3,4,5} VM, per un totale di 20 configurazioni. L'obiettivo è misurare come il tempo di calcolo, la qualità della soluzione e il gap QAOA/Classico si comportano al crescere delle dimensioni del problema.
Schema del flusso scaling

text
config_generator.py
  └── generate_all_configs(maxM=4, maxN=5)
       │  20 configurazioni con parametri deterministici
       ▼
scaling_runner.py  (entry point)
  └── chiama runconfiguration() per ogni (M, N)
       ▼
single_run.py  (motore di calcolo)
  ├── build_qp(M, N, params)      → QuadraticProgram
  ├── get_qubo_info(M, N, params)  → dimensioni QUBO
  ├── ADMM Classico                → risultato + metriche
  ├── ADMM QAOA (se n_qubits ≤ 20) → risultato + metriche
  └── runconfiguration() → dict metriche completo
       ▼
scaling_results.json  (output)
  └── 20 entry con tutte le metriche per (M, N)


SEZIONE 2 — DOCUMENTAZIONE FILE PYTHON (scaling module)
2.1 — config_generator.py
Scopo generale: Genera tutte le 20 combinazioni (M, N) con M ∈ [1, MAXM=4] e N ∈ [1, MAXN=5], assegnando a ciascuna un set di parametri fisici realistici e deterministici (seed = M×100 + N). Garantisce che ogni VM abbia carico ≥ 1 e ≤ capacità minima del server, rendendo il problema sempre fattibile in linea di principio.
Funzioni documentate:
Funzione	Input	Output	Logica principale
generate_all_configs(maxM, maxN)	int, int (default 4, 5)	list[dict]	Doppio loop M×N; per ogni coppia usa random.Random(M*100+N) per generare P_idle [80-150], P_dynamic [40-80], C_capacity [4-10], u_cpu [1, u_max] dove u_max = max(C_min-1, 1)
print_config_summary(configs)	list[dict]	nessuno (stampa)	Tabella ASCII con M, N, n_bin, n_cont, P_idle, u_cpu per ogni configurazione
Parametri generati per configurazione:
Campo	Range	Metodo	Nota
P_idle	W	rng.randint(80, 150)	1 valore per server
P_dynamic	W	rng.randint(40, 80)	1 valore per server
C_capacity	unità	rng.randint(4, 10)	1 valore per server
u_cpu	[1, u_max]	rng.randint(1, u_max)	u_max = max(C_min-1, 1)
Particolarità tecniche:
    • Il seed M*100+N garantisce riproducibilità totale: rieseguire generate_all_configs() produce sempre gli stessi parametri
    • La formula u_max = max(min(C_capacity)-1, 1) assicura che ogni singola VM non superi la capacità minima, rendendo teoricamente possibile l'assegnamento anche con un solo server attivo
    • MAXM e MAXN sono costanti modificabili senza toccare il codice del solver
[PPT] Bullet per slide:
    • Genera 20 istanze (M×N = 4×5) in modo completamente deterministico
    • Parametri fisicamente realistici: P_idle 80-150 W, P_dyn 40-80 W, CPU 4-10 unità
    • Seed = M×100+N: stessi risultati ad ogni esecuzione, confronto equo
    • Garantisce u_cpu ≤ C_min - 1: problemi con soluzioni esistenti (teoricamente)

2.2 — scaling_runner.py
Scopo generale: Entry point dello scaling study. Itera sequenzialmente le 20 configurazioni generate da config_generator, lancia runconfiguration per ciascuna, mostra una barra di progresso nel terminale e salva i risultati aggregati in scaling_results.json. Include gestione degli errori per configurazioni che falliscono completamente.
Funzioni documentate:
Funzione	Input	Output	Logica principale
progress_bar(current, total, width, label)	int, int, int, str	stringa con barra	Calcola percentuale, riempie barra ASCII █░, aggiunge label con M/N corrente
main()	nessuno	nessuno (side effects)	Loop sulle 20 config; chiama runconfiguration; cattura eccezioni; assembla output con metadata; salva JSON; stampa riepilogo
Struttura dell'output scaling_results.json:

json
{
  "metadata": {
    "generated_at": "2026-03-16T09:28:35",
    "total_configs": 20,
    "successful_classical": 20,
    "successful_qaoa": 20,
    "qiskit_optimization_version": "0.7.0",
    "max_M": 4, "max_N": 5,
    "total_time_sec": 1247.08
  },
  "results": [ ... 20 dict ... ]
}
Particolarità tecniche:
    • Esecuzione seriale (non parallela): le 20 configurazioni vengono eseguite una alla volta — single_run.py offre anche versione parallela ma scaling_runner.py usa quella seriale per semplicità e riproducibilità
    • In caso di eccezione completa su una configurazione, viene salvato comunque un dict con errore (tutti i campi a None/False/0) per non interrompere la sequenza
    • Il contatore n_qaoa_ok conta le config dove QAOA non ha errori (include gli "skipped" per n_qubits > 20)
    • Riepilogo finale: mostra la config con più qubit, quella con max gap QAOA, quella più lenta
[PPT] Bullet per slide:
    • Esecuzione sequenziale con progress bar in tempo reale
    • Gestione robusta degli errori: nessun crash anche se una config fallisce
    • Output unico: tutto in scaling_results.json con metadata completo
    • Durata totale del run: 1247 secondi (≈ 20.8 minuti)

2.3 — single_run.py
Scopo generale: Motore di calcolo dello scaling study. Contiene tutte le funzioni necessarie per un singolo run (M, N): costruzione del QP, analisi QUBO, risoluzione classica, risoluzione QAOA con threshold sui qubit, estrazione metriche. Include anche versione parallela e seriale per batch di configurazioni. È il modulo più ricco e complesso dei 4 nuovi file.
Funzioni documentate:
Funzione	Input	Output	Logica
build_qp(M, N, params)	int, int, dict	QuadraticProgram	Come problem_formulation.py ma usa params dict invece di DEFAULT_*; tenta DOcplex con fallback manuale
build_binary_only_qp(M, N, params)	int, int, dict	QuadraticProgram	Versione ausiliaria solo-binaria per analisi QUBO dimensionale
get_qubo_info(M, N, params)	int, int, dict	dict con n_qubo, n_slack, n_qubits, sparsity	Costruisce QP binario, converte in QUBO, calcola sparsità; fallback con stima se conversione fallisce
decode_result(result, M, N)	ADMMOptimizationResult, int, int	dict s, v, l	Usa cu.asarray (GPU-aware) per np.round, poi converte in numpy con cu.asnumpy
extract_residuals(result)	ADMMOptimizationResult	(list[float], int, bool)	Accede a result.state.residuals; ritorna (lista, n_iterazioni, converged)
compute_energy(dec, M, N, params)	dict, int, int, dict	(float, int)	Calcola energia = Σ P_idles_i + Σ P_dynu_j*v_ji; conta server accesi
check_feasibility(result, M, N, params)	ADMMOptimizationResult, int, int, dict	bool	3 check: ogni VM su esattamente 1 server; carico ≤ C_i*s_i; vincoli binari
get_admm_params()	nessuno	ADMMParameters	Stessi parametri di admm_solver.py: rho=10, factor_c=100000, beta=10000
run_configuration(M, N, params, ...)	vari keyword args	dict metriche completo	Funzione centrale: costruisce QP → ADMM classico → ADMM QAOA (con threshold n_qubits) → calcola ratios → ritorna dict con 30+ metriche
worker_run(job)	tupla (M, N, params, ...)	dict metriche	Wrapper top-level picklabile per ProcessPoolExecutor; cattura eccezioni e produce dict di errore
run_configurations_parallel(...)	lista config, vari kwargs	list[dict]	Usa ProcessPoolExecutor + as_completed con timeout globale; salva CSV e JSON opzionali
run_configurations_serial(...)	lista config, vari kwargs	list[dict]	Versione semplice senza multiprocessing; più debuggabile
summarize_results(results)	list[dict]	dict riassunto	Conta OK/errori, medie tempi classico/QAOA/walltime
Parametri di run_configuration e loro significato:
Parametro	Default	Significato
timeout_sec	120	Timeout singolo run (non usato attivamente in seriale)
run_qaoa	True	Se False, salta completamente QAOA
qaoa_max_qubits	20	Soglia: se n_qubits > 20, QAOA viene skippato (qaoa_skipped=True)
qaoa_reps	1	Numero di layer del circuito QAOA (ridotto a 1 rispetto al main.py dove era 2)
qaoa_max_iter	100	Max iterazioni COBYLA per ottimizzazione parametri QAOA
Particolarità tecniche:
    • GPU-aware decoding: decode_result usa cu.xp.rint(x_dev[:M]) — se CuPy è disponibile, l'arrotondamento avviene su GPU
    • Threshold QAOA: il parametro qaoa_max_qubits=20 è il cutoff pratico: oltre i 20 qubit il simulatore statevector richiederebbe 2²⁰ = 1M stati e diventerebbe intrattabile
    • reps=1 vs reps=2: nello scaling study si usa reps=1 (invece di 2 del main.py) per contenere i tempi; ogni iterazione ADMM chiamerebbe QAOA con al massimo 100 valutazioni COBYLA
    • Metriche derivate calcolate:
        ○ obj_diff_pct = (QAOA_obj - Classical_obj) / |Classical_obj| × 100
        ○ iter_ratio = QAOA_iter / Classical_iter
        ○ time_ratio = QAOA_time / Classical_time
    • check_feasibility verifica 3 condizioni hard: assegnamento univoco VM, capacità server rispettata, valori binari in {0,1}
Snippet chiave — Logica di skip QAOA con soglia qubit:

python
if not run_qaoa:
    qaoa_skipped = True; qaoa_skip_reason = "run_qaoa=False"
elif qaoa_max_qubits is not None and qubo_info["n_qubits"] > qaoa_max_qubits:
    qaoa_skipped = True
    qaoa_skip_reason = f"n_qubits>{qaoa_max_qubits}"
else:
    # ... lancia QAOA normalmente
Snippet chiave — Calcolo metriche derivate al termine del run:

python
if qaoa_obj is not None and result_cl.fval != 0:
    obj_diff_pct = (qaoa_obj - result_cl.fval) / abs(result_cl.fval) * 100
if qaoa_iter is not None and cl_iter > 0:
    iter_ratio = qaoa_iter / cl_iter
if qaoa_time is not None and cl_time > 0:
    time_ratio = qaoa_time / cl_time
[PPT] Bullet per slide:
    • Motore unico per tutti i 20 run dello scaling study
    • Threshold QAOA: n_qubits ≤ 20 → eseguito; > 20 → skippato (log tracciato)
    • GPU-aware: se CuPy disponibile, decodifica soluzione su GPU
    • Versione parallela con ProcessPoolExecutor disponibile per speedup su multi-core
    • reps=1 (vs reps=2 del main): bilanciamento qualità/velocità per lo scaling

SEZIONE 3 — ANALISI COMPLETA scaling_results.json [EXCEL]
Metadata del Run
Campo	Valore
generated_at	2026-03-16T09:28:35.776277
total_configs	20
successful_classical	20 (100%)
successful_qaoa	20 (include skipped — 0 errori)
qiskit_optimization_version	0.7.0
max_M	4
max_N	5
Tempo totale run	1247.08 secondi (≈ 20.8 minuti)
[EXCEL] Tabella Completa — Tutte le 20 Configurazioni
M	N	Bin.orig	Slack	QUBO tot	Qubits	Sparsità%	Cont.	Feasible
1	1	2	3	5	5	68.00	1	✅
1	2	3	3	6	6	66.67	1	❌
1	3	4	4	8	8	62.50	1	❌
1	4	5	4	9	9	61.73	1	❌
1	5	6	4	10	10	61.00	1	❌
2	1	4	7	11	11	33.88	2	✅
2	2	6	7	13	13	33.73	2	✅
2	3	8	6	14	14	34.18	2	❌
2	4	10	6	16	16	33.59	2	✅
2	5	12	7	19	19	32.41	2	❌
3	1	6	9	15	15	24.00	3	✅
3	2	9	9	18	18	24.07	3	✅
3	3	12	11	23	23	22.87	3	✅
3	4	15	11	26	26	22.63	3	✅
3	5	18	9	27	27	23.05	3	❌
4	1	8	14	22	22	17.77	4	✅
4	2	12	15	27	27	17.70	4	✅
4	3	16	15	31	31	17.69	4	❌
4	4	20	12	32	32	18.36	4	❌
4	5	24	14	38	38	17.59	4	❌
Nota: Le righe in grassetto hanno n_qubits > 20 → QAOA skippato.
[EXCEL] Risultati Solver — Tutte le 20 Configurazioni
M	N	Obj Classico	Iter Cl	T.Classico (s)	Obj QAOA	Iter QAOA	T.QAOA (s)	Gap%	T.Ratio	QAOA stato
1	1	400.00	49	1.36	400.00	49	8.84	0.00%	6.50	✅ Eseguito
1	2	572.00	71	2.50	572.00	71	14.72	0.00%	5.89	✅ Eseguito
1	3	615.00	72	3.54	615.00	72	17.17	0.00%	4.85	✅ Eseguito
1	4	550.00	74	6.56	550.00	74	22.95	0.00%	3.50	✅ Eseguito
1	5	894.00	74	7.82	894.00	74	27.19	0.00%	3.47	✅ Eseguito
2	1	241.00	68	3.90	241.00	67	16.80	0.00%	4.31	✅ Eseguito
2	2	400.00	68	6.15	400.00	68	24.06	0.00%	3.91	✅ Eseguito
2	3	687.00	73	9.20	765.00	70	37.65	+11.35%	4.09	⚠️ Gap
2	4	822.00	69	11.58	864.00	85	77.51	+5.11%	6.69	⚠️ Gap
2	5	1106.00	76	19.24	1254.00	100	175.13	+13.38%	9.10	⚠️ Gap grande
3	1	338.00	69	6.64	308.00	70	25.74	−8.88%	3.87	✅ QAOA migliore!
3	2	511.00	70	11.95	616.00	79	55.95	+20.61%	4.68	⚠️ Gap molto grande
3	3	890.39	71	19.47	—	—	—	—	—	⛔ Skipped (23>20)
3	4	478.00	69	20.22	—	—	—	—	—	⛔ Skipped (26>20)
3	5	1076.00	73	41.71	—	—	—	—	—	⛔ Skipped (27>20)
4	1	355.00	70	9.73	—	—	—	—	—	⛔ Skipped (22>20)
4	2	1023.00	71	15.74	—	—	—	—	—	⛔ Skipped (27>20)
4	3	766.00	74	26.80	—	—	—	—	—	⛔ Skipped (31>20)
4	4	535.00	74	79.17	—	—	—	—	—	⛔ Skipped (32>20)
4	5	1364.00	73	439.90	—	—	—	—	—	⛔ Skipped (38>20)
[EXCEL] Crescita Dimensionale — Qubit vs (M, N)
M\N	N=1	N=2	N=3	N=4	N=5
M=1	5	6	8	9	10
M=2	11	13	14	16	19
M=3	15	18	23	26	27
M=4	22	27	31	32	38
Soglia QAOA (20 qubit) superata in 8/20 configurazioni.
[EXCEL] Crescita Sparsità Matrice Q vs (M, N)
M\N	N=1	N=2	N=3	N=4	N=5
M=1	68.0%	66.7%	62.5%	61.7%	61.0%
M=2	33.9%	33.7%	34.2%	33.6%	32.4%
M=3	24.0%	24.1%	22.9%	22.6%	23.1%
M=4	17.8%	17.7%	17.7%	18.4%	17.6%
Osservazione chiave: la sparsità diminuisce drasticamente al crescere di M (da 68% con M=1 a 17.6% con M=4), mentre N ha impatto minore. Questo perché aggiungere server introduce molte variabili di assegnamento e nuovi vincoli di capacità, che densificano la matrice Q.
[EXCEL] Tempi di Calcolo — Scaling del Classico
M	N	T.Classico (s)	T.QAOA (s)	T.Wall (s)	Time Ratio
1	1	1.36	8.84	10.21	6.50×
1	2	2.50	14.72	17.23	5.89×
1	3	3.54	17.17	20.71	4.85×
1	4	6.56	22.95	29.52	3.50×
1	5	7.82	27.19	35.02	3.47×
2	1	3.90	16.80	20.70	4.31×
2	2	6.15	24.06	30.22	3.91×
2	3	9.20	37.65	46.87	4.09×
2	4	11.58	77.51	89.10	6.69×
2	5	19.24	175.13	194.37	9.10×
3	1	6.64	25.74	32.39	3.87×
3	2	11.95	55.95	67.91	4.68×
3	3	19.47	—	19.48	—
3	4	20.22	—	20.23	—
3	5	41.71	—	41.72	—
4	1	9.73	—	9.74	—
4	2	15.74	—	15.75	—
4	3	26.80	—	26.81	—
4	4	79.17	—	79.19	—
4	5	439.90	—	439.91	—
Caso critico M=4, N=5: il solver classico impiega già 7.3 minuti da solo per 38 qubit. Il QAOA su statevector (2³⁸ ≈ 274 miliardi di stati) sarebbe computazionalmente impossibile.

SEZIONE 4 — ANALISI SCALING: OSSERVAZIONI CHIAVE [PPT + EXCEL]
Crescita Qubit vs (M, N)
La formula per il numero di variabili QUBO è approssimativamente:

n_qubo≈⏟(M+N⋅M)┬"binarie" +⏟(n_slack )┬"slack per vincoli ≤" 
 
I qubit crescono come O(M·N), con le variabili slack aggiuntive che dipendono dal numero di vincoli di capacità (M) e dai loro encoding binari.
Comportamento del Gap QAOA/Classico
Range qubit	N. config	Gap medio	Tendenza
5-10 qubit (M=1)	5	0.00%	Perfetta equivalenza
11-14 qubit (M=2, N=1..3)	3	0.00%	Perfetta equivalenza
15-18 qubit (M=3, N=1..2)	2	5.87% (media di -8.88% e +20.61%)	Iniziano le deviazioni
16-19 qubit (M=2, N=4..5)	2	9.25%	Gap significativo
Soglia critica: intorno ai 14-15 qubit, QAOA con reps=1 inizia a perdere qualità rispetto al classico. Sotto i 14 qubit, le soluzioni sono identiche.
Caso Anomalo M=3, N=1: QAOA MIGLIORE del Classico
Il classico trova obiettivo = 338 W, mentre il QAOA trova 308 W (gap = −8.88%). Questo sembra controintuitivo, ma ha una spiegazione: con reps=1 il QAOA esplora uno spazio diverso di soluzioni, e in questo caso specifico il suo punto di convergenza ha valore obiettivo minore. Non si tratta necessariamente di una soluzione più "giusta": entrambi usano un algoritmo euristico (ADMM non garantisce l'ottimo globale). La soluzione QAOA potrebbe essere non fattibile (da verificare con check_feasibility).
[EXCEL] Riepilogo Gap QAOA per config dove è stato eseguito
Config (M,N)	Qubits	Gap%	Giudizio
(1,1)	5	0.00%	✅ Identico
(1,2)	6	0.00%	✅ Identico
(1,3)	8	0.00%	✅ Identico
(1,4)	9	0.00%	✅ Identico
(1,5)	10	0.00%	✅ Identico
(2,1)	11	0.00%	✅ Identico
(2,2)	13	0.00%	✅ Identico
(2,3)	14	+11.35%	⚠️ Degradazione QAOA
(2,4)	16	+5.11%	⚠️ Degradazione QAOA
(2,5)	19	+13.38%	⚠️ Degradazione significativa
(3,1)	15	−8.88%	🔵 QAOA migliore (anomalia)
(3,2)	18	+20.61%	❌ Forte degradazione QAOA

SEZIONE 5 — SCALING DELLA SPARSITÀ E STRUTTURA QUBO [PPT]
Legge di Sparsità Osservata
Al crescere di M, la matrice Q diventa progressivamente più densa:
    • M=1: ~61-68% non-zero → matrice quasi piena
    • M=2: ~33% non-zero → semi-densa
    • M=3: ~23% non-zero → più sparsa
    • M=4: ~18% non-zero → sparsa
Paradossalmente, la sparsità aumenta (la matrice diventa meno densa) al crescere di M. Questo perché con più server, la struttura a blocchi della matrice Q si diversifica: i blocchi di server non interagiscono tra loro direttamente, riducendo il numero relativo di termini fuori-diagonale. N ha invece impatto minore sulla sparsità.

SEZIONE 6 — FEASIBILITY ANALYSIS [EXCEL]
[EXCEL] Tabella Feasibility per Configurazione
M	N	Feasible	Server accesi (classico)	Note
1	1	✅	1	1 VM su 1 server, OK
1	2	❌	1	2 VM su 1 server, possibile sovraccarico
1	3	❌	1	3 VM su 1 server
1	4	❌	1	Carico totale > C del server
1	5	❌	1	Carico totale >> C del server
2	1	✅	1	1 VM basta su 1 server
2	2	✅	1	2 VM su 1 server, OK
2	3	❌	2	Carico distribuito ma vincoli violati
2	4	✅	2	4 VM su 2 server, bilanciato
2	5	❌	2	Carico totale eccede capacità
3	1	✅	1	1 VM basta su 1 dei 3 server
3	2	✅	1	2 VM su 1 server
3	3	✅	2	Distribuzione possibile
3	4	✅	1	4 VM su 1 server se capacità sufficienti
3	5	❌	3	Tutti i 3 server necessari + vincolo
4	1	✅	1	1 VM su 1 server
4	2	✅	2	2 VM su 2 server
4	3	❌	2	Vincoli capacity violati
4	4	❌	3	Distribuzione non fattibile con ADMM
4	5	❌	—	Troppo carico
Tasso di feasibility: 10/20 configurazioni (50%) producono soluzioni feasibili secondo check_feasibility.

SEZIONE 7 — PUNTI DI FORZA E LIMITAZIONI DEL MODULO SCALING [PPT]
Punti di Forza
#	Punto	Dettaglio
1	Completezza	20 configurazioni coprendo tutto il range M×N previsto
2	Riproducibilità	Seed deterministico: ogni re-run produce gli stessi parametri e risultati
3	Threshold automatico	QAOA skippato se n_qubits > 20: nessun crash per problemi troppo grandi
4	GPU-aware	decode_result usa cu.xp.rint su device
5	Parallelo disponibile	run_configurations_parallel con ProcessPoolExecutor per speedup
6	Metriche derivate	Gap%, iter ratio, time ratio calcolati automaticamente
7	Dual export	CSV + JSON opzionali da run_configurations_serial/parallel
Limitazioni
#	Limitazione	Impatto
1	reps=1 per QAOA	Qualità inferiore rispetto a reps=2; gap cresce prima della soglia teorica
2	Soglia 20 qubit arbitraria	Su macchine con molta RAM (128+ GB), il limite potrebbe essere alzato a 24-26
3	Nessuna ripetizione	1 solo run per configurazione: QAOA è stocastico, i risultati potrebbero variare
4	Scaling runner seriale	Le 20 config vengono eseguite in serie: totale 1247s; con il parallelo si potrebbe scendere a ~400s
5	Parametri fissi	rho=10, factor_c=100000 non vengono ottimizzati per ogni (M,N)
6	M=4,N=5 esplosione	440s solo per il classico: il problema è già al limite della gestibilità classica
7	check_feasibility limitato	Non verifica la vincolo di uguaglianza load_def_i per le variabili continue

SEZIONE 8 — TABELLA RIASSUNTIVA FINALE SCALING [EXCEL + PPT]
Metrica	Valore
Configurazioni totali	20
Range M (server)	1 – 4
Range N (VM)	1 – 5
Config con QAOA eseguito	12/20
Config con QAOA skippato (n_qubits>20)	8/20
Config con QAOA = Classico (gap 0%)	7/12
Config con QAOA peggiore	4/12
Config con QAOA migliore	1/12 (M=3,N=1)
Max qubit testati con QAOA	19 (M=2,N=5)
Max qubit totale (classico)	38 (M=4,N=5)
Soglia skip QAOA	20 qubit
Tempo classico min	1.36 s (M=1,N=1)
Tempo classico max	439.90 s (M=4,N=5)
Tempo QAOA min	8.84 s (M=1,N=1)
Tempo QAOA max	175.13 s (M=2,N=5)
Time ratio min (QAOA/classico)	3.47× (M=1,N=5)
Time ratio max (QAOA/classico)	9.10× (M=2,N=5)
Gap QAOA min (assoluto)	−8.88% (M=3,N=1: QAOA migliore)
Gap QAOA max	+20.61% (M=3,N=2)
Feasibility rate	10/20 (50%)
Sparsità Q minima	17.59% (M=4,N=5)
Sparsità Q massima	68.00% (M=1,N=1)
Tempo totale run scaling	1247.08 s (≈ 20.8 min)
QAOA reps usati	1 (vs reps=2 del main.py)
qaoa_max_qubits threshold	20
qaoa_max_iter (COBYLA)	100
Tutti i run classici OK	✅ 20/20
Errori QAOA	0

SEZIONE 9 — SUGGERIMENTI SLIDE POWERPOINT (modulo scaling) [PPT]
Slide 1 — Titolo Scaling Study
Titolo: Scaling Study: QAOA vs Classico da M=1 a M=4
    • 20 configurazioni: M ∈ {1,2,3,4} server × N ∈ {1,2,3,4,5} VM
    • Parametri deterministici: seed = M×100+N
    • Soglia QAOA: n_qubits ≤ 20 → eseguito; > 20 → skippato
    • Durata totale: 1247 secondi (≈ 20.8 minuti)
Slide 2 — Crescita delle Dimensioni
Titolo: Come Cresce il Problema con M e N
    • Qubit = M + N×M + slack (crescita O(M·N))
    • Da 5 qubit (M=1,N=1) a 38 qubit (M=4,N=5)
    • 12/20 configurazioni accessibili al QAOA (≤ 20 qubit)
    • La sparsità di Q scende da 68% (M=1) a 18% (M=4): matrice più densa
Slide 3 — Risultati QAOA: Zona di Equivalenza
Titolo: QAOA Equivalente al Classico fino a ~13 Qubit
    • M=1 (5-10 qubit): gap = 0% in tutte le 5 configurazioni
    • M=2, N=1,2 (11-13 qubit): gap = 0% — identici
    • Oltre 14 qubit: inizia la degradazione del QAOA (+5% … +21%)
    • Anomalia M=3,N=1: QAOA trova soluzione −8.88% migliore (euristica)
Slide 4 — Costo Temporale QAOA
Titolo: Il QAOA è 3.5× – 9.1× Più Lento del Classico
    • Time ratio minimo: 3.47× (M=1,N=5) — problema piccolo, QAOA maturo
    • Time ratio massimo: 9.10× (M=2,N=5) — 175s vs 19s per 19 qubit
    • Il classico scala meglio: da 1.36s a 440s (+323×); QAOA da 8.84s a 175s (+20×) ma si ferma a 19 qubit
    • Conclusione: il QAOA non ha ancora vantaggio di scaling su simulatore
Slide 5 — Feasibility e Qualità delle Soluzioni
Titolo: 50% delle Soluzioni è Fattibile (check_feasibility)
    • Soluzioni feasibili: 10/20 (vincoli tutti rispettati)
    • Infeasibilità tipica: carico CPU supera capacità server con i parametri casuali
    • ADMM euristico: non garantisce fattibilità né ottimalità globale
    • Prossimo passo: warm-start, aumento rho, rilassamento penalità

DATI DA COPIARE IN EXCEL — Blocco Aggregato Scaling

text
TAB_SCALA_1: Dimensioni per (M,N)
M, N, Bin_orig, Slack, QUBO_tot, Qubits, Sparsità%, Cont_vars, Feasible
1,1,2,3,5,5,68.00,1,SI
1,2,3,3,6,6,66.67,1,NO
1,3,4,4,8,8,62.50,1,NO
1,4,5,4,9,9,61.73,1,NO
1,5,6,4,10,10,61.00,1,NO
2,1,4,7,11,11,33.88,2,SI
2,2,6,7,13,13,33.73,2,SI
2,3,8,6,14,14,34.18,2,NO
2,4,10,6,16,16,33.59,2,SI
2,5,12,7,19,19,32.41,2,NO
3,1,6,9,15,15,24.00,3,SI
3,2,9,9,18,18,24.07,3,SI
3,3,12,11,23,23,22.87,3,SI
3,4,15,11,26,26,22.63,3,SI
3,5,18,9,27,27,23.05,3,NO
4,1,8,14,22,22,17.77,4,SI
4,2,12,15,27,27,17.70,4,SI
4,3,16,15,31,31,17.69,4,NO
4,4,20,12,32,32,18.36,4,NO
4,5,24,14,38,38,17.59,4,NO
TAB_SCALA_2: Risultati solver (M,N)
M,N,Obj_Cl,Iter_Cl,T_Cl_s,Obj_QAOA,Iter_QAOA,T_QAOA_s,Gap%,TimeRatio,QAOA_stato
1,1,400.00,49,1.36,400.00,49,8.84,0.00%,6.50,eseguito
1,2,572.00,71,2.50,572.00,71,14.72,0.00%,5.89,eseguito
1,3,615.00,72,3.54,615.00,72,17.17,0.00%,4.85,eseguito
1,4,550.00,74,6.56,550.00,74,22.95,0.00%,3.50,eseguito
1,5,894.00,74,7.82,894.00,74,27.19,0.00%,3.47,eseguito
2,1,241.00,68,3.90,241.00,67,16.80,0.00%,4.31,eseguito
2,2,400.00,68,6.15,400.00,68,24.06,0.00%,3.91,eseguito
2,3,687.00,73,9.20,765.00,70,37.65,+11.35%,4.09,eseguito
2,4,822.00,69,11.58,864.00,85,77.51,+5.11%,6.69,eseguito
2,5,1106.00,76,19.24,1254.00,100,175.13,+13.38%,9.10,eseguito
3,1,338.00,69,6.64,308.00,70,25.74,-8.88%,3.87,eseguito (QAOA migliore)
3,2,511.00,70,11.95,616.00,79,55.95,+20.61%,4.68,eseguito
3,3,890.39,71,19.47,,,,,skipped (23>20)
3,4,478.00,69,20.22,,,,,skipped (26>20)
3,5,1076.00,73,41.71,,,,,skipped (27>20)
4,1,355.00,70,9.73,,,,,skipped (22>20)
4,2,1023.00,71,15.74,,,,,skipped (27>20)
4,3,766.00,74,26.80,,,,,skipped (31>20)
4,4,535.00,74,79.17,,,,,skipped (32>20)
4,5,1364.00,73,439.90,,,,,skipped (38>20)
TAB_SCALA_3: Metadati run
Metrica,Valore
Configurazioni totali,20
QAOA eseguito,12
QAOA skippato,8
Gap=0%,7
QAOA peggiore,4
QAOA migliore,1
Max qubit QAOA testato,19
Max qubit totale,38
T.classico min (s),1.36
T.classico max (s),439.90
T.QAOA min (s),8.84
T.QAOA max (s),175.13
Time ratio min,3.47x
Time ratio max,9.10x
Feasibility rate,50%
Tempo totale run (s),1247.08
QAOA reps,1
qaoa_max_qubits,20
