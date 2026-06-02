# MailProgettoTesi/12

## Descrizione generale

La cartella `MailProgettoTesi/12` raccoglie un caso di studio di ottimizzazione sviluppato con Qiskit 1.x e finalizzato al confronto tra una pipeline classica e una pipeline quantistica simulata per un problema di allocazione del carico tra server fisici e macchine virtuali.

L'esperimento adotta la stessa formulazione matematica per entrambe le strategie di soluzione e differenzia unicamente il metodo usato per il sottoproblema combinatorio. Le due configurazioni considerate sono:

- soluzione classica basata su `ADMM + NumPyMinimumEigensolver + COBYLA`
- soluzione quantistica simulata basata su `ADMM + QAOA + COBYLA`

Le istanze studiate coprono il dominio `N, M in {1, ..., 6}`, dove `N` rappresenta il numero di server e `M` il numero di macchine virtuali.

## Problema di ottimizzazione

Il modello implementato in [qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/12/qiskit_opt.py) descrive un problema di minimizzazione dei costi in ambiente data center. Le variabili decisionali principali sono:

- variabili binarie `s_i` per lo stato di attivazione dei server
- variabili continue `v_{j,i}` per il carico della VM `j` assegnato al server `i`
- variabili continue `u_j` per il livello minimo di CPU associato alla VM `j`

La funzione obiettivo combina:

- un termine di costo fisso per l'attivazione dei server
- un termine di costo variabile legato all'utilizzo delle risorse computazionali

I vincoli principali richiedono che:

- ogni server riceva almeno un livello minimo di carico `capacities[i] - 1`
- ogni VM rispetti un limite massimo di allocazione
- ogni VM soddisfi una CPU minima pari a `min_cpu_per_vm`
- tutti i server siano attivi quando `require_all_on = True`

Il modello viene costruito in DOcplex e poi tradotto in `QuadraticProgram` tramite `qiskit-optimization`.

## Impostazione metodologica

La strategia di risoluzione adottata si basa su ADMM, utilizzato per separare:

- il sottoproblema discreto, trattato come QUBO
- il sottoproblema continuo, risolto con COBYLA

Il confronto tra pipeline classica e pipeline quantistica simulata dipende quindi esclusivamente dal metodo scelto per il sottoproblema QUBO:

- `NumPyMinimumEigensolver` nella configurazione classica
- `QAOA` con `StatevectorSampler` nella configurazione quantistica simulata

Un elemento importante del workflow e la funzione `snap_to_feasible()`, introdotta per correggere piccoli scostamenti numerici prodotti da ADMM. Questo passaggio evita che soluzioni praticamente ammissibili vengano etichettate come `INFEASIBLE` a causa dei controlli di fattibilita rigorosi eseguiti da Qiskit.

Prima di avviare la risoluzione, lo script esegue inoltre un controllo strutturale di fattibilita basato sulla condizione:

```text
sum(vm_allocation_limits) >= sum(capacities) - n_servers
```

## Struttura della pipeline

La cartella e organizzata secondo una pipeline sperimentale lineare.

### 1. Risoluzione di una singola istanza

[qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/12/qiskit_opt.py) e lo script principale. Le sue funzioni principali sono:

- costruzione del modello DOcplex
- conversione in `QuadraticProgram`
- risoluzione sia classica sia quantistica simulata
- misura dei tempi di esecuzione
- salvataggio dei risultati in JSON
- produzione di una PNG riassuntiva per ogni istanza

Esempio di esecuzione:

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

L'opzione `--fast` consente una configurazione sperimentale piu leggera, riducendo il costo computazionale di QAOA.

### 2. Esecuzione batch

[run_batch_qiskit.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/12/run_batch_qiskit.py) automatizza l'esecuzione dell'intera griglia `1..6 x 1..6`. Lo script:

- genera i valori di default per `capacities`
- costruisce `vm_allocation_limits` con margine di sicurezza
- richiama [qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/12/qiskit_opt.py) come subprocess
- salva i log testuali nella directory `batch_runs/`

Esempi di utilizzo:

```bash
python run_batch_qiskit.py
python run_batch_qiskit.py --fast
python run_batch_qiskit.py --seed 42
```

### 3. Aggregazione dei risultati

[merge_results.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/12/merge_results.py) aggrega i file `*results.json` e produce:

- un JSON complessivo delle run
- un CSV piatto per analisi comparative

Tra i campi piu utili del CSV:

- `classic_objective`, `quantum_objective`
- `classic_time_s`, `quantum_time_s`
- `speedup_x = classic_time / quantum_time`
- `best_solver`

Esempi di utilizzo:

```bash
python merge_results.py
python merge_results.py --dir batch_runs --recursive
python merge_results.py --sort quantum_time
```

## Artefatti presenti

La cartella contiene gia una campagna sperimentale completa. I principali artefatti presenti sono:

- `q_20260328_*.json`, risultati delle singole istanze
- `q_20260328_*.png`, grafici di confronto per ogni istanza
- `merged_20260328_154707.json`, merge completo delle 36 esecuzioni
- `merged_20260328_154707.csv`, vista tabellare aggregata dei risultati
- `batch_runs/`, log delle esecuzioni batch

## Evidenze sperimentali

Dai file gia presenti nella cartella emerge che:

- il merge aggregato raccoglie 36 istanze, corrispondenti alla griglia completa `6 x 6`
- tutte le istanze risultano `SUCCESS` sia per il solver classico sia per il solver quantistico simulato
- il solver indicato come migliore in termini di tempo e sempre `classical`
- il rapporto `speedup_x = classic_time / quantum_time` rimane sempre inferiore a 1

Questi risultati indicano che, nel dataset salvato in questa directory, la configurazione quantistica simulata non fornisce un vantaggio prestazionale rispetto alla controparte classica. L'interesse del caso di studio e quindi soprattutto metodologico: la cartella documenta una pipeline riproducibile per il confronto tra due strategie di soluzione applicate allo stesso problema.

## Dipendenze principali

Gli script utilizzano in particolare:

- `qiskit`
- `qiskit-aer`
- `qiskit-algorithms`
- `qiskit-optimization`
- `docplex`
- `cplex`
- `numpy`
- `matplotlib`

Lo script principale tenta anche l'installazione automatica di alcuni pacchetti via `pip`, ma per una riproduzione controllata e preferibile predisporre l'ambiente in anticipo.

## Riproduzione del workflow

1. Eseguire una singola istanza:

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

L'interpretazione dei risultati deve tenere conto di alcuni aspetti:

- la componente quantistica e simulata tramite statevector e non eseguita su hardware quantistico reale
- il problema contiene una parte continua risolta tramite COBYLA, quindi il confronto non isola esclusivamente la componente combinatoria
- la metrica `speedup_x` misura un confronto di tempo a livello di pipeline complessiva, non una speedup quantistica in senso stretto

## Sintesi

La cartella `MailProgettoTesi/12` costituisce un caso di studio completo per:

- formalizzare un problema di allocazione server/VM
- confrontare una soluzione classica e una soluzione quantistica simulata all'interno di ADMM
- eseguire campagne batch su piu istanze
- salvare e aggregare i risultati in JSON e CSV

Nel suo stato attuale, la directory rappresenta una base ordinata e riproducibile per analisi comparative e discussione metodologica.