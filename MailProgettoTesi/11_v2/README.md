# MailProgettoTesi/11

## Descrizione generale

La presente cartella documenta uno studio sperimentale di ottimizzazione combinatoria sviluppato con Qiskit 1.x, finalizzato al confronto tra una procedura di risoluzione classica e una procedura quantistica simulata applicate al medesimo problema di allocazione del carico tra server fisici e macchine virtuali.

L'interesse del caso di studio risiede nella possibilita di confrontare, a parita di formulazione matematica e di schema risolutivo, due varianti della stessa pipeline:

- una configurazione classica basata su `ADMM + NumPyMinimumEigensolver + COBYLA`
- una configurazione quantistica simulata basata su `ADMM + QAOA + COBYLA`

Le istanze considerate coprono il dominio `N, M in {1, ..., 6}`, dove `N` rappresenta il numero di server e `M` il numero di macchine virtuali.

## Inquadramento del problema

Il modello implementato in [qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/qiskit_opt.py) descrive un problema di minimizzazione dei costi in ambiente data center. Le variabili decisionali sono le seguenti:

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

Un elemento metodologico rilevante e la funzione `snap_to_feasible()`, introdotta per correggere lievi scostamenti numerici prodotti da ADMM. Tale passaggio e necessario per evitare che soluzioni praticamente ammissibili vengano classificate come `INFEASIBLE` a causa della verifica rigorosa di fattibilita eseguita da Qiskit.

Prima della risoluzione viene inoltre eseguito un controllo strutturale di fattibilita, basato sulla condizione:

```text
sum(vm_allocation_limits) >= sum(capacities) - n_servers
```

Se tale condizione non risulta soddisfatta, l'istanza viene scartata prima dell'avvio del solver.

## Organizzazione della cartella

La cartella e organizzata come una pipeline sperimentale completa, dalla definizione del modello alla produzione dei materiali finali di sintesi.

### 1. Script principale

[qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/qiskit_opt.py) costituisce il nucleo dell'esperimento. Lo script:

- costruisce il modello DOcplex
- converte il modello in `QuadraticProgram`
- esegue la risoluzione classica e la risoluzione quantistica simulata
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

L'opzione `--fast` consente di eseguire test rapidi riducendo il costo computazionale della configurazione QAOA.

### 2. Esecuzione batch

[run_batch_qiskit.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/run_batch_qiskit.py) automatizza l'esecuzione della griglia completa di istanze `1..6 x 1..6`. Lo script:

- genera i valori di default per `capacities`
- costruisce limiti `vm_allocation_limits` con margine di sicurezza
- richiama [qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/qiskit_opt.py) come processo separato
- salva i log testuali nella directory `batch_runs/`

Esempi di utilizzo:

```bash
python run_batch_qiskit.py
python run_batch_qiskit.py --fast
python run_batch_qiskit.py --seed 42
```

### 3. Aggregazione dei risultati

[merge_results.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/merge_results.py) esegue il merge dei file `*results.json` e produce:

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

### 4. Produzione della presentazione finale

[create_pptx.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/create_pptx.py) genera la presentazione `Qiskit_Optimization_Results.pptx` mediante `python-pptx`, utilizzando il CSV aggregato e una serie di grafici intermedi salvati in `_pptx_charts/`.

La presentazione include:

- inquadramento del problema
- formulazione e approccio ADMM
- discussione della procedura `snap_to_feasible()`
- grafici comparativi su tempi, speedup e scalabilita

Lo script contiene un riferimento hardcoded al file CSV da utilizzare:

```python
CSV = HERE / "merged_20260325_165037.csv"
```

In caso di rigenerazione del merge con timestamp differente, tale riferimento deve essere aggiornato manualmente prima della creazione del file PPTX.

## Script ausiliari di analisi

La cartella include anche alcuni script di diagnosi sviluppati durante il debugging delle condizioni di fattibilita e della classificazione `INFEASIBLE`:

- [check_batch.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/check_batch.py)
- [diagnose.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/diagnose.py)
- [inspect_infeasible.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/inspect_infeasible.py)
- [test_feasibility.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/test_feasibility.py)

Sono inoltre presenti gli script `check_7a*.py`, che fanno riferimento a materiali storici della directory `MailProgettoTesi/7a/`. Questi file non appartengono alla pipeline principale della cartella `11`, ma costituiscono una traccia utile del percorso di verifica e correzione del modello.

## Artefatti presenti

La directory contiene gia gli output di una campagna sperimentale completa. I principali artefatti sono:

- `q_20260325_*.json`, contenenti i risultati delle singole istanze
- `q_20260325_*.png`, contenenti i grafici di confronto per ogni singola run
- `merged_20260325_165037.json`, contenente il merge complessivo delle esecuzioni
- `merged_20260325_165037.csv`, contenente la vista tabellare aggregata
- `batch_runs/`, contenente i log delle esecuzioni automatiche
- `_pptx_charts/`, contenente le immagini intermedie utilizzate per la presentazione
- `Qiskit_Optimization_Results.pptx`, contenente la sintesi finale in formato slide

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

4. Rigenerare la presentazione finale, aggiornando prima il nome del CSV in [create_pptx.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/11/create_pptx.py) se necessario:

```bash
python create_pptx.py
```

## Limiti del caso di studio

L'interpretazione dei risultati richiede alcune cautele:

- la componente quantistica e simulata tramite statevector e non corrisponde a un'esecuzione su hardware quantistico reale
- il problema contiene una parte continua risolta con COBYLA, pertanto il confronto non isola esclusivamente la componente combinatoria
- una parte degli script ausiliari ha natura esplorativa e contiene riferimenti hardcoded a file o cartelle specifiche

## Sintesi conclusiva

La directory `MailProgettoTesi/11` puo essere considerata come un caso di studio completo, strutturato per:

- definire formalmente un problema di allocazione server/VM
- confrontare una soluzione classica e una soluzione quantistica simulata all'interno della stessa architettura ADMM
- eseguire campagne batch su piu istanze
- aggregare e analizzare i risultati in formato JSON, CSV e PowerPoint

Nel suo stato attuale, la cartella costituisce quindi una base solida sia per la discussione metodologica in sede di tesi, sia per successive estensioni sperimentali.