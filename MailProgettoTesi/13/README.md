# MailProgettoTesi/13

## Descrizione generale

La cartella `MailProgettoTesi/13` documenta un esperimento di ottimizzazione con Qiskit 1.x sostanzialmente allineato a quello presente in `MailProgettoTesi/12`. Anche in questo caso l'obiettivo e confrontare una pipeline classica e una pipeline quantistica simulata per un problema di allocazione del carico tra server fisici e macchine virtuali.

Le due configurazioni messe a confronto sono:

- soluzione classica basata su `ADMM + NumPyMinimumEigensolver + COBYLA`
- soluzione quantistica simulata basata su `ADMM + QAOA + COBYLA`

Le istanze considerate coprono il dominio `N, M in {1, ..., 6}`.

## Relazione con la cartella 12

Dal confronto dei file principali risulta che `MailProgettoTesi/13` contiene gli stessi script e gli stessi file di merge presenti in `MailProgettoTesi/12`:

- [qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/qiskit_opt.py)
- [run_batch_qiskit.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/run_batch_qiskit.py)
- [merge_results.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/merge_results.py)
- `merged_20260328_154707.csv`
- `merged_20260328_154707.json`

In pratica, `13` puo essere letta come una variante o copia operativa di `12`, arricchita da alcuni artefatti supplementari di reportistica ed esportazione.

Un indizio ulteriore di questa relazione e dato dal fatto che il file `merged_20260328_154707.json` conserva nel campo `source_directory` il percorso della cartella `12`, segnalando che il merge aggregato presente in `13` e una copia del merge originario.

## Problema e metodologia

Il problema modellato in [qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/qiskit_opt.py) e lo stesso gia adottato nella cartella `12`: un problema di minimizzazione dei costi in cui si decide come distribuire il carico di un insieme di VM su un insieme di server.

Le variabili decisionali sono:

- variabili binarie `s_i` per lo stato dei server
- variabili continue `v_{j,i}` per l'allocazione del carico della VM `j` sul server `i`
- variabili continue `u_j` per il livello minimo di CPU associato alle VM

La funzione obiettivo combina costo fisso di attivazione e costo variabile di utilizzo. I vincoli impongono soglie minime di carico sui server, limiti massimi di allocazione per le VM e una CPU minima per ogni VM.

La procedura di soluzione e basata su ADMM:

- il sottoproblema discreto e trattato come QUBO
- il sottoproblema continuo e risolto tramite COBYLA
- la differenza tra pipeline classica e pipeline quantistica simulata riguarda esclusivamente il metodo usato per il QUBO

Come in `12`, e presente la funzione `snap_to_feasible()`, introdotta per correggere piccole imprecisioni numeriche e rendere le soluzioni accettabili rispetto ai controlli di fattibilita di Qiskit.

## Struttura operativa

### 1. Risoluzione di singole istanze

[qiskit_opt.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/qiskit_opt.py) costruisce il modello, risolve l'istanza nelle due configurazioni, salva il JSON dei risultati e genera una PNG riassuntiva.

Esempio di esecuzione:

```bash
python qiskit_opt.py --n_servers 3 --n_vms 2
```

### 2. Esecuzione batch

[run_batch_qiskit.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/run_batch_qiskit.py) esegue la griglia completa `1..6 x 1..6` e salva i log della campagna.

Esempi:

```bash
python run_batch_qiskit.py
python run_batch_qiskit.py --fast
python run_batch_qiskit.py --seed 42
```

### 3. Aggregazione

[merge_results.py](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/merge_results.py) produce il merge delle run in JSON e CSV.

Esempi:

```bash
python merge_results.py
python merge_results.py --dir batch_runs --recursive
```

## Artefatti presenti

Oltre ai file principali gia descritti, la cartella contiene:

- `q_20260328_*.json` e `q_20260328_*.png` per le singole run
- `batch_runs/` con 36 log batch
- ulteriori 36 log `run_s*_full_*.log` duplicati al livello radice della cartella
- `3664b707.csv`, che appare come un riepilogo sintetico delle metriche aggregate
- `688d1b39.pptx` e `9d86b98f.pptx`, che appaiono come esportazioni supplementari in formato PowerPoint

La presenza di questi artefatti aggiuntivi suggerisce che `13` sia stata utilizzata anche come cartella di raccolta per materiali di presentazione o reportistica successiva al batch principale.

## Evidenze sperimentali

Poiche i file di merge coincidono con quelli di `12`, le evidenze sperimentali principali sono le stesse:

- 36 istanze complessive nella griglia `6 x 6`
- tutte le esecuzioni classiche e quantistiche simulate risultano `SUCCESS`
- il solver classico risulta sempre piu rapido del solver quantistico simulato
- il rapporto `speedup_x = classic_time / quantum_time` rimane sempre inferiore a 1

Il file supplementare [3664b707.csv](/Users/lgandolfi/Desktop/AI/Github/Qiskit-QuantumComputing/MailProgettoTesi/13/3664b707.csv) riporta inoltre una sintesi aggregata coerente con tali risultati, includendo il numero totale di run, il successo completo delle due pipeline e le medie dei tempi di esecuzione.

## Dipendenze principali

Gli script principali utilizzano:

- `qiskit`
- `qiskit-aer`
- `qiskit-algorithms`
- `qiskit-optimization`
- `docplex`
- `cplex`
- `numpy`
- `matplotlib`

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

## Osservazioni finali

La cartella `MailProgettoTesi/13` non introduce una nuova pipeline rispetto a `12`, ma rappresenta piuttosto una sua replica arricchita con materiali accessori di sintesi. Per questo motivo puo essere interpretata come una directory di lavoro usata sia per l'esecuzione sperimentale sia per la preparazione di report o presentazioni.

Nel complesso, `13` conserva valore documentale perche unisce:

- la pipeline completa di esecuzione e merge
- i risultati aggregati della campagna sperimentale
- artefatti aggiuntivi di riepilogo e presentazione