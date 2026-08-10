# Report Breve di Consegna

## Quantum vs Classical Optimization for Server-VM Workload Placement

**Cartella di riferimento:** `MailProgettoTesi/11`  
**Formato:** sintesi breve del progetto da allegare alle slide  
**Data:** 30 maggio 2026  

---

## 1. Obiettivo del progetto

Il progetto ha come obiettivo il confronto tra una pipeline di ottimizzazione classica e una pipeline di ottimizzazione quantistica simulata applicate allo stesso problema di allocazione del carico tra server fisici e macchine virtuali.

L'idea centrale e verificare se un approccio ibrido classico-quantistico, costruito con Qiskit, possa essere inserito in una pipeline di ottimizzazione reale mantenendo correttezza numerica, fattibilita delle soluzioni e confrontabilita sperimentale con un metodo classico.

In particolare, il progetto vuole:

- formalizzare il problema come modello di ottimizzazione
- risolverlo con una pipeline classica e una quantistica simulata
- confrontare tempi, valori obiettivo e qualita delle soluzioni
- produrre una base sperimentale riproducibile e documentata

---

## 2. Problema affrontato

Il caso di studio rappresenta un problema di resource allocation in ambiente cloud o data center. Si considera un insieme di server fisici e un insieme di macchine virtuali, con l'obiettivo di distribuire il carico in modo da minimizzare un costo totale composto da:

- costo fisso di attivazione dei server
- costo variabile legato all'utilizzo delle risorse computazionali

Le variabili del modello sono:

- `s_i`: variabili binarie che rappresentano lo stato del server `i`
- `v_{j,i}`: variabili continue che rappresentano la quota di carico della VM `j` allocata al server `i`
- `u_j`: variabili continue associate al livello minimo di CPU della VM `j`

I vincoli principali impongono che:

- ogni server riceva almeno un livello minimo di carico
- ogni VM rispetti un limite massimo di allocazione
- ogni VM soddisfi una CPU minima
- nel dataset salvato, tutti i server risultino attivi (`require_all_on = True`)

Il modello viene costruito in DOcplex e successivamente convertito in `QuadraticProgram` di Qiskit.

---

## 3. Metodologia adottata

L'approccio di soluzione e basato su una decomposizione ADMM, che separa il problema in:

- un sottoproblema discreto, trattato come QUBO
- un sottoproblema continuo, risolto con COBYLA

Le due pipeline confrontate sono:

### Pipeline classica

- `ADMM + NumPyMinimumEigensolver + COBYLA`

### Pipeline quantistica simulata

- `ADMM + QAOA + COBYLA`

Il confronto e corretto perche la formulazione matematica e identica in entrambi i casi. L'unica differenza riguarda il metodo usato per risolvere il sottoproblema combinatorio.

Un punto metodologico importante del progetto e la funzione `snap_to_feasible()`, introdotta per correggere piccoli errori numerici prodotti da ADMM. Senza questa correzione, diverse soluzioni quasi corrette verrebbero classificate come `INFEASIBLE` dai controlli rigorosi di Qiskit.

---

## 4. Organizzazione della cartella 11

La cartella `11` contiene una pipeline completa di progetto e non soltanto gli script di calcolo. I file principali sono:

- [qiskit_opt.py](qiskit_opt.py): costruzione del modello e risoluzione di una singola istanza
- [run_batch_qiskit.py](run_batch_qiskit.py): esecuzione batch della campagna sperimentale
- [merge_results.py](merge_results.py): aggregazione dei risultati in JSON e CSV
- [create_pptx.py](create_pptx.py): generazione della presentazione finale
- [merged_20260325_165037.csv](merged_20260325_165037.csv): tabella aggregata dei risultati
- [merged_20260325_165037.json](merged_20260325_165037.json): merge completo delle run
- [Qiskit_Optimization_Results.pptx](Qiskit_Optimization_Results.pptx): presentazione conclusiva

Sono inoltre presenti script di diagnosi e debug, utili per comprendere la fase di verifica della fattibilita numerica.

---

## 5. Setup sperimentale

La campagna di test eseguita nella cartella `11` comprende 36 istanze, ottenute dalla griglia:

- `n_servers in {1, 2, 3, 4, 5, 6}`
- `n_vms in {1, 2, 3, 4, 5, 6}`

Parametri principali usati nelle run salvate:

- `require_all_on = True`
- `min_cpu_per_vm = 1.0`
- `pi_i = 1.0` per tutti i server
- `pd_i = 1.0` per tutti i server
- `capacities = [11, 11, 11, 10, 10, 10]` troncato in base al numero di server

I limiti di allocazione delle VM vengono generati con una procedura che garantisce la fattibilita strutturale del problema prima dell'avvio del solver.

---

## 6. Risultati principali

I risultati aggregati, verificati direttamente sui file CSV e JSON presenti nella cartella, sono i seguenti.

### Risultati globali

| Metrica | Valore |
|---|---|
| Numero totale di istanze | 36 |
| Successi pipeline classica | 36 / 36 |
| Successi pipeline quantistica simulata | 36 / 36 |
| Solver piu veloce in tutte le istanze | classico |
| Differenza massima tra obiettivi classico e quantum | 0.0 |
| Vettori soluzione identici tra i due solver | 36 / 36 |

### Tempi di esecuzione

| Metrica | Classico | Quantum simulato |
|---|---:|---:|
| Tempo medio | 2.294 s | 11.499 s |
| Tempo mediano | 1.047 s | 4.945 s |
| Tempo minimo | 0.080 s | 1.256 s |
| Tempo massimo | 15.080 s | 70.791 s |

### Speedup

Nel progetto si usa la metrica:

```text
speedup_x = classic_time / quantum_time
```

Quindi un valore inferiore a `1` indica che il solver classico e piu rapido. Nel dataset salvato:

- media `speedup_x = 0.229`
- mediana `speedup_x = 0.222`
- minimo `speedup_x = 0.021`
- massimo `speedup_x = 0.477`

In tutte le 36 istanze il solver quantistico simulato risulta piu lento della controparte classica.

### Casi significativi

- istanza minima `1s x 1v`: `0.080 s` classico contro `1.256 s` quantum
- caso intermedio `3s x 3v`: `0.744 s` classico contro `2.765 s` quantum
- caso piu lento per il quantum `6s x 3v`: `5.278 s` classico contro `70.791 s` quantum
- caso piu costoso per il classico `4s x 6v`: `15.080 s` classico contro `31.640 s` quantum

---

## 7. Interpretazione dei risultati

Il risultato piu forte del progetto non riguarda la velocita, ma la coerenza numerica.

Infatti, per tutte le 36 istanze:

- le due pipeline producono lo stesso valore obiettivo
- le due pipeline producono esattamente lo stesso vettore di soluzione
- entrambe raggiungono stato `SUCCESS`

Questo significa che la pipeline quantistica simulata e corretta dal punto di vista della soluzione trovata, anche se non competitiva dal punto di vista temporale nel contesto di simulazione scelto.

La maggiore lentezza del quantum simulato e attesa, per almeno tre motivi:

- QAOA viene eseguito in simulazione statevector e non su hardware quantistico reale
- il costo della simulazione cresce rapidamente con il numero di qubit
- il workflow include comunque una parte continua risolta classicamente

Dal punto di vista accademico, il progetto dimostra quindi la validita della pipeline ibrida come strumento sperimentale, non l'esistenza di un vantaggio prestazionale quantistico.

---

## 8. Limiti del progetto

Il limite metodologico piu importante e che tutte le run salvate hanno `require_all_on = True`. In pratica, le variabili binarie dei server sono vincolate a `1`, per cui la parte discreta del problema perde parte del proprio contenuto decisionale.

Altri limiti importanti sono:

- costi `pi_i` e `pd_i` uniformi, che rendono il problema molto regolare
- assenza di rumore quantistico reale, poiche QAOA e simulato con `StatevectorSampler`
- scala ancora limitata del benchmark, fermo a `6 x 6`

Questi aspetti non invalidano il progetto, ma ne definiscono correttamente il perimetro scientifico.

**Nota sulla coerenza con la consegna.** Il progetto puo essere interpretato come uno sviluppo coerente della traccia descritta in [Assignment.pdf](Assignment.pdf). Restano infatti invariati il dominio applicativo, l'uso dell'`ADMMOptimizer` di Qiskit e l'obiettivo di studiare un problema server/VM con una decomposizione tra parte discreta e parte continua. Le differenze di modellazione introdotte nel codice rispondono soprattutto all'esigenza di ottenere una pipeline stabile, verificabile e riproducibile. In questo senso, il lavoro si presenta come un caso di studio metodologicamente allineato alla consegna, con margini chiari di estensione verso una formulazione ancora piu vicina al PDF.

---

## 9. Valore del progetto

Il valore del lavoro svolto e principalmente metodologico e progettuale. La cartella `11` mostra infatti come:

- costruire un problema di ottimizzazione realistico con DOcplex e Qiskit
- integrare ADMM in una pipeline ibrida classico-quantistica
- affrontare in modo esplicito il problema della fattibilita numerica
- organizzare un benchmark riproducibile con output aggregati, grafici e slide

Per una consegna universitaria, questo e un punto di forza rilevante: il progetto non e un semplice esercizio di codice, ma una mini-pipeline di ricerca ben documentata.

---

## 10. Conclusione

Il progetto contenuto in `MailProgettoTesi/11` fornisce un confronto chiaro e rigoroso tra una pipeline classica e una pipeline quantistica simulata per un problema di allocazione server/VM.

Le conclusioni principali sono:

- entrambe le pipeline sono corrette e restituiscono sempre soluzioni fattibili
- i risultati classici e quantistici coincidono esattamente su tutte le istanze testate
- il solver classico e sempre piu veloce nel contesto di simulazione adottato
- il contributo piu importante del progetto e la costruzione di una pipeline sperimentale completa, verificata e riproducibile

Nel suo stato attuale, il progetto e pienamente adatto a essere presentato come elaborato universitario, soprattutto se accompagnato dalle slide [Qiskit_Optimization_Results.pptx](Qiskit_Optimization_Results.pptx) e dal report esteso [REPORT_PROGETTO_UNIVERSITARIO.md](REPORT_PROGETTO_UNIVERSITARIO.md).