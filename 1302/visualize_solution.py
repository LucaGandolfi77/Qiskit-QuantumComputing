import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
from pathlib import Path

def natural_sort_key(s):
    """
    Permette di ordinare le variabili in modo umano (x2 viene prima di x10).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def visualize_model_results(json_file_path, model_name=None):
    """
    Carica il JSON, estrae la soluzione e genera i grafici.
    """
    path = Path(json_file_path)
    if not path.exists():
        print(f"❌ Errore: File {path} non trovato.")
        return

    # 1. Caricamento Dati
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Se è una lista (il report completo), prendiamo il primo modello o quello specificato
        target_result = None
        if isinstance(data, list):
            if model_name:
                for res in data:
                    if res.get('model_name') == model_name:
                        target_result = res
                        break
            else:
                target_result = data[0] # Prendi il primo se non specificato
        elif isinstance(data, dict):
             # Caso in cui il JSON è un singolo risultato
             target_result = data
        
        if not target_result:
            print("❌ Nessun risultato trovato nel file.")
            return

        # Preferiamo la soluzione Gurobi, altrimenti QAOA
        sol_data = target_result.get('gurobi', {}).get('solution')
        source = "Gurobi"
        
        if not sol_data:
            sol_data = target_result.get('qaoa', {}).get('solution')
            source = "QAOA"
            
        if not sol_data:
            print(f"⚠️ Nessuna soluzione trovata per {target_result.get('model_name')}")
            return

        print(f"✅ Visualizzazione soluzione per: {target_result.get('model_name')} (Source: {source})")

    except Exception as e:
        print(f"❌ Errore nel parsing del file: {e}")
        return

    # 2. Preparazione Dati
    # Ordiniamo le variabili per nome (x0, x1, x2...) per avere senso nella heatmap
    sorted_keys = sorted(sol_data.keys(), key=natural_sort_key)
    values = np.array([sol_data[k] for k in sorted_keys])
    
    # Calcolo dimensioni per la heatmap (griglia quadrata o quasi)
    num_vars = len(values)
    grid_side = math.ceil(math.sqrt(num_vars))
    
    # Padding con NaN per riempire la griglia se non è un quadrato perfetto
    padded_length = grid_side * grid_side
    padded_values = np.pad(values, (0, padded_length - num_vars), constant_values=np.nan)
    matrix_values = padded_values.reshape(grid_side, grid_side)

    # 3. Creazione Grafici
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # --- Grafico 1: Istogramma Distribuzione ---
    plt.subplot(1, 2, 1)
    
    # Determina se i dati sono binari o continui per un plotting migliore
    unique_vals = np.unique(values)
    is_binary = len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals)
    
    if is_binary:
        sns.countplot(x=values, palette="viridis")
        plt.title(f"Distribuzione Variabili (Binario)\nTotale: {num_vars}", fontsize=14)
        plt.xlabel("Valore Assegnato (0 o 1)")
    else:
        sns.histplot(values, bins=20, kde=True, color="blue")
        plt.title(f"Distribuzione Valori (Continuo)\nTotale: {num_vars}", fontsize=14)
        plt.xlabel("Valore Variabile")
    
    plt.ylabel("Conteggio")

    # --- Grafico 2: Heatmap (Mappa di Colore) ---
    plt.subplot(1, 2, 2)
    
    # Maschera per nascondere le celle vuote (NaN)
    mask = np.isnan(matrix_values)
    
    sns.heatmap(
        matrix_values, 
        annot=True if num_vars < 100 else False, # Mostra numeri solo se poche var
        fmt=".1f", 
        cmap="coolwarm", 
        cbar_kws={'label': 'Valore Variabile'},
        mask=mask,
        linewidths=0.5,
        linecolor='lightgray',
        square=True
    )
    
    plt.title(f"Mappa Variabili ({grid_side}x{grid_side})\nOrd: {sorted_keys[0]} → {sorted_keys[-1]}", fontsize=14)
    plt.axis('off') # Nascondi assi x/y perché sono indici di matrice

    plt.suptitle(f"Analisi Soluzione: {target_result.get('model_name')}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Salva e Mostra
    out_filename = f"viz_{target_result.get('model_name')}.png"
    plt.savefig(out_filename, dpi=300)
    print(f"📊 Grafico salvato come: {out_filename}")
    plt.show()

# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Esempio: Crea un file dummy se non ne hai uno
    dummy_data = [{
        "model_name": "Test_Model_01",
        "gurobi": {
            "solution": {f"x_{i}": np.random.choice([0, 1, 0.5], p=[0.4, 0.4, 0.2]) for i in range(50)}
        }
    }]
    
    with open("dummy_results.json", "w") as f:
        json.dump(dummy_data, f)
        
    # Esegui la visualizzazione
    visualize_model_results("dummy_results.json")
    
    # Se hai il file vero generato dallo script precedente, usa:
    # visualize_model_results("reports/comparison_report_2026....json")
