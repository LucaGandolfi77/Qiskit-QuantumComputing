import json
import os
import re

# Paths
data_json_path = '1302/DATA.json'
notebook_dir = 'QISKIT'

def extract_objective_from_notebook(filepath):
    """
    Reads a notebook and looks for mdl.maximize(...) or mdl.minimize(...)
    Returns (type, expression) or None.
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Look for minimize or maximize
                # Simple regex, assuming single line or simple structure for now
                # Match mdl.maximize(EXPR)
                
                # Check for maximize
                match_max = re.search(r'mdl\.maximize\((.*)\)', source, re.DOTALL)
                if match_max:
                    # simplistic extraction, might capture trailing stuff if complex
                    # Better to look line by line if possible
                    lines = source.split('\n')
                    for line in lines:
                        if 'mdl.maximize' in line:
                            # extract content inside first ( and last )
                            # Assuming it is on one line as per example
                            start = line.find('mdl.maximize(') + len('mdl.maximize(')
                            end = line.rfind(')')
                            if start > -1 and end > -1:
                                return ('MAXIMIZE', line[start:end].strip())
                
                # Check for minimize
                match_min = re.search(r'mdl\.minimize\((.*)\)', source, re.DOTALL)
                if match_min:
                    lines = source.split('\n')
                    for line in lines:
                        if 'mdl.minimize' in line:
                            start = line.find('mdl.minimize(') + len('mdl.minimize(')
                            end = line.rfind(')')
                            if start > -1 and end > -1:
                                return ('MINIMIZE', line[start:end].strip())
                                
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    return None

def main():
    print(f"Reading {data_json_path}...")
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_count = 0
    
    for entry in data:
        source_file = entry.get('source_file')
        # Filter for Qiskit0X as requested
        # 'found like: ... Qiskit0X.ipynb' implies we specifically target these
        if source_file and source_file.startswith('Qiskit0') and len(source_file) == 8: # Qiskit01 to Qiskit09
             nb_path = os.path.join(notebook_dir, f"{source_file}.ipynb")
             print(f"Checking {nb_path}...")
             
             obj_info = extract_objective_from_notebook(nb_path)
             if obj_info:
                 obj_type, obj_expr = obj_info
                 print(f"  Found {obj_type}: {obj_expr}")
                 
                 # clean up expression if needed (remove comments etc)
                 # Update entry
                 entry['objective'] = {
                     'type': obj_type,
                     'expression': obj_expr
                 }
                 modified_count += 1
             else:
                 print(f"  No objective found in {nb_path}")

    if modified_count > 0:
        print(f"Updating {data_json_path} with {modified_count} objectives...")
        with open(data_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("Done.")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    main()
