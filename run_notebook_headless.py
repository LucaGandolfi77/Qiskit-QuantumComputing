#!/usr/bin/env python3
import sys
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

def print_outputs(nb):
    for idx, cell in enumerate(nb.cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            continue
        print(f"--- Cell {idx} ---")
        for out in outputs:
            otype = out.get("output_type")
            if otype == "stream":
                print(out.get("text", ""), end="")
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                text = data.get("text/plain")
                if text is None:
                    # fallback to other mime types (note: binary/image outputs omitted)
                    for k in ("text/html", "image/png", "image/jpeg"):
                        if k in data:
                            text = f"<{k} output omitted>"
                            break
                if text:
                    print(text)
            elif otype == "error":
                tb = out.get("traceback", [])
                print("".join(tb), file=sys.stderr)

def main():
    nb_path = sys.argv[1] if len(sys.argv) > 1 else "QISKIT/Qiskit01.ipynb"
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except FileNotFoundError:
        print(f"Notebook not found: {nb_path}", file=sys.stderr)
        sys.exit(2)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    try:
        client.execute()
    except CellExecutionError as e:
        print(f"Execution stopped with error: {e}", file=sys.stderr)

    print_outputs(nb)

if __name__ == "__main__":
    main()
