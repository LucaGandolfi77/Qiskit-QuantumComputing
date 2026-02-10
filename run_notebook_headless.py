#!/usr/bin/env python3
import sys
import nbformat
import traceback
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import time
import datetime
import html


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
    # optional: parse extra args
    max_code_cells = None
    per_cell_mode = False
    for a in sys.argv[2:]:
        if a in ("--per-cell", "percell"):
            per_cell_mode = True
            continue
        try:
            if max_code_cells is None:
                max_code_cells = int(a)
        except Exception:
            pass

    try:
        nb = nbformat.read(nb_path, as_version=4)
    except FileNotFoundError:
        print(f"Notebook not found: {nb_path}", file=sys.stderr)
        sys.exit(2)

    # execute the notebook
    # increase timeouts to allow longer-running cells
    client = NotebookClient(nb, timeout=1800, startup_timeout=120, kernel_name="python3")
    # ensure kernel is started for single-client execution mode
    try:
        # prefer start_new_kernel (available on current nbclient)
        # ensure kernel manager exists first
        try:
            client.create_kernel_manager()
        except Exception:
            pass
        client.start_new_kernel()
        # also try to start a kernel client if available
        try:
            client.start_new_kernel_client()
        except Exception:
            pass
    except Exception:
        try:
            # older versions expose setup_kernel
            client.setup_kernel()
        except Exception:
            # fallback: some nbclient versions start the kernel lazily; continue
            pass

    executed_nb_path = nb_path.replace('.ipynb', '_executed.ipynb')
    log_path = nb_path.replace('.ipynb', '.log.txt')

    def sanitize_outputs(nb_obj):
        # Ensure outputs are plain dicts (nbformat can accept dicts but some kernels may return objects)
        for cell in nb_obj.cells:
            if cell.get('cell_type') != 'code':
                continue
            outs = cell.get('outputs', [])
            sanitized = []
            for o in outs:
                if isinstance(o, dict):
                    # wrap dict into NotebookNode so attribute access works
                    try:
                        sanitized.append(nbformat.NotebookNode(o))
                    except Exception:
                        sanitized.append(o)
                else:
                    # fallback: coerce to stream text and wrap
                    try:
                        sanitized.append(nbformat.NotebookNode({
                            'output_type': 'stream',
                            'name': 'stdout',
                            'text': str(o)
                        }))
                    except Exception:
                        sanitized.append({
                            'output_type': 'stream',
                            'name': 'stdout',
                            'text': str(o)
                        })
            cell['outputs'] = sanitized

    # record per-cell timings and results
    cell_logs = []

    def render_outputs_text(outputs):
        lines = []
        for out in outputs:
            otype = out.get('output_type') if isinstance(out, dict) else None
            if otype == 'stream':
                lines.append(out.get('text', ''))
            elif otype in ('execute_result', 'display_data'):
                data = out.get('data', {})
                text = data.get('text/plain')
                if text is None:
                    for k in ('text/html', 'image/png', 'image/jpeg'):
                        if k in data:
                            text = f"<{k} output omitted>"
                            break
                if text:
                    lines.append(str(text))
            elif otype == 'error':
                tb = out.get('traceback', [])
                lines.append(''.join(tb))
            else:
                # unknown output; stringify
                lines.append(str(out))
        return ''.join(lines)

    def write_reports(cell_logs):
        # ANSI color codes
        ANSI = {'reset':'\x1b[0m','red':'\x1b[31m','green':'\x1b[32m','yellow':'\x1b[33m','cyan':'\x1b[36m'}
        # plain text with color (for terminal)
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write(f"Run report for {nb_path} -- {datetime.datetime.utcnow().isoformat()}Z\n\n")
            for entry in cell_logs:
                status = entry['status']
                color = ANSI['green'] if status=='success' else ANSI['yellow'] if status=='skipped' else ANSI['red']
                lf.write(f"{color}--- Cell {entry['index']} [{status.upper()}] ({entry['duration']:.3f}s) ---{ANSI['reset']}\n")
                lf.write(entry['text'] + '\n')
        # HTML report
        html_lines = []
        html_lines.append('<!doctype html>')
        html_lines.append('<meta charset="utf-8"><style>body{font-family:Arial,Helvetica,sans-serif;padding:20px} .cell{border:1px solid #ddd;padding:10px;margin:10px 0;border-radius:6px} .hdr{font-weight:600} .success{background:#e6ffed;border-color:#b7f3c3} .error{background:#ffecec;border-color:#f5c2c2} .skipped{background:#f0f8ff;border-color:#cfe8ff} .meta{color:#666;font-size:0.9em}</style>')
        html_lines.append(f'<h1>Run report for {html.escape(nb_path)}</h1>')
        html_lines.append(f'<p><em>Generated: {html.escape(datetime.datetime.utcnow().isoformat())}Z</em></p>')
        for entry in cell_logs:
            cls = 'success' if entry['status']=='success' else 'skipped' if entry['status']=='skipped' else 'error'
            html_lines.append(f'<div class="cell {cls}">')
            html_lines.append(f'<div class="hdr">Cell {entry["index"]} — {entry["status"].upper()} — <span class="meta">{entry["duration"]:.3f}s</span></div>')
            html_lines.append('<pre>' + html.escape(entry['text']) + '</pre>')
            html_lines.append('</div>')
        report_path = executed_nb_path.replace('_executed.ipynb', '_report.html')
        with open(report_path, 'w', encoding='utf-8') as hf:
            hf.write('\n'.join(html_lines))

    try:
        # execute cells one by one and save partial results after each cell
        total = len(nb.cells)
        code_cells_executed = 0
        for idx, cell in enumerate(nb.cells):
            if cell.get('cell_type') != 'code':
                continue
            if max_code_cells is not None and code_cells_executed >= max_code_cells:
                break

            # detect and skip cells that do in-notebook package installation to avoid kernel instability
            src = cell.get('source', '')
            if isinstance(src, list):
                src_text = '\n'.join(src)
            else:
                src_text = str(src)

            entry = {'index': idx+1, 'status': None, 'start': None, 'end': None, 'duration': None, 'text': ''}
            if ('pip install' in src_text) or src_text.strip().startswith('!'):
                # mark skipped installation cell in outputs
                cell.setdefault('outputs', []).append({
                    'output_type': 'stream',
                    'name': 'stdout',
                    'text': 'Skipped pip/install cell during headless automated run.\n'
                })
                entry['status'] = 'skipped'
                entry['start'] = time.time()
                entry['end'] = time.time()
                entry['duration'] = 0.0
                entry['text'] = 'Skipped installation cell.'
                cell_logs.append(entry)
                try:
                    sanitize_outputs(nb)
                except Exception:
                    pass
                nbformat.write(nb, executed_nb_path)
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f"--- Skipped Cell {idx+1} ---\nSkipped installation cell.\n")
                print(f"Skipped install cell {idx+1}/{total}")
                continue

            print(f"Executing cell {idx+1}/{total}...", flush=True)
            entry['start'] = time.time()
            try:
                if per_cell_mode:
                    # run this cell in its own fresh notebook + kernel
                    single_nb = nbformat.v4.new_notebook()
                    single_nb.cells = [nbformat.v4.new_code_cell(cell.get('source', ''))]
                    single_client = NotebookClient(single_nb, timeout=1800, startup_timeout=120, kernel_name="python3")
                    try:
                        single_client.execute()
                        # copy outputs back
                        cell['outputs'] = single_nb.cells[0].get('outputs', [])
                    finally:
                        # no explicit shutdown needed; client context handles it
                        pass
                else:
                    client.execute_cell(cell, idx)

                entry['end'] = time.time()
                entry['duration'] = entry['end'] - entry['start']
                entry['status'] = 'success'
                entry['text'] = render_outputs_text(cell.get('outputs', []))

            except Exception as e:
                entry['end'] = time.time()
                entry['duration'] = entry['end'] - entry['start']
                entry['status'] = 'error'
                entry['text'] = f"Error: {e}\n" + render_outputs_text(cell.get('outputs', []))
                print(f"Error executing cell {idx+1}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # save partial state
                try:
                    sanitize_outputs(nb)
                except Exception:
                    pass
                try:
                    nbformat.write(nb, executed_nb_path)
                except Exception as e2:
                    print(f"Failed to write partial executed notebook: {e2}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                # append and write reports immediately
                cell_logs.append(entry)
                write_reports(cell_logs)
                raise

            # after each successful cell execution, persist progress and update logs
            cell_logs.append(entry)
            try:
                try:
                    sanitize_outputs(nb)
                except Exception:
                    pass
                nbformat.write(nb, executed_nb_path)
                write_reports(cell_logs)
            except Exception as e:
                print(f"Failed to persist progress after cell {idx+1}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # continue trying next cells

    except Exception as e:
        print(f"Execution stopped with error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # final persistence (may be same as last saved state)
    try:
        try:
            sanitize_outputs(nb)
        except Exception:
            pass
        nbformat.write(nb, executed_nb_path)
    except Exception as e:
        print(f"Failed to write final executed notebook: {e}", file=sys.stderr)

    print(f"Executed notebook saved to: {executed_nb_path}")
    print(f"Log file saved to: {log_path}")

if __name__ == "__main__":
    main()
