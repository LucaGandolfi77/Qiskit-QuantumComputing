import json
import gurobipy as gp
from gurobipy import GRB
import time
from collections import defaultdict
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import math
import numpy as np

# Set non-interactive backend for matplotlib
plt.switch_backend('Agg')


# -------------------------------------------------------------------------
# Symbolic Helper for Expression Parsing
# -------------------------------------------------------------------------
class SymbolicExpr:
    def __init__(self, linear=None, quadratic=None, constant=0.0):
        self.linear = defaultdict(float, linear or {})
        self.quadratic = defaultdict(float, quadratic or {})
        self.constant = constant

    @classmethod
    def var(cls, name):
        return cls(linear={name: 1.0})

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return SymbolicExpr(self.linear, self.quadratic, self.constant + other)
        if isinstance(other, SymbolicExpr):
            l = self.linear.copy()
            for k, v in other.linear.items(): l[k] += v
            q = self.quadratic.copy()
            for k, v in other.quadratic.items(): q[k] += v
            return SymbolicExpr(l, q, self.constant + other.constant)
        return NotImplemented

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other): return self.__add__(other * -1)
    
    def __rsub__(self, other): return (self * -1).__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            l = {k: v * other for k, v in self.linear.items()}
            q = {k: v * other for k, v in self.quadratic.items()}
            return SymbolicExpr(l, q, self.constant * other)
        if isinstance(other, SymbolicExpr):
            # Simple Linear * Linear or Linear * Constant logic
            res = SymbolicExpr(constant=self.constant * other.constant)
            for v, c in self.linear.items(): res.linear[v] += c * other.constant
            for v, c in other.linear.items(): res.linear[v] += c * self.constant
            # Quadratic term approximation (assuming Gurobi handles expansion)
            # Actually we just need to collect coeffs for (v1, v2)
            for v1, c1 in self.linear.items():
                for v2, c2 in other.linear.items():
                    k = tuple(sorted((v1, v2)))
                    res.quadratic[k] += c1 * c2
            for k, v in self.quadratic.items(): res.quadratic[k] += v * other.constant
            for k, v in other.quadratic.items(): res.quadratic[k] += v * self.constant
            return res
        return NotImplemented

    def __rmul__(self, other): return self.__mul__(other)
    def __neg__(self): return self * -1
    def __truediv__(self, other): 
        if isinstance(other, (int, float)) and other != 0: return self * (1.0/other)
        return NotImplemented
    
    # Comparison operators return (lhs - rhs, sense)
    def __le__(self, other): return (self - other, GRB.LESS_EQUAL)
    def __eq__(self, other): return (self - other, GRB.EQUAL)
    def __ge__(self, other): return (self - other, GRB.GREATER_EQUAL)

# -------------------------------------------------------------------------
# Solver
# -------------------------------------------------------------------------
def solve_model_from_json(m_data, model_id):
    model_name = m_data.get('source_file', f'Model_{model_id}')
    
    # Init Gurobi Model
    mdl = gp.Model(model_name)
    mdl.setParam('OutputFlag', 1)  # Enable output to see logs in console too if needed 

    # 1. Variables
    variables = {}
    var_defs = m_data.get('variables', {})
    
    # Check if 'constants' are defined in JSON
    constants = m_data.get('constants', {})

    # Create variables
    for name, cfg in var_defs.items():
        if name in constants: continue # Skip if it is a constant placeholder
            
        vtype_str = cfg.get('type', 'binary').lower()
        if 'binary' in vtype_str:
            variables[name] = mdl.addVar(vtype=GRB.BINARY, name=name)
        elif 'integer' in vtype_str:
            variables[name] = mdl.addVar(vtype=GRB.INTEGER, lb=cfg.get('min', 0), name=name)
        else:
            variables[name] = mdl.addVar(vtype=GRB.CONTINUOUS, lb=cfg.get('min', 0), name=name)
            
    mdl.update()

    # 2. Context for parsing
    # Map variable names to SymbolicExpr(var_name)
    context = {name: SymbolicExpr.var(name) for name in variables}
    # Add constants
    context.update(constants)
    
    # 3. Constraints
    constraints_added = 0
    raw_constraints = m_data.get('constraints', [])
    
    # Iterative parsing to handle dependencies 
    # (Just assume order is roughly correct or repeat passes)
    for _ in range(2): 
        for c_item in raw_constraints:
            c_name = c_item.get('name')
            expr_str = c_item.get('expression')
            
            if not expr_str: continue

            try:
                # Parse
                val = eval(expr_str, {"__builtins__": {}}, context)
                
                # If it evaluates to a Gurobi constraint relation
                if isinstance(val, tuple) and len(val) == 2:
                    lhs_expr, sense = val
                    
                    # Convert SymbolicExpr to Gurobi expression
                    g_expr = gp.LinExpr()
                    g_expr += lhs_expr.constant
                    for v, coeff in lhs_expr.linear.items():
                        if v in variables: g_expr.add(variables[v], coeff)
                    
                    # Handle quadratic terms if Gurobi supports QuadExpr in addConstr
                    final_expr = g_expr
                    if lhs_expr.quadratic:
                         g_quad = gp.QuadExpr()
                         g_quad += g_expr
                         for (u, v), coeff in lhs_expr.quadratic.items():
                             if u in variables and v in variables:
                                 g_quad.add(variables[u] * variables[v], coeff)
                         final_expr = g_quad
                    
                    if sense == GRB.LESS_EQUAL:
                        mdl.addConstr(final_expr <= 0, name=c_name or "")
                    elif sense == GRB.EQUAL:
                        mdl.addConstr(final_expr == 0, name=c_name or "")
                    elif sense == GRB.GREATER_EQUAL:
                        mdl.addConstr(final_expr >= 0, name=c_name or "")
                    
                    constraints_added += 1
                elif isinstance(val, SymbolicExpr):
                    # It's a defined expression, store in context
                    if c_name:
                        context[c_name] = val
                    # print(f"Defined expression: {c_name}")
            except Exception as e:
                # print(f"Skipping constraint '{c_name}': {e}")
                pass

    # 4. Objective
    obj_data = m_data.get('objective', {})
    obj_str = obj_data.get('expression', '')
    if obj_str:
        try:
            val = eval(obj_str, {"__builtins__": {}}, context)
            if isinstance(val, SymbolicExpr):
                g_obj = gp.QuadExpr()
                g_obj += val.constant
                for v, coeff in val.linear.items():
                    if v in variables: g_obj.add(variables[v], coeff)
                for (u, v), coeff in val.quadratic.items():
                    if u in variables and v in variables:
                        g_obj.add(variables[u] * variables[v], coeff)
                
                sense = GRB.MAXIMIZE if obj_data.get('type', 'MAXIMIZE') == 'MAXIMIZE' else GRB.MINIMIZE
                mdl.setObjective(g_obj, sense)
        except Exception as e:
            print(f"Error parsing objective: {e}")

    # Solve
    start_time = time.time()
    mdl.optimize()
    elapsed = time.time() - start_time

    # Collect Results
    res = {
        'name': model_name,
        'status': mdl.status,
        'time': elapsed,
        'obj_val': 0.0,
        'vars': {},
        'constraints_count': constraints_added
    }

    if mdl.status == GRB.OPTIMAL:
        res['obj_val'] = mdl.objVal
        for v in mdl.getVars():
            # Store all variables regardless of value
            res['vars'][v.VarName] = v.X
                
    return res

# -------------------------------------------------------------------------
# HTML Generator
# -------------------------------------------------------------------------
def _generate_single_group_plots(items, title_prefix):
    """Helper to generate hist + heatmap for a list of (name, value) tuples"""
    if not items:
        return None, None
        
    names = [x[0] for x in items]
    values = [x[1] for x in items]

    # --- 1. Histogram ---
    plt.figure(figsize=(4, 3))
    plt.hist(values, bins=min(20, len(values)), color='#0056b3', alpha=0.7, edgecolor='black')
    plt.title(f'{title_prefix} Distrib.', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    hist_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- 2. Heatmap (Square Grid) ---
    n = len(values)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols) if cols > 0 else 0
    
    # Pad
    padded = np.array(values + [np.nan] * (rows * cols - n))
    grid = padded.reshape((rows, cols))
    
    plt.figure(figsize=(5, 4))
    im = plt.imshow(grid, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im)
    plt.title(f'{title_prefix} Heatmap', fontsize=10)
    plt.axis('off')
    
    # Annotate
    vmin, vmax = min(values), max(values)
    rng = vmax - vmin if vmax > vmin else 1.0
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                val = values[idx]
                norm = (val - vmin) / rng
                color = 'black' if norm > 0.6 else 'white'
                # Dynamic font size
                fontsize = 8
                if n > 30: fontsize = 6
                if n > 70: fontsize = 4
                
                plt.text(j, i, names[idx], ha="center", va="center", color=color, 
                         fontsize=fontsize, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return hist_b64, heatmap_b64

def create_plots(vars_dict):
    """Generates plots for u-vars and v-vars separately"""
    if not vars_dict:
        return {}
    
    sorted_items = sorted(vars_dict.items())
    u_items = [x for x in sorted_items if x[0].startswith('u')]
    v_items = [x for x in sorted_items if x[0].startswith('v')]
    
    plots = {}
    if u_items:
        plots['u'] = _generate_single_group_plots(u_items, 'u-Vars')
    if v_items:
        plots['v'] = _generate_single_group_plots(v_items, 'v-Vars')
        
    return plots

def generate_html(results, filename="report_q003.html"):
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    final_filename = f"{name}_{timestamp}{ext}"

    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f0f2f5; color: #333; }
            .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 3px solid #0056b3; padding-bottom: 10px; }
            h2 { color: #0056b3; margin-top: 0; }
            .badge { padding: 5px 10px; border-radius: 4px; font-weight: bold; color: white; display: inline-block; font-size: 0.9em; }
            .opt { background: #28a745; }
            .fail { background: #dc3545; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }
            th, td { padding: 10px; border-bottom: 1px solid #eee; text-align: left; }
            th { background-color: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Optimization Report</h1>
        <p>Results from <code>DATA.json</code></p>
    """
    
    for r in results:
        status_code = r['status']
        if status_code == GRB.OPTIMAL:
            status_text = "OPTIMAL"
            status_cls = "opt"
        elif status_code == GRB.UNBOUNDED:
            status_text = "UNBOUNDED"
            status_cls = "fail"
        elif status_code == GRB.INFEASIBLE:
            status_text = "INFEASIBLE"
            status_cls = "fail"
        else:
            status_text = f"STATUS {status_code}"
            status_cls = "fail"
        
        html += f"""
        <div class="card">
            <h2>{r['name']} <span class="badge {status_cls}">{status_text}</span></h2>
            <div style="display: flex; gap: 20px; font-size: 0.95em; color: #555;">
                <div>⏱ <strong>Time:</strong> {r['time']:.4f}s</div>
                <div>🎯 <strong>Objective:</strong> {r['obj_val']:.4f}</div>
                <div>🔒 <strong>Constraints:</strong> {r['constraints_count']}</div>
            </div>
        """
        
        if r['vars']:
            # Generate plots
            plot_data = {}
            try:
                plot_data = create_plots(r['vars'])
            except Exception as e:
                print(f"Error generating plots: {e}")

            # --- Render Plots ---
            # 1. U-Vars
            if 'u' in plot_data and plot_data['u']:
                u_h, u_hm = plot_data['u']
                html += f"""
                <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">
                    <h4 style="margin:5px 0; color:#555;">u-Variables Analysis</h4>
                    <div style="display: flex; gap: 15px;">
                        <img src="data:image/png;base64,{u_h}" style="border:1px solid #ddd; border-radius:4px;">
                        <img src="data:image/png;base64,{u_hm}" style="border:1px solid #ddd; border-radius:4px;">
                    </div>
                </div>
                """
            
            # 2. V-Vars
            if 'v' in plot_data and plot_data['v']:
                v_h, v_hm = plot_data['v']
                html += f"""
                <div style="margin-top: 15px; border-top: 1px dashed #eee; padding-top: 10px;">
                    <h4 style="margin:5px 0; color:#555;">v-Variables Analysis</h4>
                    <div style="display: flex; gap: 15px;">
                        <img src="data:image/png;base64,{v_h}" style="border:1px solid #ddd; border-radius:4px;">
                        <img src="data:image/png;base64,{v_hm}" style="border:1px solid #ddd; border-radius:4px;">
                    </div>
                </div>
                """

            html += "<h3>Computed Variables</h3><table><tr><th>Variable</th><th>Value</th></tr>"
            # Sort by name for cleaner look
            sorted_vars = sorted(r['vars'].items())
            for name, val in sorted_vars:
                # Highlight integers and non-zero values
                val_fmt = f"{val:.0f}" if abs(val - round(val)) < 1e-4 else f"{val:.4f}"
                style = "font-weight: bold; color: #0056b3;" if abs(val) > 1e-4 else "color: #999;"
                html += f"<tr><td><code>{name}</code></td><td style='{style}'>{val_fmt}</td></tr>"
            html += "</table>"
        else:
            html += "<p style='color: #888; margin-top: 15px;'><em>No variables found or solution not optimal.</em></p>"
            
        html += "</div>"

    html += """
    </body>
    </html>
    """
    
    with open(final_filename, 'w', encoding='utf-8') as f:
        f.write(html)
    return final_filename

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    json_path = Path(__file__).parent / "DATA_01.json"
    if not json_path.exists():
        # Fallback to current file dir if name is different
        json_path = Path("DATA.json")
    
    if not json_path.exists():
        print("DATA.json not found!")
        return

    print(f"Loading {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return
        
    print(f"Found {len(data)} models.")
    results = []
    
    for i, m_data in enumerate(data):
        print(f"[{i+1}/{len(data)}] Solving model...")
        try:
            res = solve_model_from_json(m_data, i+1)
            results.append(res)
        except Exception as e:
            print(f"Failed to solve model {i+1}: {e}")
            import traceback
            traceback.print_exc()

    out_file = generate_html(results)
    print(f"Report generated: {out_file}")

if __name__ == "__main__":
    main()
