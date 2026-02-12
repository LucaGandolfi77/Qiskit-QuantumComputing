"""run_and_compare_gui_fixed.py (Headless/CLI version)"""

import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time
import json
import re
import argparse
import sys
import traceback
import warnings

# Suppress SciPy sparse efficiency warnings
warnings.filterwarnings('ignore', message='.*splu converted its input to CSC format.*')
warnings.filterwarnings('ignore', message='.*spsolve is more efficient when sparse b is in the CSC matrix format.*')

# ------------------------- Optional deps -------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    # Try new Qiskit 1.0+ structure
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization import QuadraticProgram
    from qiskit.primitives import StatevectorSampler as Sampler
    QISKIT_AVAILABLE = True
    QISKIT_VERSION = "1.x"
except ImportError:
    try:
        # Fallback to old structure (< 1.0)
        from qiskit import Aer
        from qiskit.algorithms import QAOA
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization import QuadraticProgram
        QISKIT_AVAILABLE = True
        QISKIT_VERSION = "0.x"
    except ImportError:
        QISKIT_AVAILABLE = False
        QISKIT_VERSION = None

# ------------------------- Symbolic Helper -------------------------
class SymbolicExpr:
    """Helper to parse linear/quadratic expressions via eval()."""
    def __init__(self, linear=None, quadratic=None, constant=0.0):
        self.linear = defaultdict(float, linear or {})      # {var_name: coeff}
        self.quadratic = defaultdict(float, quadratic or {}) # {tuple(sorted((v1, v2))): coeff}
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

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rsub__(self, other):
        return (self * -1).__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            l = {k: v * other for k, v in self.linear.items()}
            q = {k: v * other for k, v in self.quadratic.items()}
            return SymbolicExpr(l, q, self.constant * other)
        
        if isinstance(other, SymbolicExpr):
            # (L1 + C1) * (L2 + C2)
            # Result C = C1*C2
            res = SymbolicExpr(constant=self.constant * other.constant)
            
            # Linear terms: L1*C2 + L2*C1
            for v, c in self.linear.items():
                res.linear[v] += c * other.constant
            for v, c in other.linear.items():
                res.linear[v] += c * self.constant
            
            # Quadratic terms from Linear*Linear: L1*L2
            for v1, c1 in self.linear.items():
                for v2, c2 in other.linear.items():
                    k = tuple(sorted((v1, v2)))
                    res.quadratic[k] += c1 * c2
            
            # Quadratic terms from Quad*Constant: Q1*C2 + Q2*C1
            for k, v in self.quadratic.items():
                res.quadratic[k] += v * other.constant
            for k, v in other.quadratic.items():
                res.quadratic[k] += v * self.constant

            return res
            
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        return self * -1

# ------------------------- Helpers -------------------------
def fmt_num(x, decimals=2, suffix=''):
    if isinstance(x, (int, float)):
        return f"{x:.{decimals}f}{suffix}"
    return "N/A"


def safe_preview(solution: dict, max_items: int = 5) -> str:
    if not isinstance(solution, dict) or not solution:
        return "N/A"
    parts = []
    for k, v in list(solution.items())[:max_items]:
        if isinstance(v, float):
            v_str = f"{v:.3f}"
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    if len(solution) > max_items:
        parts.append(f"... (+{len(solution) - max_items})")
    return ', '.join(parts)


# ------------------------- Solvers -------------------------
class GurobiSolver:
    def solve(self, problem_data: dict) -> dict:
        start = time.time()

        if not GUROBI_AVAILABLE:
            return {'status': 'unavailable', 'time': 0.0, 'error': 'gurobipy not installed'}

        var_cfg = problem_data.get('variables_config', {})
        obj_coeffs = problem_data.get('objective_coefficients', {})
        quad_coeffs = problem_data.get('quadratic_coefficients', {})
        constants = problem_data.get('constants', {})

        if not var_cfg or (not obj_coeffs and not quad_coeffs):
             # It implies empty objective, but maybe constant?
             pass

        try:
            # Create environment
            try:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
            except gp.GurobiError:
                env = None
            
            mdl = gp.Model("gurobi_model", env=env)
            mdl.setParam('OutputFlag', 0)

            variables = {}
            for var_name, cfg in var_cfg.items():
                # Skip if it is actually a constant
                if var_name in constants:
                    continue

                t = cfg.get('type', 'binary')
                if t == 'binary':
                    variables[var_name] = mdl.addVar(vtype=GRB.BINARY, name=var_name)
                elif t == 'integer':
                    variables[var_name] = mdl.addVar(vtype=GRB.INTEGER, lb=0, name=var_name)
                elif t == 'decimal':
                    variables[var_name] = mdl.addVar(vtype=GRB.CONTINUOUS, lb=cfg.get('min', 0.0), ub=cfg.get('max', 1.0), name=var_name)
                else:
                    variables[var_name] = mdl.addVar(vtype=GRB.CONTINUOUS, name=var_name)

            mdl.update()

            obj_expr = gp.quicksum(float(obj_coeffs.get(v, 0.0)) * variables[v] for v in variables if v in obj_coeffs)
            
            if quad_coeffs:
                for (u, v), c in quad_coeffs.items():
                    if u in variables and v in variables:
                        obj_expr += float(c) * variables[u] * variables[v]

            mdl.setObjective(obj_expr, GRB.MAXIMIZE)

            mdl.optimize()

            elapsed = time.time() - start

            if mdl.status == GRB.OPTIMAL:
                sol = {v.VarName: v.X for v in mdl.getVars() if abs(v.X) > 1e-6}
                return {
                    'status': 'optimal',
                    'objective_value': float(mdl.objVal),
                    'time': elapsed,
                    'solution': sol,
                    'num_solutions': len(sol)
                }
            elif mdl.status == GRB.UNBOUNDED:
                return {'status': 'unbounded', 'time': elapsed, 'error': 'Model is unbounded (Status 5)'}
            elif mdl.status == GRB.INF_OR_UNBD:
                return {'status': 'unbounded', 'time': elapsed, 'error': 'Model is infeasible or unbounded (Status 4)'}
            elif mdl.status == GRB.INFEASIBLE:
                 return {'status': 'infeasible', 'time': elapsed, 'error': 'Model is infeasible (Status 3)'}

            return {'status': 'infeasible', 'time': elapsed, 'error': f'Status code: {mdl.status}'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'time': time.time() - start, 'error': str(e)}


class QAOASolver:
    def solve(self, problem_data: dict) -> dict:
        start = time.time()

        if not QISKIT_AVAILABLE:
            return {'status': 'skipped', 'reason': 'Qiskit not installed', 'time': 0.0}

        var_cfg = problem_data.get('variables_config', {})
        obj_coeffs = problem_data.get('objective_coefficients', {})
        quad_coeffs = problem_data.get('quadratic_coefficients', {})

        bin_vars = [v for v, cfg in var_cfg.items() if cfg.get('type') == 'binary']
        if len(bin_vars) == 0:
            return {'status': 'skipped', 'reason': 'No binary variables', 'time': time.time() - start}
        if len(bin_vars) > 20:
            return {'status': 'skipped', 'reason': 'Too many binary variables (>20)', 'time': time.time() - start}

        try:
            qp = QuadraticProgram("qaoa_model")
            for v in bin_vars:
                qp.binary_var(name=v)

            # Maximize Obj -> Minimize (-Obj)
            linear = {v: -float(obj_coeffs.get(v, 0.0)) for v in bin_vars if v in obj_coeffs}
            quadratic = {}
            if quad_coeffs:
                for (u, v), c in quad_coeffs.items():
                    if u in bin_vars and v in bin_vars:
                         quadratic[(u, v)] = -float(c)
            
            # Check for empty objective
            if not linear and not quadratic:
                return {
                    'status': 'completed',
                    'objective_value': 0.0,
                    'time': time.time() - start,
                    'solution': {v: 0 for v in bin_vars}, # Default to 0? Or maybe skipped?
                    'num_iterations': 0,
                    'note': 'Empty objective -> Trivial 0'
                }

            # Need to confirm qp.minimize supports quadratic dict. Yes it does.
            qp.minimize(linear=linear, quadratic=quadratic)

            if QISKIT_VERSION == "1.x":
                # Qiskit 1.x
                sampler = Sampler() # StatevectorSampler
                # Reduced maxiter and reps for speed on large datasets
                qaoa_optimizer = COBYLA(maxiter=50) 
                qaoa = QAOA(optimizer=qaoa_optimizer, reps=1, sampler=sampler)
                res = MinimumEigenOptimizer(qaoa).solve(qp)
                # Extract parameters if available
                optimal_params = None
                if res.min_eigen_solver_result and hasattr(res.min_eigen_solver_result, 'optimal_point'):
                    optimal_params = [float(x) for x in res.min_eigen_solver_result.optimal_point] if res.min_eigen_solver_result.optimal_point is not None else None
            else:
                # Old Qiskit
                backend = Aer.get_backend('qasm_simulator')
                # Reduced maxiter and reps for speed on large datasets
                qaoa_optimizer = COBYLA(maxiter=50)
                qaoa = QAOA(optimizer=qaoa_optimizer, reps=1, quantum_instance=backend)
                res = MinimumEigenOptimizer(qaoa).solve(qp)
                optimal_params = None
                if res.min_eigen_solver_result and hasattr(res.min_eigen_solver_result, 'optimal_point'):
                    optimal_params = [float(x) for x in res.min_eigen_solver_result.optimal_point] if res.min_eigen_solver_result.optimal_point is not None else None

            elapsed = time.time() - start

            sol = {name: int(val) for name, val in zip([v.name for v in qp.variables], res.x) if val > 0.5}
            fval_max = -float(res.fval)

            return {
                'status': 'completed',
                'objective_value': fval_max,
                'time': elapsed,
                'solution': sol,
                'num_iterations': 50,
                'optimal_parameters': optimal_params,
                'debug_info': f"reps=1, vars={len(bin_vars)}"
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'time': time.time() - start, 'error': str(e)}


# ------------------------- Runner -------------------------
class ModelRunner:
    def __init__(self, problems: dict):
        self.problems = problems
        self.gurobi = GurobiSolver()
        self.qaoa = QAOASolver()

    def run_all(self) -> list:
        results = []
        items = list(self.problems.items())

        for i, (name, pdata) in enumerate(items, 1):
            print(f"\n[{i}/{len(items)}] {name}")

            result = {
                'model_name': name,
                'num_variables': len(pdata.get('variables_config', {})),
                'num_constraints': len(pdata.get('constraints', [])) if isinstance(pdata.get('constraints', []), list) else 0,
                'constants': pdata.get('constants', {}) if isinstance(pdata.get('constants', {}), dict) else {},
                'source': pdata.get('source_file') or pdata.get('source')
            }

            print("  Gurobi...", end=' ', flush=True)
            g = self.gurobi.solve(pdata)
            print("Done" if g.get('status') == 'optimal' else "Fail")
            if g.get('status') != 'optimal':
                 print(f"    -> Gurobi Error: {g.get('error', 'unknown error')}")
            result['gurobi'] = g

            print("  QAOA...", end=' ', flush=True)
            # Peek at binary variable count for user info
            vcfg = pdata.get('variables_config', {})
            n_bin = len([v for v, cfg in vcfg.items() if cfg.get('type') == 'binary'])
            if n_bin > 15:
                 print(f"({n_bin} vars, ~{(2**n_bin)/1000:.0f}k states, please wait)...", end=' ', flush=True)
            
            q = self.qaoa.solve(pdata)
            print("Done" if q.get('status') == 'completed' else "-" if q.get('status') == 'skipped' else "Fail")
            if q.get('status') != 'completed':
                 print(f"    -> QAOA Status: {q.get('status')} Reason: {q.get('reason', q.get('error', 'unknown'))}")
            result['qaoa'] = q

            results.append(result)

        return results


# ------------------------- HTML report -------------------------
class HTMLReportGenerator:
    def __init__(self, results: list):
        self.results = results

    def generate(self, output_file: Path):
        html = self._generate_html()
        output_file.write_text(html, encoding='utf-8')

    def _generate_html(self) -> str:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        total = len(self.results)
        g_opt = sum(1 for r in self.results if r.get('gurobi', {}).get('status') == 'optimal')
        q_done = sum(1 for r in self.results if r.get('qaoa', {}).get('status') == 'completed')
        avg_g_time = 0.0
        if total:
            times = [r.get('gurobi', {}).get('time') for r in self.results]
            nums = [t for t in times if isinstance(t, (int, float))]
            avg_g_time = sum(nums) / len(nums) if nums else 0.0

        # Data for chart
        labels = [r.get('model_name', f'Model {i}') for i, r in enumerate(self.results)]
        g_values = [r.get('gurobi', {}).get('objective_value', 0) if isinstance(r.get('gurobi', {}).get('objective_value'), (int, float)) else 0 for r in self.results]
        q_values = [r.get('qaoa', {}).get('objective_value', 0) if isinstance(r.get('qaoa', {}).get('objective_value'), (int, float)) else 0 for r in self.results]
        
        # Serialize for JS
        import json
        js_labels = json.dumps(labels)
        js_g_values = json.dumps(g_values)
        js_q_values = json.dumps(q_values)

        css = """
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:Segoe UI,Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px}
        .container{max-width:1400px;margin:0 auto;background:#fff;border-radius:15px;box-shadow:0 20px 60px rgba(0,0,0,0.3);overflow:hidden}
        .header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:40px;text-align:center}
        .header h1{font-size:2.5em;margin-bottom:10px}
        .header p{font-size:1.1em;opacity:.9}
        .summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;padding:30px;background:#f8f9fa}
        .summary-card{background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);text-align:center}
        .summary-card h3{color:#667eea;font-size:2em;margin-bottom:5px}
        .summary-card p{color:#666;font-size:.9em}
        .chart-container{padding:30px;background:#fff;border-bottom:1px solid #eee;height:400px}
        .models{padding:30px}
        .model-card{background:#fff;border:2px solid #e0e0e0;border-radius:10px;padding:25px;margin-bottom:25px;transition:transform .2s,box-shadow .2s}
        .model-card:hover{transform:translateY(-5px);box-shadow:0 10px 30px rgba(0,0,0,0.15)}
        .model-header{border-bottom:2px solid #667eea;padding-bottom:15px;margin-bottom:20px}
        .model-header h2{color:#333;font-size:1.5em}
        .model-info{display:flex;gap:30px;margin-bottom:20px;flex-wrap:wrap}
        .info-item{flex:1;min-width:150px}
        .info-label{font-weight:700;color:#666;font-size:.85em;text-transform:uppercase;margin-bottom:5px}
        .info-value{font-size:1.05em;color:#333;word-break:break-word}
        table{width:100%;border-collapse:collapse;margin-top:20px}
        th{background:#667eea;color:#fff;padding:15px;text-align:left;font-weight:600}
        td{padding:12px 15px;border-bottom:1px solid #e0e0e0}
        tr:hover{background:#f8f9fa}
        .status-optimal{background:#4caf50;color:#fff;padding:5px 10px;border-radius:5px;font-size:.85em;font-weight:700}
        .status-error{background:#f44336;color:#fff;padding:5px 10px;border-radius:5px;font-size:.85em;font-weight:700}
        .status-skipped{background:#9e9e9e;color:#fff;padding:5px 10px;border-radius:5px;font-size:.85em;font-weight:700}
        .winner{background:#fff9c4;font-weight:700}
        .footer{background:#333;color:#fff;text-align:center;padding:20px;font-size:.9em}
        .mono{font-family:Consolas,Monaco,monospace;font-size:10px}
        """

        html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Gurobi vs QAOA - Comparison Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{{css}}</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Gurobi vs QAOA</h1>
<p>Comparison Report - Optimization Results</p>
<p style="font-size:0.9em;margin-top:10px;">{{now}}</p>
</div>
<div class="summary">
  <div class="summary-card"><h3>{{total}}</h3><p>Modelli Testati</p></div>
  <div class="summary-card"><h3>{{g_opt}}</h3><p>Gurobi Optimal</p></div>
  <div class="summary-card"><h3>{{q_done}}</h3><p>QAOA Completati</p></div>
  <div class="summary-card"><h3>{{avg_g_time:.3f}}s</h3><p>Tempo Medio Gurobi</p></div>
</div>
<div class="chart-container">
    <canvas id="comparisonChart"></canvas>
</div>
<div class="models">
<h2 style="margin-bottom:20px;color:#333;">Dettagli Modelli</h2>
"""

        for r in self.results:
            html += self._generate_model_card(r)

        html += f"""
</div>
<div class="footer">
<p>Generated by run_and_compare_gui_fixed.py | Powered by Gurobi & Qiskit</p>
</div>
</div>
<script>
const ctx = document.getElementById('comparisonChart');
new Chart(ctx, {{
    type: 'bar',
    data: {{
        labels: {{js_labels}},
        datasets: [
            {{
                label: 'Gurobi Objective',
                data: {{js_g_values}},
                backgroundColor: 'rgba(76, 175, 80, 0.6)',
                borderColor: 'rgba(76, 175, 80, 1)',
                borderWidth: 1
            }},
            {{
                label: 'QAOA Objective',
                data: {{js_q_values}},
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{
                beginAtZero: true
            }}
        }}
    }}
}});
</script>
</body>
</html>
"""
        return html

    def _generate_model_card(self, result: dict) -> str:
        g = result.get('gurobi', {})
        q = result.get('qaoa', {})

        g_status = g.get('status', 'error')
        q_status = q.get('status', 'error')

        g_status_class = 'status-optimal' if g_status == 'optimal' else 'status-skipped' if g_status == 'unavailable' else 'status-error'
        q_status_class = 'status-optimal' if q_status == 'completed' else 'status-skipped' if q_status == 'skipped' else 'status-error'

        g_winner = ''
        q_winner = ''
        if g_status == 'optimal' and q_status == 'completed':
            g_time = g.get('time')
            q_time = q.get('time')
            if isinstance(g_time, (int, float)) and isinstance(q_time, (int, float)):
                if g_time <= q_time:
                    g_winner = 'winner'
                else:
                    q_winner = 'winner'

        g_obj_str = fmt_num(g.get('objective_value'), 2)
        q_obj_str = fmt_num(q.get('objective_value'), 2)
        if q_status == 'skipped' and not isinstance(q.get('objective_value'), (int, float)):
            q_obj_str = q.get('reason', 'N/A')

        g_time_str = fmt_num(g.get('time'), 4)
        q_time_str = fmt_num(q.get('time'), 4)

        g_num_sol = g.get('num_solutions', 0)
        q_num_sol = len(q.get('solution', {}) or {})
        
        # New: Parameters
        q_params = q.get('optimal_parameters')
        if q_params:
            q_params_str = ", ".join([f"{{p:.3f}}" for p in q_params])
        else:
            q_params_str = "-"

        consts = result.get('constants', {})
        consts_str = ', '.join([f"{{k}}={{v}}" for k, v in list(consts.items())[:10]]) if isinstance(consts, dict) and consts else "-"
        if isinstance(consts, dict) and len(consts) > 10:
            consts_str += f" ... (+{{len(consts)-10}})"

        g_preview = safe_preview(g.get('solution', {}), 5)
        q_preview = safe_preview(q.get('solution', {}), 5)

        return f"""
<div class="model-card">
  <div class="model-header">
    <h2>Model: {{result.get('model_name','Unknown')}}</h2>
  </div>
  <div class="model-info">
    <div class="info-item"><div class="info-label">Variabili</div><div class="info-value">{{result.get('num_variables','?')}}</div></div>
    <div class="info-item"><div class="info-label">Vincoli</div><div class="info-value">{{result.get('num_constraints','?')}}</div></div>
    <div class="info-item"><div class="info-label">Parametri</div><div class="info-value">{{consts_str}}</div></div>
  </div>

  <table>
    <tr><th>Metodo</th><th>Status</th><th>Obiettivo</th><th>Tempo (s)</th><th>Soluzioni</th><th>Parametri (QAOA)</th><th>Preview</th></tr>
    <tr class="{{g_winner}}">
      <td><strong>Gurobi</strong></td>
      <td><span class="{{g_status_class}}">{{str(g_status).upper()}}</span></td>
      <td>{{g_obj_str}}</td>
      <td>{{g_time_str}}</td>
      <td>{{g_num_sol}}</td>
       <td>-</td>
      <td class="mono">{{g_preview}}</td>
    </tr>
    <tr class="{{q_winner}}">
      <td><strong>QAOA</strong></td>
      <td><span class="{{q_status_class}}">{{str(q_status).upper()}}</span></td>
      <td>{{q_obj_str}}</td>
      <td>{{q_time_str}}</td>
      <td>{{q_num_sol}}</td>
      <td class="mono">{{q_params_str}}</td>
      <td class="mono">{{q_preview}}</td>
    </tr>
  </table>
</div>
"""


# ------------------------- Input loaders -------------------------
def load_problems_from_json(json_path: Path) -> dict:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        problems = {}
        for m in data:
            name = Path(m.get('source_file', 'model')).stem
            
            vars_config = m.get('variables', {})
            obj_coeffs = m.get('objective_coefficients', {})
            quad_coeffs = m.get('quadratic_coefficients', {})
            
            # If objective coefficients missing, try to parse from m['objective']['expression']
            if not obj_coeffs and not quad_coeffs and 'objective' in m and m['objective']:
                try:
                    expr_str = m['objective'].get('expression', '')
                    if expr_str:
                        context = {v: SymbolicExpr.var(v) for v in vars_config}
                        consts = m.get('constants', {})
                        context.update(consts)
                        
                        constraints = m.get('constraints', [])
                        expr_constraints = [c for c in constraints if c.get('type') == 'expression']
                        
                        max_iter = 5
                        for _ in range(max_iter):
                            for c in expr_constraints:
                                cname = c.get('name')
                                cexpr = c.get('expression')
                                if cname and cexpr and cname not in context:
                                     try:
                                         val = eval(cexpr, {"__builtins__": {}}, context)
                                         context[cname] = val
                                     except NameError:
                                         pass
                                     except Exception:
                                         pass

                        if 'objective' in m and 'expression' in m['objective']:
                            obj_expr_str = m['objective']['expression']
                            res = eval(obj_expr_str, {"__builtins__": {}}, context)
                            
                            if isinstance(res, SymbolicExpr):
                                obj_coeffs = dict(res.linear)
                                quad_coeffs = dict(res.quadratic)
                except Exception as e:
                    print(f"Warning: Failed to parse objective for {name}: {e}")

            problems[name] = {
                'variables_config': vars_config,
                'objective_coefficients': obj_coeffs,
                'quadratic_coefficients': quad_coeffs,
                'constraints': m.get('constraints', []),
                'constants': m.get('constants', {}),
                'source_file': m.get('source_file')
            }
        return problems

    if isinstance(data, dict):
        return data

    return {}


def load_problems_from_csv(csv_path: Path) -> dict:
    import csv
    import re

    sections_by_nb: dict[str, dict[str, list[str]]] = {}
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            nb = row[0].strip()
            sec = row[1].strip()
            con = row[2].strip() if len(row) > 2 else ''
            if not nb or nb.lower() == 'notebook' or not sec:
                continue
            if con:
                sections_by_nb.setdefault(nb, {}).setdefault(sec, []).append(con)

    def parse_vars(s: str) -> list[str]:
        s = (s or '').replace('\\n', ' ')
        raw = [v.strip() for v in re.split(r'[,\s]+', s) if v.strip()]
        bad = {'(continuous)', '(integer)'}
        out = [v for v in raw if v not in bad]
        seen = set()
        uniq = []
        for v in out:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq

    def extract_coeffs(expr: str) -> dict:
        expr = (expr or '').replace(' ', '')
        coeffs = {}
        for m in re.finditer(r'([+-]?[0-9]*\.?[0-9]+)\*([a-zA-Z_]\w*)', expr):
            coeffs[m.group(2)] = float(m.group(1))
        for m in re.finditer(r'(?<![\w])([+-])([a-zA-Z_]\w*)', expr):
            v = m.group(2)
            if v not in coeffs:
                coeffs[v] = -1.0 if m.group(1) == '-' else 1.0
        return coeffs

    problems = {}
    for nb, secs in sections_by_nb.items():
        bins = []
        for s in secs.get('Binaries', []):
            bins.extend(parse_vars(s))
        for s in secs.get('Generals', []):
            bins.extend(parse_vars(s))

        maximize = ' '.join(secs.get('Maximize', []))
        obj = extract_coeffs(maximize)

        all_vars = sorted(set(bins) | set(obj.keys()))
        if not all_vars:
            continue

        var_cfg = {v: {'type': 'binary'} for v in all_vars}

        problems[nb] = {
            'variables_config': var_cfg,
            'objective_coefficients': obj,
            'constants': {'maximize_raw': maximize}
        }

    return problems


# ------------------------- CLI Entry Point -------------------------
def main():
    parser = argparse.ArgumentParser(description="Gurobi vs QAOA Comparator (Headless)")
    parser.add_argument("input_file", nargs="?", default="DATA.json", help="Path to input CSV or JSON file")
    parser.add_argument("--output-dir", default="reports", help="Directory for output reports")
    
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        # Try looking in current dir if not found relative to workspace root
        if Path(input_path.name).exists():
             input_path = Path(input_path.name)
        else:
             print(f"Error: Input file '{input_path}' not found.")
             sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Gurobi vs QAOA Comparator (CLI)")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    try:
        # Load problems
        print("Loading problems...")
        if input_path.suffix.lower() == '.json':
            problems = load_problems_from_json(input_path)
        else:
            problems = load_problems_from_csv(input_path)
        
        if not problems:
            print("No problems found in input file.")
            sys.exit(1)
        
        print(f"Loaded {len(problems)} problems.")

        # Run models
        runner = ModelRunner(problems)
        results = runner.run_all()

        # Generate Report
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = output_dir / f"comparison_report_{ts}.html"
        
        print(f"\nGenerating HTML report at {html_path}...")
        HTMLReportGenerator(results).generate(html_path)
        
        print("Done!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
