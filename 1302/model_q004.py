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
from scipy.optimize import minimize

# Qiskit Imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorSampler, StatevectorEstimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit components not found. QAOA will be skipped.")

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
# QAOA Solver Integration
# -------------------------------------------------------------------------
def solve_with_qaoa(qubo_expr, variables_map, alpha=0.25):
    """
    Solves a QUBO using Qiskit QAOA with CVaR expectation.
    qubo_expr: SymbolicExpr representing the objective.
    variables_map: List of variable names to map to qubits.
    alpha: CVaR parameter (0 < alpha <= 1).
    """
    if not QISKIT_AVAILABLE:
        return {'status': 'SKIPPED', 'obj_val': 0.0, 'vars': {}}
    
    start_time = time.time()
    n = len(variables_map)
    var_to_idx = {name: i for i, name in enumerate(variables_map)}
    
    # 1. Build Hamiltonian
    # Map binary x in {0,1} to Pauli Z in {1,-1}: x = (I - Z)/2
    # This means x_i = 0.5 * I - 0.5 * Z_i
    
    pauli_terms = defaultdict(complex)
    offset = 0.0
    
    def add_term(indices, coeff):
        # indices: list of qubit indices involved
        # coeff: float coefficient
        # term is coeff * prod( (I - Z_k)/2 )
        
        # Expand the product
        # For 1 var: coeff * (0.5 I - 0.5 Z)
        # For 2 vars: coeff * (0.5 I - 0.5 Z_i) * (0.5 I - 0.5 Z_j)
        #           = coeff * (0.25 I - 0.25 Z_i - 0.25 Z_j + 0.25 Z_i Z_j)
        
        k = len(indices)
        factor = coeff / (2**k)
        
        # Iterate over all subsets of indices to form Pauli strings
        from itertools import combinations
        
        # 0 Zs (Identity part)
        pauli_terms[tuple()] += factor
        
        for r in range(1, k + 1):
            sign = (-1)**r
            for combo in combinations(indices, r):
                # Combo is tuple of qubit indices
                key = tuple(sorted(combo))
                pauli_terms[key] += factor * sign

    # Linear terms
    if isinstance(qubo_expr, SymbolicExpr):
        for v, coeff in qubo_expr.linear.items():
            if v in var_to_idx:
                add_term([var_to_idx[v]], coeff)
            else:
                offset += coeff # Constant if var not found (shouldn't happen)
        
        # Quadratic terms
        for (u, v), coeff in qubo_expr.quadratic.items():
            if u in var_to_idx and v in var_to_idx:
                if u == v: # x^2 = x for binary
                    add_term([var_to_idx[u]], coeff)
                else:
                    add_term([var_to_idx[u], var_to_idx[v]], coeff)
        offset += qubo_expr.constant
    
    # Convert pauli_terms to SparsePauliOp
    # keys are tuples of qubit indices. Value is coeff.
    op_list = []
    for qubits, coeff in pauli_terms.items():
        if not qubits:
            # Identity
            s = 'I' * n
        else:
            chars = ['I'] * n
            for q in qubits:
                chars[n - 1 - q] = 'Z' # Little endian in Qiskit (q0 is rightmost)
            s = "".join(chars)
        
        if abs(coeff) > 1e-10:
            op_list.append((s, coeff))
            
    if not op_list:
        return {'status': 'TRIVIAL', 'obj_val': offset, 'vars': {name: 0 for name in variables_map}}

    hamiltonian = SparsePauliOp.from_list(op_list)
    
    # 2. QAOA Circuit (p=1 for simplicity, can increase)
    p = 1
    
    # Use Qiskit's library for robustness
    try:
        from qiskit.circuit.library import QAOAAnsatz
        ansatz = QAOAAnsatz(hamiltonian, reps=p)
    except ImportError:
        return {'status': 'QAOA_LIB_MISSING', 'obj_val': 0.0, 'vars': {}}

    def get_real(c): return c.real

    # 3. CVaR Cost Function
    def cost_func_cvar(params):
        # Bind params
        bound_qc = ansatz.assign_parameters(params)
        bound_qc.measure_all()
        
        sampler = StatevectorSampler()
        try:
            job = sampler.run([(bound_qc,)], shots=1024)
            result = job.result()
            counts = result[0].data.meas.get_counts()
            total = sum(counts.values())
            dist = {int(k, 2): v / total for k, v in counts.items()}
        except Exception as e:
            print(f"Sampler error: {e}")
            return 0.0

        # Precompute term masks (only once would be better, but acceptable here)
        term_data = []
        for pauli_str, coeff in op_list:
            z_indices = [k for k, char in enumerate(reversed(pauli_str)) if char == 'Z']
            term_data.append((z_indices, get_real(coeff)))

        samples = []
        for state_int, prob in dist.items():
             energy = 0.0
             for z_idxs, c in term_data:
                 parity = 0
                 for bit_idx in z_idxs:
                     if (state_int >> bit_idx) & 1:
                         parity += 1
                 energy += c * (1 if parity % 2 == 0 else -1)
             energy += offset
             samples.append((energy, prob))
        
        samples.sort(key=lambda x: x[0])
        cvar_sum = 0.0
        prob_sum = 0.0
        for e, prob in samples:
            if prob_sum >= alpha:
                break
            taken_prob = min(prob, alpha - prob_sum)
            cvar_sum += e * taken_prob
            prob_sum += taken_prob
            
        return cvar_sum / alpha

    # --- STANDARD QAOA (Estimator-based) ---
    estimator = StatevectorEstimator()

    def cost_func_standard(params):
        """Standard QAOA using Estimator to find expectation value <H>"""
        # Estimator takes circuits and observables
        # params must be passed/bound
        job = estimator.run([(ansatz, hamiltonian, params)])
        result = job.result()
        # Returns expectation value
        return float(result[0].data.evs) + offset

    # Select method
    if alpha < 1.0:
        # Use CVaR
        cost_function = cost_func_cvar
        method_name = "CVaR"
    else:
        # Use Standard Estimator (faster/standard)
        if estimator:
            cost_function = cost_func_standard
            method_name = "Standard_Estimator"
        else:
            # Fallback to CVaR logic with alpha=1.0 if Estimator missing
            cost_function = cost_func_cvar
            method_name = "Standard_Sampler"

    # 4. Minimize
    init_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)
    res = minimize(cost_function, init_params, method='COBYLA', options={'maxiter': 50})
    
    # 5. Extract Best Solution

    opt_params = res.x
    
    # Run one last time to get counts
    bound_qc = ansatz.assign_parameters(opt_params)
    bound_qc.measure_all()
    sampler = StatevectorSampler()
    job = sampler.run([(bound_qc,)], shots=2048)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    dist = {int(k, 2): v / total for k, v in counts.items()}
    
    # Find most probable state
    best_state_int = max(dist.items(), key=lambda x: x[1])[0]
    
    # Convert to var dict
    # int to binary
    bin_str = format(best_state_int, f'0{n}b')[::-1] # Little endian reverse to map 0->var[0]
    
    final_vars = {}
    for i, name in enumerate(variables_map):
        val = int(bin_str[i]) if i < len(bin_str) else 0
        final_vars[name] = float(val)

    return {
        'status': 'QAOA_CVaR_OPTIMAL',
        'time': time.time() - start_time,
        'obj_val': res.fun, # This is CVaR value, not exact expectation
        'vars': final_vars
    }

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

    # Pre-calculate total bits needed to determine if we need to reduce precision
    total_bits_needed = 0
    max_bits_cap = 100 
    
    for name in variables:
        cfg = var_defs.get(name, {})
        vtype_str = cfg.get('type', 'binary').lower()
        if 'integer' in vtype_str:
            min_v = int(cfg.get('min', 0))
            max_v = int(cfg.get('max', 7)) if 'max' in cfg else 7
            span = max(1, max_v - min_v)
            import math
            nbits = math.floor(math.log2(span)) + 1
            total_bits_needed += nbits
        elif 'binary' in vtype_str:
            total_bits_needed += 1
            
    if total_bits_needed > 20:
        print(f"Warning: Model too large ({total_bits_needed} qubits > 20). Limiting integers to 2 bits to fit.")
        max_bits_cap = 2

    # 2. Context for parsing
    context = {}
    var_map_to_bits = {} # name -> list of (bit_name, coeff, offset)

    # Iterate over copy keys to avoid runtime error when adding bits
    for name in list(variables.keys()):
        cfg = var_defs.get(name, {})
        vtype_str = cfg.get('type', 'binary').lower()
        
        if 'integer' in vtype_str:
            min_val = int(cfg.get('min', 0))
            max_val = cfg.get('max', 1) 
            # If max not set, assume something reasonable or use Gurobi logic? 
            # Better to rely on what user provided. If max missing, default to 7 (3 bits)?
            if 'max' not in cfg: max_val = 7
            else: max_val = int(max_val)

            span = max_val - min_val
            if span < 1: span = 1
            import math
            nbits = math.floor(math.log2(span)) + 1
            
            # Apply Cap
            if nbits > max_bits_cap:
                nbits = max_bits_cap
            
            expr = SymbolicExpr(constant=float(min_val))
            bits_info = []
            for i in range(nbits):
                bname = f"{name}__b{i}"
                coeff = 2**i
                expr = expr + SymbolicExpr.var(bname) * coeff
                bits_info.append((bname, coeff))
                
                # Add bit var to Gurobi and variables dict
                variables[bname] = mdl.addVar(vtype=GRB.BINARY, name=bname)
            
            context[name] = expr
            var_map_to_bits[name] = {'min': min_val, 'bits': bits_info}
            
        elif 'binary' in vtype_str:
            context[name] = SymbolicExpr.var(name)
            var_map_to_bits[name] = {'min': 0, 'bits': [(name, 1)]}
            
        else:
            context[name] = SymbolicExpr.var(name)

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
    qaoa_objective = None
    obj_data = m_data.get('objective', {})
    obj_str = obj_data.get('expression', '')
    if obj_str:
        try:
            val = eval(obj_str, {"__builtins__": {}}, context)
            if isinstance(val, SymbolicExpr):
                qaoa_objective = val
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

    # Solve (Gurobi)
    start_time = time.time()
    mdl.optimize()
    elapsed = time.time() - start_time

    # Collect Gurobi Results
    res = {
        'name': model_name,
        'status': mdl.status,
        'time': elapsed,
        'obj_val': 0.0,
        'vars': {},
        'constraints_count': constraints_added,
        'qaoa_cvar': None,
        'qaoa_std': None
    }

    if mdl.status == GRB.OPTIMAL:
        res['obj_val'] = mdl.objVal
        for v in mdl.getVars():
            # Store all variables regardless of value
            res['vars'][v.VarName] = v.X
    
    # ----------------------------------------------------
    # QAOA ATTEMPT (Simulation/Placeholder)
    # ----------------------------------------------------
    # Default failure/skip states
    res['qaoa_cvar'] = {'status': 'SKIPPED', 'obj_val': 0.0, 'time': 0.0, 'vars': {}}
    res['qaoa_std'] = {'status': 'SKIPPED', 'obj_val': 0.0, 'time': 0.0, 'vars': {}}

    # Determine if we can run QAOA (supports integers via binary expansion now)
    has_continuous = any('continuous' in var_defs.get(v.VarName,{}).get('type','').lower() for v in mdl.getVars())
    
    if not QISKIT_AVAILABLE:
        res['qaoa_cvar']['status'] = 'QISKIT_MISSING'
        res['qaoa_std']['status'] = 'QISKIT_MISSING'
    elif has_continuous:
        res['qaoa_cvar']['status'] = 'NON_BINARY' 
        res['qaoa_std']['status'] = 'NON_BINARY'
    elif not qaoa_objective:
        res['qaoa_cvar']['status'] = 'NO_OBJECTIVE'
        res['qaoa_std']['status'] = 'NO_OBJECTIVE'
    else:
        # Prepare Variable Map for QAOA from the OBJECTIVE expression terms
        # The objective expression now contains the BIT variables
        q_vars = set()
        for v in qaoa_objective.linear: q_vars.add(v)
        for (u,v) in qaoa_objective.quadratic: 
            q_vars.add(u)
            q_vars.add(v)
            
        qaoa_var_list = sorted(list(q_vars))
        
        if len(qaoa_var_list) > 20: 
            res['qaoa_cvar']['status'] = 'TOO_LARGE'
            res['qaoa_std']['status'] = 'TOO_LARGE'
        else:
            # Post-process results (reconstruct integers)
            def reconstruct_vars(bit_vars):
                final = {}
                for name, info in var_map_to_bits.items():
                    val = info['min']
                    for bname, coeff in info['bits']:
                        val += bit_vars.get(bname, 0) * coeff
                    final[name] = val
                return final
            
            # Solve CVaR
            try:
                print("  Running QAOA (CVaR)...")
                cvar_res = solve_with_qaoa(qaoa_objective, qaoa_var_list, alpha=0.25)
                if cvar_res.get('vars'):
                    cvar_res['vars'] = reconstruct_vars(cvar_res['vars'])
                res['qaoa_cvar'] = cvar_res
            except Exception as e:
                print(f"QAOA CVaR failed: {e}")
                res['qaoa_cvar']['status'] = f"ERROR: {str(e)}"

            # Solve Standard
            try:
                print("  Running QAOA (Standard)...")
                std_res = solve_with_qaoa(qaoa_objective, qaoa_var_list, alpha=1.0)
                if std_res.get('vars'):
                    std_res['vars'] = reconstruct_vars(std_res['vars'])
                res['qaoa_std'] = std_res
            except Exception as e:
                print(f"QAOA Standard failed: {e}")
                res['qaoa_std']['status'] = f"ERROR: {str(e)}"

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
            
            <!-- 1. GUROBI SECTION -->
            <div style="margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                <h3 style="color: #28a745; margin-bottom: 10px;">🟢 Gurobi (Classical)</h3>
                <div style="display: flex; gap: 20px; font-size: 0.95em; color: #555; margin-bottom: 15px;">
                    <div>⏱ <strong>Time:</strong> {r['time']:.4f}s</div>
                    <div>🎯 <strong>Objective:</strong> {r['obj_val']:.4f}</div>
                    <div>🔒 <strong>Constraints:</strong> {r['constraints_count']}</div>
                </div>
        """
        
        if r['vars']:
            # Generate plots (Gurobi)
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
                <div style="margin-top: 10px;">
                    <h4 style="margin:5px 0; color:#555;">u-Variables Visualization</h4>
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
                <div style="margin-top: 10px;">
                    <h4 style="margin:5px 0; color:#555;">v-Variables Visualization</h4>
                    <div style="display: flex; gap: 15px;">
                        <img src="data:image/png;base64,{v_h}" style="border:1px solid #ddd; border-radius:4px;">
                        <img src="data:image/png;base64,{v_hm}" style="border:1px solid #ddd; border-radius:4px;">
                    </div>
                </div>
                """

            html += """
                <div style="margin-top: 15px;">
                    <details>
                        <summary style="cursor: pointer; color: #0056b3; font-weight: bold;">View Gurobi Variables Table</summary>
                        <table><tr><th>Variable</th><th>Value</th></tr>
            """
            sorted_vars = sorted(r['vars'].items())
            for name, val in sorted_vars:
                val_fmt = f"{val:.0f}" if abs(val - round(val)) < 1e-4 else f"{val:.4f}"
                style = "font-weight: bold; color: #0056b3;" if abs(val) > 1e-4 else "color: #999;"
                html += f"<tr><td><code>{name}</code></td><td style='{style}'>{val_fmt}</td></tr>"
            html += """
                        </table>
                    </details>
                </div>
            </div>
            """

        # --- QAOA STANDARD SECTION ---
        # Always display section, even if failed/skipped
        q_std = r.get('qaoa_std', {'status': 'MISSING', 'obj_val': 0.0, 'time': 0.0})
        
        # Check for error status
        std_status = q_std.get('status', 'UNKNOWN')
        std_is_error = std_status.startswith(('ERROR', 'SKIPPED', 'MISSING', 'NON_BINARY', 'TOO_LARGE', 'NO_OBJECTIVE', 'QISKIT_MISSING'))
        std_style = "color: #dc3545; font-weight: bold;" if std_is_error else "color: #28a745; font-weight: bold;"

        html += f"""
        <div style="margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
            <h3 style="color: #6f42c1; margin-bottom: 10px;">🟣 Qiskit Standard (Estimator)</h3>
            <div style="display: flex; gap: 20px; font-size: 0.95em; color: #555;">
                <div>⚡ <strong>Status:</strong> <span style="{std_style}">{std_status}</span></div>
        """
        if not std_is_error and 'time' in q_std:
            html += f"""
                <div>⏱ <strong>Time:</strong> {q_std['time']:.4f}s</div>
                <div>🎯 <strong>Approx Objective:</strong> {q_std['obj_val']:.4f}</div>
            """
        html += """</div>"""

        if q_std.get('vars'):
            q_plot_data = create_plots(q_std['vars'])
            if 'u' in q_plot_data and q_plot_data['u']:
                    qh, qhm = q_plot_data['u']
                    html += f"""
                <div style="margin-top: 10px;">
                    <h4 style="margin:5px 0; color:#555;">u-Variables (Standard QAOA)</h4>
                    <div style="display: flex; gap: 15px;">
                        <img src="data:image/png;base64,{qh}" style="border:1px solid #ddd; border-radius:4px;">
                        <img src="data:image/png;base64,{qhm}" style="border:1px solid #ddd; border-radius:4px;">
                    </div>
                </div>
                """
        elif std_is_error:
             html += f"""<p style="color: #666; margin-top: 10px; font-style: italic;">No results generated due to status: {std_status}</p>"""
        
        html += "</div>"
    
        # --- QAOA CVAR SECTION ---
        q_cvar = r.get('qaoa_cvar', {'status': 'MISSING', 'obj_val': 0.0, 'time': 0.0})
        
        cvar_status = q_cvar.get('status', 'UNKNOWN')
        cvar_is_error = cvar_status.startswith(('ERROR', 'SKIPPED', 'MISSING', 'NON_BINARY', 'TOO_LARGE', 'NO_OBJECTIVE', 'QISKIT_MISSING'))
        cvar_style = "color: #dc3545; font-weight: bold;" if cvar_is_error else "color: #28a745; font-weight: bold;"

        html += f"""
        <div>
            <h3 style="color: #0056b3; margin-bottom: 10px;">🔵 Advanced QAOA (CVaR)</h3>
            <div style="display: flex; gap: 20px; font-size: 0.95em; color: #555;">
                <div>⚡ <strong>Status:</strong> <span style="{cvar_style}">{cvar_status}</span></div>
        """
        if not cvar_is_error and 'time' in q_cvar:
             html += f"""
                <div>⏱ <strong>Time:</strong> {q_cvar['time']:.4f}s</div>
                <div>🎯 <strong>CVaR Objective:</strong> {q_cvar['obj_val']:.4f}</div>
             """
        html += """</div>"""

        if q_cvar.get('vars'):
            q_plot_data = create_plots(q_cvar['vars'])
            if 'u' in q_plot_data and q_plot_data['u']:
                    qh, qhm = q_plot_data['u']
                    html += f"""
                <div style="margin-top: 10px;">
                    <h4 style="margin:5px 0; color:#555;">u-Variables (CVaR QAOA)</h4>
                    <div style="display: flex; gap: 15px;">
                        <img src="data:image/png;base64,{qh}" style="border:1px solid #ddd; border-radius:4px;">
                        <img src="data:image/png;base64,{qhm}" style="border:1px solid #ddd; border-radius:4px;">
                    </div>
                </div>
                """
        elif cvar_is_error:
             html += f"""<p style="color: #666; margin-top: 10px; font-style: italic;">No results generated due to status: {cvar_status}</p>"""

        html += "</div>"

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
