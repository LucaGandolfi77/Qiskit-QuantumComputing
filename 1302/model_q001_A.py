import gurobipy as gp
from gurobipy import GRB

# Crea modello Gurobi
mdl = gp.Model("max_computational_load")

# Parametri
c = 100  # Capacità CPU per server
pi = 10  # (non usato nella massimizzazione pura)
pd = 5   # Peso carico dinamico

# Variabili server (binarie)
s0 = mdl.addVar(vtype=GRB.BINARY, name="s0")
s1 = mdl.addVar(vtype=GRB.BINARY, name="s1")
s2 = mdl.addVar(vtype=GRB.BINARY, name="s2")

# Variabili assegnazione VM (binarie)
v00 = mdl.addVar(vtype=GRB.BINARY, name="v00")
v10 = mdl.addVar(vtype=GRB.BINARY, name="v10")
v20 = mdl.addVar(vtype=GRB.BINARY, name="v20")
v30 = mdl.addVar(vtype=GRB.BINARY, name="v30")
v40 = mdl.addVar(vtype=GRB.BINARY, name="v40")
v01 = mdl.addVar(vtype=GRB.BINARY, name="v01")
v11 = mdl.addVar(vtype=GRB.BINARY, name="v11")
v21 = mdl.addVar(vtype=GRB.BINARY, name="v21")
v31 = mdl.addVar(vtype=GRB.BINARY, name="v31")
v41 = mdl.addVar(vtype=GRB.BINARY, name="v41")
v02 = mdl.addVar(vtype=GRB.BINARY, name="v02")
v12 = mdl.addVar(vtype=GRB.BINARY, name="v12")
v22 = mdl.addVar(vtype=GRB.BINARY, name="v22")
v32 = mdl.addVar(vtype=GRB.BINARY, name="v32")
v42 = mdl.addVar(vtype=GRB.BINARY, name="v42")

# CPU usage per VM (intere, >= 0)
u00 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u00")
u10 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u10")
u20 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u20")
u30 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u30")
u40 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u40")
u01 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u01")
u11 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u11")
u21 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u21")
u31 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u31")
u41 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u41")
u02 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u02")
u12 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u12")
u22 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u22")
u32 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u32")
u42 = mdl.addVar(vtype=GRB.INTEGER, lb=0, name="u42")

# IMPORTANTE: Aggiorna il modello dopo aver aggiunto variabili
mdl.update()

# Carichi computazionali per server
sum0 = (u00*v00 + u10*v10 + u20*v20 + u30*v30 + u40*v40)
sum1 = (u01*v01 + u11*v11 + u21*v21 + u31*v31 + u41*v41)
sum2 = (u02*v02 + u12*v12 + u22*v22 + u32*v32 + u42*v42)

# VINCOLI
# Ogni VM su esattamente 1 server
mdl.addConstr(v00 + v01 + v02 == 1, "vm0_assigned")
mdl.addConstr(v10 + v11 + v12 == 1, "vm1_assigned")
mdl.addConstr(v20 + v21 + v22 == 1, "vm2_assigned")
mdl.addConstr(v30 + v31 + v32 == 1, "vm3_assigned")
mdl.addConstr(v40 + v41 + v42 == 1, "vm4_assigned")

# Capacità server
mdl.addConstr(sum0 <= c * s0, "capacity_server0")
mdl.addConstr(sum1 <= c * s1, "capacity_server1")
mdl.addConstr(sum2 <= c * s2, "capacity_server2")

# CPU usage limitato
mdl.addConstr(u00 <= c * v00, "cpu_limit_u00")
mdl.addConstr(u10 <= c * v10, "cpu_limit_u10")
mdl.addConstr(u20 <= c * v20, "cpu_limit_u20")
mdl.addConstr(u30 <= c * v30, "cpu_limit_u30")
mdl.addConstr(u40 <= c * v40, "cpu_limit_u40")
mdl.addConstr(u01 <= c * v01, "cpu_limit_u01")
mdl.addConstr(u11 <= c * v11, "cpu_limit_u11")
mdl.addConstr(u21 <= c * v21, "cpu_limit_u21")
mdl.addConstr(u31 <= c * v31, "cpu_limit_u31")
mdl.addConstr(u41 <= c * v41, "cpu_limit_u41")
mdl.addConstr(u02 <= c * v02, "cpu_limit_u02")
mdl.addConstr(u12 <= c * v12, "cpu_limit_u12")
mdl.addConstr(u22 <= c * v22, "cpu_limit_u22")
mdl.addConstr(u32 <= c * v32, "cpu_limit_u32")
mdl.addConstr(u42 <= c * v42, "cpu_limit_u42")

# OBIETTIVO: Massimizza carico computazionale totale
mdl.setObjective(sum0 + sum1 + sum2, GRB.MAXIMIZE)

# Risolvi
mdl.optimize()

# Stampa risultati
if mdl.status == GRB.OPTIMAL:
    print(f"Carico totale massimizzato: {mdl.objVal}")
    print(f"Server 0 attivo: {s0.X}, carico: {sum0.getValue()}")
    print(f"Server 1 attivo: {s1.X}, carico: {sum1.getValue()}")
    print(f"Server 2 attivo: {s2.X}, carico: {sum2.getValue()}")
    
    # Dettagli assegnazione VM
    print("\nAssegnazioni VM:")
    for i in range(5):
        for j in range(3):
            var = mdl.getVarByName(f"v{i}{j}")
            if var.X > 0.5:  # Variabile binaria = 1
                u_var = mdl.getVarByName(f"u{i}{j}")
                print(f"  VM{i} → Server{j} (CPU usage: {u_var.X})")
else:
    print(f"Nessuna soluzione ottimale trovata. Status: {mdl.status}")
