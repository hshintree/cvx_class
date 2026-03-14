import numpy as np
import cvxpy as cp

def pick_solver():
    installed = cp.installed_solvers()
    for s in ["CLARABEL", "ECOS", "OSQP", "SCS"]:
        if s in installed:
            return s
    return None

def solve_blood_banks(kappa=0.5, allow_shipments=True):
    # data
    p  = np.array([4, 2, 2, 1], float)
    d  = np.array([20,  5, 10, 15], float)
    s  = np.array([30, 10,  5,  0], float)
    dt = np.array([10, 25,  5, 15], float)
    st = np.array([ 5, 20, 15, 20], float)

    n = 4
    ones = np.ones(n)

    # variables
    B  = cp.Variable((n, n), nonneg=True)
    Bt = cp.Variable((n, n), nonneg=True)
    t  = cp.Variable(n)  # ALWAYS variable

    constraints = []

    # meet demand
    constraints += [B @ ones == d]
    constraints += [Bt @ ones == dt]

    # forbid shipments if requested
    if not allow_shipments:
        constraints += [t == 0]

    # post-shipment supplies nonnegative
    constraints += [s - t >= 0]
    constraints += [st + t >= 0]

    # usage <= post-shipment supply
    constraints += [B.T  @ ones <= s - t]
    constraints += [Bt.T @ ones <= st + t]

    # substitution zeros
    forbid = [(0,1),(0,2),(0,3),   # O demand: only O
              (1,2),(1,3),         # A demand: O or A
              (2,1),(2,3)]         # B demand: O or B
    for i,j in forbid:
        constraints += [B[i,j]  == 0]
        constraints += [Bt[i,j] == 0]

    # objective
    used1 = B.T  @ ones
    used2 = Bt.T @ ones
    usage_cost = p @ (used1 + used2)

    if allow_shipments:
        obj = cp.Minimize(usage_cost + kappa * cp.norm1(t))
    else:
        obj = cp.Minimize(usage_cost)

    prob = cp.Problem(obj, constraints)

    solver = pick_solver()
    if solver is None:
        raise RuntimeError(f"No suitable solver found. Installed: {cp.installed_solvers()}")

    solve_kwargs = {}
    if solver == "SCS":
        solve_kwargs = {"max_iters": 200000, "eps": 1e-6}

    prob.solve(solver=solver, **solve_kwargs)

    return prob, B, Bt, t, usage_cost, solver

if __name__ == "__main__":
    prob, B, Bt, t, usage_cost, solver = solve_blood_banks(kappa=0.5, allow_shipments=True)
    print("=== Shipments allowed ===")
    print("solver:", solver)
    print("status:", prob.status)
    print("optimal total cost:", prob.value)
    print("t*:", t.value)

    # infeasible check
    prob0, *_ = solve_blood_banks(kappa=0.5, allow_shipments=False)
    print("\n=== No shipments (t=0) ===")
    print("status:", prob0.status)