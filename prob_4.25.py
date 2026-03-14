import cvxpy as cp
import numpy as np

# Joint pmf: p[x1, x2, x3, x4], each index in {0,1}
p = cp.Variable((2, 2, 2, 2), nonneg=True)

constraints = []

# Total probability
constraints += [cp.sum(p) == 1]

# Marginals
constraints += [cp.sum(p[1, :, :, :]) == 0.9]  # P(X1=1)
constraints += [cp.sum(p[:, 1, :, :]) == 0.9]  # P(X2=1)
constraints += [cp.sum(p[:, :, 1, :]) == 0.1]  # P(X3=1)

# Conditional: P(X1=1, X4=0 | X3=1)=0.7  => P(X1=1,X3=1,X4=0)=0.07
constraints += [cp.sum(p[1, :, 1, 0]) == 0.07]

# Conditional: P(X4=1 | X2=1, X3=0)=0.6
lhs = cp.sum(p[:, 1, 0, 1])           # P(X4=1, X2=1, X3=0)
rhs = 0.6 * cp.sum(p[:, 1, 0, :])     # 0.6 * P(X2=1, X3=0)
constraints += [lhs == rhs]

# Objective quantity: P(X4=1)
p_x4_1 = cp.sum(p[:, :, :, 1])

# Pick an installed solver (prefer LP/QP solvers if available; fall back to SCS)
installed = set(cp.installed_solvers())
for cand in ["HIGHS", "GLPK", "GLPK_MI", "CBC", "CLARABEL", "OSQP", "SCS"]:
    if cand in installed:
        SOLVER = cand
        break
else:
    raise RuntimeError(f"No suitable solver found. Installed: {sorted(installed)}")

print("Using solver:", SOLVER)

# Minimize
prob_min = cp.Problem(cp.Minimize(p_x4_1), constraints)
min_val = prob_min.solve(solver=SOLVER)

# Maximize
prob_max = cp.Problem(cp.Maximize(p_x4_1), constraints)
max_val = prob_max.solve(solver=SOLVER)

print("status (min):", prob_min.status)
print("min P(X4=1):", min_val)

print("status (max):", prob_max.status)
print("max P(X4=1):", max_val)

# Optional sanity checks (tolerances matter with first-order solvers like SCS)
p_val = p.value
print("\nSanity checks:")
print("sum p:", p_val.sum())
print("P(X1=1):", p_val[1, :, :, :].sum())
print("P(X2=1):", p_val[:, 1, :, :].sum())
print("P(X3=1):", p_val[:, :, 1, :].sum())
print("P(X1=1,X4=0,X3=1):", p_val[1, :, 1, 0].sum())
den = p_val[:, 1, 0, :].sum()
print("P(X4=1|X2=1,X3=0):", (p_val[:, 1, 0, 1].sum() / den) if den > 1e-12 else np.nan)
