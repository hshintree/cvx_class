import numpy as np
import cvxpy as cp

# ---------------------------
# Generate censored-fit data (same as prompt)
# ---------------------------
np.random.seed(15)

n = 20
M = 25
K = 100

c_true = np.random.randn(n, 1)
X = np.random.randn(n, K)
y_full = X.T @ c_true + 0.1 * np.sqrt(n) * np.random.randn(K, 1)  # (K,1)

# Sort by y and reorder X accordingly
sort_ind = np.argsort(y_full[:, 0])          # (K,)
y_sorted = y_full[sort_ind, :]              # (K,1)
X_sorted = X[:, sort_ind]                   # (n,K)

# Censor threshold between y[M-1] and y[M]
D = ((y_sorted[M-1, 0] + y_sorted[M, 0]) / 2.0).item()

# Uncensored / censored split
y_u = y_sorted[:M, :]                       # (M,1)
X_u = X_sorted[:, :M]                       # (n,M)
X_c = X_sorted[:, M:]                       # (n,K-M)

# Ensure all are plain 2D numpy arrays (CVXPY-friendly)
X_u = np.asarray(X_u, dtype=float)
X_c = np.asarray(X_c, dtype=float)
y_u = np.asarray(y_u, dtype=float)

# ---------------------------
# Censored least squares (convex QP)
# ---------------------------
c = cp.Variable((n, 1))
t = cp.Variable((K - M, 1))                 # slack = max(D - pred, 0)

pred_u = cp.matmul(X_u.T, c)                # (M,1)
pred_c = cp.matmul(X_c.T, c)                # (K-M,1)

constraints = [
    t >= 0,
    t >= D - pred_c
]

objective = cp.Minimize(cp.sum_squares(y_u - pred_u) + cp.sum_squares(t))
prob = cp.Problem(objective, constraints)

# Choose an available solver
installed = set(cp.installed_solvers())
for cand in ["OSQP", "CLARABEL", "SCS"]:
    if cand in installed:
        SOLVER = cand
        break
else:
    raise RuntimeError(f"No suitable solver found. Installed: {sorted(installed)}")

print("Using solver:", SOLVER)
prob.solve(solver=SOLVER)

c_hat = c.value
y_c_hat = (X_c.T @ c_hat) + t.value         # reconstructed censored y (>= D)

# ---------------------------
# Naive least squares (ignore censored points)
# ---------------------------
c_ls, *_ = np.linalg.lstsq(X_u.T, y_u, rcond=None)

# ---------------------------
# Relative errors
# ---------------------------
def rel_err(true, est):
    return np.linalg.norm(true - est) / np.linalg.norm(true)

err_censored = rel_err(c_true, c_hat)
err_ls = rel_err(c_true, c_ls)

# ---------------------------
# Report
# ---------------------------
np.set_printoptions(precision=6, suppress=True)

print("\nD =", D)
print("\nRelative error (censored fit) :", err_censored)
print("Relative error (ignore cens.)  :", err_ls)

print("\nFirst 10 entries of c_true:")
print(c_true[:20, 0])

print("\nFirst 10 entries of c_hat (censored fit):")
print(c_hat[:20, 0])

print("\nFirst 10 entries of c_ls (ignore censored):")
print(c_ls[:20, 0])

print("\nSanity check: min reconstructed censored y =", float(np.min(y_c_hat)))
print("Should be >= D (up to tolerance):", D)
