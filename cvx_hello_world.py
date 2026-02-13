import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# -----------------------------
# Data (matches your script)
# -----------------------------
np.random.seed(1)
n = 20
pbar = np.ones((n, 1)) * 0.03
pbar += np.r_[np.random.rand(n - 1, 1), np.zeros((1, 1))] * 0.12

S = np.random.randn(n, n)
S = S.T @ S
S = S / max(np.abs(np.diag(S))) * 0.2
S[:, -1] = np.zeros(n)
S[-1, :] = np.zeros(n)

x_unif = np.ones((n, 1)) / n

# Scalars/arrays convenient for CVXPY
p = pbar.reshape(-1)           # shape (n,)
Sigma = S                      # shape (n,n)
x_unif_vec = x_unif.reshape(-1)

mu_unif = float(p @ x_unif_vec)
var_unif = float(x_unif_vec @ Sigma @ x_unif_vec)
sd_unif = float(np.sqrt(var_unif))

print("Uniform portfolio:")
print(f"  mean return = {mu_unif:.6f}")
print(f"  variance    = {var_unif:.6f}")
print(f"  stdev       = {sd_unif:.6f}\n")


# -----------------------------
# Helper: solve min-variance with constraints + target mean
# -----------------------------
def min_variance_portfolio(mu_target, mode):
    """
    mode in {"unconstrained", "long_only", "short_limit"}
    """
    x = cp.Variable(n)

    # objective: minimize variance
    var = cp.quad_form(x, Sigma)
    constraints = [
        cp.sum(x) == 1,
        p @ x == mu_target,   # same expected return as uniform (part a)
    ]

    if mode == "long_only":
        constraints += [x >= 0]
    elif mode == "short_limit":
        x_minus = cp.pos(-x)                 # elementwise max{-x,0}
        constraints += [cp.sum(x_minus) <= 0.5]
    elif mode == "unconstrained":
        pass
    else:
        raise ValueError("unknown mode")

    prob = cp.Problem(cp.Minimize(var), constraints)

    prob.solve()

    x_val = x.value
    var_val = float(x_val @ Sigma @ x_val)
    sd_val = float(np.sqrt(max(var_val, 0.0)))
    mu_val = float(p @ x_val)
    return x_val, mu_val, var_val, sd_val


# -----------------------------
# (a) Three portfolios at mu = mu_unif
# -----------------------------
modes = ["unconstrained", "long_only", "short_limit"]
results_a = {}

for mode in modes:
    x_val, mu_val, var_val, sd_val = min_variance_portfolio(mu_unif, mode)
    results_a[mode] = (x_val, mu_val, var_val, sd_val)

print("(a) Minimum-variance portfolios with mean = uniform mean:\n")
for mode in modes:
    x_val, mu_val, var_val, sd_val = results_a[mode]
    print(f"{mode:14s}: mean={mu_val:.6f}  var={var_val:.6f}  sd={sd_val:.6f}")

print("\nCompare to uniform:")
print(f"uniform         : mean={mu_unif:.6f}  var={var_unif:.6f}  sd={sd_unif:.6f}")

print("\nRisk ratios (sd / uniform sd):")
for mode in modes:
    _, _, _, sd_val = results_a[mode]
    print(f"{mode:14s}: {sd_val/sd_unif:.4f}")


# -----------------------------
# (b) Efficient frontiers (risk-return tradeoff curves)
#     For each target return r, solve:
#        minimize x^T Sigma x
#        s.t. sum x = 1, p^T x >= r, constraints...
# -----------------------------
def frontier(mode, r_grid):
    sds = []
    mus = []

    for r in r_grid:
        x = cp.Variable(n)
        var = cp.quad_form(x, Sigma)
        constraints = [cp.sum(x) == 1,
                       p @ x >= r]

        if mode == "long_only":
            constraints += [x >= 0]
        elif mode == "short_limit":
            x_minus = cp.pos(-x)
            constraints += [cp.sum(x_minus) <= 0.5]
        else:
            raise ValueError("frontier only for long_only or short_limit here")

        prob = cp.Problem(cp.Minimize(var), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            # infeasible r (too high), skip
            continue

        x_val = x.value
        mu_val = float(p @ x_val)
        var_val = float(x_val @ Sigma @ x_val)
        sd_val = float(np.sqrt(max(var_val, 0.0)))

        mus.append(mu_val)
        sds.append(sd_val)

    return np.array(sds), np.array(mus)


# Choose a reasonable return range from the data.
# For long-only, max return is basically max(p); min is min(p) (with sum=1, x>=0).
# For short-limit, returns can go higher, but we’ll still grid around the natural scale.
p_min = float(np.min(p))
p_max = float(np.max(p))

# grid of target returns
r_grid = np.linspace(p_min, p_max, 60)

sd_long, mu_long = frontier("long_only", r_grid)
sd_short, mu_short = frontier("short_limit", r_grid)

# Plot
plt.figure(figsize=(7.5, 5.0))
plt.plot(sd_long, mu_long, marker="o", markersize=3, linewidth=1, label="Long-only (x ≥ 0)")
plt.plot(sd_short, mu_short, marker="o", markersize=3, linewidth=1, label="Short limit (1^T x_- ≤ 0.5)")

# mark uniform point
plt.scatter([sd_unif], [mu_unif], marker="x", s=80, label="Uniform")

plt.xlabel("Standard deviation of return  (x^T Σ x)^{1/2}")
plt.ylabel("Mean return  p̄^T x")
plt.title("Risk–return tradeoff curves")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
