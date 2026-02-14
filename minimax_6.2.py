import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

k = 201
t = np.linspace(-3, 3, k)
y = np.exp(t)

a0 = cp.Variable()
a1 = cp.Variable()
a2 = cp.Variable()
b1 = cp.Variable()
b2 = cp.Variable()

eps = 1e-6

lo, hi = 0.0, 10.0

for _ in range(60):
    gamma = 0.5 * (lo + hi)

    cons = []
    for ti, yi in zip(t, y):
        p = a0 + a1*ti + a2*(ti**2)
        q = 1.0 + b1*ti + b2*(ti**2)
        cons += [q >= eps]
        cons += [p <= (yi + gamma)*q]
        cons += [p >= (yi - gamma)*q]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=cp.ECOS)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        hi = gamma
    else:
        lo = gamma

gamma_star = hi

cons = []
for ti, yi in zip(t, y):
    p = a0 + a1*ti + a2*(ti**2)
    q = 1.0 + b1*ti + b2*(ti**2)
    cons += [q >= eps]
    cons += [p <= (yi + gamma_star)*q]
    cons += [p >= (yi - gamma_star)*q]

prob = cp.Problem(cp.Minimize(0), cons)
try:
    prob.solve(solver=cp.ECOS)
except cp.error.SolverError:
    prob.solve(solver=cp.SCS)

a0v, a1v, a2v, b1v, b2v = a0.value, a1.value, a2.value, b1.value, b2.value

def f_eval(tt):
    num = a0v + a1v*tt + a2v*(tt**2)
    den = 1.0 + b1v*tt + b2v*(tt**2)
    return num / den

f_t = f_eval(t)
err = f_t - y

print(f"a0 = {a0v:.6f}")
print(f"a1 = {a1v:.6f}")
print(f"a2 = {a2v:.6f}")
print(f"b1 = {b1v:.6f}")
print(f"b2 = {b2v:.6f}")
print(f"optimal objective (max |f(t_i)-y_i|) ≈ {gamma_star:.3f}")

tt = np.linspace(-3, 3, 600)
ff = f_eval(tt)

fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=False)

axes[0,0].plot(t, y, label="data: exp(t)")
axes[0,0].plot(tt, ff, label="rational fit")
axes[0,0].set_title("Data and fit")
axes[0,0].grid(True)
axes[0,0].legend()

axes[1,0].plot(t, err)
axes[1,0].set_title("Error: f(t_i) - y_i")
axes[1,0].grid(True)

axes[0,1].axis("off")
axes[0,2].axis("off")
axes[0,3].axis("off")
axes[1,1].axis("off")
axes[1,2].axis("off")
axes[1,3].axis("off")

plt.tight_layout()
plt.show()
