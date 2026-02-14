import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

n = 4
m = 2

A = np.array([
    [ 0.95,  0.16,  0.12,  0.01],
    [-0.12,  0.98, -0.11, -0.03],
    [-0.16,  0.02,  0.98,  0.03],
    [-0.00,  0.02, -0.04,  1.03],
])

B = np.array([
    [ 0.8 , 0. ],
    [ 0.1 , 0.2],
    [ 0.  , 0.8],
    [-0.2 , 0.1],
])

x_init = np.ones(n)
T = 100

u = cp.Variable((m, T))
x = cp.Variable((n, T + 1))

cons = [x[:, 0] == x_init, x[:, T] == 0]
for t in range(T):
    cons += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

objs = [
    cp.Minimize(cp.sum_squares(u)),
    cp.Minimize(cp.sum([cp.norm(u[:, t], 2) for t in range(T)])),
    cp.Minimize(cp.max(cp.hstack([cp.norm(u[:, t], 2) for t in range(T)]))),
    cp.Minimize(cp.sum([cp.norm(u[:, t], 1) for t in range(T)])),
]
labels = [
    r"(a) $\sum_t \|u_t\|_2^2$",
    r"(b) $\sum_t \|u_t\|_2$",
    r"(c) $\max_t \|u_t\|_2$",
    r"(d) $\sum_t \|u_t\|_1$",
]

plt.figure(figsize=(18, 6))
tgrid = np.arange(T)

for i, (obj, label) in enumerate(zip(objs, labels)):
    prob = cp.Problem(obj, cons)
    try:
        if i == 0:
            prob.solve(solver=cp.OSQP, verbose=False)
        else:
            prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    U = u.value
    u2 = np.linalg.norm(U, axis=0)

    plt.subplot(2, 4, i + 1)
    plt.plot(tgrid, U[0, :], label=r"$u_{t,1}$")
    plt.plot(tgrid, U[1, :], label=r"$u_{t,2}$")
    if i == 0:
        plt.ylabel(r"$u_t$")
    plt.title(label)
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.subplot(2, 4, 4 + i + 1)
    plt.plot(tgrid, u2)
    if i == 0:
        plt.ylabel(r"$\|u_t\|_2$")
    plt.xlabel(r"$t$")
    plt.grid(True)

plt.tight_layout()
plt.show()
