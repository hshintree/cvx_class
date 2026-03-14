import numpy as np
import matplotlib.pyplot as plt

def residuals(A, b, c, x, nu):
    rp = A @ x - b
    rd = c - 1 / x + A.T @ nu
    return rp, rd

def residual_norm(A, b, c, x, nu):
    rp, rd = residuals(A, b, c, x, nu)
    return np.linalg.norm(np.hstack([rp, rd]))

def newton_step_block_elimination(A, b, c, x, nu):
    rp, rd = residuals(A, b, c, x, nu)
    Hinv = x**2
    AHinv = A * Hinv
    S = AHinv @ A.T
    rhs = rp - AHinv @ rd
    dnu = np.linalg.solve(S, rhs)
    dx = -Hinv * (rd + A.T @ dnu)
    return dx, dnu

def newton_step_kkt(A, b, c, x, nu):
    m, n = A.shape
    rp, rd = residuals(A, b, c, x, nu)
    H = np.diag(1 / x**2)
    KKT = np.block([
        [H, A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = -np.hstack([rd, rp])
    sol = np.linalg.solve(KKT, rhs)
    dx = sol[:n]
    dnu = sol[n:]
    return dx, dnu

def infeasible_start_newton_lp_centering(A, b, c, x0, alpha=0.01, beta=0.5, tol=1e-6, max_iter=50, check_kkt=False):
    x = x0.astype(float).copy()
    m, n = A.shape
    nu = np.zeros(m)
    hist = [residual_norm(A, b, c, x, nu)]

    for k in range(max_iter):
        dx, dnu = newton_step_block_elimination(A, b, c, x, nu)

        if check_kkt:
            dx_kkt, dnu_kkt = newton_step_kkt(A, b, c, x, nu)
            print("step difference:", np.linalg.norm(dx - dx_kkt), np.linalg.norm(dnu - dnu_kkt))

        if hist[-1] <= tol:
            break

        t = 1.0
        neg = dx < 0
        if np.any(neg):
            t = min(t, 0.99 * np.min(-x[neg] / dx[neg]))

        r0 = hist[-1]
        while True:
            x_trial = x + t * dx
            nu_trial = nu + t * dnu
            if np.min(x_trial) <= 0:
                t *= beta
                continue
            r_trial = residual_norm(A, b, c, x_trial, nu_trial)
            if r_trial <= (1 - alpha * t) * r0:
                break
            t *= beta

        x = x_trial
        nu = nu_trial
        hist.append(r_trial)

    return x, nu, len(hist) - 1, np.array(hist)

def generate_feasible_bounded_instance(m=30, n=100, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    while np.linalg.matrix_rank(A) < m:
        A = rng.standard_normal((m, n))
    x_feas = rng.uniform(0.5, 2.0, size=n)
    b = A @ x_feas
    c = rng.uniform(0.5, 2.0, size=n)
    row = rng.integers(0, m)
    A[row, :] = rng.uniform(0.2, 1.5, size=n)
    b = A @ x_feas
    x0 = rng.uniform(0.5, 2.5, size=n)
    return A, b, c, x0, x_feas

def generate_infeasible_instance(m=30, n=100, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    while np.linalg.matrix_rank(A) < m:
        A = rng.standard_normal((m, n))
    x_feas = rng.uniform(0.5, 2.0, size=n)
    b = A @ x_feas
    b = b.copy()
    b[0] += 100.0
    c = rng.uniform(0.5, 2.0, size=n)
    x0 = rng.uniform(0.5, 2.5, size=n)
    return A, b, c, x0

def generate_unbounded_instance(m=30, n=100, seed=2):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    while np.linalg.matrix_rank(A) < m:
        A = rng.standard_normal((m, n))
    x_feas = rng.uniform(0.5, 2.0, size=n)
    b = A @ x_feas
    c = -rng.uniform(0.5, 2.0, size=n)
    x0 = rng.uniform(0.5, 2.5, size=n)
    return A, b, c, x0

def centering_objective(c, x):
    return c @ x - np.sum(np.log(x))

A, b, c, x0, x_feas = generate_feasible_bounded_instance(m=25, n=80, seed=3)
x_star, nu_star, steps, hist = infeasible_start_newton_lp_centering(
    A, b, c, x0, alpha=0.01, beta=0.5, tol=1e-6, max_iter=50, check_kkt=True
)

print("feasible instance")
print("steps:", steps)
print("final residual norm:", hist[-1])
print("objective value:", centering_objective(c, x_star))
print("equality residual:", np.linalg.norm(A @ x_star - b))
print("min x:", np.min(x_star))

plt.figure(figsize=(7, 4))
plt.semilogy(range(len(hist)), hist, marker="o")
plt.xlabel("iteration k")
plt.ylabel("||(r_p, r_d)||_2")
plt.title("Infeasible-start Newton method")
plt.grid(True)
plt.tight_layout()
plt.show()

alphas = [0.01, 0.1, 0.25]
betas = [0.3, 0.5, 0.8]

print("\nparameter sweep")
for alpha in alphas:
    for beta in betas:
        x_tmp, nu_tmp, steps_tmp, hist_tmp = infeasible_start_newton_lp_centering(
            A, b, c, x0, alpha=alpha, beta=beta, tol=1e-6, max_iter=50, check_kkt=False
        )
        print(f"alpha={alpha:.2f}, beta={beta:.2f}, steps={steps_tmp}, final_residual={hist_tmp[-1]:.3e}")

A_inf, b_inf, c_inf, x0_inf = generate_infeasible_instance(m=25, n=80, seed=4)
x_inf, nu_inf, steps_inf, hist_inf = infeasible_start_newton_lp_centering(
    A_inf, b_inf, c_inf, x0_inf, alpha=0.01, beta=0.5, tol=1e-6, max_iter=50
)
print("\ninfeasible-like instance")
print("steps:", steps_inf)
print("final residual norm:", hist_inf[-1])

A_unb, b_unb, c_unb, x0_unb = generate_unbounded_instance(m=25, n=80, seed=5)
x_unb, nu_unb, steps_unb, hist_unb = infeasible_start_newton_lp_centering(
    A_unb, b_unb, c_unb, x0_unb, alpha=0.01, beta=0.5, tol=1e-6, max_iter=50
)
print("\nunbounded-below-like instance")
print("steps:", steps_unb)
print("final residual norm:", hist_unb[-1])