import numpy as np, time
import numpy.linalg as la

np.random.seed(1)

def run_demo(n=800, k=30, jitter=1e-8):
    F = np.random.randn(n, k)

    G = np.random.randn(k, k)
    Q = G @ G.T + 0.5*np.eye(k)          # SPD
    Q = Q + jitter*np.eye(k)             # extra safety

    d = np.random.rand(n) + 0.5
    D = np.diag(d)

    mu = np.random.randn(n)
    ones = np.ones(n)

    def portfolio_from_solver(solve_sigma_inv):
        x = solve_sigma_inv(mu)
        y = solve_sigma_inv(ones)
        nu = (ones @ x - 1.0) / (ones @ y)
        return x - nu*y

    t0 = time.time()
    Sigma = F @ Q @ F.T + D
    L = la.cholesky(Sigma)

    def solve_a(v):
        z = la.solve(L, v)
        return la.solve(L.T, z)

    w_a = portfolio_from_solver(solve_a)
    time_a = time.time() - t0

    t1 = time.time()

    Dinv = 1.0 / d
    DinvF = F * Dinv[:, None]            # D^{-1}F
    M = F.T @ DinvF                      # F^T D^{-1} F  (k x k), SPD
    S = la.inv(Q) + M
    S = 0.5*(S + S.T) + jitter*np.eye(k) # symmetrize + jitter
    LS = la.cholesky(S)

    def solve_S(b):
        z = la.solve(LS, b)
        return la.solve(LS.T, z)

    def solve_b(v):
        r = Dinv * v                     # D^{-1} v
        b = F.T @ r                      # F^T D^{-1} v
        y = solve_S(b)                   # (Q^{-1}+M)^{-1} b
        corr = Dinv * (F @ y)            # D^{-1} F y
        return r - corr

    w_b = portfolio_from_solver(solve_b)
    time_b = time.time() - t1

    rel_diff = la.norm(w_a - w_b) / max(1e-12, la.norm(w_a))
    cons = (ones @ w_a, ones @ w_b)

    return time_a, time_b, rel_diff, cons

print(run_demo())