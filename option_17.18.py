import numpy as np
import cvxpy as cp

m = 200
S = np.linspace(0.5, 2.0, m)

r = 1.05
S0 = 1.0

def call(K): return np.maximum(S - K, 0.0)
def put(K):  return np.maximum(K - S, 0.0)

Kc1, Pc1 = 1.1, 0.06
Kc2, Pc2 = 1.2, 0.03
Kp1, Pp1 = 0.8, 0.02
Kp2, Pp2 = 0.7, 0.01

F, C = 0.9, 1.15
phi = np.minimum(C, np.maximum(F, S))

pay_rf    = np.ones(m) * r
pay_stock = S
pay_c1    = call(Kc1)
pay_c2    = call(Kc2)
pay_p1    = put(Kp1)
pay_p2    = put(Kp2)

pi = cp.Variable(m, nonneg=True)

cons = [
    pi @ pay_rf    == 1.0,
    pi @ pay_stock == S0,
    pi @ pay_c1    == Pc1,
    pi @ pay_c2    == Pc2,
    pi @ pay_p1    == Pp1,
    pi @ pay_p2    == Pp2,
]

prob_min = cp.Problem(cp.Minimize(pi @ phi), cons)
prob_min.solve()   # if CLARABEL fails, do prob_min.solve(solver=cp.ECOS) or cp.SCS
lower = prob_min.value

prob_max = cp.Problem(cp.Maximize(pi @ phi), cons)
prob_max.solve()
upper = prob_max.value

print(f"no-arb collar price in [{lower:.3f}, {upper:.3f}]")
