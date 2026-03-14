import cvxpy as cp
import numpy as np

A = 10000.0
alpha = np.array([1e-5, 1e-2, 1e-2, 1e-2], dtype=float)
M = np.array([0.1, 5.0, 10.0, 10.0], dtype=float)
n = 4

a   = cp.Variable(n, pos=True)      # gains
Sin = cp.Variable(pos=True)         # chosen input level (will become S_in,max)
t   = cp.Variable(n, pos=True)      # t[i] = N_i^2

constraints = []

constraints += [cp.prod(a) == A]

for i in range(n):
    constraints += [Sin * cp.prod(a[:i+1]) <= M[i]]

eps = 1e-30
t_prev = eps
for i in range(n):
    constraints += [t[i] >= (a[i]**2) * (t_prev + alpha[i]**2)]
    t_prev = t[i]

D = A * Sin * t[n-1]**(-0.5)
prob = cp.Problem(cp.Maximize(D), constraints)

prob.solve(gp=True)

print("status:", prob.status)
print("optimal dynamic range D*:", D.value)
print("optimal gains a*:", a.value)
print("optimal Sin (S_in,max):", Sin.value)
print("optimal N_out:", np.sqrt(t.value[-1]))
print("optimal S_max:", A * Sin.value)