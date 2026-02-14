import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

PLOT_FIGURES = True
N = 96

fig_size = (14, 3)
xtick_vals = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
xtick_labels = (
    "0:00","2:00am","4:00am","6:00am","8:00am","10:00am",
    "12:00pm","2:00pm","4:00pm","6:00pm","8:00pm","10:00pm","12:00am"
)

partial_peak_start = 34
peak_start = 48
peak_end = 72
partial_peak_end = 86

off_peak_inds = np.concatenate([np.arange(partial_peak_start), np.arange(partial_peak_end, N)])
partial_peak_inds = np.concatenate([np.arange(partial_peak_start, peak_start), np.arange(peak_end, partial_peak_end)])
peak_inds = np.arange(peak_start, peak_end)

off_peak_buy = 0.14
partial_peak_buy = 0.25
peak_buy = 0.45

off_peak_sell = (1 - 0.20) * off_peak_buy
partial_peak_sell = (1 - 0.12) * partial_peak_buy
peak_sell = (1 - 0.11) * peak_buy

R_buy = np.zeros(N)
R_buy[off_peak_inds] = off_peak_buy
R_buy[partial_peak_inds] = partial_peak_buy
R_buy[peak_inds] = peak_buy

R_sell = np.zeros(N)
R_sell[off_peak_inds] = off_peak_sell
R_sell[partial_peak_inds] = partial_peak_sell
R_sell[peak_inds] = peak_sell

shift = N / 2
p_pv = np.power(np.cos((np.arange(N) - shift) * 2 * np.pi / N), 2)
p_pv = np.maximum(p_pv * 35, 0)
p_pv[: int(shift / 2)] = 0
p_pv[-int(shift / 2) :] = 0

points = np.array([
    [0, 7],[10, 8],[20, 10],[28, 15],[36, 21],[45, 23],
    [52, 21],[56, 18],[60, 22.5],[66, 24.3],[70, 25],
    [73, 24],[83, 19],[95, 7],
], dtype=float)

p_fit = cp.Variable(N)
obj_val = 0
constr = [p_fit[0] == p_fit[-1]]
constr += [(p_fit[1] - p_fit[0]) == (p_fit[-1] - p_fit[-2])]

for pt in points:
    obj_val += cp.square(p_fit[int(pt[0])] - pt[1])

for i in range(N):
    obj_val += 100 * cp.square(p_fit[(i + 1) % N] - 2 * p_fit[i] + p_fit[(i - 1) % N])

prob_fit = cp.Problem(cp.Minimize(obj_val), constr)
prob_fit.solve()
p_ld = p_fit.value

D = 10
C = 8
Q = 27

p_buy  = cp.Variable(N, nonneg=True)
p_sell = cp.Variable(N, nonneg=True)
p_grid = p_buy - p_sell

p_batt = cp.Variable(N)
q = cp.Variable(N)

power_balance = (p_grid + p_batt + p_pv == p_ld)

cons = []
cons += [power_balance]
cons += [q[1:] == q[:-1] - 0.25 * p_batt[:-1]]
cons += [q[0]  == q[-1]  - 0.25 * p_batt[-1]]
cons += [q >= 0, q <= Q]
cons += [p_batt >= -C, p_batt <= D]

cost = 0.25 * (R_buy @ p_buy - R_sell @ p_sell)
prob = cp.Problem(cp.Minimize(cost), cons)
prob.solve()

print("status:", prob.status)
print("optimal grid cost ($):", prob.value)

nu = power_balance.dual_value
LMP = 4 * nu
if np.mean(LMP) < 0:
    LMP = -LMP
    nu = -nu

p_buy_v  = p_buy.value
p_sell_v = p_sell.value
p_grid_v = p_grid.value
p_batt_v = p_batt.value
q_v      = q.value

i = np.arange(N)

plt.figure(figsize=(14, 9))

plt.subplot(5, 1, 1)
plt.plot(i, p_grid_v, label="p_grid")
plt.plot(i, p_buy_v,  label="p_buy",  linestyle="--")
plt.plot(i, -p_sell_v, label="-p_sell", linestyle="--")
plt.ylabel("kW")
plt.title("Grid power (net, buy, sell)")
plt.grid(True)
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(i, p_ld)
plt.ylabel("p_load (kW)")
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(i, p_pv)
plt.ylabel("p_pv (kW)")
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(i, p_batt_v)
plt.ylabel("p_batt (kW)")
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(i, q_v)
plt.ylabel("q (kWh)")
plt.xlabel("period i")
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=fig_size)
plt.plot(i, LMP, label="LMP ($/kWh)")
plt.plot(i, R_buy, label="R_buy ($/kWh)")
plt.plot(i, R_sell, label="R_sell ($/kWh)")
plt.legend()
plt.ylabel("Price ($/kWh)")
plt.title("LMP vs Grid Buy/Sell Prices", fontsize=16)
plt.xticks(xtick_vals, xtick_labels)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- (c) Payments ----
# Problem statement uses v^T p for payments, with v the dual variable of power balance.
# Here nu is the dual for (p_grid + p_batt + p_pv - p_ld == 0).
# Using their convention, use v = nu. The kWh conversion factor is 0.25 hours per interval.
v = nu

pay_load = 0.25 * (v @ p_ld)
pay_pv   = 0.25 * (v @ p_pv)
pay_batt = 0.25 * (v @ p_batt_v)
pay_grid = 0.25 * (v @ p_grid_v)

print("\nLMP payments (in $):")
print("load pays:     ", pay_load)
print("PV is paid:    ", pay_pv)
print("battery is paid:", pay_batt)
print("grid is paid:  ", pay_grid)

balance_resid = pay_load - (pay_pv + pay_batt + pay_grid)
print("\npayment balance residual (should be ~0):", balance_resid)
