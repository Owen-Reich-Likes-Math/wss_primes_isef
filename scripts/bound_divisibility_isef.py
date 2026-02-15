import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange, legendre_symbol

# ---- User input ----
print("--- Lucas Universe Overlay (skip periodic + P^2 = kQ for k=0..3; omit zeros) ---")
try:
    P_min = int(input("Enter P_min (e.g. 1): "))
    P_max = int(input("Enter P_max (>= P_min, e.g. 5): "))
    Q_abs_max = int(input("Enter max |Q| (positive integer, e.g. 5): "))
    prime_bound = int(input("Enter Prime Bound (e.g. 200): "))
except Exception:
    print("Invalid input — using defaults: P_min=1, P_max=5, Q_abs_max=3, prime_bound=200")
    P_min, P_max, Q_abs_max, prime_bound = 1, 5, 3, 200

if P_max < P_min:
    P_min, P_max = P_max, P_min

# ---- Explicit periodic universes to skip ----
periodic_skip_set = {
    (0, -1), (0, 1),
    (1, 1), (-1, 1),
    (2, 1), (-2, 1)
}

# ---- Helper functions ----
def matrix_pow(A, n, m):
    """Compute A^n (2x2 matrix) modulo m using binary exponentiation."""
    res = np.array([[1, 0], [0, 1]], dtype=object)
    base = A.copy()
    while n > 0:
        if n & 1:
            res = (res @ base) % m
        base = (base @ base) % m
        n >>= 1
    return res

def get_raw_valuation(P, Q, p, max_k=16):
    """
    Returns:
      - None if prime divides Q (skip),
      - 0 if p does not divide the target U_n,
      - k>0 integer if p^k || U_n.
    """
    D = P * P - 4 * Q

    if Q % p == 0 and P % p != 0:
        return None

    if D % p == 0:
        n = p
    else:
        n = p - legendre_symbol(D, p)

    T = np.array([[P, -Q], [1, 0]], dtype=object)

    # quick test modulo p
    res_mod_p = matrix_pow(T, n, p)
    u_n_mod_p = int(res_mod_p[1, 0] % p)
    if u_n_mod_p != 0:
        return 0

    # p divides U_n; find exact p-adic valuation up to max_k
    val = 0
    for k in range(1, max_k + 1):
        m = p ** k
        res = matrix_pow(T, n, m)
        u_n_mod_m = int(res[1, 0] % m)
        if u_n_mod_m == 0:
            val = k
        else:
            break
    return val

# ---- Build requested universes ----
requested_universes = [
    (P, Q)
    for P in range(P_min, P_max + 1)
    for Q in range(-Q_abs_max, Q_abs_max + 1)
    if Q != 0
]

if not requested_universes:
    raise SystemExit("No universes to scan (maybe Q_abs_max was 0).")

# ---- Additional filter: skip P^2 = k*Q for k in {0,1,2,3} ----
def violates_k_relation(P, Q):
    for k in range(0, 4):
        # integer equality test
        if P * P == k * Q:
            return True
    return False

# Apply both filters (explicit periodic set + k-relation)
filtered_universes = [
    (P, Q) for (P, Q) in requested_universes
    if (P, Q) not in periodic_skip_set and not violates_k_relation(P, Q)
]

skipped = [u for u in requested_universes if u not in filtered_universes]

print(f"Requested universes: {len(requested_universes)}; skipping {len(skipped)} universes (periodic or P^2=kQ).")
if skipped:
    print("Skipped examples (up to 40):", skipped[:40])

# ---- Prime list ----
primes = list(primerange(3, prime_bound + 1))
if not primes:
    raise SystemExit("No primes found in the given bound.")

# Shared theoretical bound line
bound_multiplier = 2400
bound_line = np.array([bound_multiplier * p for p in primes])

# ---- Compute valuations, include only v>0 points (skip zeros) ----
all_data = []
print(f"Scanning {len(filtered_universes)} universes; primes up to {prime_bound}...")
for (P, Q) in filtered_universes:
    xs = []
    ys = []
    anomalies = []
    for p in primes:
        v = get_raw_valuation(P, Q, p)
        if v is None:
            continue
        # include only strictly positive valuations (skip zeros)
        if v > 0:
            xs.append(p)
            ys.append(v)
            if v >= 2:
                anomalies.append((p, v))
    all_data.append({"P": P, "Q": Q, "primes": np.array(xs), "vals": np.array(ys), "anoms": anomalies})

# ---- Identify Fibonacci and the max-universe (by observed max valuation) ----
fib_pair = (1, -1)  # Fibonacci corresponds to (P=1, Q=-1)
fib_index = None
for i, data in enumerate(all_data):
    if (data["P"], data["Q"]) == fib_pair:
        fib_index = i
        break

# find universe with largest observed valuation (only among those with any positive vals)
max_val = -1
max_idx = None
for i, data in enumerate(all_data):
    if data["vals"].size > 0:
        local_max = int(data["vals"].max())
        if local_max > max_val:
            max_val = local_max
            max_idx = i

# ---- Plot overlay ----
plt.figure(figsize=(14, 8))
ax = plt.gca()
cmap = plt.get_cmap('tab20')

for i, data in enumerate(all_data):
    xs = data["primes"]
    ys = data["vals"]
    P, Q = data["P"], data["Q"]

    if xs.size == 0:
        continue  # nothing to plot for this universe (no positive valuations)

    color = cmap(i % cmap.N)
    is_fib = (i == fib_index)
    is_max = (i == max_idx)

    if is_fib or is_max:
        lw = 1.8 if is_max else 1.6
        alpha = 0.95
        marker_size = 4
        label = ("Fibonacci (1,-1)" if is_fib else f"Highest max v: ({P},{Q}) v={int(ys.max())}")
        ax.plot(xs, ys, marker='o', markersize=marker_size, linestyle='-', linewidth=lw,
                alpha=alpha, color=color, label=label)
    else:
        ax.plot(xs, ys, marker='o', markersize=2.5, linestyle='-', linewidth=0.6,
                alpha=0.45, color=color, label='_nolegend_')

# plot theoretical bound across primes
ax.plot(primes, bound_line, linestyle='--', linewidth=1.0, color='k', alpha=0.8)

# mark anomalies (v>=2) red, no legend entry
anom_ps = []
anom_vs = []
for data in all_data:
    for (p, v) in data["anoms"]:
        anom_ps.append(p)
        anom_vs.append(v)
if anom_ps:
    ax.scatter(anom_ps, anom_vs, color='red', s=30, zorder=10)

# Only add legend if there are labeled lines (Fibonacci or a max universe)
handles, labels = ax.get_legend_handles_labels()
if labels:
    ax.legend(loc='upper left', fontsize='small')

ax.set_yscale('log')
ax.set_xlabel("Prime p")
ax.set_ylabel(r"Valuation $\nu_p(U_{rank})$ (only v > 0 shown)")
ax.set_title(f"Overlay: Lucas universes P in [{P_min},{P_max}], |Q| ≤ {Q_abs_max} (skipped periodic & P^2=kQ)")
ax.grid(True, which='both', linestyle=':', alpha=0.3)
plt.tight_layout()
plt.show()
