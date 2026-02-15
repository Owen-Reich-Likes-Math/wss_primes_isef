# Acts as if 9600p^2 is asymptotically correct and heuristically predicts # of WSS primes
# Not true and comes from worse bound 9600p^2 (which comes from worse constant optimization from Bugeaud--Laurent)
# Unused and mainly just a toy script

import math
from math import isqrt
import matplotlib.pyplot as plt

# ---------------------------
# Helper functions
# ---------------------------
def primes_upto(n):
    sieve = [True]*(n+1)
    sieve[0:2] = [False, False]
    for i in range(2, isqrt(n)+1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [p for p in range(2, n+1) if sieve[p]]

def legendre_symbol(a: int, p: int) -> int:
    """Return (a|p): 1 if quadratic residue mod p, -1 if nonresidue, 0 if p | a."""
    a = a % p
    if a == 0:
        return 0
    ls = pow(a, (p-1)//2, p)
    if ls == 1:
        return 1
    elif ls == p-1:
        return -1
    else:
        return 0

def mat_mult(A, B, mod):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod, (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod, (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]
    ]

def mat_pow(M, e, mod):
    result = [[1,0],[0,1]]
    base = [[M[0][0] % mod, M[0][1] % mod],[M[1][0] % mod, M[1][1] % mod]]
    while e > 0:
        if e & 1:
            result = mat_mult(result, base, mod)
        base = mat_mult(base, base, mod)
        e >>= 1
    return result

def U_n_mod(P, Q, n, m):
    """
    Lucas U_n modulo m with U_0=0, U_1=1 and recurrence U_n = P*U_{n-1} - Q*U_{n-2}.
    Uses matrix exponentiation: M = [[P,-Q],[1,0]], U_n is first entry of M^(n-1) * [1;0].
    """
    if n == 0:
        return 0 % m
    if n == 1:
        return 1 % m
    M = [[P % m, (-Q) % m], [1 % m, 0]]
    Mn_1 = mat_pow(M, n-1, m)
    Un = (Mn_1[0][0] * 1 + Mn_1[0][1] * 0) % m
    return Un

# ---------------------------
# Interactive input
# ---------------------------
print("Lucas universe expectation vs actual (primes up to 200).")
P = int(input("Enter integer P (e.g. 1 for Fibonacci-like): ").strip())
Q = int(input("Enter integer Q (e.g. -1 for Fibonacci-like): ").strip())

# ---------------------------
# Constants and precompute primes
# ---------------------------
MAX_PRIME = 200
odd_primes = [p for p in primes_upto(MAX_PRIME) if p % 2 == 1]

D = P*P - 4*Q
print(f"\nParameters: P={P}, Q={Q}, D = {D}\n(Primes dividing D will be skipped.)")

# compute phi (may be complex) and its modulus for bottom heuristic
if D >= 0:
    sqrtD = math.sqrt(D)
else:
    sqrtD = complex(0, math.sqrt(-D))
phi = (P + sqrtD) / 2
phi_abs = abs(phi) if isinstance(phi, complex) else float(phi)

if phi_abs <= 1:
    print("Warning: |phi| <= 1. Bottom heuristic uses ln(|phi|) and may be <= 0 (not very meaningful).")
if isinstance(phi, complex):
    print("Note: D < 0; phi is complex. Using ln(|phi|) (modulus) for bottom heuristic.")

# ---------------------------
# Sweep primes and compute cumulative series
# ---------------------------
primes_x = []                 # x-axis: prime values considered
cumul_top = []                # cumulative sum of top heuristic
cumul_bottom = []             # cumulative sum of bottom heuristic
cumul_actual = []             # cumulative integer count of primes with p^3 | U_n

top_sum = 0.0
bottom_sum = 0.0
actual_count = 0

for p in odd_primes:
    if D % p == 0:
        # skip primes dividing D (per user's spec)
        continue
    leg = legendre_symbol(D, p)
    if leg == 0:
        continue
    f = 1 if leg == 1 else 2         # residue -> f=1, nonresidue -> f=2
    n = p - leg                      # n = p - (D/p): p-1 if leg=1, p+1 if leg=-1

    # compute U_n mod p^3 and test divisibility
    mod = p**3
    Un_mod_p3 = U_n_mod(P, Q, n, mod)
    divisible_p3 = (Un_mod_p3 % mod == 0)
    if divisible_p3:
        actual_count += 1

    # top heuristic term and bottom heuristic term
    top_term = 9600.0 / (p * (f**2))
    top_sum += top_term

    if phi_abs > 0 and p > 1:
        bottom_term = math.log(phi_abs) / (p * math.log(p))
    else:
        bottom_term = 0.0
    bottom_sum += bottom_term

    # record
    primes_x.append(p)
    cumul_top.append(top_sum)
    cumul_bottom.append(bottom_sum)
    cumul_actual.append(actual_count)

# ---------------------------
# Plotting
# ---------------------------
plt.figure(figsize=(10,6))
# plt.plot(primes_x, cumul_top, marker='o', label='Cumulative Top Expectation (9600/(p f^2))')
plt.plot(primes_x, cumul_bottom, marker='s', label='Cumulative Bottom Expectation (ln(|phi|)/(p ln p))')
plt.step(primes_x, cumul_actual, where='post', label='Cumulative Actual Count (p^3 | U_n)')

plt.xlabel('Prime p (odd primes ≤ 200, excluding p | D)')
plt.ylabel('Cumulative value / count')
plt.title(f"Lucas universe (P={P}, Q={Q}) — Expectations vs Actual (primes ≤ {MAX_PRIME})")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show
outname = f"lucas_expectations_P{P}_Q{Q}.png"
plt.savefig(outname, dpi=300)
print(f"Plot saved to: {outname}")
plt.show()

# ---------------------------
# Print a short numeric summary
# ---------------------------
print("\nSummary:")
print(f"Number of primes considered: {len(primes_x)}")
print(f"Top expectation total (sum 9600/(p f^2)): {top_sum:.6f}")
print(f"Bottom expectation total (sum ln(|phi|)/(p ln p)): {bottom_sum:.6f}")
print(f"Actual count of primes with p^3 | U_n: {actual_count}")
print("\nDone.")

