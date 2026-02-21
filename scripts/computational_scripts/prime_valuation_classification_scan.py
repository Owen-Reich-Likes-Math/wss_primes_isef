# Calculates the number of WSS primes in a given universe up to a bound
# Made the heatmap results

# Update notes: made main functionality able to be

import math
import time
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# --------------------- Sieve to get first N primes ---------------------
def nth_prime_upper_bound(n: int) -> int:
    # For n >= 6, p_n < n (ln n + ln ln n)
    if n < 6:
        return 15
    nn = float(n)
    return int(nn * (math.log(nn) + math.log(math.log(nn)))) + 10


def first_n_primes(n: int) -> List[int]:
    if n <= 0:
        return []
    limit = nth_prime_upper_bound(n)
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = b"\x00" * (((limit - start) // step) + 1)
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    # If estimate was too small (rare for small n), extend limit
    while len(primes) < n:
        limit *= 2
        sieve = bytearray(b"\x01") * (limit + 1)
        sieve[0:2] = b"\x00\x00"
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start:limit + 1:step] = b"\x00" * (((limit - start) // step) + 1)
        primes = [i for i, is_p in enumerate(sieve) if is_p]
    return primes[:n]


# --------------------- Legendre symbol (D|p) -----------------------------
def legendre_symbol_mod_p(a: int, p: int) -> int:
    """Return 1, -1, or 0 for (a|p) where p is an odd prime. For p==2 fallback to 0."""
    if p == 2:
        return 0
    a_mod = a % p
    if a_mod == 0:
        return 0
    r = pow(a_mod, (p - 1) // 2, p)
    if r == 1:
        return 1
    if r == p - 1:
        return -1
    return 0  # unreachable for prime p


# --------------------- 2x2 matrix pow mod m, optimized -------------------
def lucas_u_mod(P: int, Q: int, n: int, mod: int) -> int:
    """
    Return U_n(P,Q) mod `mod` where U_0=0, U_1=1, U_n = P U_{n-1} - Q U_{n-2}.
    Uses M = [[P, -Q],[1,0]] and U_n = (M^(n-1))[0,0] for n>=1.
    """
    if n == 0:
        return 0 % mod
    if n == 1:
        return 1 % mod

    # initialize M = (a b; c d) = (P, -Q; 1, 0) mod
    a = P % mod
    b = (-Q) % mod
    c = 1 % mod
    d = 0

    # result = identity matrix
    r00, r01, r10, r11 = 1, 0, 0, 1

    e = n - 1
    while e:
        if e & 1:
            # multiply r = r * M
            t00 = (r00 * a + r01 * c) % mod
            t01 = (r00 * b + r01 * d) % mod
            t10 = (r10 * a + r11 * c) % mod
            t11 = (r10 * b + r11 * d) % mod
            r00, r01, r10, r11 = t00, t01, t10, t11
        # M = M * M
        t00 = (a * a + b * c) % mod
        t01 = (a * b + b * d) % mod
        t10 = (c * a + d * c) % mod
        t11 = (c * b + d * d) % mod
        a, b, c, d = t00, t01, t10, t11
        e >>= 1

    return r00 % mod


# --------------------- Valuation using mod p and mod p^2 -----------------
def valuation_up_to_p2(prime: int, P: int, Q: int) -> int:
    """
    Returns:
      0 if prime does not divide U_n,
      1 if prime divides U_n but not p^2,
      2 if prime divides U_n and p^2 divides U_n (we return 2 to mean ">=2")
    Uses modular exponentiation: compute U_n mod p; if 0 then compute mod p^2.
    """
    p = prime
    if p == 2:
        # treat p=2 specially: choose n=2 as a fallback (safe)
        n = 2
    else:
        D = P * P - 4 * Q
        chi = legendre_symbol_mod_p(D, p)
        n = p if chi == 0 else p - chi

    # compute U_n mod p
    up_mod_p = lucas_u_mod(P, Q, n, p)
    if up_mod_p % p != 0:
        return 0
    # compute U_n mod p^2 to see if p^2 divides it
    pp = p * p
    up_mod_p2 = lucas_u_mod(P, Q, n, pp)
    if up_mod_p2 % pp != 0:
        return 1
    return 2  # means valuation >= 2


# --------------------- Main orchestrator --------------------------------
def compute_prime_valuations_for_lucas(P: int, Q: int, prime_count: int, show_progress: bool = True):
    primes = first_n_primes(prime_count)
    start = time.time()
    counts = [0, 0, 0]  # index: 0 -> v=0, 1 -> v=1, 2 -> v>=2
    vals = [0] * len(primes)

    for i, p in enumerate(primes):
        v = valuation_up_to_p2(p, P, Q)
        vals[i] = v
        counts[v] += 1
        if show_progress and (i % 100000 == 0 and i > 0):
            elapsed = time.time() - start
            print(f"Processed {i:,}/{prime_count:,} primes in {elapsed:.1f}s, rates: {counts}")

    elapsed = time.time() - start
    # print(f"Done: processed {prime_count:,} primes in {elapsed:.2f}s")
    return primes, vals, counts, elapsed


# --------------------- Example usage ------------------------------------

print(
    "Welcome to 'prime_valuation_classification_scan.py', a Python script from Owen Reich's ISEF project on Wall--Sun--Sun primes. ")
print()

print("Functionality 1: scans a single Lucas universe and returns the amount of WSS primes in it. ")
print()
print(
    "Functionality 2: scans Lucas universes in a window and returns the average # of WSS primes, frequency of primes, and best universe. ")
print(
    "Graphs the frequency of primes against the heuristic frequency of 1/p and computes the constant for which O(1/p) is most accurate. ")
print()

response = input("Enter which functionality you'd like [1/2/q]: ")

if response == "1":

    print("Lucas sequence valuation experiment")
    print("U_n = P*U_{n-1} - Q*U_{n-2}")
    print("Examples:")
    print("  Pell:       P=2, Q=-1")
    print("  Fibonacci:  P=1, Q=-1")
    print()

    P = int(input("Enter P: ").strip())
    Q = int(input("Enter Q: ").strip())
    N = int(input("How many primes to process? ").strip())
    # N = 50000000

    primes, valuations, counts, elapsed = compute_prime_valuations_for_lucas(P, Q, N)

    n = len(primes)
    first_count = counts[1]
    higher_count = counts[2]

    higher_primes = [p for p, v in zip(primes, valuations) if v >= 2]

    print()
    print(f"{first_count} primes divide U_n exactly once.")
    print(f"{higher_count} primes divide U_n more than once.")
    print()

    if len(higher_primes) <= 500:
        print("Primes dividing U_n with valuation ≥ 2:")
        print(higher_primes)
    else:
        print("Likely all primes divide. Check though.")

elif response == "2":

    print("Lucas sequence valuation experiment")
    print("U_n = P*U_{n-1} - Q*U_{n-2}")
    print("Examples:")
    print("  Pell:       P=2, Q=-1")
    print("  Fibonacci:  P=1, Q=-1")
    print()

    P_min = int(input("Enter P minimum: ").strip())
    P_max = int(input("Enter P maximum: ").strip())
    Q_min = int(input("Enter Q minimum: ").strip())
    Q_max = int(input("Enter Q maximum: ").strip())
    N = int(input("How many primes to process? ").strip())

    if P_max < P_min:
        P_max, P_min = P_min, P_max
    if Q_max < Q_min:
        Q_max, Q_min = Q_min, Q_max

    a = 0
    a_max = 0
    prime_frequency = Counter()

    degenerate_universes = [(0, -1), (0, 1), (1, 1), (-1, 1), (2, 1), (-2, 1)]
    for p in range(P_min, P_max + 1):
        for q in range(Q_min, Q_max + 1):
            for k in range(0, 4):
                if p * p == k * q:
                    degenerate_universes.append((p, q))

    for p in range(P_min, P_max + 1):
        for q in range(Q_min, Q_max + 1):
            if (p, q) in degenerate_universes:
                # print(f"Universe P = {p}, Q = {q} is degenerate. ")
                continue
            else:
                primes, valuations, counts, elapsed = compute_prime_valuations_for_lucas(p, q, N)

                n = len(primes)
                first_count = counts[1]
                higher_count = counts[2]

                higher_primes = [a for a, v in zip(primes, valuations) if
                                 (v >= 2 and (not (p % a == 0 and q % a == 0)) and a > 2 and a != (p ** 2 - 4 * q))]

                prime_frequency.update(higher_primes)

                # print(f"P={p}, Q={q}: {higher_primes}")
                if q == Q_max:
                    print(f"{p - P_min + 1} / {P_max - P_min + 1} done. ")
                a += len(higher_primes)

                if len(higher_primes) > a_max:
                    a_max = len(higher_primes)
                    P_highest = p
                    Q_highest = q

    if prime_frequency:
        # most_common(1) returns a list like [(prime, count)]
        most_common_prime, total_occurrences = prime_frequency.most_common(1)[0]

        print()
        print(f"MOST FREQUENT PRIME: {most_common_prime}")
        print(f"Total occurrences across all universes: {total_occurrences}")
        print()

        if prime_frequency:
            plt.style.use('dark_background')

            # 1. Prepare data as numpy arrays
            x_data = np.array(sorted(prime_frequency.keys()), dtype=float)
            y_data = np.array(
                [prime_frequency[p] / ((P_max - P_min + 1) * (Q_max - Q_min + 1) - len(degenerate_universes)) for p in
                 x_data], dtype=float)

            # 2. Calculate the optimal constant c
            # We use the formula for a linear regression through the origin (y = c * X)
            # where X = 1/x
            inv_x = 1.0 / x_data
            c = np.sum(y_data * inv_x) / np.sum(inv_x ** 2)

            print(f"Calculated constant c: {c:.4f}")
            print(f"Heuristic model: f(p) = {c:.2f} / p")

            # 3. Set up the plot
            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_data, color='lightblue', label='Observed Frequencies', alpha=0.6)

            # 4. Generate points for the regression line
            # We create a smooth range of x-values for the curve
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
            y_smooth = c / x_smooth

            plt.plot(x_smooth, y_smooth, color='red', linewidth=2,
                     label=f'Best Fit: {c:.2f}/x')

            plt.title("WSS Prime Frequency with $1/x$ Regression")
            plt.xlabel("Prime Size ($p$)")
            plt.ylabel("Total Occurrences")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.show()

    else:
        print("\nNo WSS primes were found in the given range.")

    print(
        f"The average number of WSS primes up to the {N}th prime in Lucas sequences with P in [{P_min}, {P_max}] and Q in [{Q_min}, {Q_max}] is: {a / ((P_max - P_min + 1) * (Q_max - Q_min + 1) - len(degenerate_universes))}")
    print(
        f"The universe with the most WSS primes found was P = {P_highest}, Q = {Q_highest}, with {a_max} WSS primes observed. ")

elif response == "q":
    print("Quitting program. Have a nice day. ")
else:
    print("Invalid input. Have a nice day. ")
