# Finds universes with high divisibility - 47, 11, for example

import argparse
import time
import numpy as np
from sympy import primerange, legendre_symbol


def matrix_pow(A, n, m):
    res = np.array([[1, 0], [0, 1]], dtype=object)
    while n > 0:
        if n % 2 == 1:
            res = (res @ A) % m
        A = (A @ A) % m
        n //= 2
    return res


def get_valuation(P, Q, p, max_k=5):
    D = P * P - 4 * Q
    if D % p == 0:
        return 0
    n = p - legendre_symbol(D, p)
    T = np.array([[P, -Q], [1, 0]], dtype=object)
    val = 0
    for k in range(1, max_k + 1):
        m = p ** k
        res_matrix = matrix_pow(T, n, m)
        if int(res_matrix[1, 0]) % m == 0:
            val = k
        else:
            break
    return val


def scan_universes(b, prime_limit, max_k=5, top_n=1, verbose=True):
    primes = list(primerange(3, prime_limit))
    P_range = range(1, b + 1)
    Q_range = range(-b // 2, (b // 2) + 1)

    universe_list = [(P, Q) for P in P_range for Q in Q_range if P * P - 4 * Q != 0]
    total_universes = len(universe_list)

    if verbose:
        print(f"Primes: {len(primes)} (3..{prime_limit-1})")
        print(f"Universes to scan: {total_universes} (P in 1..{b}, Q in -{b//2}..{b//2})")
        print(f"max_k (valuation depth) = {max_k}")
        print("Starting scan...")

    start = time.perf_counter()

    # We'll keep a small leaderboard of best universes (value, complexity, P, Q)
    leaderboard = []  # list of tuples (value, complexity, P, Q, delta_array)

    for idx, (P, Q) in enumerate(universe_list, start=1):
        vals = [get_valuation(P, Q, p, max_k=max_k) for p in primes]
        delta = np.maximum.accumulate(vals).astype(int)
        cur_max = int(np.max(delta)) if delta.size > 0 else 0
        complexity = abs(P) + abs(Q)

        # maintain leaderboard of top_n entries sorted by (value desc, complexity asc)
        entry = (cur_max, complexity, P, Q, delta)
        if len(leaderboard) < top_n:
            leaderboard.append(entry)
            leaderboard.sort(key=lambda x: (-x[0], x[1]))
        else:
            # compare to worst in leaderboard (last entry)
            worst = leaderboard[-1]
            if (cur_max > worst[0]) or (cur_max == worst[0] and complexity < worst[1]):
                leaderboard[-1] = entry
                leaderboard.sort(key=lambda x: (-x[0], x[1]))

        # occasional progress print
        if verbose and (idx % 1000 == 0 or idx == total_universes):
            elapsed = time.perf_counter() - start
            print(f"  scanned {idx}/{total_universes} universes — time {elapsed:.1f}s")

    elapsed_total = time.perf_counter() - start

    # Output results
    print("\nScan complete.")
    print(f"Time elapsed: {elapsed_total:.2f} seconds")
    print(f"Universes checked: {total_universes}")
    print("Top universes found (value, |P|+|Q|, P, Q):")
    for rank, (val, comp, P, Q, delta) in enumerate(leaderboard, start=1):
        print(f" {rank}. max={val:2d}, complexity={comp:3d}, P={P:3d}, Q={Q:4d}")

    # Return leaderboard for programmatic use
    return leaderboard, primes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the most 'explosive' Lucas universe in a b-range.")
    parser.add_argument("-b", "--b", type=int, default=100,
                        help="Even integer b (P in 1..b, Q in -b//2..b//2). Default 100 (~10000 universes).")
    parser.add_argument("-p", "--prime_limit", type=int, default=500,
                        help="Upper bound (exclusive) for primes to check. Default 500.")
    parser.add_argument("--max_k", type=int, default=5,
                        help="Max valuation exponent to check (default 5 for p^1..p^5).")
    parser.add_argument("--top_n", type=int, default=1,
                        help="Print the top N universes (default 1).")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output (only final result).")
    args = parser.parse_args()

    lb, primes = scan_universes(args.b, args.prime_limit, max_k=args.max_k, top_n=args.top_n, verbose=not args.quiet)

