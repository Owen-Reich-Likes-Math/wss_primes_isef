# Unused script, checks "wss composites"

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from typing import Tuple

# ---------- Miller-Rabin probable-prime ----------
_small_primes = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
    139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199
]


def is_probable_prime(n: int, rounds: int = 10) -> bool:
    """Return True if n is probably prime (Miller-Rabin), False if composite.
    We do small-prime trial division first. For very small n result is exact.
    """
    if n < 2:
        return False
    for p in _small_primes:
        if n % p == 0:
            return n == p
    # write n-1 as d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # deterministic bases for 64-bit integers
    if n < 2 ** 64:
        bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    else:
        bases = [random.randrange(2, n - 1) for _ in range(rounds)]

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in bases:
        if a % n == 0:
            continue
        if not check(a):
            return False
    return True


# ---------- Legendre-like symbol over 5 ----------
def legendre_over_5(n: int) -> int:
    r = n % 5
    if r == 0:
        return 0
    return 1 if r in (1, 4) else -1


# ---------- Iterative fast-doubling Fibonacci modulo ----------
def fib_fast_doubling_mod(n: int, mod: int) -> Tuple[int, int]:
    """Iterative fast-doubling: returns (F_n mod mod, F_{n+1} mod mod)."""
    if mod == 1:
        return 0, 0
    n = int(n)
    a, b = 0 % mod, 1 % mod
    if n == 0:
        return a, b
    bits = bin(n)[3:]  # skip '0b' and leading 1
    for bit in bits:
        # c = F_{2k}, d = F_{2k+1}
        c = (a * ((2 * b - a) % mod)) % mod
        d = (a * a + b * b) % mod
        if bit == '0':
            a, b = c, d
        else:
            a, b = d, (c + d) % mod
    return a, b


def compute_wss_for_m(m: int) -> Tuple[int, bool, int]:
    """Compute q and verification for composite m.
    Returns (q, is_wss_verified, idx_used) where:
      - q = floor(F_idx / m) mod m
      - is_wss_verified = True iff F_idx ≡ 0 (mod m^2)
      - idx_used = m - (m/5) using legendre_over_5
    """
    chi = legendre_over_5(m)
    idx = m - chi
    mod = m * m
    f_mod, _ = fib_fast_doubling_mod(idx, mod)
    q = (f_mod // m) % m
    is_wss = (f_mod == 0)
    return int(q), bool(is_wss), int(idx)


# ---------- Generators for the two families ----------
def generate_family_A(x: int):
    """Yield tuples (k, exp, m) for m = 2^{4k+3} + 1, k = 0..x."""
    for k in range(0, x + 1):
        exp = 4 * k + 3
        m = (1 << exp) + 1  # 2^exp + 1
        yield k, exp, m


def generate_family_B(y: int):
    """Yield tuples (n, m) for m = 2^n - 1 where n<=y and n mod 12 in {2,3,6,10,11}."""
    residues = {2, 3, 6, 10, 11}
    for n in range(2, y + 1):
        if (n % 12) in residues:
            m = (1 << n) - 1
            yield n, m


# ---------- CLI & interactive prompts ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Check WSS composites on two special families of numbers (interactive prompts for x,y).")
    # make x,y optional; if not provided, script will prompt for them
    p.add_argument("--x", type=int, default=None,
                   help="Upper bound for k in family A (m = 2^{4k+3}+1). If omitted, script will prompt.")
    p.add_argument("--y", type=int, default=None,
                   help="Upper bound for n in family B (m = 2^n - 1). If omitted, script will prompt.")
    p.add_argument("--max-bitlen", type=int, default=10000,
                   help="Maximum bit-length of m to process (skip larger). Default 10000")
    p.add_argument("--mr-rounds", type=int, default=10,
                   help="Miller-Rabin random rounds for large integers (default 10)")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional CSV filename to write verified WSS composites (columns: family, param, exp/n, m, idx, is_wss)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def prompt_for_nonneg_int(prompt_text: str) -> int:
    """Prompt repeatedly until user supplies a non-negative integer or 'q' to quit."""
    while True:
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if raw.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        if raw == "":
            print("Please enter a non-negative integer (or q to quit).")
            continue
        try:
            v = int(raw)
            if v < 0:
                print("Enter a non-negative integer.")
                continue
            return v
        except ValueError:
            print("Not an integer — try again.")


def main():
    args = parse_args()
    # Interactive prompts if x/y not provided
    if args.x is None:
        x = prompt_for_nonneg_int("Enter x (upper bound for k in family A; k runs 0..x): ")
    else:
        x = args.x
    if args.y is None:
        y = prompt_for_nonneg_int("Enter y (upper bound for n in family B; n runs 2..y): ")
    else:
        y = args.y

    max_bitlen = args.max_bitlen
    mr_rounds = args.mr_rounds
    csvfile = args.csv
    verbose = args.verbose

    if csvfile:
        outdir = os.path.dirname(csvfile)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        csv_fh = open(csvfile, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["family", "param_k_or_n", "exp_or_n", "m", "idx", "is_wss"])
    else:
        csv_fh = None
        csv_writer = None

    print(f"Scanning family A: m = 2^(4k+3)+1 for k=0..{x}")
    verified_A = []
    skipped_A = []
    start = time.perf_counter()
    for k, exp, m in generate_family_A(x):
        bl = m.bit_length()
        if bl > max_bitlen:
            skipped_A.append((k, exp, m, "bitlen>max"))
            if verbose:
                print(f"[A k={k}] Skipped m (bitlen {bl} > {max_bitlen})")
            continue
        # primality test
        if is_probable_prime(m, rounds=mr_rounds):
            skipped_A.append((k, exp, m, "probable-prime"))
            if verbose:
                print(f"[A k={k}] m appears prime (skipping)")
            continue
        # m appears composite → compute WSS test
        q, is_wss, idx = compute_wss_for_m(m)
        if is_wss:
            verified_A.append((k, exp, m, idx))
            print(f"[A k={k}] VERIFIED WSS composite: m={m} (2^{exp}+1), idx={idx}")
            if csv_writer:
                csv_writer.writerow(["A", k, exp, str(m), idx, "1"])
        else:
            skipped_A.append((k, exp, m, f"q={q} (not wss)"))
            if verbose:
                print(f"[A k={k}] m composite but NOT WSS: q={q}")
            if csv_writer:
                csv_writer.writerow(["A", k, exp, str(m), idx, "0"])

    tA = time.perf_counter() - start
    print(f"Family A done. Verified {len(verified_A)} entries; scanned k=0..{x} in {tA:.2f}s")

    print(f"\nScanning family B: m = 2^n - 1 for n <= {y}, n mod 12 in {{2,3,6,10,11}}")
    verified_B = []
    skipped_B = []
    start = time.perf_counter()
    for n, m in generate_family_B(y):
        bl = m.bit_length()
        if bl > max_bitlen:
            skipped_B.append((n, m, "bitlen>max"))
            if verbose:
                print(f"[B n={n}] Skipped m (bitlen {bl} > {max_bitlen})")
            continue
        # primality test
        if is_probable_prime(m, rounds=mr_rounds):
            skipped_B.append((n, m, "probable-prime"))
            if verbose:
                print(f"[B n={n}] m appears prime (skipping)")
            continue
        # m composite -> compute WSS test
        q, is_wss, idx = compute_wss_for_m(m)
        if is_wss:
            verified_B.append((n, m, idx))
            print(f"[B n={n}] VERIFIED WSS composite: m=2^{n}-1 (n={n}), idx={idx}")
            if csv_writer:
                csv_writer.writerow(["B", n, n, str(m), idx, "1"])
        else:
            skipped_B.append((n, m, f"q={q} (not wss)"))
            if verbose:
                print(f"[B n={n}] m composite but NOT WSS: q={q}")
            if csv_writer:
                csv_writer.writerow(["B", n, n, str(m), idx, "0"])

    tB = time.perf_counter() - start
    print(f"Family B done. Verified {len(verified_B)} entries; scanned n up to {y} in {tB:.2f}s")

    if csv_fh:
        csv_fh.close()
        print(f"CSV written: {csvfile}")

    # Summary
    print("\n=== Summary ===")
    print(f"Family A: verified WSS composites (k, exp, m): {len(verified_A)} found")
    for rec in verified_A:
        k, exp, m, idx = rec
        print(f"  k={k}, exp={exp}, m(bitlen={m.bit_length()})")

    print(f"\nFamily B: verified WSS composites (n, m): {len(verified_B)} found")
    for rec in verified_B:
        n, m, idx = rec
        print(f"  n={n}, m(bitlen={m.bit_length()})")

    print("\n(Entries labelled 'probable-prime' were skipped by the probable-prime test;")
    print(" very large entries may be skipped due to max bit-length.)")


if __name__ == "__main__":
    main()
