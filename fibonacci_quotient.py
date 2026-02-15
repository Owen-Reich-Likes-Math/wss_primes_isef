#!/usr/bin/env python3
from __future__ import annotations
import argparse
from fractions import Fraction
from math import isfinite
import matplotlib.pyplot as plt
from typing import List


def fibonacci(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")

    def mat_mult(A: tuple, B: tuple) -> tuple:
        return (
            (A[0][0] * B[0][0] + A[0][1] * B[1][0],
             A[0][0] * B[0][1] + A[0][1] * B[1][1]),
            (A[1][0] * B[0][0] + A[1][1] * B[1][0],
             A[1][0] * B[0][1] + A[1][1] * B[1][1])
        )

    def mat_pow(mat: tuple, exp: int) -> tuple:
        result = ((1, 0), (0, 1))
        base = mat
        n_local = exp
        while n_local > 0:
            if n_local & 1:
                result = mat_mult(result, base)
            base = mat_mult(base, base)
            n_local >>= 1
        return result

    M = ((1, 1), (1, 0))
    return mat_pow(M, n)[0][1]


def legendre_symbol_5(p: int) -> int:
    r = p % 5
    if r == 0:
        return 0
    if r in (1, 4):
        return 1
    return -1


def fib_quo_p2_fraction(p: int) -> Fraction:
    k = legendre_symbol_5(p)
    num = fibonacci(p - k)
    denom = p * p
    return Fraction(num, denom)


def sieve_primes(start: int, end: int) -> List[int]:
    if end < 2 or start > end:
        return []
    sieve = bytearray(b"\x01") * (end + 1)
    sieve[0:2] = b"\x00\x00"
    p = 2
    while p * p <= end:
        if sieve[p]:
            step = p
            start_idx = p * p
            sieve[start_idx:end + 1:step] = b"\x00" * (((end - start_idx) // step) + 1)
        p += 1
    primes = [i for i in range(max(2, start), end + 1) if sieve[i]]
    return primes


def compute_and_plot(start: int, end: int, save_path: str | None = None, show: bool = True, log_scale: bool = False) -> None:
    primes = sieve_primes(start, end)
    if not primes:
        print(f"No primes in range [{start}, {end}].")
        return

    xs = []
    ys = []
    printed = []

    for p in primes:
        frac = fib_quo_p2_fraction(p)
        try:
            fval = float(frac)
        except (OverflowError, ValueError):
            fval = None

        if fval is None or not isfinite(fval):
            printed.append((p, frac, None))
            continue

        xs.append(p)
        ys.append(fval)
        printed.append((p, frac, fval))

    print("p   -> F_(p-(5/p)) / p^2  (exact)    -> (float plotted)")
    for p, frac, fval in printed:
        if fval is None:
            print(f"{p:3d} -> {frac} -> (skipped for plot: non-finite float)")
        else:
            print(f"{p:3d} -> {frac} -> {fval:.6g}")

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker='o', linestyle='-', linewidth=1)
    plt.xlabel("prime p")
    plt.ylabel(r"$F_{p-(5/p)} / p^2$")

    # IMPORTANT FIX: escape braces around the LaTeX subscript so .format doesn't treat them as fields
    plt.title(r"Values of $F_{{p-(5/p)}} / p^2$ for primes $p$ in [{}, {}]".format(start, end))

    plt.grid(True)
    plt.tight_layout()

    if log_scale:
        plt.yscale("log")  # optional: use a log scale if values vary hugely

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot F_{p-(5/p)} / p^2 for primes in a range.")
    parser.add_argument("--start", type=int, default=2, help="Start of inclusive range (default: 2)")
    parser.add_argument("--end", type=int, default=200, help="End of inclusive range (default: 200)")
    parser.add_argument("--save", "-s", type=str, default=None, help="Optional file path to save the figure (e.g. output.png).")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful when saving only).")
    parser.add_argument("--log", action="store_true", help="Plot y-axis on a log scale (helpful when values grow quickly).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise SystemExit("Error: --start must be <= --end.")
    compute_and_plot(args.start, args.end, save_path=args.save, show=not args.no_show, log_scale=args.log)


if __name__ == "__main__":
    main()
