import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange, legendre_symbol
from datetime import datetime, timezone
import os
import multiprocessing as mp
from functools import partial

# matrix multiplier mod m for calculating terms
def mat_mul_mod(A, B, m):
    return (
        (A[0] * B[0] + A[1] * B[2]) % m,
        (A[0] * B[1] + A[1] * B[3]) % m,
        (A[2] * B[0] + A[3] * B[2]) % m,
        (A[2] * B[1] + A[3] * B[3]) % m
    )

def get_Un_mod(P, Q, n, m):
    if n == 0: return 0
    if n == 1: return 1 % m

    r0, r1, r2, r3 = 1, 0, 0, 1
    b0, b1, b2, b3 = P, -Q, 1, 0

    while n > 0:
        if n & 1:
            r0, r1, r2, r3 = (
                (r0 * b0 + r1 * b2) % m,
                (r0 * b1 + r1 * b3) % m,
                (r2 * b0 + r3 * b2) % m,
                (r2 * b1 + r3 * b3) % m
            )
        b0, b1, b2, b3 = (
            (b0 * b0 + b1 * b2) % m,
            (b0 * b1 + b1 * b3) % m,
            (b2 * b0 + b3 * b2) % m,
            (b2 * b1 + b3 * b3) % m
        )
        n >>= 1
    return r2

def get_Un_int(P, Q, n):
    if n == 0: return 0
    if n == 1: return 1

    r0, r1, r2, r3 = 1, 0, 0, 1
    b0, b1, b2, b3 = P, -Q, 1, 0

    while n > 0:
        if n & 1:
            r0, r1, r2, r3 = (
                r0 * b0 + r1 * b2, r0 * b1 + r1 * b3,
                r2 * b0 + r3 * b2, r2 * b1 + r3 * b3
            )
        b0, b1, b2, b3 = (
            b0 * b0 + b1 * b2, b0 * b1 + b1 * b3,
            b2 * b0 + b3 * b2, b2 * b1 + b3 * b3
        )
        n >>= 1
    return r2

# core logic, iterated over for every universe
def compute_universe(args):
    P, Q = args
    xs = []
    ys = []
    anomalies = []
    
    global primes_global
    
    for p in primes_global:
        D = P * P - 4 * Q

        # check degen
        if Q % p == 0 or D % p == 0:
            continue

        n = p - legendre_symbol(D, p)

        # check for no divis (shouldnt happen)
        if get_Un_mod(P, Q, n, p) != 0:
            val = 0
        else:
            # compute int if divis by p
            u_n_int = get_Un_int(P, Q, n)
            if u_n_int == 0:
                continue # should not happen if p > n, but safety
            
            val = 1
            # check higher powers
            for k in range(2, 101):
                m = p ** k
                if get_Un_mod(P, Q, n, m) == 0:
                    val = k
                else:
                    break
        
        if val > 0:
            xs.append(p)
            ys.append(val)
            if val >= 2:
                anomalies.append((p, val))

    if not xs:
        return None
    
    return {
        "P": P, 
        "Q": Q, 
        "primes": np.array(xs, dtype=np.int64), 
        "vals": np.array(ys, dtype=np.int64), 
        "anoms": anomalies
    }

# cpu workers parallel
def init_worker(primes):
    global primes_global
    primes_global = primes

if __name__ == '__main__':
    mp.freeze_support()

    print("Lucas Universe Overlay")
    try:
        P_min = int(input("Enter P_min: "))
        P_max = int(input("Enter P_max (>= P_min): "))
        Q_abs_max = int(input("Enter max |Q|: "))
        prime_bound = int(input("Enter Prime Bound: "))
    except Exception:
        print("Invalid input — using defaults: P_min=1, P_max=5, Q_abs_max=3, prime_bound=200")
        P_min, P_max, Q_abs_max, prime_bound = 1, 5, 3, 200

    if P_max < P_min:
        P_min, P_max = P_max, P_min

    # periodic universes that are stupid
    periodic_skip_set = {
        (0, -1), (0, 1),
        (1, 1), (-1, 1),
        (2, 1), (-2, 1)
    }

    def violates_k_relation(P, Q):
        P2 = P * P
        if P2 == 0: return True
        if P2 == Q: return True
        if P2 == 2 * Q: return True
        if P2 == 3 * Q: return True
        return False

    # gen universes
    requested_universes = [
        (P, Q)
        for P in range(P_min, P_max + 1)
        for Q in range(-Q_abs_max, Q_abs_max + 1)
        if Q != 0
    ]

    if not requested_universes:
        raise SystemExit("No universes to scan.")

    filtered_universes = [
        (P, Q) for (P, Q) in requested_universes
        if (P, Q) not in periodic_skip_set and not violates_k_relation(P, Q)
    ]
    
    total_unis = len(filtered_universes)
    print(f"Scanning {total_unis} universes; primes up to {prime_bound}...")

    primes_list = list(primerange(3, prime_bound + 1))
    if not primes_list:
        raise SystemExit("No primes found.")

    # parallel start
    all_data = []
    
    # get num of cores
    num_cores = mp.cpu_count()
    print(f" utilizing {num_cores} CPU cores.")

    start_time = time.time()
    
    # share primes
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(primes_list,)) as pool:
        chunk_size = max(1, total_unis // (num_cores * 4))
        results_iter = pool.imap_unordered(compute_universe, filtered_universes, chunksize=chunk_size)
        
        bar_width = 30
        for i, result in enumerate(results_iter):
            if result is not None:
                all_data.append(result)
            
            # prog bar
            if (i + 1) % 10 == 0 or (i + 1) == total_unis:
                progress = (i + 1) / total_unis
                filled = int(bar_width * progress)
                bar = "=" * filled + "-" * (bar_width - filled)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}% | {i + 1}/{total_unis} | {rate:.1f} u/s")
                sys.stdout.flush()

    print("\nScan complete.")

    # plotting
    
    max_val = -1
    max_indices = []
    
    for i, data in enumerate(all_data):
        if data["vals"].size > 0:
            local_max = int(np.max(data["vals"]))
            if local_max > max_val:
                max_val = local_max
    
    if max_val > -1:
        for i, data in enumerate(all_data):
            if data["vals"].size > 0 and int(np.max(data["vals"])) == max_val:
                max_indices.append(i)
                print(f"Universe with max_v={max_val}: P={data['P']}, Q={data['Q']}")
    else:
        print("No positive valuations found.")

    print("Generating plot...")
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    cmap = plt.get_cmap('tab20')

    # make primes list for bound plotting
    primes_np = np.array(primes_list)
    bound_multiplier = 2400
    bound_line = bound_multiplier * primes_np

    # plot lines
    for i, data in enumerate(all_data):
        xs = data["primes"]
        ys = data["vals"]
        
        color = cmap(i % cmap.N)
        is_top_universe = (i in max_indices)

        if is_top_universe:
            label = f"Highest max v={max_val} (P={data['P']}, Q={data['Q']})"
            ax.plot(xs, ys, marker='o', markersize=4, linestyle='-', linewidth=1.8,
                    alpha=0.95, color=color, label=label, zorder=5)
        else:
            # Lighter plot for bulk
            ax.plot(xs, ys, marker='o', markersize=2.5, linestyle='-', linewidth=0.6,
                    alpha=0.45, color=color, label='_nolegend_')

    ax.plot(primes_np, bound_line, linestyle='--', linewidth=1.0, color='k', alpha=0.8)

    # mark big points
    anom_ps = []
    anom_vs = []
    for data in all_data:
        for (p, v) in data["anoms"]:
            anom_ps.append(p)
            anom_vs.append(v)
            
    if anom_ps:
        ax.scatter(anom_ps, anom_vs, color='red', s=30, zorder=10)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper left', fontsize='small')

    ax.set_yscale('log')
    ax.set_xlabel("Prime p")
    ax.set_ylabel(r"Valuation $\nu_p(U_{rank})$")
    ax.set_title(rf"Lucas Universes Scan (Parallelized)")
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tight_layout()

    # export data to csv
    output_dir = "exported_data"
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, "all_universes_optimized.csv")

    print(f"Writing data to {outfile}...")
    with open(outfile, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Lucas universes export\n# generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        for data in all_data:
            f.write(f"Universe {data['P']}, {data['Q']}\n")
            xs = data["primes"]
            ys = data["vals"]
            for p, v in zip(xs, ys):
                f.write(f"{int(p)} | {int(v)}\n")
            f.write("\n")

    print("Done.")
    plt.show()
