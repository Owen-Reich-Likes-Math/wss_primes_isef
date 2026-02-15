from sympy import primerange, legendre_symbol
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import argparse
import math
import time
import textwrap

# optional progress bar
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False


# ---- matrix utilities (pure Python) ----
def mat_mult(A, B, mod=None):
    if mod is None:
        a11 = A[0][0] * B[0][0] + A[0][1] * B[1][0]
        a12 = A[0][0] * B[0][1] + A[0][1] * B[1][1]
        a21 = A[1][0] * B[0][0] + A[1][1] * B[1][0]
        a22 = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    else:
        a11 = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod
        a12 = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod
        a21 = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod
        a22 = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod
    return ((a11, a12), (a21, a22))


def mat_pow(A, n, mod=None):
    res = ((1, 0), (0, 1))
    base = A
    while n > 0:
        if n & 1:
            res = mat_mult(res, base, mod)
        base = mat_mult(base, base, mod)
        n >>= 1
    return res


def U_n_integer(P, Q, n):
    if n == 0:
        return 0
    T = ((P, -Q), (1, 0))
    M = mat_pow(T, n, mod=None)
    return M[1][0]


def detect_period(P, Q, max_search=30):
    for n in range(1, max_search + 1):
        if U_n_integer(P, Q, n) == 0:
            return n
    return None


def get_valuation(P, Q, p, max_k=5, seq_zero_period=None):
    D = P * P - 4 * Q
    if D % p == 0:
        return 0
    leg = int(legendre_symbol(D, p))
    n = p - leg
    if seq_zero_period is not None and n % seq_zero_period == 0:
        return -1
    val = 0
    for k in range(1, max_k + 1):
        m = p ** k
        T_mod = ((P % m, (-Q) % m), (1, 0))
        M = mat_pow(T_mod, n, mod=m)
        u_n = M[1][0] % m
        if u_n == 0:
            val = k
        else:
            break
    return val


# ---- main generator ----
def generate_universe_page(P, Q, prime_limit=100000, max_k=15,
                           filename="Universe_Profile_final2.pdf",
                           show_progress=True,
                           left_margin_inches=0.6):
    start_time = time.time()
    D = P * P - 4 * Q
    primes = list(primerange(3, prime_limit))
    if show_progress and _use_tqdm:
        iterator = tqdm(primes, desc="Computing primes")
    else:
        iterator = primes

    period = detect_period(P, Q, max_search=30)
    is_cycle = period is not None

    valuations = []
    seq_zero_primes = []
    anomalies = []
    for p in iterator:
        v = get_valuation(P, Q, p, max_k=max_k, seq_zero_period=period)
        if v == -1:
            seq_zero_primes.append(p)
            valuations.append(None)
        else:
            valuations.append(v)
            if v >= 2:
                anomalies.append((p, v))

    # staircase running max (ignore None)
    running = 0
    delta = []
    for v in valuations:
        if v is None:
            delta.append(running)
            continue
        if v > running:
            running = v
        delta.append(running)

    # My bound
    bound_f2 = [2400 * (p ** 2) for p in primes]

    # stats
    total_primes = len(primes)
    count_v1 = valuations.count(1)
    wss_count = len(anomalies)
    seq_zero_count = len(seq_zero_primes)
    max_obs = max([v for v in valuations if isinstance(v, int)] + [0])
    safety_margin = None
    if bound_f2 and max(delta) > 0:
        safety_margin = int(math.log10(bound_f2[-1] / max(1, max(delta))))

    # --- figure setup ---
    plt.rcParams.update({'font.size': 9})
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    fig_width_in = 8.5
    left_frac = left_margin_inches / fig_width_in
    # move plot a bit right so stats fit on the right without overlap
    plot_left = left_frac + 0.02
    plot_width = 0.64
    stats_left = plot_left + plot_width + 0.02
    stats_width = 0.97 - stats_left

    # header
    ax_header = fig.add_axes([left_frac, 0.92, 0.82, 0.06])
    ax_header.axis('off')
    title = f"UNIVERSE PROFILE: U_n({P}, {Q}) — Discriminant D = {D}"
    ax_header.text(0.5, 0.6, title, ha='center', va='center', fontsize=14, weight='bold')
    subtitle = f"Field: Q(√{D})   |   Tested primes: 3 .. {primes[-1] if primes else prime_limit}"
    ax_header.text(0.5, 0.05, subtitle, ha='center', va='center', fontsize=9, style='italic', color='gray')

    # main plot (single y-axis, log scale)
    ax_plot = fig.add_axes([plot_left, 0.38, plot_width, 0.52])
    # convert delta for plotting on log scale: replace 0 by a tiny positive so log works
    delta_for_plot = [d if d > 0 else 1e-3 for d in delta]
    ax_plot.step(primes, delta_for_plot, where='post', linewidth=2, label='Empirical δ(U, b)', zorder=2)
    # plot bound on same axis
    ax_plot.plot(primes, bound_f2, linestyle='--', linewidth=1.0, label='Bound (2400·p²)', zorder=1, alpha=0.9)

    # set log scale so bound soars
    ax_plot.set_yscale('log')
    # limit y-axis lower bound small but >0
    ax_plot.set_ylim(bottom=1e-3)
    # format y ticks as scientific where large, but show small integers reasonably
    ax_plot.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}" if x < 100 else f"{x:.2e}"))
    ax_plot.set_xlabel('Prime p')
    # reposition ylabel so it is not clipped and is inside page margin
    ax_plot.yaxis.set_label_coords(-0.08, 0.5)

    ax_plot.set_title('Staircase valuations and bound (single axis, log scale)')
    ax_plot.grid(True, linestyle=':', alpha=0.6)
    ax_plot.legend(loc='upper left', fontsize=8)

    # stats panel (right) with smaller font and wrapped lines to avoid overflow
    ax_stats = fig.add_axes([stats_left, 0.55, stats_width, 0.35])
    ax_stats.axis('off')
    stats_lines = [
        f"Total primes tested : {total_primes:,}",
        f"Standard primes (v=1): {count_v1}",
        f"WSS candidates (v>=2): {wss_count}",
        f"Exact zeros (periodic primes): {seq_zero_count}",
        f"Max observed valuation : {max_obs}",
        f"p-adic depth (max_k): {max_k}",
        f"Theoretical safety margin (log10): {safety_margin if safety_margin is not None else 'N/A'}",
    ]
    ax_stats.text(0, 0.95, "Statistical Summary", fontsize=10, weight='bold', va='top')
    for i, ln in enumerate(stats_lines):
        # wrap longer lines
        wrapped = textwrap.fill(ln, width=28)
        ax_stats.text(0, 0.82 - i * 0.12, wrapped, fontsize=8.5, va='top', family='monospace')

    # anomaly log area across bottom
    ax_log = fig.add_axes([left_frac, 0.06, 0.94 - left_frac, 0.28])
    ax_log.axis('off')
    ax_log.text(0.0, 0.97, "ANOMALY LOG — primes with ν_p(U_n) ≥ 2", fontsize=11, weight='bold', va='top')

    # helper format
    def fmt(pv):
        p, v = pv
        return f"{p:6d}  v={v}"

    N = len(anomalies)
    # layout:
    # - If >= 90: special grouping (30 | 15+15 | 30) into 3 columns
    # - Else: balanced two columns (ceil(N/2) rows)
    if N >= 90:
        first30 = anomalies[0:30]
        next15 = anomalies[30:45]
        mid15 = anomalies[-45:-30]
        last30 = anomalies[-30:]
        col1 = first30
        col2 = next15 + mid15
        col3 = last30
        cols = [col1, col2, col3]
        col_x = [0.01, 0.36, 0.71]
        max_rows = max(len(c) for c in cols)
        y0 = 0.92
        y_step = 0.88 / max_rows
        for ci, col in enumerate(cols):
            x = col_x[ci]
            for ri, entry in enumerate(col):
                y = y0 - ri * y_step
                ax_log.text(x, y, fmt(entry), fontsize=9, family='monospace', va='top')
    else:
        # two-column balanced layout
        rows = math.ceil(N / 2)
        left_col = anomalies[:rows]
        right_col = anomalies[rows:rows*2]
        col_x = [0.01, 0.50]
        y0 = 0.92
        y_step = 0.88 / max(max(len(left_col), len(right_col), 1), rows)
        # left
        for ri, entry in enumerate(left_col):
            y = y0 - ri * y_step
            ax_log.text(col_x[0], y, fmt(entry), fontsize=9, family='monospace', va='top')
        # right
        for ri, entry in enumerate(right_col):
            y = y0 - ri * y_step
            ax_log.text(col_x[1], y, fmt(entry), fontsize=9, family='monospace', va='top')

    # footer / note
    if is_cycle:
        note = (f"NOTE — Sequence periodic: smallest n>0 with U_n = 0 is n = {period}. "
                "Exact integer zeros occur when n is a multiple of this period; they are listed separately.")
        fig.text(0.5, 0.02, textwrap.fill(note, 110), ha='center', fontsize=8, color='darkred')
    else:
        foot = f"Note: primes tested up to {primes[-1] if primes else prime_limit}. Bound plotted on same axis (log scale)."
        fig.text(0.5, 0.02, textwrap.fill(foot, 110), ha='center', fontsize=8, color='gray')

    # save pdf
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    elapsed = time.time() - start_time
    print(f"Saved '{filename}' — primes: {total_primes:,}, anomalies: {wss_count}, seq-zero primes: {seq_zero_count}. Elapsed {elapsed:.1f}s.")


# ---- CLI ----
def _parse_args():
    p = argparse.ArgumentParser(description="Universe Profile final2 (single-page) PDF generator.")
    p.add_argument("--P", type=int, help="P parameter")
    p.add_argument("--Q", type=int, help="Q parameter")
    p.add_argument("--prime_limit", type=int, default=100000, help="Upper bound for primes")
    p.add_argument("--max_k", type=int, default=5, help="p-adic depth (default 5)")
    p.add_argument("--out", type=str, default=None, help="Output PDF filename")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm")
    p.add_argument("--left_margin", type=float, default=0.6,
                   help="Left margin in inches (default 0.6) to allow hole punching")
    return p.parse_args()


def main():
    args = _parse_args()
    global _use_tqdm
    if args.no_tqdm:
        _use_tqdm = False

    if args.P is None or args.Q is None:
        try:
            P = int(input("Enter integer P: ").strip())
            Q = int(input("Enter integer Q: ").strip())
        except Exception:
            print("Invalid input; please provide integer values for P and Q.")
            return
    else:
        P, Q = args.P, args.Q

    filename = args.out or f"Profile_P{P}_Q{Q}_final2.pdf"
    generate_universe_page(P=P, Q=Q, prime_limit=args.prime_limit, max_k=args.max_k,
                           filename=filename, show_progress=True, left_margin_inches=args.left_margin)


if __name__ == "__main__":
    main()
