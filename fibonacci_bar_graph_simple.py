import matplotlib.pyplot as plt
import math
import numpy as np
import sympy as sp

def fib(n):
    # Use iterative approach for speed
    if n == 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fib1(b):
    # constants
    phi = (1 + math.sqrt(5)) / 2

    # integer points
    x_vals = list(range(b + 1))
    y_vals = [fib(n) for n in x_vals]

    # continuous x for asymptotic/Binet curve
    x_cont_vals = np.arange(0, b + 0.5, 0.1)
    y_asymp_vals = phi**x_cont_vals / math.sqrt(5)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_vals, y_vals, color='cornflowerblue', edgecolor='black', label='Fibonacci numbers')
    ax.plot(x_cont_vals, y_asymp_vals, color='r', linestyle='dashed', linewidth=2, label='Asymptotic (Binet)')

    # annotate each bar with its Fibonacci number, placed so it does not overlap the asymptotic line
    for n, y in zip(x_vals, y_vals):
        y_line = phi**n / math.sqrt(5)           # asymptotic value exactly at integer n
        # vertical padding chosen relative to the larger of the two values,
        # with a small minimum so labels don't sit on top of bars for tiny numbers
        pad = max(0.10 * max(y, y_line), 0.1)
        y_text = max(y, y_line) + pad
        ax.text(n, y_text, str(y), ha='center', va='bottom', fontsize=9, weight='bold')

    ax.set_xticks(x_vals)
    ax.set_xlim(-0.5, b + 0.5)
    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('n')
    ax.set_title('Fibonacci numbers and their asymptotic (Binet) approximation')
    plt.tight_layout()
    plt.show()

def _text_contrast_color(rgba):
    r, g, b, _ = rgba
    lum = 0.299*r + 0.587*g + 0.114*b
    return 'white' if lum < 0.5 else 'black'

def fib2(b, method='log', figsize=(10, 6)):
    """
    Plots F_0..F_b with stacked bars broken into *unit* prime factors.
    - Each prime-power p^k is expanded into k boxes labeled 'p'.
    - Color is assigned by factor-position index (first unit = color0, second unit = color1, ...).
    - method: 'log'  -> unit weight = log(p)
              'value'-> unit weight = p
    """
    if method not in ('log', 'value'):
        raise ValueError("method must be 'log' or 'value'")

    ns = list(range(b+1))
    fibs = [fib(n) for n in ns]

    # For each Fibonacci number, get list of prime units sorted by prime ascending,
    # where a prime p with exponent e becomes [p, p, ..., p] (e times).
    unit_lists = []
    for F in fibs:
        if F > 1:
            factor_dict = sp.factorint(F)
            units = []
            for p in sorted(factor_dict.keys()):
                units.extend([p] * factor_dict[p])
            unit_lists.append(units)
        else:
            unit_lists.append([])

    # maximum number of unit positions across all bars (for color assignment)
    max_units = max((len(u) for u in unit_lists), default=0)

    # colors per position index (0 = lowest unit, 1 = second unit, ...)
    cmap = plt.cm.tab20
    position_colors = [cmap(i % 20) for i in range(max_units)]

    plt.figure(figsize=figsize)

    for i, (F, units) in enumerate(zip(fibs, unit_lists)):
        if F <= 1:
            # trivial 0 or 1 bar
            plt.bar(i, F, color='lightgray', edgecolor='black')
            if F > 0:
                plt.text(i, F/2, str(F), ha='center', va='center', fontsize=9)
            continue

        if not units:
            plt.bar(i, F, color='lightgray', edgecolor='black')
            plt.text(i, F/2, str(F), ha='center', va='center', fontsize=9)
            continue

        # compute raw value per unit according to method
        if method == 'log':
            raw_vals = [math.log(p) for p in units]          # each multiplicity unit gets log(p)
        else:  # 'value'
            raw_vals = [p for p in units]                    # each unit weighted by p

        sum_vals = sum(raw_vals)
        if sum_vals == 0:
            shares = [F / len(raw_vals)] * len(raw_vals)
        else:
            shares = [(v / sum_vals) * F for v in raw_vals]

        # stack units bottom-up; color by position index (consistent across bars)
        bottom = 0.0
        for pos_index, (p, share) in enumerate(zip(units, shares)):
            color = position_colors[pos_index]
            plt.bar(i, share, bottom=bottom, color=color, edgecolor='black')
            # label inside or above depending on relative size
            rel_share = share / F if F > 0 else 0
            label = str(p)
            txt_color = _text_contrast_color(color)
            if rel_share >= 0.08:
                fontsize = 10
                plt.text(i, bottom + share/2, label, ha='center', va='center', fontsize=fontsize, color=txt_color, clip_on=True)
            elif rel_share >= 0.03:
                fontsize = 8
                plt.text(i, bottom + share/2, label, ha='center', va='center', fontsize=fontsize, color=txt_color, clip_on=True)
            else:
                # tiny segment: put label just above the segment to keep it readable
                plt.text(i, bottom + share + max(0.01*F, 1e-6), label, ha='center', va='bottom', fontsize=7)
            bottom += share

        # thin outline of the true Fibonacci value
        plt.bar(i, F, bottom=0, fill=False, edgecolor='black', linewidth=1)

    plt.xticks(ns, [f"F{n}" for n in ns])
    plt.ylabel("Value (stacked sub-bars sum to Fibonacci number)")
    plt.xlabel("n")
    plt.title(rf"Fibonacci numbers up to $F({b})$ split into prime units (prime box height scaled with {method})")
    plt.tight_layout()
    plt.show()

def fib3(b, figsize=(10, 10), cmap=plt.cm.tab20, default_color='lightgray'):
    """
    Plot F_0..F_b where:
      - each bar is the Fibonacci number F_n (single bar per n)
      - when a prime p appears for the *first time* as a factor of some F_n,
        that bar is shaded in a unique color assigned to p
      - if multiple primes first-appear in the same bar, the bar color is chosen
        deterministically as the smallest new prime; all new primes are annotated above the bar
    """
    # prepare sequence
    ns = list(range(b+1))
    fibs = [fib(n) for n in ns]

    # factor each Fibonacci number
    factor_dicts = [sp.factorint(v) if v > 1 else {} for v in fibs]

    # first-pass: record the first index where each prime appears
    first_occurrence = {}
    for idx, d in enumerate(factor_dicts):
        for p in sorted(d.keys()):
            if p not in first_occurrence:
                first_occurrence[p] = idx

    # order primes by their first occurrence (ascending by index) to assign colors
    primes_ordered = sorted(first_occurrence.keys(), key=lambda p: first_occurrence[p])
    k = len(primes_ordered)
    if k > 0:
        color_palette = cmap(np.linspace(0, 1, max(k,1)))
        prime_color_map = {p: tuple(color_palette[i]) for i, p in enumerate(primes_ordered)}
    else:
        prime_color_map = {}

    # choose bar colors: default unless this bar is where at least one prime first appears
    bar_colors = []
    new_primes_by_index = {}  # index -> list of primes that first appear here
    for idx, d in enumerate(factor_dicts):
        new_primes = [p for p in sorted(d.keys()) if first_occurrence.get(p) == idx]
        new_primes_by_index[idx] = new_primes
        if new_primes:
            # deterministic choice for single color: smallest new prime
            chosen_prime = min(new_primes)
            bar_colors.append(prime_color_map[chosen_prime])
        else:
            bar_colors.append(default_color)

    # plot
    plt.figure(figsize=figsize)
    bars = plt.bar(ns, fibs, color=bar_colors, edgecolor='black')

    # annotate new-prime appearances above the bar
    ymax = max(fibs) if fibs else 1
    y_offset = 0.02 * ymax
    for idx, new_primes in new_primes_by_index.items():
        if not new_primes:
            continue
        label = "first: " + ", ".join(str(p) for p in new_primes)
        plt.text(idx, fibs[idx] + y_offset, label, ha='center', va='bottom', fontsize=8, rotation=0)

    plt.xticks(ns, [f"F{n}" for n in ns])
    plt.xlabel("n")
    plt.ylabel("Fibonacci value")
    plt.title(f"Fibonacci numbers F0..F{b} — bars colored where primes first appear")
    plt.tight_layout()
    plt.show()

def _legendre_p_over_5(p):
    # Legendre symbol (p/5): 0 if p%5==0, 1 if p%5 in {1,4}, -1 if p%5 in {2,3}
    r = p % 5
    if r == 0:
        return 0
    if r in (1, 4):
        return 1
    return -1

def _text_contrast_color(rgba):
    r, g, b, _ = rgba
    lum = 0.299*r + 0.587*g + 0.114*b
    return 'white' if lum < 0.5 else 'black'

def fib4(b, method='log', figsize=(10,6)):
    """
    Like the previous function but does NOT plot F0.
    Plots F1..Fb on a logarithmic y-axis, highlighting 'special' primes p that satisfy
    p | F[p - (p/5)] by placing unit boxes for those primes at the bottom of the bar.
    Other prime factors are aggregated into a single gray block on top.
    method: 'log' (unit weight = log(p)) or 'value' (unit weight = p)
    """
    if b < 1:
        raise ValueError("b must be at least 1 (this function does not plot F0).")
    if method not in ('log', 'value'):
        raise ValueError("method must be 'log' or 'value'")

    # compute Fibonacci numbers for 0..b (we need them to check divisibility),
    # but we'll only plot 1..b
    fibs_full = [fib(n) for n in range(b+1)]

    # find special primes: p up to b+1 with idx = p - (p/5)
    special_by_index = {n: [] for n in range(b+1)}
    for p in sp.primerange(2, b+2):
        leg = _legendre_p_over_5(p)
        idx = p - leg
        if 0 <= idx <= b:
            if fibs_full[idx] % p == 0:
                special_by_index[idx].append(p)

    # prepare the list of indices we will plot (skip 0)
    plotted_ns = list(range(1, b+1))
    plotted_fibs = [fibs_full[n] for n in plotted_ns]

    # build unit lists for each plotted index: only units from special primes (p repeated exp times)
    unit_lists = []
    for n in plotted_ns:
        F = fibs_full[n]
        if F > 1:
            factors = sp.factorint(F)
            specials = [p for p in sorted(special_by_index.get(n, [])) if p in factors]
            units = []
            for p in specials:
                e = factors[p]
                units.extend([p] * e)
            unit_lists.append(units)
        else:
            unit_lists.append([])

    # colors for unit positions (consistent across bars)
    max_units = max((len(u) for u in unit_lists), default=0)
    cmap = plt.cm.tab20
    position_colors = [cmap(i % 20) for i in range(max_units)]

    plt.figure(figsize=figsize)

    # compute a reasonable tiny value scale (only used for placeholders — not needed here since we skip F0)
    max_fib = max(plotted_fibs) if plotted_fibs else 1
    tiny = 10 ** (math.floor(math.log10(max(1, max_fib))) - 9)

    # Plot bars at x positions equal to their Fibonacci index (1..b)
    for x, n, F, units in zip(plotted_ns, plotted_ns, plotted_fibs, unit_lists):
        # compute raw values for special units
        if units:
            if method == 'log':
                unit_raw = [math.log(p) for p in units]
            else:
                unit_raw = [p for p in units]
            special_raw_sum = sum(unit_raw)
        else:
            unit_raw = []
            special_raw_sum = 0.0

        # compute raw for aggregated 'other' units (non-special prime factors)
        factors = sp.factorint(F) if F > 1 else {}
        non_special_units = []
        for p, e in sorted(factors.items()):
            if p not in special_by_index.get(n, []):
                non_special_units.extend([p] * e)
        if non_special_units:
            if method == 'log':
                other_raw = sum(math.log(p) for p in non_special_units)
            else:
                other_raw = sum(p for p in non_special_units)
        else:
            other_raw = 0.0

        total_raw = special_raw_sum + other_raw
        if total_raw == 0:
            # nothing to highlight; draw a single gray bar
            plt.bar(x, F, color='lightgray', edgecolor='black')
            plt.text(x, F, str(F), ha='center', va='bottom', fontsize=8)
            continue

        # compute shares that sum to F
        unit_shares = [(r / total_raw) * F for r in unit_raw]
        other_share = (other_raw / total_raw) * F

        # plot bottom-up: special units (colored by position), then other (gray)
        bottom = 0.0
        for pos_idx, (p, share) in enumerate(zip(units, unit_shares)):
            if share <= 0:
                continue
            color = position_colors[pos_idx]
            plt.bar(x, share, bottom=bottom, color=color, edgecolor='black')
            # label the unit with the prime when there's room
            txt_color = _text_contrast_color(color)
            rel_share = share / F if F > 0 else 0
            if rel_share >= 0.06:
                fontsize = 9
                plt.text(x, bottom + share/2, str(p), ha='center', va='center', fontsize=fontsize, color=txt_color, clip_on=True)
            elif rel_share >= 0.03:
                fontsize = 7
                plt.text(x, bottom + share/2, str(p), ha='center', va='center', fontsize=fontsize, color=txt_color, clip_on=True)
            else:
                plt.text(x, bottom + share + max(0.01*F, 1e-9), str(p), ha='center', va='bottom', fontsize=7)
            bottom += share

        # top 'other' aggregated segment (light gray) — draw only if positive
        if other_share > 0:
            plt.bar(x, other_share, bottom=bottom, color='lightgray', edgecolor='black')

        # thin outline for the exact Fibonacci value
        plt.bar(x, F, bottom=0, fill=False, edgecolor='black', linewidth=1)

        # optional small annotation listing the special primes above the bar
        specials_here = sorted(set(units))
        if specials_here:
            plt.text(x, F * 1.06, ", ".join(str(p) for p in specials_here),
                     ha='center', va='bottom', fontsize=8)

    plt.yscale('log')
    plt.xticks(plotted_ns, [f"F{n}" for n in plotted_ns])
    plt.xlabel("n")
    plt.ylabel("Fibonacci value (log scale)")
    plt.title(f"Fibonacci F1..F{b} — bottom boxes highlight special primes (log y-scale, method={method})")
    plt.tight_layout()
    plt.show()

# fib1 = plot fibonacci sequence
# fib2 = plot fibonacci sequence with prime factors shown
# fib3 = plot fibonacci sequence with first term with a prime colored individually
# fib4 = plot fibonacci sequence without F0 with each prime bar for F(p - (p/5)) shown in log scale (best with b between 15 and 30)

fib1(15)
fib2(15)
fib4(25)