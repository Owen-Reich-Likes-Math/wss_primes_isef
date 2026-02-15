#!/usr/bin/env python3
"""wall_sun_sun_composites_safe_wss_fixed.py

Patched version with these fixes:
 - Keep full WSS list while <= display_wss_limit (wss_all), otherwise keep head/tail only.
 - Create directories for HTML/CSV outputs before writing.
 - Use an iterative fast-doubling Fibonacci (reduces recursion overhead).
 - Only mark a composite as WSS when F_{idx} ≡ 0 (mod n^2) (i.e., f_mod == 0).
 - Stream WSS to CSV as discovered; keep only head/tail in memory when the list grows.
 - Return a summary dict from the main generator function (handy for programmatic checks).
"""

from __future__ import annotations
import argparse
import base64
import csv
import io
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from heapq import heappush, heappushpop
from typing import Iterator, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional extras
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

try:
    import scipy.stats as ss
    from scipy.stats import chi2 as _chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# -----------------------
# Sieve & composite iterator
# -----------------------
def simple_sieve(n: int) -> List[int]:
    """Sieve primes up to n (small n), used to seed segmented sieve."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    limit = int(n**0.5) + 1
    for p in range(2, limit):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n+1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i, isprime in enumerate(sieve) if isprime]


def segmented_composites(bound: int, segment_size: int = 1_000_000) -> Iterator[int]:
    """
    Yield composite numbers <= bound using a segmented sieve approach.
    Memory ~ segment_size bytes + primes up to sqrt(bound).
    """
    if bound < 4:
        return
    limit = int(math.isqrt(bound)) + 1
    small_primes = simple_sieve(limit)
    small_prime_set = set(small_primes)
    # small composites in [4..min(limit,bound)]
    for n in range(4, min(limit, bound) + 1):
        if n not in small_prime_set:
            yield n
    low = max(limit + 1, 2)
    while low <= bound:
        high = min(low + segment_size - 1, bound)
        size = high - low + 1
        segment = bytearray(b"\x01") * size
        for p in small_primes:
            if p * p > high:
                break
            start = ((low + p - 1) // p) * p
            for m in range(start, high + 1, p):
                segment[m - low] = 0
        for i in range(size):
            val = low + i
            if val >= 4 and segment[i] == 0:
                yield val
        low += segment_size


# -----------------------
# Legendre-like helper (mod 5)
# -----------------------
def legendre_over_5(n: int) -> int:
    """
    Return (n/5) like Legendre symbol with denominator 5:
      - 0 if n % 5 == 0
      - 1 if n % 5 in {1,4}
      - -1 if n % 5 in {2,3}
    """
    r = n % 5
    if r == 0:
        return 0
    return 1 if r in (1, 4) else -1


# -----------------------
# Fibonacci fast-doubling modulo (iterative)
# -----------------------
def fib_fast_doubling_mod(n: int, mod: int) -> Tuple[int, int]:
    """Iterative fast-doubling returning (F_n mod mod, F_{n+1} mod mod)."""
    if mod == 1:
        return 0, 0
    n = int(n)
    a, b = 0 % mod, 1 % mod  # F0, F1
    if n == 0:
        return a, b
    # process bits from most-significant to least, skipping the leading 1
    bits = bin(n)[3:]  # drop '0b' and the highest '1' bit
    for bit in bits:
        # double: from F_k=a, F_{k+1}=b compute F_{2k}, F_{2k+1}
        c = (a * ((2 * b - a) % mod)) % mod
        d = (a * a + b * b) % mod
        if bit == '0':
            a, b = c, d
        else:
            a, b = d, (c + d) % mod
    return a, b


def compute_qn_for_composite(n: int) -> Tuple[int, bool, int, int]:
    """
    For composite n compute q_n = (F_{n - (n/5)} / n) mod n using F_n mod n^2.
    Returns (q_n, warned, idx_used, f_mod).
      - q_n: the quotient (floor(F / n) mod n), equals 0 is candidate but we verify with f_mod
      - warned: True when F_{idx} % n != 0 (i.e., remainder nonzero mod n)
      - idx_used: the Fibonacci index computed (n - (n/5))
      - f_mod: F_idx mod n^2 (0 <= f_mod < n^2)
    """
    chi = legendre_over_5(n)
    idx = n - chi
    mod = n * n
    f_mod, _ = fib_fast_doubling_mod(idx, mod)
    # q = floor(f_mod / n) mod n; but q == 0 iff f_mod == 0 (since 0 <= f_mod < n^2)
    q = (f_mod // n) % n
    warned = (f_mod % n != 0)
    return int(q), bool(warned), int(idx), int(f_mod)


# -----------------------
# Utilities: fig->base64
# -----------------------
def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# -----------------------
# Online statistics helpers
# -----------------------
class Welford:
    """Online mean/variance (population variance)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def result(self):
        if self.n == 0:
            return math.nan, math.nan
        var = self.m2 / self.n
        return self.mean, math.sqrt(var)

    def sample_variance(self):
        if self.n < 2:
            return math.nan
        return self.m2 / (self.n - 1)


class OnlinePearson:
    """Online Pearson components to compute correlation later."""
    def __init__(self):
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0

    def add(self, x: float, y: float):
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    def result(self):
        if self.n < 2:
            return math.nan
        num = self.sxy - (self.sx * self.sy / self.n)
        den = math.sqrt((self.sxx - (self.sx * self.sx / self.n)) * (self.syy - (self.sy * self.sy / self.n)))
        if den == 0:
            return math.nan
        return num / den


# -----------------------
# Main streaming composite analysis & HTML generation (memory-safe)
# -----------------------
def generate_composite_report_streaming_wss(bound: int,
                                            sample_size: int = 100_000,
                                            bins: int = 50,
                                            htmlfile: Optional[str] = None,
                                            csvfile: Optional[str] = None,
                                            wss_csvfile: Optional[str] = None,
                                            no_wss_csv: bool = False,
                                            display_wss_limit: int = 5000,
                                            head_wss: int = 100,
                                            tail_wss: int = 100,
                                            z_threshold: float = 3.0,
                                            progress_interval: int = 10_000,
                                            segment_size: int = 1_000_000,
                                            max_composites: Optional[int] = None,
                                            store_warnings_limit: int = 1000):
    """
    Streaming composite analysis with memory-safe handling of WSS composites and warnings.

    Returns a dict with a summary including verification info.
    """
    t0 = time.perf_counter()
    if htmlfile is None:
        now = datetime.now()
        date_part = now.strftime("%m-%d-%y")
        safe_bound = str(bound).replace(",", "")
        htmlfile = f"report-{date_part}-{safe_bound}-composites.html"
    print(f"Output HTML: {htmlfile}")

    # default WSS CSV file derived from htmlfile
    if wss_csvfile is None:
        wss_csvfile = htmlfile.replace(".html", "-wss-composites.csv")

    # Ensure output directories exist for html and csv if requested
    def ensure_parent_dir(path: Optional[str]):
        if not path:
            return
        outdir = os.path.dirname(path)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

    ensure_parent_dir(htmlfile)
    ensure_parent_dir(csvfile)
    ensure_parent_dir(wss_csvfile)

    composite_iter = segmented_composites(bound, segment_size=segment_size)

    # streaming state
    count = 0
    welf = Welford()
    zeros = 0
    lt10 = 0
    eps = 0.01
    near0 = 0
    near1 = 0
    bin_counts = [0] * bins
    p5_counts = defaultdict(int)
    p5_sums = defaultdict(float)
    modk_online = {k: OnlinePearson() for k in range(2, 11)}
    serial_online = OnlinePearson()
    prev_norm = None
    sample: List[Tuple[int, int, float, int]] = []
    sample_t = 0

    # warnings: keep only a small stored sample + a count
    warnings_count = 0
    warnings_sample: List[int] = []

    # top-k outlier heap (bounded)
    top_k_heap = []
    top_k_size = 200

    # WSS handling: maintain head list and tail deque; write all discovered WSS to CSV if allowed
    wss_head: List[int] = []
    wss_tail = deque(maxlen=tail_wss)
    wss_count = 0
    wrote_wss_csv = False
    wss_csv_fh = None
    # keep full list while it's small enough to embed in HTML later
    wss_all: Optional[List[int]] = [] if display_wss_limit and display_wss_limit > 0 else None

    if not no_wss_csv:
        try:
            # ensure directory exists if wss_csvfile has dir (done earlier) then open
            wss_csv_fh = open(wss_csvfile, "w", newline="", encoding="utf-8")
            wss_csv_writer = csv.writer(wss_csv_fh)
            wss_csv_writer.writerow(["n"])
            wrote_wss_csv = True
        except Exception as e:
            print(f"Warning: failed to open WSS CSV {wss_csvfile} for writing: {e}")
            wss_csv_fh = None
            wrote_wss_csv = False

    # Keep track of verification failures (if any)
    verification_failures: List[Tuple[int, int]] = []  # (n, f_mod)

    # Iterate composites streamingly
    for n in composite_iter:
        count += 1
        if max_composites is not None and count > max_composites:
            break

        q, warned, n_used, f_mod = compute_qn_for_composite(n)
        if warned:
            warnings_count += 1
            if len(warnings_sample) < store_warnings_limit:
                warnings_sample.append(n)

        # Treat true WSS composites as those where f_mod == 0 (i.e., n^2 divides the Fibonacci term).
        is_wss_verified = (f_mod == 0)
        if q == 0 and not is_wss_verified:
            # candidate (q == 0) but final f_mod != 0; record as verification failure
            verification_failures.append((n, f_mod))
            warned = True
            warnings_count += 1
            if len(warnings_sample) < store_warnings_limit:
                warnings_sample.append(n)

        if is_wss_verified:
            wss_count += 1
            if wss_all is not None:
                wss_all.append(n)
                if wss_count > display_wss_limit:
                    # drop full list to conserve memory
                    wss_all = None
            if len(wss_head) < head_wss:
                wss_head.append(n)
            else:
                wss_tail.append(n)
            # stream-write to CSV if available
            if wss_csv_fh is not None:
                try:
                    wss_csv_writer.writerow([int(n)])
                except Exception:
                    # disable further writes on error
                    try:
                        wss_csv_fh.close()
                    except Exception:
                        pass
                    wss_csv_fh = None
                    wrote_wss_csv = False

        norm = q / n
        # streaming stat updates
        welf.add(norm)
        if q == 0:
            zeros += 1
        if q < 10:
            lt10 += 1
        if norm < eps:
            near0 += 1
        if norm > 1 - eps:
            near1 += 1

        # histogram bin
        idx = min(int(norm * bins), bins - 1)
        bin_counts[idx] += 1

        # p%5 grouping
        r5 = n % 5
        p5_counts[r5] += 1
        p5_sums[r5] += norm

        # p%k online Pearson
        for k, online in modk_online.items():
            online.add(norm, float(n % k))

        # serial lag-1 correlation stream
        if prev_norm is not None:
            serial_online.add(prev_norm, norm)
        prev_norm = norm

        # reservoir sampling
        sample_t += 1
        if len(sample) < sample_size:
            sample.append((n, q, norm, n_used))
        else:
            j = random.randrange(0, sample_t)
            if j < sample_size:
                sample[j] = (n, q, norm, n_used)

        # top-k outlier heap heuristic
        mean_so_far = welf.mean
        absdev = abs(norm - mean_so_far)
        if len(top_k_heap) < top_k_size:
            heappush(top_k_heap, (absdev, n, q, norm))
        else:
            if absdev > top_k_heap[0][0]:
                heappushpop(top_k_heap, (absdev, n, q, norm))

        # periodic progress & memory info
        if (count % progress_interval) == 0:
            now = time.perf_counter()
            elapsed = now - t0
            per_sec = count / elapsed if elapsed > 0 else float('inf')
            per_million = (elapsed / count) * 1_000_000 if count > 0 else float('nan')
            mem_info = ""
            if PSUTIL_AVAILABLE:
                rss = psutil.Process().memory_info().rss / (1024 ** 2)
                mem_info = f", RSS={rss:.1f} MB"
            print(f"[{count:,}] composites processed — {per_sec:.1f} comp/s, {per_million:.1f} s per million{mem_info}")

    # close wss csv handle if opened
    if wss_csv_fh is not None:
        try:
            wss_csv_fh.close()
        except Exception:
            pass

    total_time = time.perf_counter() - t0
    comps_per_sec = count / total_time if total_time > 0 else float('inf')
    sec_per_million = (total_time / count) * 1_000_000 if count > 0 else float('nan')
    mean_norm, std_norm = welf.result()
    if math.isnan(std_norm) or std_norm == 0:
        std_norm = float('nan')

    # sample arrays for sample-based tests/plots
    sample_arr = np.array([s[2] for s in sample], dtype=float) if sample else np.array([], dtype=float)

    # chi-square on streamed bins
    expected = count / bins if count > 0 else 0
    chi2_stat = float(sum(((c - expected) ** 2) / expected for c in bin_counts)) if expected > 0 else float('nan')
    chi2_df = bins - 1
    chi2_p = float(_chi2.sf(chi2_stat, chi2_df)) if SCIPY_AVAILABLE and not math.isnan(chi2_stat) else None

    # KS on sample
    if sample_arr.size > 0:
        if SCIPY_AVAILABLE:
            ks_res = ss.kstest(sample_arr, 'uniform')
            ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)
        else:
            sorted_norm = np.sort(sample_arr)
            m = sample_arr.size
            ecdf = np.arange(1, m + 1) / m
            d1 = np.max(ecdf - sorted_norm)
            d2 = np.max(sorted_norm - (np.arange(0, m) / m))
            ks_stat = max(d1, d2)
            ks_p = None
    else:
        ks_stat, ks_p = float('nan'), None

    # modular stats
    p5_stats = {r: (p5_counts[r], (p5_sums[r] / p5_counts[r] if p5_counts[r] > 0 else float('nan'))) for r in sorted(p5_counts.keys())}
    modk_results = {k: online.result() for k, online in modk_online.items()}
    serial_r = serial_online.result()

    # outliers detection from sample + heap
    outlier_candidates = []
    if not math.isnan(std_norm):
        for (n_val, q_val, norm_val, _) in sample:
            z = (norm_val - mean_norm) / std_norm if std_norm != 0 else float('nan')
            if abs(z) > z_threshold:
                outlier_candidates.append((abs(z), n_val, q_val, norm_val, z))
        for absdev, n_val, q_val, norm_val in top_k_heap:
            z = (norm_val - mean_norm) / std_norm if std_norm != 0 else float('nan')
            if abs(z) > z_threshold:
                outlier_candidates.append((abs(z), n_val, q_val, norm_val, z))
        outlier_by_n = {}
        for rec in outlier_candidates:
            _, num, qv, normv, z = rec
            if num not in outlier_by_n or abs(z) > abs(outlier_by_n[num][2]):
                outlier_by_n[num] = (qv, normv, z)
        outliers_sorted = sorted([(num, qv, normv, z) for num, (qv, normv, z) in outlier_by_n.items()], key=lambda x: -abs(x[3]))
    else:
        outliers_sorted = []

    # visuals from reservoir (cap viz points)
    viz_limit = min(len(sample), 100_000)
    viz_sample = random.sample(sample, viz_limit) if len(sample) > viz_limit else list(sample)
    viz_norms = np.array([s[2] for s in viz_sample], dtype=float) if viz_sample else np.array([], dtype=float)
    viz_ns = np.array([s[0] for s in viz_sample], dtype=int) if viz_sample else np.array([], dtype=int)

    images = {}
    # histogram image
    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.linspace(0, 1, bins, endpoint=False), bin_counts, width=1.0 / bins, align='edge', edgecolor='black')
    ax.set_xlabel("Normalized q_n / n")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram (binned) of normalized q_n (n_composites={count:,})")
    images['hist'] = fig_to_base64_png(fig)

    # QQ
    if viz_norms.size > 0:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        sorted_norm = np.sort(viz_norms)
        m = sorted_norm.size
        uniform_q = (np.arange(1, m + 1) - 0.5) / m
        ax.plot(uniform_q, sorted_norm, marker='.', linestyle='none')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel("Uniform quantiles")
        ax.set_ylabel("Empirical normalized quantiles")
        ax.set_title("Q-Q plot vs Uniform(0,1) (sample)")
        images['qq'] = fig_to_base64_png(fig)
    else:
        images['qq'] = ""

    # scatter
    if viz_norms.size > 0:
        fig = plt.figure(figsize=(10, 3.6))
        ax = fig.add_subplot(1, 1, 1)
        colors = (viz_ns % 5)
        sc = ax.scatter(viz_ns, viz_norms, c=colors, cmap='tab10', s=6, alpha=0.6)
        ax.set_xlabel("composite n (sample)")
        ax.set_ylabel("normalized q_n / n")
        ax.set_title("normalized q_n vs n (color = n % 5) [sample]")
        plt.colorbar(sc, ax=ax, ticks=range(5), label="n % 5")
        images['scatter'] = fig_to_base64_png(fig)
    else:
        images['scatter'] = ""

    # sequence & FFT
    if viz_norms.size > 0:
        fig = plt.figure(figsize=(10, 2.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(viz_norms.size), viz_norms, marker='.', markersize=2, linestyle='-', alpha=0.6)
        ax.set_xlabel("sample index")
        ax.set_ylabel("normalized q_n/n (sample)")
        ax.set_title("Sequence (sample)")
        images['sequence'] = fig_to_base64_png(fig)

        if viz_norms.size > 1:
            centered = viz_norms - viz_norms.mean()
            fft_vals = np.fft.rfft(centered)
            fft_freqs = np.fft.rfftfreq(viz_norms.size)
            fft_power = np.abs(fft_vals)
            fig = plt.figure(figsize=(8, 2.6))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(fft_freqs[1:], fft_power[1:])
            ax.set_xlabel("Frequency (sample-normalized)")
            ax.set_ylabel("FFT magnitude")
            ax.set_title("Spectral analysis (sample)")
            images['fft'] = fig_to_base64_png(fig)
        else:
            images['fft'] = ""
    else:
        images['sequence'] = ""
        images['fft'] = ""

    # Build HTML report (embed WSS composite list head/tail or full if small)
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    html_parts: List[str] = []
    html_parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    html_parts.append(f"<title>Composite q_n analysis report (bound={bound:,})</title>")
    html_parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:18px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;}th{background:#f4f7fb;}code{font-family:monospace;}</style>")
    html_parts.append("</head><body>")
    html_parts.append(f"<h1>Wall–Sun–Sun style q_n analysis — composites up to {bound:,}</h1>")
    html_parts.append(f"<p>Generated: {now}</p>")

    # WSS block at top
    html_parts.append("<div style='border:1px solid #ccc;padding:10px;margin-bottom:12px;background:#f9faff;'>")
    html_parts.append(f"<h2>WSS composites (q_n == 0) — verified count = {wss_count:,}</h2>")
    if wss_count == 0:
        html_parts.append("<p><em>No WSS composites found in this run.</em></p>")
    else:
        if wss_all is not None:
            # we kept the full list in memory
            full_list = wss_all
            chunk_size = 200
            lines = []
            for i in range(0, len(full_list), chunk_size):
                chunk = full_list[i:i + chunk_size]
                lines.append(", ".join(str(x) for x in chunk))
            html_parts.append("<pre style='font-family:monospace;max-height:400px;overflow:auto;'>")
            html_parts.append("\n".join(lines))
            html_parts.append("</pre>")
        else:
            # partial listing: show head and tail, indicate partial
            html_parts.append(f"<p><strong>Note:</strong> The list of verified WSS composites is large ({wss_count:,} entries). Only an excerpt is shown below.</p>")
            html_parts.append("<p><strong>First entries:</strong></p>")
            html_parts.append("<pre style='font-family:monospace;max-height:200px;overflow:auto;'>")
            html_parts.append(", ".join(str(x) for x in wss_head))
            html_parts.append("</pre>")
            html_parts.append("<p><strong>Last entries:</strong></p>")
            html_parts.append("<pre style='font-family:monospace;max-height:200px;overflow:auto;'>")
            html_parts.append(", ".join(str(x) for x in list(wss_tail)))
            html_parts.append("</pre>")
            if wrote_wss_csv:
                html_parts.append(f"<p>Full list was written to CSV: <code>{wss_csvfile}</code></p>")
            else:
                if no_wss_csv:
                    html_parts.append("<p>Full list was not saved to CSV because --no-wss-csv was set.</p>")
                else:
                    html_parts.append("<p>Attempted to save full list to CSV but failed — check console output.</p>")
    if verification_failures:
        html_parts.append("<p style='color:darkred'><strong>Verification failures:</strong> Some entries flagged as q==0 failed the stricter f_mod==0 check. See console/log for list.</p>")
    html_parts.append("</div>")

    # Performance table
    html_parts.append("<h2>Performance</h2>")
    html_parts.append("<table>")
    html_parts.append(f"<tr><th>Composites processed</th><td>{count:,}</td></tr>")
    html_parts.append(f"<tr><th>Total time (s)</th><td>{total_time:.3f}</td></tr>")
    html_parts.append(f"<tr><th>Composites / sec</th><td>{comps_per_sec:.1f}</td></tr>")
    html_parts.append(f"<tr><th>Seconds per million composites</th><td>{sec_per_million:.1f}</td></tr>")
    if PSUTIL_AVAILABLE:
        rss = psutil.Process().memory_info().rss / (1024 ** 2)
        html_parts.append(f"<tr><th>Memory RSS (MB)</th><td>{rss:.1f}</td></tr>")
    else:
        html_parts.append("<tr><th>Memory RSS (MB)</th><td>psutil not installed</td></tr>")
    html_parts.append(f"<tr><th>SciPy available</th><td>{SCIPY_AVAILABLE}</td></tr>")
    if warnings_count > 0:
        sample_warns = ", ".join(str(x) for x in warnings_sample)
        more = "" if warnings_count <= store_warnings_limit else f" (and {warnings_count - len(warnings_sample)} more)"
        html_parts.append(f"<tr><th>Warnings (F mod n^2 not divisible by n)</th><td class='mono'>{sample_warns}{more}</td></tr>")
    html_parts.append("</table>")

    # Distribution & sample-based tests
    html_parts.append("<h2>Distribution & sample-based tests</h2>")
    html_parts.append("<table>")
    html_parts.append(f"<tr><th>Sample size (reservoir)</th><td>{len(sample):,}</td></tr>")
    html_parts.append(f"<tr><th>Chi-square (binned)</th><td>stat={chi2_stat:.4f}, df={chi2_df}, p={'{:.4g}'.format(chi2_p) if chi2_p is not None else 'install scipy'}</td></tr>")
    html_parts.append(f"<tr><th>KS (sample)</th><td>D={'{:.6f}'.format(ks_stat) if not math.isnan(ks_stat) else 'N/A'}, p={'{:.4g}'.format(ks_p) if ks_p is not None else 'N/A'}</td></tr>")
    html_parts.append("</table>")

    # histogram table
    html_parts.append("<h3>Histogram (binned counts)</h3>")
    html_parts.append("<table><tr><th>bin</th><th>interval</th><th>count</th></tr>")
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        html_parts.append(f"<tr><td>{i+1}</td><td>[{lo:.4f},{hi:.4f})</td><td>{bin_counts[i]:,}</td></tr>")
    html_parts.append("</table>")

    html_parts.append("<div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:12px;'>")
    if images['hist']:
        html_parts.append(f"<div style='flex:1;min-width:360px'><h4>Histogram</h4><img src='{images['hist']}' style='max-width:100%'></div>")
    if images['qq']:
        html_parts.append(f"<div style='flex:1;min-width:360px'><h4>Q-Q (sample)</h4><img src='{images['qq']}' style='max-width:100%'></div>")
    html_parts.append("</div>")

    # modular & correlations
    html_parts.append("<h2>Modular & correlation summaries</h2>")
    html_parts.append("<h3>n % 5 — counts & mean(normalized)</h3>")
    html_parts.append("<table><tr><th>residue</th><th>count</th><th>mean(normalized)</th></tr>")
    for r, (cnt, meanr) in p5_stats.items():
        html_parts.append(f"<tr><td>{r}</td><td>{cnt:,}</td><td>{meanr:.6f}</td></tr>")
    html_parts.append("</table>")

    html_parts.append("<h3>Pearson streaming estimates vs n % k</h3>")
    html_parts.append("<table><tr><th>k</th><th>Pearson r (estimate)</th></tr>")
    for k in sorted(modk_results.keys()):
        html_parts.append(f"<tr><td>{k}</td><td>{modk_results[k]:.6f}</td></tr>")
    html_parts.append("</table>")
    html_parts.append(f"<p>Serial lag-1 Pearson (estimate): r = {serial_r:.6f}</p>")

    # outliers & small values
    html_parts.append("<h2>Small-value counts & outliers</h2>")
    html_parts.append("<table><tr><th>criterion</th><th>count</th><th>proportion</th></tr>")
    html_parts.append(f"<tr><td>q_n == 0</td><td>{zeros:,}</td><td>{zeros/count if count>0 else 0:.6f}</td></tr>")
    html_parts.append(f"<tr><td>q_n &lt; 10</td><td>{lt10:,}</td><td>{lt10/count if count>0 else 0:.6f}</td></tr>")
    html_parts.append(f"<tr><td>normalized &lt; {eps}</td><td>{near0:,}</td><td>{near0/count if count>0 else 0:.6f}</td></tr>")
    html_parts.append(f"<tr><td>normalized &gt; {1-eps:.2f}</td><td>{near1:,}</td><td>{near1/count if count>0 else 0:.6f}</td></tr>")
    html_parts.append("</table>")

    html_parts.append("<h3>Outliers (sample-based)</h3>")
    if outliers_sorted:
        html_parts.append("<table><tr><th>n</th><th>q_n</th><th>normalized</th><th>zscore</th></tr>")
        for n_val, q_val, norm_val, z_val in outliers_sorted[:200]:
            html_parts.append(f"<tr><td>{n_val:,}</td><td>{q_val:,}</td><td>{norm_val:.6f}</td><td>{z_val:.3f}</td></tr>")
        html_parts.append("</table>")
    else:
        html_parts.append("<p>(no outliers above threshold in sample)</p>")

    # visualizations
    html_parts.append("<h2>Visualizations (sampled)</h2>")
    if images['scatter']:
        html_parts.append("<h3>Scatter: n vs normalized (color = n % 5)</h3>")
        html_parts.append(f"<img src='{images['scatter']}' style='max-width:100%'>")
    if images['sequence']:
        html_parts.append("<h3>Sequence (sample)</h3>")
        html_parts.append(f"<img src='{images['sequence']}' style='max-width:100%'>")
    if images['fft']:
        html_parts.append("<h3>Spectral (sample)</h3>")
        html_parts.append(f"<img src='{images['fft']}' style='max-width:100%'>")

    # reservoir sample snippet
    html_parts.append("<h2>Reservoir sample (first rows) — copy/paste friendly</h2>")
    html_parts.append("<pre style='font-family:monospace'>n,n_used,q_n,normalized</pre>")
    html_parts.append("<pre style='font-family:monospace'>")
    for (n_val, q_val, norm_val, n_used_val) in sample[:200]:
        html_parts.append(f"{n_val},{n_used_val},{q_val},{norm_val:.6f}")
    html_parts.append("</pre>")

    # mention WSS CSV location if written
    if wss_count > display_wss_limit and wrote_wss_csv:
        html_parts.append(f"<p>Full WSS composite list written to CSV: <code>{wss_csvfile}</code></p>")

    # save reservoir CSV if requested
    if csvfile:
        try:
            with open(csvfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["n", "n_used", "q_n", "normalized"])
                for (n_val, q_val, norm_val, n_used_val) in sample:
                    writer.writerow([int(n_val), int(n_used_val), int(q_val), float(norm_val)])
            html_parts.append(f"<p>Reservoir CSV written to: <code>{csvfile}</code></p>")
        except Exception as e:
            html_parts.append(f"<p>Failed to write reservoir CSV: {e}</p>")

    html_parts.append("<hr>")
    html_parts.append("<p style='font-size:0.9em;color:#666'>Notes: histogram & counts are exact by streaming; sample-based tests/plots operate on a reservoir sample. For accurate inferential p-values install SciPy (<code>pip install scipy</code>). For memory usage install psutil (<code>pip install psutil</code>).</p>")
    html_parts.append("</body></html>")

    # write HTML file
    html_text = "\n".join(html_parts)
    try:
        with open(htmlfile, "w", encoding="utf-8") as f:
            f.write(html_text)
        print(f"HTML report written: {htmlfile}")
    except Exception as e:
        print(f"Failed to write HTML: {e}")

    # If wss CSV not saved and WSS count small, try to save it (convenience)
    if wss_count > 0 and not no_wss_csv and not wrote_wss_csv:
        try:
            with open(wss_csvfile, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["n"])
                # if only head/tail present, reconstruct
                if wss_all is not None:
                    full_list = wss_all
                    for x in full_list:
                        writer.writerow([int(x)])
                else:
                    # we didn't keep full list — warn user
                    pass
            print(f"WSS CSV saved (attempt): {wss_csvfile}")
        except Exception:
            # ignore
            pass

    print(f"Done. Processed {count:,} composite numbers in {total_time:.3f}s — {comps_per_sec:.1f} comp/s, {sec_per_million:.1f}s per million.")

    return {
        'count': count,
        'wss_verified_count': wss_count,
        'wss_head': wss_head,
        'wss_tail': list(wss_tail),
        'wss_all_kept': (wss_all is not None),
        'wss_csv_written': wrote_wss_csv,
        'verification_failures': verification_failures,
        'warnings_count': warnings_count,
        'summary_time_s': total_time,
    }


# -----------------------
# CLI & helpers
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Streaming composite q_n analysis with memory-safe WSS composite handling")
    p.add_argument("--bound", "-b", type=int, help="Upper bound for composites (required). E.g., 1_000_000")
    p.add_argument("--sample-size", type=int, default=100_000, help="Reservoir sample size for plots/tests (default 100000)")
    p.add_argument("--bins", type=int, default=50, help="Histogram bins (default 50)")
    p.add_argument("--html", type=str, default=None, help="Output HTML filename (default auto report-MM-DD-YY-<bound>-composites.html)")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV filename to save reservoir sample")
    p.add_argument("--wss-csv", type=str, default=None, help="Optional CSV filename to save full WSS composites list (overrides auto name)")
    p.add_argument("--no-wss-csv", action="store_true", help="Do not auto-save full WSS list to CSV (not recommended for large lists)")
    p.add_argument("--display-wss-limit", type=int, default=5000, help="Embed full WSS list in HTML only if count <= this (default 5000)")
    p.add_argument("--wss-head", type=int, default=100, help="When truncated, show this many head entries in HTML (default 100)")
    p.add_argument("--wss-tail", type=int, default=100, help="When truncated, show this many tail entries in HTML (default 100)")
    p.add_argument("--z-threshold", type=float, default=3.0, help="Z-score threshold for outlier detection (default 3.0)")
    p.add_argument("--progress-interval", type=int, default=10_000, help="Report every N composites processed (default 10000)")
    p.add_argument("--segment-size", type=int, default=1_000_000, help="Segmented sieve block size (default 1_000_000)")
    p.add_argument("--max-composites", type=int, default=None, help="Optional cap on number of composites to process (for testing)")
    p.add_argument("--store-warnings-limit", type=int, default=1000, help="How many warning indices to store (default 1000)")
    return p.parse_args()


def prompt_for_bound():
    while True:
        try:
            raw = input("Enter upper bound x (integer ≥ 4) for composites (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        try:
            val = int(raw)
            if val >= 4:
                return val
            print("Enter integer >= 4.")
        except ValueError:
            print("Not an integer — try again.")


def main():
    args = parse_args()
    if args.bound is None:
        bound = prompt_for_bound()
    else:
        bound = args.bound

    generate_composite_report_streaming_wss(
        bound=bound,
        sample_size=args.sample_size,
        bins=args.bins,
        htmlfile=args.html,
        csvfile=args.csv,
        wss_csvfile=args.wss_csv,
        no_wss_csv=args.no_wss_csv,
        display_wss_limit=args.display_wss_limit,
        head_wss=args.wss_head,
        tail_wss=args.wss_tail,
        z_threshold=args.z_threshold,
        progress_interval=args.progress_interval,
        segment_size=args.segment_size,
        max_composites=args.max_composites,
        store_warnings_limit=args.store_warnings_limit
    )


if __name__ == "__main__":
    main()
