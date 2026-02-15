# makes an HTML report (I think), I don't really remember what this one does
# not used?

from __future__ import annotations
import argparse
import base64
import csv
import io
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from heapq import heappush, heappushpop, nlargest
from typing import Iterator, List, Tuple, Optional

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
# Number-theory helpers
# -----------------------
def simple_sieve(n: int) -> List[int]:
    """Sieve primes up to n (n small, used to seed segmented sieve)."""
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


def segmented_sieve(bound: int, segment_size: int = 1_000_000) -> Iterator[int]:
    """
    Yield primes up to bound using a segmented sieve.
    Memory ~ segment_size bytes + primes up to sqrt(bound).
    """
    if bound < 2:
        return
    limit = int(math.isqrt(bound)) + 1
    small_primes = simple_sieve(limit)
    yield from (p for p in small_primes if p <= bound) if limit <= bound else (p for p in small_primes if p <= bound)
    # segments [low, high)
    low = max(limit, 2)
    while low <= bound:
        high = min(low + segment_size - 1, bound)
        size = high - low + 1
        segment = bytearray(b"\x01") * size
        for p in small_primes:
            if p * p > high:
                break
            # find first multiple >= low
            start = ((low + p - 1) // p) * p
            for m in range(start, high + 1, p):
                segment[m - low] = 0
        for i in range(size):
            if segment[i]:
                yield low + i
        low += segment_size


def legendre_p_over_5(p: int) -> int:
    if p == 5:
        return 0
    r = p % 5
    return 1 if r in (1, 4) else -1


def fib_fast_doubling_mod(n: int, mod: int) -> Tuple[int, int]:
    """Fast doubling (recursive) returning (F_n mod mod, F_{n+1} mod mod)."""
    if n == 0:
        return 0 % mod, 1 % mod
    a, b = fib_fast_doubling_mod(n >> 1, mod)
    c = (a * ((b << 1) - a)) % mod
    d = (a * a + b * b) % mod
    if n & 1:
        return d, (c + d) % mod
    else:
        return c, d


def compute_qp_for_prime(p: int) -> Tuple[int, bool, int]:
    """
    Compute q_p = (F_{p - (p/5)} / p) mod p using F_n mod p^2.
    Returns (q_p, warned, n_used).
    """
    chi = legendre_p_over_5(p)
    n = p - chi
    mod = p * p
    f_n_mod_p2, _ = fib_fast_doubling_mod(n, mod)
    quotient_mod_p = (f_n_mod_p2 // p) % p
    warned = (f_n_mod_p2 % p != 0)
    return int(quotient_mod_p), bool(warned), int(n)


# -----------------------
# Utilities: reservoir sampling, fig->base64
# -----------------------
def reservoir_sample_iter(it, k: int):
    """Yield a reservoir sample of size k from iterator `it` (returns list)."""
    sample = []
    for t, item in enumerate(it, start=1):
        if t <= k:
            sample.append(item)
        else:
            j = random.randrange(0, t)
            if j < k:
                sample[j] = item
    return sample


def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# -----------------------
# Online stats helpers
# -----------------------
class Welford:
    """Online mean/variance (population variance by default)."""
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
        var = self.m2 / self.n  # population var
        return self.mean, math.sqrt(var)

    def sample_variance(self):
        if self.n < 2:
            return math.nan
        return self.m2 / (self.n - 1)


# For online Pearson between x and y we keep sums
class OnlinePearson:
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
# Main streaming analysis
# -----------------------
def generate_report_streaming(bound: int,
                              sample_size: int = 100_000,
                              bins: int = 50,
                              htmlfile: Optional[str] = None,
                              csvfile: Optional[str] = None,
                              z_threshold: float = 3.0,
                              progress_interval: int = 10_000,
                              segment_size: int = 1_000_000,
                              max_primes: Optional[int] = None):
    start_time = time.perf_counter()
    if htmlfile is None:
        now = datetime.now()
        date_part = now.strftime("%m-%d-%y")
        htmlfile = f"report-{date_part}-{bound:,}.html".replace(",", "")
    print(f"Output HTML: {htmlfile}")

    # Prepare iterators
    prime_iter = segmented_sieve(bound, segment_size=segment_size)

    # State for streaming stats
    n_processed = 0
    welf = Welford()
    zeros = 0
    lt10 = 0
    near0 = 0
    near1 = 0
    eps = 0.01

    # histogram bins counts (equal-width on [0,1))
    bin_counts = [0] * bins

    # p%5 grouping sums & counts
    p5_counts = defaultdict(int)
    p5_sums = defaultdict(float)

    # Pearson online vs p%k for k=2..10
    modk_online = {k: OnlinePearson() for k in range(2, 11)}

    # Pearson online between successive normalized values (serial correlation)
    serial_online = OnlinePearson()
    prev_norm = None

    # reservoir sample for tests/plots (store tuples (p, q, normalized))
    sample = []
    sample_t = 0

    # keep detected warnings
    warnings_list = []

    # keep top outliers by absolute deviation using reservoir-sample-based approach:
    # we'll extract outliers from the sample later. For streaming detection we also
    # maintain a small heap of largest q (in absolute normalized) as *candidates*
    top_k_outlier_heap = []  # min-heap of (abs(normalized), p, q, normalized)
    top_k_size = 200

    # performance tracking
    last_report_time = start_time
    times_checkpoint = [start_time]

    # We need small_primes for segmented sieve - handled inside segmented_sieve
    # Iterate primes streamingly
    for p in prime_iter:
        n_processed += 1
        if max_primes is not None and n_processed > max_primes:
            break

        q, warned, n_used = compute_qp_for_prime(p)
        if warned:
            warnings_list.append(p)

        norm = q / p  # normalized in [0,1)
        # update streaming stats
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
        bin_idx = min(int(norm * bins), bins - 1)
        bin_counts[bin_idx] += 1

        # p%5 grouping
        mod5 = p % 5
        p5_counts[mod5] += 1
        p5_sums[mod5] += norm

        # modk online Pearson
        for k, online in modk_online.items():
            online.add(norm, float(p % k))

        # serial correlation (lag-1)
        if prev_norm is not None:
            serial_online.add(prev_norm, norm)
        prev_norm = norm

        # maintain reservoir sample for visuals & sample tests
        sample_t += 1
        if len(sample) < sample_size:
            sample.append((p, q, norm))
        else:
            # reservoir replacement
            j = random.randrange(0, sample_t)
            if j < sample_size:
                sample[j] = (p, q, norm)

        # maintain top-k outlier candidates (based on abs dev from current mean)
        # Note: mean changes over time; this is heuristic. We'll recompute final z-scores on sample.
        # Push small heap of largest |norm - mean_so_far|
        mean_so_far = welf.mean
        absdev = abs(norm - mean_so_far)
        if len(top_k_outlier_heap) < top_k_size:
            heappush(top_k_outlier_heap, (absdev, p, q, norm))
        else:
            if absdev > top_k_outlier_heap[0][0]:
                heappushpop(top_k_outlier_heap, (absdev, p, q, norm))

        # periodic progress & memory
        if (n_processed % progress_interval) == 0:
            now = time.perf_counter()
            elapsed = now - start_time
            per_sec = n_processed / elapsed if elapsed > 0 else float('inf')
            per_million = (elapsed / n_processed) * 1_000_000 if n_processed > 0 else float('nan')
            mem_info = ""
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                rss = process.memory_info().rss / (1024 ** 2)
                mem_info = f", RSS={rss:.1f} MB"
            print(f"[{n_processed:,}] primes processed — {per_sec:.1f} p/s, {per_million:.1f} s per million{mem_info}")
            times_checkpoint.append(now)

    total_time = time.perf_counter() - start_time
    primes_per_sec = n_processed / total_time if total_time > 0 else float('inf')
    sec_per_million = (total_time / n_processed) * 1_000_000 if n_processed > 0 else float('nan')

    # finalize summaries
    mean_norm, std_norm = welf.result()
    count_zero = zeros
    count_lt_10 = lt10

    # prepare sample-based statistics (use reservoir)
    sample_arr = np.array([s[2] for s in sample], dtype=float) if sample else np.array([], dtype=float)
    sample_ps = [s[0] for s in sample]
    sample_qs = [s[1] for s in sample]

    # chi-square on histogram counts
    expected = n_processed / bins if n_processed > 0 else 0
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
            # approximate
            # small-sample approx on sample of size m
            ks_p = None
    else:
        ks_stat, ks_p = float('nan'), None

    # Pearson vs p%5 (exact via streaming sums)
    pear_p5_val = None
    spear_p5_val = None
    # compute mean per residue p%5
    p5_stats = {}
    for r in sorted(p5_counts.keys()):
        cnt = p5_counts[r]
        meanr = (p5_sums[r] / cnt) if cnt > 0 else float('nan')
        p5_stats[r] = (cnt, meanr)

    # compute modk Pearson from online sums
    modk_results = {}
    for k, online in modk_online.items():
        r = online.result()
        modk_results[k] = r

    # serial lag-1 correlation from serial_online sums
    serial_r = serial_online.result()

    # Outliers: examine reservoir & top heap candidates to create list
    outlier_candidates = []
    # use final mean/std to compute z on sample and on heap candidates
    if not math.isnan(std_norm) and std_norm != 0:
        # check sampled points
        for (p, q, norm) in sample:
            z = (norm - mean_norm) / std_norm
            if abs(z) > z_threshold:
                outlier_candidates.append((abs(z), p, q, norm, z))
        # check heap candidates too (might have points not in sample)
        for absdev, p, q, norm in top_k_outlier_heap:
            z = (norm - mean_norm) / std_norm if std_norm != 0 else float('nan')
            if abs(z) > z_threshold:
                outlier_candidates.append((abs(z), p, q, norm, z))
        # unique by p
        outlier_by_p = {}
        for rec in outlier_candidates:
            _, p, q, norm, z = rec
            if p not in outlier_by_p or abs(z) > abs(outlier_by_p[p][3]):
                outlier_by_p[p] = (q, norm, z)
        # convert to sorted list
        outliers_sorted = sorted([(p, q, norm, z) for p, (q, norm, z) in outlier_by_p.items()], key=lambda x: -abs(x[3]))
    else:
        outliers_sorted = []

    # Prepare images from sample (downsample for visuals if sample is large)
    viz_sample_limit = min(len(sample), 100_000)
    if len(sample) > viz_sample_limit:
        viz_sample = random.sample(sample, viz_sample_limit)
    else:
        viz_sample = list(sample)

    viz_norms = np.array([s[2] for s in viz_sample], dtype=float) if viz_sample else np.array([], dtype=float)
    viz_ps = np.array([s[0] for s in viz_sample], dtype=int) if viz_sample else np.array([], dtype=int)

    images = {}
    # histogram plot (from bin_counts)
    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.linspace(0, 1, bins, endpoint=False), bin_counts, width=1.0 / bins, align='edge', edgecolor='black')
    ax.set_xlabel("Normalized q_p / p")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram (binned) of normalized q_p (n={n_processed:,})")
    images['hist'] = fig_to_base64_png(fig)

    # QQ plot using sample
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

    # scatter p vs normalized colored by p%5 (sample)
    if viz_norms.size > 0:
        fig = plt.figure(figsize=(10, 3.6))
        ax = fig.add_subplot(1, 1, 1)
        colors = (viz_ps % 5)
        sc = ax.scatter(viz_ps, viz_norms, c=colors, cmap='tab10', s=6, alpha=0.6)
        ax.set_xlabel("prime p (sample)")
        ax.set_ylabel("normalized q_p / p")
        ax.set_title("normalized q_p vs p (color = p % 5) [sample]")
        plt.colorbar(sc, ax=ax, ticks=range(5), label="p % 5")
        images['scatter'] = fig_to_base64_png(fig)
    else:
        images['scatter'] = ""

    # sequence plot sample
    if viz_norms.size > 0:
        fig = plt.figure(figsize=(10, 2.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(viz_norms.size), viz_norms, marker='.', markersize=2, linestyle='-', alpha=0.6)
        ax.set_xlabel("sample index")
        ax.set_ylabel("normalized q_p/p (sample)")
        ax.set_title("Sequence (sample)")
        images['sequence'] = fig_to_base64_png(fig)
    else:
        images['sequence'] = ""

    # FFT on sample
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

    # Build compact HTML (only sample plots + summary tables)
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append(f"<title>Wall–Sun–Sun q_p report (bound={bound:,})</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:18px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;}th{background:#f4f7fb;}code{font-family:monospace;}</style>")
    html.append("</head><body>")
    html.append(f"<h1>Wall–Sun–Sun q_p analysis — bound={bound:,}</h1>")
    html.append(f"<p>Generated: {now}</p>")
    html.append("<h2>Performance</h2>")
    html.append("<table>")
    html.append(f"<tr><th>Primes processed</th><td>{n_processed:,}</td></tr>")
    html.append(f"<tr><th>Total time (s)</th><td>{total_time:.3f}</td></tr>")
    html.append(f"<tr><th>Primes / sec</th><td>{primes_per_sec:.1f}</td></tr>")
    html.append(f"<tr><th>Seconds per million primes</th><td>{sec_per_million:.1f}</td></tr>")
    if PSUTIL_AVAILABLE:
        mem = psutil.Process().memory_info().rss / (1024 ** 2)
        html.append(f"<tr><th>Memory RSS (MB)</th><td>{mem:.1f}</td></tr>")
    else:
        html.append("<tr><th>Memory RSS (MB)</th><td>psutil not installed</td></tr>")
    html.append("</table>")

    html.append("<h2>Distribution & sample-based tests</h2>")
    html.append("<table>")
    html.append(f"<tr><th>Sample size used for tests/plots</th><td>{len(sample)}</td></tr>")
    html.append(f"<tr><th>Chi-square (binned)</th><td>stat={chi2_stat:.4f}, df={chi2_df}, p={'{:.4g}'.format(chi2_p) if chi2_p is not None else 'install scipy'}</td></tr>")
    html.append(f"<tr><th>KS (on sample)</th><td>D={'{:.6f}'.format(ks_stat) if not math.isnan(ks_stat) else 'N/A'}, p={'{:.4g}'.format(ks_p) if ks_p is not None else 'N/A'}</td></tr>")
    html.append("</table>")

    html.append("<h3>Histogram (binned counts)</h3>")
    html.append("<table><tr><th>bin</th><th>interval</th><th>count</th></tr>")
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        html.append(f"<tr><td>{i+1}</td><td>[{lo:.4f},{hi:.4f})</td><td>{bin_counts[i]:,}</td></tr>")
    html.append("</table>")

    html.append("<div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:12px;'>")
    if images['hist']:
        html.append(f"<div style='flex:1;min-width:360px'><h4>Histogram</h4><img src='{images['hist']}' style='max-width:100%'></div>")
    if images['qq']:
        html.append(f"<div style='flex:1;min-width:360px'><h4>Q-Q (sample)</h4><img src='{images['qq']}' style='max-width:100%'></div>")
    html.append("</div>")

    html.append("<h2>Modular & correlation summaries</h2>")
    html.append("<h3>p % 5 — counts & mean(normalized)</h3>")
    html.append("<table><tr><th>residue</th><th>count</th><th>mean(normalized)</th></tr>")
    for r in sorted(p5_stats.keys()):
        cnt, meanr = p5_stats[r]
        html.append(f"<tr><td>{r}</td><td>{cnt:,}</td><td>{meanr:.6f}</td></tr>")
    html.append("</table>")

    html.append("<h3>Pearson correlations (streaming estimates)</h3>")
    html.append("<table><tr><th>k</th><th>Pearson r (normalized vs p%k)</th></tr>")
    for k in sorted(modk_results.keys()):
        html.append(f"<tr><td>{k}</td><td>{modk_results[k]:.6f}</td></tr>")
    html.append("</table>")
    html.append(f"<p>Serial lag-1 Pearson (estimate): r = {serial_r:.6f}</p>")

    html.append("<h2>Small-value & outlier summary</h2>")
    html.append("<table><tr><th>criterion</th><th>count</th><th>proportion</th></tr>")
    html.append(f"<tr><td>q_p == 0</td><td>{count_zero:,}</td><td>{count_zero/n_processed:.6f}</td></tr>")
    html.append(f"<tr><td>q_p &lt; 10</td><td>{count_lt_10:,}</td><td>{count_lt_10/n_processed:.6f}</td></tr>")
    html.append(f"<tr><td>normalized &lt; {eps}</td><td>{near0:,}</td><td>{near0/n_processed:.6f}</td></tr>")
    html.append(f"<tr><td>normalized &gt; {1-eps:.2f}</td><td>{near1:,}</td><td>{near1/n_processed:.6f}</td></tr>")
    html.append("</table>")

    html.append("<h3>Outliers (from sample & candidates)</h3>")
    if outliers_sorted:
        html.append("<table><tr><th>p</th><th>q_p</th><th>normalized</th><th>zscore</th></tr>")
        for p, q, norm, z in outliers_sorted[:200]:
            html.append(f"<tr><td>{p:,}</td><td>{q:,}</td><td>{norm:.6f}</td><td>{z:.3f}</td></tr>")
        html.append("</table>")
    else:
        html.append("<p>(no outliers in sample above threshold)</p>")

    html.append("<h2>Visualizations (sampled)</h2>")
    if images['scatter']:
        html.append("<h3>Scatter: p vs normalized (color = p % 5)</h3>")
        html.append(f"<img src='{images['scatter']}' style='max-width:100%'>")
    if images['sequence']:
        html.append("<h3>Sequence (sample)</h3>")
        html.append(f"<img src='{images['sequence']}' style='max-width:100%'>")
    if images['fft']:
        html.append("<h3>Spectral (sample)</h3>")
        html.append(f"<img src='{images['fft']}' style='max-width:100%'>")

    # Samples (first N of the reservoir only) for copy/paste
    html.append("<h2>Sample (first reservoir rows) — copy/paste friendly</h2>")
    html.append("<pre style='font-family:monospace'>p,n_used,q_p,normalized</pre>")
    html.append("<pre style='font-family:monospace'>")
    for i, (p, q, norm) in enumerate(sample[:200]):
        html.append(f"{p},{n_used if i==0 else ''},{q},{norm:.6f}")
    html.append("</pre>")

    # Save CSV of the reservoir sample if requested
    if csvfile:
        try:
            with open(csvfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["p", "q_p", "normalized"])
                for p, q, norm in sample:
                    writer.writerow([int(p), int(q), float(norm)])
            html.append(f"<p>CSV written to: <code>{csvfile}</code></p>")
        except Exception as e:
            html.append(f"<p>Failed to write CSV: {e}</p>")

    html.append("<hr>")
    html.append("<p style='font-size:0.9em;color:#666'>Notes: sample-based tests (KS, Q-Q, FFT) operate on a reservoir sample; "
                "histogram & counts are exact via streaming binning. For full inferential p-values, install SciPy: <code>pip install scipy</code>.</p>")
    html.append("</body></html>")

    # Write HTML file
    try:
        with open(htmlfile, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"HTML report written: {htmlfile}")
    except Exception as e:
        print(f"Failed to write HTML: {e}")

    # final performance print
    print(f"Done. Processed {n_processed:,} primes in {total_time:.3f}s — {primes_per_sec:.1f} p/s, {sec_per_million:.1f}s per million.")


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Streaming Wall–Sun–Sun q_p report (HTML, sample-based visuals, scalable)")
    p.add_argument("--bound", "-b", type=int, help="Upper bound for primes (required)")
    p.add_argument("--sample-size", type=int, default=100_000, help="Reservoir sample size for plots/tests (default 100000)")
    p.add_argument("--bins", type=int, default=50, help="Histogram bins (default 50)")
    p.add_argument("--html", type=str, default=None, help="Output HTML filename (default auto report-MM-DD-YY-N.html)")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV filename to save reservoir sample")
    p.add_argument("--z-threshold", type=float, default=3.0, help="Z-score threshold for outlier detection (default 3.0)")
    p.add_argument("--progress-interval", type=int, default=10_000, help="Report every N primes (default 10000)")
    p.add_argument("--segment-size", type=int, default=1_000_000, help="Segmented sieve block size (default 1_000_000)")
    p.add_argument("--max-primes", type=int, default=None, help="Optional cap on number of primes to process (for testing)")
    return p.parse_args()


def prompt_for_bound():
    while True:
        try:
            raw = input("Enter upper bound x (integer ≥ 2) for primes (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        try:
            val = int(raw)
            if val >= 2:
                return val
            print("Enter integer >= 2.")
        except ValueError:
            print("Not an integer — try again.")


def main():
    args = parse_args()
    if args.bound is None:
        bound = prompt_for_bound()
    else:
        bound = args.bound

    generate_report_streaming(bound=bound,
                              sample_size=args.sample_size,
                              bins=args.bins,
                              htmlfile=args.html,
                              csvfile=args.csv,
                              z_threshold=args.z_threshold,
                              progress_interval=args.progress_interval,
                              segment_size=args.segment_size,
                              max_primes=args.max_primes)


if __name__ == "__main__":
    main()

