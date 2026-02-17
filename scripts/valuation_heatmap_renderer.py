# Makes a heatmap of the number of WSS primes in the first 100,000 primes in universes
# With P in [1, 5] and Q in [-5, 5] \ 0
# Values symmetric across P and gets numbers from pell_wss_calculator.py, manually inputted by me

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -------------------------
# Input data (NaN = "All")
# -------------------------
data = np.array([
    [1, 3, 1, 6, 2, 0, 1, 2, 2, 1],
    [3, 4, 3, 4, 2, 2, 4, 3, 1, 2],
    [2, 2, 0, 1, 1, 0, 2, np.nan, 1, 4],
    [5, 0, 2, 1, 3, 0, np.nan, 1, np.nan, 1],
    [0, 4, 0, 2, 0, np.nan, 1, 1, 1, 3],
], dtype=float)

x_labels = ["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5"]
y_labels = ["5", "4", "3", "2", "1"]


# -------------------------
# NaN-aware smoothing (SciPy if available, otherwise small NumPy fallback)
# -------------------------
def nan_gaussian_smooth(data, sigma=1.0):
    mask = ~np.isnan(data)
    filled = np.where(mask, data, 0.0).astype(float)
    try:
        from scipy.ndimage import gaussian_filter
        blurred_filled = gaussian_filter(filled, sigma=sigma, mode='constant', cval=0.0)
        blurred_mask = gaussian_filter(mask.astype(float), sigma=sigma, mode='constant', cval=0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            result = blurred_filled / blurred_mask
        result[blurred_mask == 0] = np.nan
        return result
    except Exception:
        # small pure-NumPy kernel fallback (fine for 5x10)
        truncate = 3.0
        radius = max(1, int(truncate * sigma + 0.5))
        size = 2 * radius + 1
        x = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, x)
        kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
        kernel /= kernel.sum()
        padded_filled = np.pad(filled, pad_width=radius, mode='constant', constant_values=0.0)
        padded_mask = np.pad(mask.astype(float), pad_width=radius, mode='constant', constant_values=0.0)
        out = np.full_like(data, np.nan, dtype=float)
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                w_f = padded_filled[i:i + size, j:j + size] * kernel
                w_m = padded_mask[i:i + size, j:j + size] * kernel
                denom = w_m.sum()
                if denom > 0:
                    out[i, j] = w_f.sum() / denom
                else:
                    out[i, j] = np.nan
        return out


sigma = 1.0
smoothed = nan_gaussian_smooth(data, sigma=sigma)

# -------------------------
# Overwrite smoothed centers with raw values so peaks show at cell centers
# -------------------------
smoothed_fixed = smoothed.copy()
valid_positions = ~np.isnan(data)
smoothed_fixed[valid_positions] = data[valid_positions]  # preserve raw center values

# -------------------------
# Soft black overlay creation (distance-based alpha)
# -------------------------
nan_mask = np.isnan(data)


# distance transform (scipy if available)
def distance_to_mask(mask):
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask)  # distance to nearest True in mask
    except Exception:
        coords = np.argwhere(mask)
        rows, cols = mask.shape
        dist = np.full(mask.shape, np.inf, dtype=float)
        if coords.size == 0:
            return dist
        for i in range(rows):
            for j in range(cols):
                dmin = np.min(np.sqrt((coords[:, 0] - i) ** 2 + (coords[:, 1] - j) ** 2))
                dist[i, j] = dmin
        return dist


dist_to_nan = distance_to_mask(nan_mask)
blend_radius = 1.4
alpha_overlay = np.clip(1.0 - (dist_to_nan / blend_radius), 0.0, 1.0)
alpha_overlay[nan_mask] = 1.0
overlay = np.zeros(data.shape + (4,), dtype=float)  # RGBA
overlay[..., 3] = alpha_overlay  # black RGB with variable alpha

# -------------------------
# Plot
# -------------------------
cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#2b83ba", "#ffffbf", "#d7191c"])
vmin = 0.0
vmax = max(6.0, float(np.nanmax(data)))  # force 6 -> top color
norm = Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(10, 4.5))
img = ax.imshow(smoothed_fixed, cmap=cmap, norm=norm, interpolation='bicubic', origin='upper', aspect='auto')
ax.imshow(overlay, interpolation='bicubic', origin='upper', aspect='auto', zorder=3)

# ticks / grid
ax.set_xticks(np.arange(len(x_labels)));
ax.set_xticklabels(x_labels)
ax.set_yticks(np.arange(len(y_labels)));
ax.set_yticklabels(y_labels)
ax.set_xticks(np.arange(-.5, len(x_labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(y_labels), 1), minor=True)
ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# annotations
rows, cols = data.shape
for i in range(rows):
    for j in range(cols):
        val = data[i, j]
        if np.isnan(val):
            ax.text(j, i, "Degen", ha="center", va="center", fontsize=10, fontweight='bold', color='white', zorder=5)
        else:
            txt = str(int(val)) if float(val).is_integer() else f"{val:.2f}"
            sm_val = smoothed_fixed[i, j]
            mapped = (sm_val - vmin) / (vmax - vmin) if vmax > vmin and not np.isnan(sm_val) else 0.0
            text_color = "white" if mapped > 1 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight='bold', color=text_color, zorder=5)

cbar = fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label('Cases', rotation=270, labelpad=12)
cbar.set_ticks(np.arange(vmin, vmax + 1))

ax.set_xlabel("Q")
ax.set_ylabel("P")

ax.set_title("Heatmap of Number of Wall-Sun-Sun Primes up to 1000000 in Different Universes")
plt.tight_layout()
plt.show()
