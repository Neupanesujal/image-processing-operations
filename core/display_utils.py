"""
core/display_utils.py
---------------------
ALL visualization lives here. Every module calls these functions.
No module should import matplotlib directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ──────────────────────────────────────────────
# 1.  Generic multi-image display
# ──────────────────────────────────────────────

def show_images(images, titles=None, cmap="gray", figsize_per=4):
    """
    Display a list of images side-by-side in a single row.

    Parameters
    ----------
    images       : list of np.ndarray (2-D grayscale)
    titles       : list of str  (optional, same length as images)
    cmap         : matplotlib colormap string
    figsize_per  : width in inches allocated per image
    """
    n = len(images)
    titles = titles or [""] * n
    fig, axes = plt.subplots(1, n, figsize=(figsize_per * n, figsize_per + 1))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 2.  Image with pixel-index grid overlay
# ──────────────────────────────────────────────

def show_with_grid(image, interval=50):
    """
    Display image with a coordinate grid overlay.
    Tick marks show pixel indices at every `interval` pixels.
    """
    h, w = image.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Image with Pixel-Index Grid", fontsize=13)

    # Ticks at every `interval` pixels
    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.grid(color="red", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Column index (x)")
    ax.set_ylabel("Row index (y)")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 3.  Histogram
# ──────────────────────────────────────────────

def plot_histogram(image, title="Histogram", color="steelblue"):
    """
    Reusable histogram. Any module may call this at any time.

    Parameters
    ----------
    image : 2-D uint8 array
    title : str  – displayed as the plot title
    color : bar color
    """
    flat = image.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=256, range=(0, 255), color=color, edgecolor="none")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Intensity value")
    ax.set_ylabel("Pixel count")
    ax.set_xlim(0, 255)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 4.  Cumulative histogram
# ──────────────────────────────────────────────

def plot_cumulative_histogram(image, title="Cumulative Histogram"):
    """Plot the cumulative distribution function of pixel intensities."""
    flat = image.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=256, range=(0, 255), cumulative=True,
            color="darkorange", edgecolor="none", density=True)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Intensity value")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, 255)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 5.  Intensity profile (row or column)
# ──────────────────────────────────────────────

def plot_intensity_profile(image, axis="row", index=None):
    """
    Plot the intensity profile along a single row or column.

    Parameters
    ----------
    image  : 2-D uint8 array
    axis   : 'row' or 'col'
    index  : int – which row/column to profile
              defaults to the middle of the image
    """
    h, w = image.shape
    if axis == "row":
        idx = index if index is not None else h // 2
        idx = max(0, min(idx, h - 1))
        profile = image[idx, :]
        label = f"Row {idx}"
        xlabel = "Column index"
    else:
        idx = index if index is not None else w // 2
        idx = max(0, min(idx, w - 1))
        profile = image[:, idx]
        label = f"Column {idx}"
        xlabel = "Row index"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(profile, color="royalblue", linewidth=1.2)
    ax.set_title(f"Intensity Profile — {label}", fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intensity (0–255)")
    ax.set_ylim(0, 255)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 6.  Padded vs unpadded comparison
# ──────────────────────────────────────────────

def show_padded_vs_unpadded(image, pad_size=20, pad_mode="reflect"):
    """
    Display the original (unpadded) image next to a padded version
    so the user can see the border extension clearly.

    Parameters
    ----------
    image    : 2-D uint8 array
    pad_size : number of pixels added on each side
    pad_mode : numpy pad mode – 'reflect', 'edge', 'constant', 'wrap', etc.
    """
    padded = np.pad(image, pad_size, mode=pad_mode)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Original  ({image.shape[1]}×{image.shape[0]})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(padded, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Padded — mode='{pad_mode}', +{pad_size}px each side\n"
        f"({padded.shape[1]}×{padded.shape[0]})",
        fontsize=11
    )
    # Draw a red rectangle showing the original image boundary
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (pad_size - 0.5, pad_size - 0.5),
        image.shape[1], image.shape[0],
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[1].add_patch(rect)
    axes[1].axis("off")

    plt.suptitle("Padding Comparison  (red border = original image area)", fontsize=13)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 7.  FFT 4-panel result display
# ──────────────────────────────────────────────

def display_fft_result(original, mask, filtered_magnitude, reconstructed,
                       filter_name="Filter"):
    """
    Standard 4-panel layout for every frequency-domain operation.

    Panels:  [Original]  [Mask]  [Filtered Spectrum]  [Reconstructed]
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    panels = [
        (original,            "Original",              "gray"),
        (mask,                f"Mask — {filter_name}", "gray"),
        (filtered_magnitude,  "Filtered Spectrum",     "inferno"),
        (reconstructed,       "Reconstructed",         "gray"),
    ]

    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle(f"Frequency Domain — {filter_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 8.  Butterworth order comparison  (3-panel)
# ──────────────────────────────────────────────

def display_butterworth_orders(original, results, cutoff, kind="Low-Pass"):
    """
    Show Butterworth results for n=1, n=2, and user-chosen n side-by-side.

    Parameters
    ----------
    original : 2-D uint8 array
    results  : list of (order, reconstructed_image) tuples
    cutoff   : the cutoff frequency used
    kind     : 'Low-Pass' or 'High-Pass'
    """
    n_panels = 1 + len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for ax, (order, img) in zip(axes[1:], results):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Butterworth {kind}\nn={order}, D₀={cutoff}", fontsize=11)
        ax.axis("off")

    plt.suptitle(f"Butterworth {kind} — Order Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 9.  Ideal vs Gaussian comparison  (3-panel)
# ──────────────────────────────────────────────

def display_ideal_vs_gaussian(original, ideal_result, gaussian_result,
                               cutoff, kind="Low-Pass"):
    """
    Side-by-side comparison of Ideal filter vs Gaussian filter result.
    Demonstrates Gibbs ringing artifacts from the ideal filter.

    Parameters
    ----------
    original        : 2-D uint8 array
    ideal_result    : reconstructed image after ideal filter
    gaussian_result : reconstructed image after Gaussian filter
    cutoff          : cutoff frequency used for both
    kind            : 'Low-Pass' or 'High-Pass'
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(ideal_result, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Ideal {kind}  (D₀={cutoff})\n⚠ May show ringing (Gibbs effect)",
        fontsize=11
    )
    axes[1].axis("off")

    axes[2].imshow(gaussian_result, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        f"Gaussian {kind}  (σ={cutoff})\nSmooth roll-off, no ringing",
        fontsize=11
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Ideal vs Gaussian {kind} Filter — Cutoff = {cutoff}\n"
        "Notice ringing/blurring differences",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.show()
