"""
modules/intensity_transforms.py
--------------------------------
Module 2 — Intensity Transformations
  1. Negative
  2. Binarization
  3. Power-law (Gamma)
  4. Contrast Stretching
  5. Histogram Equalization (4 cases)
"""

import numpy as np
import matplotlib.pyplot as plt
from core.display_utils import show_images, plot_histogram


# ──────────────────────────────────────────────
# Transform functions
# ──────────────────────────────────────────────

def negative(image):
    return (255 - image).astype(np.uint8)


def binarize(image, threshold):
    return (np.where(image >= threshold, 255, 0)).astype(np.uint8)


def power_law(image, gamma, c=1.0):
    """s = c * r^gamma.  Result normalised back to [0,255]."""
    normalized = image.astype(np.float64) / 255.0
    result     = c * np.power(normalized, gamma)
    result     = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def contrast_stretch(image, r_min=None, r_max=None):
    """
    Linear contrast stretch:  s = (r - r_min) / (r_max - r_min) * 255
    Defaults to image min/max if not provided.
    """
    r_min = r_min if r_min is not None else int(image.min())
    r_max = r_max if r_max is not None else int(image.max())
    if r_max == r_min:
        return image.copy()
    stretched = (image.astype(np.float64) - r_min) / (r_max - r_min) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(image):
    """Standard histogram equalization via CDF mapping."""
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))
    cdf        = hist.cumsum()
    cdf_min    = cdf[cdf > 0].min()
    total      = image.size
    lut        = np.round((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
    return lut[image]


# ──────────────────────────────────────────────
# Synthetic images for histogram equalization demo
# ──────────────────────────────────────────────

def _make_bright(base):
    return np.clip(base.astype(np.int32) + 80, 0, 255).astype(np.uint8)

def _make_dark(base):
    return np.clip(base.astype(np.int32) - 80, 0, 255).astype(np.uint8)

def _make_low_contrast(base):
    return contrast_stretch(base, r_min=100, r_max=160)

def _make_high_contrast(base):
    # Stretch to emphasise extremes
    img = base.astype(np.float64)
    img = (img - img.mean()) * 3 + img.mean()
    return np.clip(img, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Module runner
# ──────────────────────────────────────────────

def run(image):
    while True:
        print("\n── Intensity Transformations ──────────")
        print("  1. Image Negative")
        print("  2. Binarization")
        print("  3. Power-Law (Gamma)")
        print("  4. Contrast Stretching")
        print("  5. Histogram Equalization")
        print("  0. Back")
        choice = input("Select: ").strip()

        if choice == "1":
            result = negative(image)
            show_images([image, result], ["Original", "Negative"])

        elif choice == "2":
            try:
                t = int(input("Threshold (0–255): "))
            except ValueError:
                print("Invalid input."); continue
            result = binarize(image, t)
            show_images([image, result], ["Original", f"Binary (t={t})"])

        elif choice == "3":
            try:
                gamma = float(input("Gamma value (e.g. 0.5 brightens, 2.0 darkens): "))
                c     = float(input("Correction factor c (default 1.0): ") or 1.0)
            except ValueError:
                print("Invalid input."); continue
            result = power_law(image, gamma, c)
            show_images([image, result], ["Original", f"Gamma={gamma}, c={c}"])

        elif choice == "4":
            print("Enter stretch range (leave blank for auto min/max):")
            try:
                r_min_in = input("  r_min: ").strip()
                r_max_in = input("  r_max: ").strip()
                r_min = int(r_min_in) if r_min_in else None
                r_max = int(r_max_in) if r_max_in else None
            except ValueError:
                print("Invalid input."); continue
            result = contrast_stretch(image, r_min, r_max)
            show_images([image, result], ["Original", "Contrast Stretched"])

        elif choice == "5":
            print("\n  Histogram Equalization cases:")
            print("  1. Original image")
            print("  2. Bright image")
            print("  3. Dark image")
            print("  4. Low contrast image")
            print("  5. High contrast image")
            sub = input("  Select case: ").strip()
            cases = {
                "1": ("Original",       image),
                "2": ("Bright",         _make_bright(image)),
                "3": ("Dark",           _make_dark(image)),
                "4": ("Low Contrast",   _make_low_contrast(image)),
                "5": ("High Contrast",  _make_high_contrast(image)),
            }
            if sub not in cases:
                print("Invalid choice."); continue
            label, src = cases[sub]
            result = histogram_equalization(src)
            plt.ion()
            show_images(
                [src, result],
                [f"{label} (before EQ)", f"{label} (after EQ)"]
            )
            # Show before/after histograms
            plot_histogram(src,    title=f"{label} — Before Equalization")
            plot_histogram(result, title=f"{label} — After Equalization")
            plt.ioff() 
        elif choice == "0":
            break
        else:
            print("Invalid choice.")
