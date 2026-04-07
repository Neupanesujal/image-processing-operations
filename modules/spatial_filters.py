"""
modules/spatial_filters.py
---------------------------
Module 5 — Spatial Filtering (Order-Statistics / Non-linear Filters)
  1. Median filter
  2. Max filter
  3. Min filter
  4. Midpoint filter
  5. Alpha-trimmed mean filter

Filters use constant (zero) border — no padding display shown.
"""

import numpy as np
from scipy.ndimage import (
    median_filter,
    maximum_filter,
    minimum_filter,
    generic_filter,
)
from core.display_utils import show_images


# ──────────────────────────────────────────────
# Filter functions  (mode='constant' → zero border, no padding tricks)
# ──────────────────────────────────────────────

def apply_median(image, size):
    return median_filter(image, size=size, mode="constant", cval=0).astype(np.uint8)


def apply_max(image, size):
    return maximum_filter(image, size=size, mode="constant", cval=0).astype(np.uint8)


def apply_min(image, size):
    return minimum_filter(image, size=size, mode="constant", cval=0).astype(np.uint8)


def apply_midpoint(image, size):
    """Midpoint = (max + min) / 2 within the neighbourhood."""
    mx = maximum_filter(image.astype(np.float64), size=size, mode="constant", cval=0)
    mn = minimum_filter(image.astype(np.float64), size=size, mode="constant", cval=0)
    return np.clip((mx + mn) / 2, 0, 255).astype(np.uint8)


def apply_alpha_trimmed_mean(image, size, alpha):
    """
    Alpha-trimmed mean:
      Sort neighbourhood, drop lowest and highest alpha/2 fraction,
      average the rest.
    """
    n_pixels = size * size
    trim     = max(1, int(np.floor(alpha * n_pixels / 2)))

    def _atm(window):
        s = np.sort(window)
        return s[trim: n_pixels - trim].mean()

    result = generic_filter(
        image.astype(np.float64), _atm,
        size=size, mode="constant", cval=0
    )
    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Module runner
# ──────────────────────────────────────────────

_FILTER_MAP = {
    "1": ("Median",             apply_median,             False),
    "2": ("Max",                apply_max,                False),
    "3": ("Min",                apply_min,                False),
    "4": ("Midpoint",           apply_midpoint,           False),
    "5": ("Alpha-Trimmed Mean", apply_alpha_trimmed_mean, True),
}


def run(image):
    while True:
        print("\n── Spatial Filters ────────────────────")
        print("  1. Median filter")
        print("  2. Max filter")
        print("  3. Min filter")
        print("  4. Midpoint filter")
        print("  5. Alpha-Trimmed Mean filter")
        print("  0. Back")
        choice = input("Select: ").strip()

        if choice == "0":
            break
        if choice not in _FILTER_MAP:
            print("Invalid choice."); continue

        name, fn, needs_alpha = _FILTER_MAP[choice]

        try:
            size = int(input(f"  Mask size (odd integer, e.g. 3, 5, 7): "))
            if size % 2 == 0:
                print("  Mask size must be odd."); continue
        except ValueError:
            print("Invalid input."); continue

        alpha = None
        if needs_alpha:
            try:
                alpha = float(input("  Alpha (0–1, e.g. 0.2 trims 10% each side): "))
                alpha = max(0.0, min(alpha, 0.9))
            except ValueError:
                print("Invalid input."); continue

        if needs_alpha:
            result = fn(image, size, alpha)
            label  = f"{name} (size={size}, α={alpha})"
        else:
            result = fn(image, size)
            label  = f"{name} (size={size})"

        show_images([image, result], ["Original", label])
